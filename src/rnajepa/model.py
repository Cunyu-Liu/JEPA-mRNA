"""RNA-JEPA: region-aware factorised latent prediction on top of mRNABERT.

Design (architecture.md v1.0), all pieces in one place so the tensor flow is
readable end to end::

    tokens + region ids
      |
      +-- student E_theta (mRNABERT-isomorphic, MLM head kept)
      |     |- H_s = E_theta(masked input)            -> L_MLM
      |     '- H'  = H_s with masked positions overwritten by the [MASK]
      |              embedding (JEPA-DNA re-masking: kills the identity path)
      |
      +-- predictor P_phi (4 x linear-attention blocks, O(L))
      |     input  = [H' ; region descriptor(one-hot region, norm. region
      |               length, relative position in region)]
      |     output = u_hat_r, the predicted summary of each region
      |
      +-- teacher E_thetabar (EMA of student, no grad)
      |     |- z_r  = W_r . meanpool(H_t[region r])   (region-specific W_r)
      |     '- z_cls = H_t[CLS]
      |
      +-- shared orthonormal factor basis P_k (k = 0..K-1, K*r_dim = d)
            u_hat^(k) = P_k^T u_hat ,  z^(k) = P_k^T z
            L_JEPA = sum_r sum_k w_r(t) [1 - cos(u_hat_r^(k), z_r^(k))]
                     + w_cls(t) [1 - cos(u_hat_cls, z_cls)]

    L = L_MLM + alpha(t) L_JEPA + lambda_var L_var + lambda_cov L_cov + L_orth

Two deliberate, documented deviations from architecture.md:

* **d = 768, K = 4, r_dim = 192** rather than the §7 starting table's
  512/4/128.  The backbone is fixed to mRNABERT's hidden size (768) by the
  "isomorphic backbone" requirement, and OPF factorisation requires
  ``K * r_dim == d``.  The §7 numbers come from JEPA-Anything's Norman setup
  and were never compatible with a 768-d backbone.
* The variance/covariance anti-collapse term is computed on a second forward
  pass run in **eval mode** (dropout and the stochastic mask switched off) as
  specified, but on the same batch rather than a batch truncated to a common
  length.  Truncating would change which tokens the statistic is measured over,
  and the eval-mode pass already removes the noise the specification was
  guarding against.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from rnajepa.tokenization import REGION_3UTR, REGION_5UTR, REGION_CDS, REGION_NONE

N_REGIONS = 3
REGION_ORDER = (REGION_5UTR, REGION_CDS, REGION_3UTR)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class JEPAConfig:
    model_path: str = ""
    mask_token_id: Optional[int] = None   # pass tokenizer.mask_token_id
    mask_prob: float = 0.15
    mask_mode: str = "token_uniform"      # token_uniform | codon_span
    codon_span_min: int = 2
    codon_span_max: int = 5
    # Which latent target the predictor is trained against.
    #   "region" -> pooled summary of each 5'UTR/CDS/3'UTR (architecture §3)
    #   "masked" -> teacher hidden states at the masked positions (JEPA-DNA /
    #               ProteinJEPA "masked-position latent prediction" recipe)
    #   "both"   -> sum of the two
    #   "cls"    -> only the global [CLS] objective.  With n_factors=1 this is the
    #               single-vector JEPA-DNA minimal arm used as ablation A1, i.e. the
    #               mixed objective without any region factorisation.
    #   "none"   -> no latent objective at all: the pure MLM baseline (ablation A0).
    #               The teacher, predictor and regularisers are skipped entirely so
    #               A0 measures MLM-only at the same step and data budget.
    # The A3 ablation crosses this with the loss form.
    jepa_target: str = "region"
    loss_form: str = "cos"                # cos | mse
    n_factors: int = 4                    # K
    predictor_layers: int = 4
    predictor_heads: int = 12
    predictor_dropout: float = 0.05
    ema_start: float = 0.996
    ema_end: float = 0.9997
    alpha: float = 0.5                    # base weight of L_JEPA
    lambda_var: float = 25.0
    lambda_cov: float = 0.5
    lambda_orth: float = 1.0
    curriculum_frac: float = 0.6          # first 60% of steps: w_r = 1.0
    w_region_start: float = 1.0
    w_region_end: float = 0.2
    w_cls_start: float = 0.2
    w_cls_end: float = 1.0
    normalize_jepa_weights: bool = True


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #
class BertBackbone(nn.Module):
    """Thin wrapper exposing the released encoder and its tied MLM head.

    ``BertForMaskedLM.forward`` only returns hidden states *at masked
    positions* (it passes ``masked_tokens_mask`` down into the encoder), which
    is useless for a region-pooled objective.  We therefore call
    ``self.bert`` and ``self.cls`` directly: ``BertLMPredictionHead`` is
    position-wise (LayerNorm + dense + tied decoder), so computing logits for
    every position and then selecting the masked ones gives exactly the same
    loss as the official subset computation.
    """

    def __init__(self, hf_model):
        super().__init__()
        self.bert = hf_model.bert
        self.cls = hf_model.cls

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return outputs[0]

    def mlm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.cls(hidden)

    @property
    def word_embeddings(self) -> nn.Embedding:
        return self.bert.embeddings.word_embeddings


class LinearAttentionBlock(nn.Module):
    """ELU+1 linear attention block: O(L) time and memory in sequence length.

    CDS sequences are thousands of tokens, so a quadratic attention in the
    predictor would dominate memory (BioM-JEPA's recipe, architecture §4.4).
    """

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        assert dim % heads == 0, "predictor dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.drop = nn.Dropout(dropout)

    def _attn(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        h, dh = self.heads, self.head_dim
        qkv = self.qkv(x).view(b, l, 3, h, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]              # b h l dh

        phi_q = F.elu(q) + 1.0
        phi_k = F.elu(k) + 1.0
        if pad_mask is not None:
            phi_k = phi_k * pad_mask[:, None, :, None].to(phi_k.dtype)

        kv = torch.einsum("bhld,bhle->bhde", phi_k, v)          # b h dh dh
        z = torch.einsum("bhld,bhde->bhle", phi_q, kv)          # b h l dh
        denom = torch.einsum("bhld,bhd->bhl", phi_q, phi_k.sum(dim=2))
        out = z / denom.clamp_min(1e-6)[..., None]
        out = out.permute(0, 2, 1, 3).reshape(b, l, h * dh)
        return self.out(out)

    def forward(self, x: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self._attn(self.norm1(x), pad_mask))
        x = x + self.drop(self.ff(self.norm2(x)))
        return x


class RegionPredictor(nn.Module):
    """Maps the re-masked student states to per-region summary predictions.

    The region descriptor (one-hot region, normalised region length, relative
    position inside the region) is projected and added to every token so the
    predictor knows *which* summary each token contributes to.
    """

    def __init__(self, dim: int, cfg: JEPAConfig):
        super().__init__()
        self.descriptor = nn.Sequential(
            nn.Linear(N_REGIONS + 2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(
            [LinearAttentionBlock(dim, cfg.predictor_heads, cfg.predictor_dropout)
             for _ in range(cfg.predictor_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, hidden: torch.Tensor, descriptors: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = hidden + self.descriptor(descriptors)
        for blk in self.blocks:
            x = blk(x, pad_mask)
        return self.norm(x)


class FactorBasis(nn.Module):
    """OPF factor basis: K shared projections P_k with P_k^T P_j ~ delta_kj.

    ``r_dim = d // K`` so the K blocks tile the hidden space exactly.  The
    orthogonality penalty keeps the factors non-redundant; the activity penalty
    keeps them from collapsing to zero (architecture §4.7, JEPA-Anything OPF).
    """

    def __init__(self, dim: int, n_factors: int):
        super().__init__()
        assert dim % n_factors == 0, "hidden size must be divisible by K"
        self.dim = dim
        self.n_factors = n_factors
        self.r_dim = dim // n_factors
        init = torch.zeros(n_factors, dim, self.r_dim)
        for k in range(n_factors):
            block = torch.zeros(dim, self.r_dim)
            block[k * self.r_dim:(k + 1) * self.r_dim] = torch.eye(self.r_dim)
            init[k] = block
        self.P = nn.Parameter(init.clone())

    def project(self, z: torch.Tensor) -> torch.Tensor:
        """``z`` is [..., d] -> [..., K, r_dim]."""
        return torch.einsum("...d,kdr->...kr", z, self.P)

    def orthogonality_loss(self) -> torch.Tensor:
        """``P_k^T P_k -> I`` for every block and ``P_k^T P_j -> 0`` across blocks.

        Two distinct residuals: the on-diagonal blocks are pulled towards the
        identity, the off-diagonal blocks towards zero.  Applying the identity
        correction to the off-diagonal blocks as well would wrongly reward
        ``P_k^T P_j = I``, which is the opposite of factor independence.
        """
        gram = torch.einsum("kdr,jde->kjre", self.P, self.P)      # K K r r
        k = self.n_factors
        eye = torch.eye(self.r_dim, device=gram.device, dtype=gram.dtype)
        same = torch.eye(k, device=gram.device, dtype=gram.dtype).bool()
        target = eye[None, None].expand(k, k, self.r_dim, self.r_dim).clone()
        target[~same] = 0.0
        return (gram - target).pow(2).mean()

    def activity(self, z_proj: torch.Tensor) -> torch.Tensor:
        """Mean activation energy per factor, for collapse monitoring."""
        return z_proj.pow(2).mean(dim=tuple(range(z_proj.dim() - 2)))


def region_descriptors(region_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Build the [B, L, N_REGIONS + 2] descriptor tensor.

    Columns: one-hot region, normalised position of the token inside its region,
    normalised length of the region.  PAD / [CLS] / [SEP] positions get an
    all-zero descriptor; they never contribute to a region summary.
    """
    b, l = region_ids.shape
    dev = region_ids.device
    desc = torch.zeros(b, l, N_REGIONS + 2, device=dev, dtype=torch.float32)

    for r_idx, region in enumerate(REGION_ORDER):
        in_region = (region_ids == region) & attention_mask.bool()
        desc[..., r_idx] = in_region.float()
        counts = in_region.sum(dim=1, keepdim=True).clamp_min(1)          # B 1
        idx = torch.arange(l, device=dev)[None, :].expand(b, l)
        # index of the token within its region == how many region tokens precede it
        cum = (in_region.cumsum(dim=1) - 1).clamp_min(0)
        rel = torch.where(in_region, cum.float() / counts.float().clamp_min(1), torch.zeros_like(idx, dtype=torch.float32))
        desc[..., N_REGIONS] += rel * in_region.float()
        desc[..., N_REGIONS + 1] += (counts.float() / float(l)) * in_region.float()
    return desc


def pool_by_region(hidden: torch.Tensor, region_ids: torch.Tensor,
                   attention_mask: torch.Tensor, region: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool ``hidden`` over positions of ``region``.

    Returns ``(pooled [B, d], present [B] bool)``.  Sequences without that
    region (a pure-CDS construct has no UTR) are flagged and skipped by the
    caller instead of being pooled from an empty set.
    """
    mask = (region_ids == region) & attention_mask.bool()
    present = mask.any(dim=1)
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    pooled = (hidden * weights).sum(dim=1) / denom
    return pooled, present


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - cos, reduced over the last dimension, kept per-sample."""
    pred = F.normalize(pred, dim=-1, eps=1e-6)
    target = F.normalize(target.detach(), dim=-1, eps=1e-6)
    return 1.0 - (pred * target).sum(dim=-1)


def variance_covariance_loss(z: torch.Tensor, eps: float = 1e-4) -> Tuple[torch.Tensor, torch.Tensor]:
    """VICReg-style variance + covariance regularisers on a [B, d] embedding."""
    if z.shape[0] < 2:
        zero = z.sum() * 0.0
        return zero, zero
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(1.0 - std).mean()
    z_centered = z - z.mean(dim=0, keepdim=True)
    n = z.shape[0] - 1
    cov = (z_centered.T @ z_centered) / max(n, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off_diag.pow(2).sum() / z.shape[1]
    return var_loss, cov_loss


def procrustes_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Orthogonal Procrustes alignment quality between two [n, d] matrices.

    ``||A_hat - B||_F / ||B||_F`` after the optimal rotation/reflection; 0 means
    perfectly alignable.  BioM-JEPA uses this as the teacher/student alignment
    monitor (Fig.6b material).
    """
    if a.shape[0] < 2:
        return float("nan")
    a = F.normalize(a, dim=-1)
    b = F.normalize(b, dim=-1)
    u, _, vh = torch.linalg.svd(a.T @ b, full_matrices=False)
    r = u @ vh
    residual = torch.linalg.norm(a @ r - b) / torch.linalg.norm(b).clamp_min(1e-8)
    return float(residual)


# --------------------------------------------------------------------------- #
# Masking
# --------------------------------------------------------------------------- #
def sample_mask_positions(region_ids: torch.Tensor, attention_mask: torch.Tensor,
                          special_ids: torch.Tensor, cfg: JEPAConfig,
                          generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Choose positions to mask, at ``cfg.mask_prob`` over eligible tokens.

    Eligible = real tokens (not [CLS]/[SEP]/[PAD]).  In ``token_uniform`` mode a
    codon is masked as a unit (that *is* the "codon-aware" behaviour of the
    codon tokenisation).  In ``codon_span`` mode CDS codons are additionally
    masked in contiguous spans, which is the arm used for ablation A4.
    """
    b, l = region_ids.shape
    dev = region_ids.device
    eligible = attention_mask.bool() & ~special_ids.bool()
    mask = torch.zeros(b, l, dtype=torch.bool, device=dev)

    if cfg.mask_mode == "token_uniform":
        rand = torch.rand(b, l, device=dev, generator=generator)
        mask = eligible & (rand < cfg.mask_prob)
        return mask

    if cfg.mask_mode != "codon_span":
        raise ValueError(f"unknown mask_mode {cfg.mask_mode!r}")

    # codon_span: UTR as single tokens, CDS as contiguous spans
    rand = torch.rand(b, l, device=dev, generator=generator)
    utr = eligible & (region_ids != REGION_CDS) & (rand < cfg.mask_prob)
    mask = mask | utr
    cds = eligible & (region_ids == REGION_CDS)
    for i in range(b):
        pos = torch.nonzero(cds[i], as_tuple=False).flatten()
        if pos.numel() == 0:
            continue
        n_target = max(1, int(round(pos.numel() * cfg.mask_prob)))
        covered = 0
        guard = 0
        while covered < n_target and guard < 200:
            guard += 1
            start = int(torch.randint(0, pos.numel(), (1,), device=dev))
            span = int(torch.randint(cfg.codon_span_min, cfg.codon_span_max + 1, (1,), device=dev))
            span = min(span, pos.numel() - start)
            mask[i, pos[start:start + span]] = True
            covered += span
    return mask


def apply_bert_masking(input_ids: torch.Tensor, mask_positions: torch.Tensor,
                       vocab_size: int, mask_token_id: int,
                       generator: Optional[torch.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Standard BERT 80/10/10 corruption; returns ``(corrupted_ids, mlm_labels)``."""
    device = input_ids.device
    corrupted = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    labels[mask_positions] = input_ids[mask_positions]

    n_mask = mask_positions.sum().item()
    if n_mask == 0:
        return corrupted, labels

    choice = torch.rand(mask_positions.shape, device=device, generator=generator)[mask_positions]
    to_mask = choice < 0.8
    to_random = (choice >= 0.8) & (choice < 0.9)

    idx = torch.nonzero(mask_positions, as_tuple=False)
    if to_mask.any():
        rows, cols = idx[to_mask, 0], idx[to_mask, 1]
        corrupted[rows, cols] = mask_token_id
    if to_random.any():
        rows, cols = idx[to_random, 0], idx[to_random, 1]
        random_ids = torch.randint(0, vocab_size, (rows.numel(),), device=device,
                                   generator=generator)
        corrupted[rows, cols] = random_ids
    return corrupted, labels


# --------------------------------------------------------------------------- #
def load_backbone(model_path: str):
    """Load the released mRNABERT as a :class:`BertBackbone` (custom code)."""
    import transformers
    from transformers.models.bert.configuration_bert import BertConfig

    config = BertConfig.from_pretrained(model_path)
    hf = transformers.AutoModelForMaskedLM.from_pretrained(
        model_path, config=config, trust_remote_code=True)
    return BertBackbone(hf), config


class RNARJEPA(nn.Module):
    """Student + EMA teacher + factorised region predictor, with loss assembly."""

    def __init__(self, cfg: JEPAConfig):
        super().__init__()
        self.cfg = cfg
        student, hf_config = load_backbone(cfg.model_path)
        self.student = student
        self.teacher = copy.deepcopy(student)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.config = hf_config

        dim = hf_config.hidden_size
        self.hidden_size = dim
        self.predictor = RegionPredictor(dim, cfg)
        self.factor_basis = FactorBasis(dim, cfg.n_factors)
        # Region-specific projection of a pooled region summary.
        #
        # architecture.md places W_r on the (gradient-free) teacher side, which
        # would leave it permanently at its initialisation: a target that is
        # detached end to end cannot train anything.  It is therefore realised as
        # a trainable student-side ``region_proj`` plus an EMA mirror
        # ``teacher_region_proj``: predictions go through W_r, targets through
        # EMA(W_r).  Both spaces adapt together, the teacher stays gradient-free,
        # and the shared factor basis P_k still guarantees comparability.
        self.region_proj = nn.ModuleDict(
            {str(r): nn.Linear(dim, dim) for r in REGION_ORDER})
        self.teacher_region_proj = copy.deepcopy(self.region_proj)
        for p in self.teacher_region_proj.parameters():
            p.requires_grad_(False)
        self.remask_embed = nn.Parameter(torch.zeros(dim))

        mask_id = cfg.mask_token_id
        if mask_id is None:
            # the released custom BertConfig does not define mask_token_id, so the
            # caller must supply the tokenizer's value; do not guess an index.
            raise ValueError(
                "JEPAConfig.mask_token_id is required: the released config has no "
                "mask_token_id. Pass tokenizer.mask_token_id.")
        if not 0 <= mask_id < hf_config.vocab_size:
            raise ValueError(f"mask_token_id {mask_id} outside vocabulary "
                             f"(size {hf_config.vocab_size})")
        with torch.no_grad():
            self.remask_embed.copy_(student.word_embeddings.weight[mask_id])
        self.mask_token_id = mask_id
        self.vocab_size = hf_config.vocab_size

    # -- EMA -----------------------------------------------------------------
    @torch.no_grad()
    def ema_momentum(self, step: int, total_steps: int) -> float:
        frac = min(1.0, step / max(1, total_steps))
        return self.cfg.ema_start + (self.cfg.ema_end - self.cfg.ema_start) * frac

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        for tp, sp in zip(self.teacher.parameters(), self.student.parameters()):
            tp.mul_(momentum).add_(sp.detach(), alpha=1.0 - momentum)
        for tb, sb in zip(self.teacher.buffers(), self.student.buffers()):
            tb.copy_(sb)
        # the target-side region projection mirrors the student's (see __init__)
        for tp, sp in zip(self.teacher_region_proj.parameters(),
                          self.region_proj.parameters()):
            tp.mul_(momentum).add_(sp.detach(), alpha=1.0 - momentum)

    # -- curriculum ----------------------------------------------------------
    def curriculum_weights(self, step: int, total_steps: int) -> Dict[str, float]:
        """Region weight decays 1.0 -> 0.2 while the CLS weight grows 0.2 -> 1.0.

        Both are held at their starting value for the first
        ``curriculum_frac`` of training, then interpolated linearly.  When
        ``normalize_jepa_weights`` is set the weighted JEPA terms are divided by
        the sum of active weights, so rebalancing the curriculum changes *where*
        the supervision goes without changing the overall scale of ``L_JEPA``
        (which would otherwise make ``alpha`` mean something different early and
        late in training).
        """
        cfg = self.cfg
        frac = step / max(1, total_steps)
        if frac <= cfg.curriculum_frac:
            t = 0.0
        else:
            t = (frac - cfg.curriculum_frac) / max(1e-9, 1.0 - cfg.curriculum_frac)
        t = min(1.0, max(0.0, t))
        w_r = cfg.w_region_start + (cfg.w_region_end - cfg.w_region_start) * t
        w_cls = cfg.w_cls_start + (cfg.w_cls_end - cfg.w_cls_start) * t
        return {"w_region": float(w_r), "w_cls": float(w_cls)}

    # -- forward -------------------------------------------------------------
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                region_ids: torch.Tensor, step: int, total_steps: int,
                special_ids: torch.Tensor, generator: Optional[torch.Generator] = None,
                compute_diagnostics: bool = True) -> Dict[str, torch.Tensor]:
        """One pre-training step.

        Pass order is chosen to minimise peak GPU memory on a shared node: both
        gradient-free passes (eval-mode variance/covariance, teacher) run *before*
        the gradient-carrying student forward, so their activations are released
        before the large retained graph exists.
        """
        cfg = self.cfg
        device = input_ids.device
        attn_bool = attention_mask.bool()

        mask_positions = sample_mask_positions(
            region_ids, attention_mask, special_ids, cfg, generator)
        corrupted, mlm_labels = apply_bert_masking(
            input_ids, mask_positions, self.vocab_size, self.mask_token_id, generator)
        if cfg.jepa_target == "none":
            # Pure MLM: no teacher forward, no predictor, no regularisers.
            h_s = self.student.encode(corrupted, attention_mask)
            mlm_logits = self.student.mlm_logits(h_s)
            flat_labels = mlm_labels.flatten()
            keep = flat_labels > 0
            loss_mlm = F.cross_entropy(mlm_logits.flatten(0, 1)[keep], flat_labels[keep])
            zero = torch.zeros((), device=device)
            return {"loss": loss_mlm, "loss_mlm": loss_mlm.detach(), "loss_jepa": zero,
                    "loss_var": zero, "loss_cov": zero, "loss_orth": zero,
                    "w_region": zero, "w_cls": zero, "n_region_terms": zero,
                    "mask_frac": mask_positions.float().mean().detach()}

        descriptors = region_descriptors(region_ids, attention_mask)
        remask = self.remask_embed.view(1, 1, -1)

        weights = self.curriculum_weights(step, total_steps)
        loss_var = torch.zeros((), device=device)
        loss_cov = torch.zeros((), device=device)
        n_eval_terms = 0

        # ---- (1) anti-collapse variance/covariance, eval mode, no grad -------
        # Measured with dropout and the stochastic mask switched off, so the
        # reported variance describes the representation rather than this step's
        # noise draw (architecture §4.7).
        if cfg.lambda_var > 0 or cfg.lambda_cov > 0:
            was_training = self.student.training
            self.student.eval()
            self.predictor.eval()
            with torch.no_grad():
                h_eval = self.student.encode(corrupted, attention_mask)
                h_prime_eval = torch.where(mask_positions.unsqueeze(-1), remask, h_eval)
                pred_eval = self.predictor(h_prime_eval, descriptors, attn_bool)
                for region in REGION_ORDER:
                    sel = (region_ids == region) & attn_bool
                    present = sel.any(dim=1)
                    if not present.any():
                        continue
                    w = sel.to(pred_eval.dtype).unsqueeze(-1)
                    denom = w.sum(dim=1).clamp_min(1.0)
                    u_eval = ((pred_eval * w).sum(dim=1) / denom)[present]
                    v, c = variance_covariance_loss(u_eval)
                    loss_var = loss_var + v
                    loss_cov = loss_cov + c
                    n_eval_terms += 1
                del h_eval, h_prime_eval, pred_eval, u_eval
            if was_training:
                self.student.train()
            self.predictor.train()
            if n_eval_terms:
                loss_var = loss_var / n_eval_terms
                loss_cov = loss_cov / n_eval_terms

        # ---- (2) teacher targets, no grad; pooled immediately to free h_t ----
        with torch.no_grad():
            self.teacher.eval()
            h_t = self.teacher.encode(input_ids, attention_mask)
            cls_pos = attn_bool & (region_ids == REGION_NONE)
            has_cls = cls_pos.any(dim=1)
            head = cls_pos.float().argmax(dim=1, keepdim=True)
            gather_idx = head.unsqueeze(-1).expand(-1, -1, h_t.shape[-1])
            t_cls_all = h_t.gather(1, gather_idx).squeeze(1)
            z_cls = t_cls_all[has_cls]
            teacher_targets = {}
            for region in REGION_ORDER:
                z_raw, _ = pool_by_region(h_t, region_ids, attention_mask, region)
                present = ((region_ids == region) & attn_bool).any(dim=1)
                if present.any():
                    teacher_targets[region] = self.teacher_region_proj[str(region)](z_raw[present])
            t_masked = None
            if cfg.jepa_target in ("masked", "both") and mask_positions.any():
                t_masked = h_t[mask_positions]
            del h_t
        if not has_cls.any():
            z_cls = None

        # ---- (3) student forward (grad) + MLM --------------------------------
        h_s = self.student.encode(corrupted, attention_mask)
        mlm_logits = self.student.mlm_logits(h_s)
        flat_labels = mlm_labels.flatten()
        keep = flat_labels > 0
        loss_mlm = F.cross_entropy(mlm_logits.flatten(0, 1)[keep], flat_labels[keep])
        del mlm_logits

        # re-masking: overwrite masked positions with the [MASK] embedding so the
        # predictor cannot read the answer off the student's own activations
        h_prime = torch.where(mask_positions.unsqueeze(-1), remask, h_s)

        # ---- (4) predictor forward (grad) ------------------------------------
        pred_hidden = self.predictor(h_prime, descriptors, attn_bool)

        # ---- (5) region + global latent objectives ---------------------------
        loss_jepa = torch.zeros((), device=device)
        weight_sum = 0.0
        n_region_terms = 0
        u_pool = torch.zeros(0, device=device)
        z_pool = torch.zeros(0, device=device)

        for region in (REGION_ORDER if cfg.jepa_target in ("region", "both") else ()):
            if region not in teacher_targets:
                continue
            sel = (region_ids == region) & attn_bool
            present = sel.any(dim=1)
            w = sel.to(pred_hidden.dtype).unsqueeze(-1)
            denom = w.sum(dim=1).clamp_min(1.0)
            u_raw = ((pred_hidden * w).sum(dim=1) / denom)[present]
            u_raw = self.region_proj[str(region)](u_raw)
            z_r = teacher_targets[region]

            u_f = self.factor_basis.project(u_raw)
            z_f = self.factor_basis.project(z_r)
            if cfg.loss_form == "cos":
                per_factor = cosine_loss(u_f, z_f).mean(dim=0)
                term = per_factor.mean()
            else:
                term = F.mse_loss(u_f, z_f.detach())
            loss_jepa = loss_jepa + weights["w_region"] * term
            weight_sum += weights["w_region"]
            n_region_terms += 1

            u_pool = torch.cat([u_pool, u_raw], dim=0)
            z_pool = torch.cat([z_pool, z_r.detach()], dim=0)

        if t_masked is not None and t_masked.shape[0] > 0:
            p_masked = pred_hidden[mask_positions]
            if cfg.loss_form == "cos":
                pf = cosine_loss(self.factor_basis.project(p_masked),
                                 self.factor_basis.project(t_masked)).mean(dim=0)
                masked_term = pf.mean()
            else:
                masked_term = F.mse_loss(self.factor_basis.project(p_masked),
                                         self.factor_basis.project(t_masked).detach())
            loss_jepa = loss_jepa + weights["w_region"] * masked_term
            weight_sum += weights["w_region"]

        if z_cls is not None and z_cls.shape[0]:
            head2 = (attn_bool & (region_ids == REGION_NONE)).float().argmax(
                dim=1, keepdim=True)
            gi = head2.unsqueeze(-1).expand(-1, -1, pred_hidden.shape[-1])
            u_cls = pred_hidden.gather(1, gi).squeeze(1)[has_cls]
            loss_jepa = loss_jepa + weights["w_cls"] * cosine_loss(u_cls, z_cls).mean()
            weight_sum += weights["w_cls"]

        if cfg.normalize_jepa_weights and weight_sum > 0:
            loss_jepa = loss_jepa / weight_sum

        loss_orth = self.factor_basis.orthogonality_loss()
        total = (loss_mlm
                 + cfg.alpha * loss_jepa
                 + cfg.lambda_var * loss_var
                 + cfg.lambda_cov * loss_cov
                 + cfg.lambda_orth * loss_orth)

        out = {
            "loss": total,
            "loss_mlm": loss_mlm.detach(),
            "loss_jepa": loss_jepa.detach(),
            "loss_var": loss_var.detach(),
            "loss_cov": loss_cov.detach(),
            "loss_orth": loss_orth.detach(),
            "w_region": torch.tensor(weights["w_region"], device=device),
            "w_cls": torch.tensor(weights["w_cls"], device=device),
            "n_region_terms": torch.tensor(float(n_region_terms), device=device),
            "mask_frac": mask_positions.float().mean().detach(),
        }
        if compute_diagnostics and u_pool.shape[0] >= 2:
            with torch.no_grad():
                out["procrustes_residual"] = torch.tensor(
                    procrustes_similarity(u_pool.detach(), z_pool), device=device)
                u_n = F.normalize(u_pool.detach(), dim=-1)
                z_n = F.normalize(z_pool, dim=-1)
                out["cos_mean"] = (u_n * z_n).sum(-1).mean()
                if has_cls.any():
                    s_cls = h_s.gather(1, gather_idx).squeeze(1)[has_cls]
                    s_n = F.normalize(s_cls.detach(), dim=-1)
                    t_n = F.normalize(t_cls_all, dim=-1)
                    out["cls_corr"] = (s_n * t_n).sum(-1).mean()
                    out["cls_var"] = s_cls.detach().var(dim=0).mean()
                out["factor_activity"] = self.factor_basis.project(
                    u_pool.detach()).pow(2).mean(dim=(0, 2))
        return out