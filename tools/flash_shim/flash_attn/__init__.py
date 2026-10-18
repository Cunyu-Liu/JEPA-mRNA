"""Pure-PyTorch drop-in shim for the `flash_attn` package (RiNALMo inference
on hosts without a compiled flash-attn). Provides exactly the surface
RiNALMo's attention.py imports:
  flash_attn_varlen_qkvpacked_func, flash_attn_qkvpacked_func,
  flash_attn.layers.rotary.RotaryEmbedding,
  flash_attn.bert_padding.unpad_input / pad_input.

Numerics: F.scaled_dot_product_attention (SDPA) in fp16/bf16 — same kernel
family flash-attn uses; output layout matches the original API.
RoPE: reimplements the non-interleaved fp32-index rotary exactly as the
flash_attn.layers.rotary default used by RiNALMo (dim=head_dim, base=10000,
interleaved=False, pos_idx_in_fp32=True).
"""
import torch
import torch.nn.functional as F
from einops import rearrange


def flash_attn_qkvpacked_func(qkv, dropout_p=0.0, softmax_scale=None,
                              causal=False, return_attn_probs=False,
                              **kwargs):
    # qkv: (B, S, 3, H, D) -> out (B, S, H, D)
    assert qkv.dtype in [torch.float16, torch.bfloat16]
    B, S, _, H, D = qkv.shape
    q, k, v = qkv.unbind(dim=2)                      # each (B, S, H, D)
    q = rearrange(q, "b s h d -> (b h) s d")
    k = rearrange(k, "b s h d -> (b h) s d")
    v = rearrange(v, "b s h d -> (b h) s d")
    out = F.scaled_dot_product_attention(             # ((B H), S, D)
        q, k, v,
        dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
        scale=softmax_scale,
        is_causal=causal,
    )
    out = rearrange(out, "(b h) s d -> b s h d", b=B)
    if return_attn_probs:
        return out, None, None
    return out


def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen,
                                     dropout_p=0.0, softmax_scale=None,
                                     causal=False, return_attn_probs=False,
                                     **kwargs):
    # qkv: (total, 3, H, D); cu_seqlens: int32 cumulative lengths
    # -> out (total, H, D)
    assert qkv.dtype in [torch.float16, torch.bfloat16]
    q, k, v = qkv.unbind(dim=1)                      # each (total, H, D)
    H, D = q.shape[-2:]
    total = q.shape[0]
    outs = []
    cu = cu_seqlens.tolist()
    for i in range(len(cu) - 1):
        s, e = cu[i], cu[i + 1]
        qs, ks, vs = q[s:e], k[s:e], v[s:e]          # (n, H, D)
        o = F.scaled_dot_product_attention(          # (H, n, D)
            qs.transpose(0, 1), ks.transpose(0, 1), vs.transpose(0, 1),
            dropout_p=dropout_p if torch.is_grad_enabled() else 0.0,
            scale=softmax_scale,
            is_causal=causal,
        ).transpose(0, 1).contiguous()               # (n, H, D)
        outs.append(o)
    out = torch.cat(outs, dim=0) if outs else q.new_empty(0, H, D)
    out = out[:total]
    if return_attn_probs:
        return out, None, None
    return out


class RotaryEmbedding(torch.nn.Module):
    """Non-interleaved rotary embedding, fp32 positions."""

    def __init__(self, dim, base=10000.0, interleaved=False, scale_base=None,
                 pos_idx_in_fp32=True, device=None, max_seq_len=None, **kw):
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2, device=device,
                                                     dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._pos_cache = {}

    def _get_pos(self, seqlen, device, dtype):
        key = (seqlen, str(device))
        if key not in self._pos_cache:
            t = torch.arange(seqlen, device=device,
                             dtype=torch.float32 if self.pos_idx_in_fp32 else dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))          # (S, D/2)
            emb = torch.cat((freqs, freqs), dim=-1)                   # (S, D)
            cos, sin = emb.cos(), emb.sin()
            self._pos_cache[key] = (cos, sin)
        return self._pos_cache[key]

    def forward(self, x, seqlen_offset=0):
        # x: (B, S, 3, H, D) qkv-packed (RiNALMo usage) — apply RoPE to q and k
        # ONLY (flash_attn's apply_rotary_emb_qkv_ semantics); v passes through.
        # Also accepts (B, S, H, D).
        is_qkv = x.dim() == 5
        if not is_qkv:
            D = x.shape[-1]
            S = x.shape[1]
            cos, sin = self._get_pos(S, x.device, x.dtype)
            cos = cos.to(x.dtype)[None, :, None, :]
            sin = sin.to(x.dtype)[None, :, None, :]
            x1, x2 = x[..., : D // 2], x[..., D // 2:]
            return torch.cat(
                (x1 * cos[..., : D // 2] - x2 * sin[..., : D // 2],
                 x2 * cos[..., : D // 2] + x1 * sin[..., : D // 2:]), dim=-1).to(x.dtype)
        D = x.shape[-1]
        S = x.shape[1]
        cos, sin = self._get_pos(S, x.device, x.dtype)
        cos = cos.to(x.dtype)[None, :, None, None, :]
        sin = sin.to(x.dtype)[None, :, None, None, :]
        out = x.clone()
        cq = cos.squeeze(2)  # (B, S, 1, D) after squeezing the 3-dim slot? cos is (B,S,1,1,D) from broadcast prep
        # cos/sin were shaped (B, S, 1, 1, D) for 5-d broadcasting; for per-t use drop to (B,S,1,D)
        cst = cos[:, :, 0]   # (B, S, 1, D)
        snt = sin[:, :, 0]   # (B, S, 1, D)
        for t in (0, 1):  # q and k only
            xt = x[:, :, t]  # (B, S, H, D)
            x1, x2 = xt[..., : D // 2], xt[..., D // 2:]
            rot = torch.cat(
                (x1 * cst[..., : D // 2] - x2 * snt[..., : D // 2],
                 x2 * cst[..., : D // 2] + x1 * snt[..., : D // 2]), dim=-1)
            out = out.index_copy(2, torch.tensor([t], device=x.device), rot.unsqueeze(2))
        return out

    def _update_cos_sin_cache(self, seqlen, device, dtype):
        self._get_pos(seqlen, device, dtype)


def unpad_input(hidden_states, attention_mask):
    # attention_mask: (B, S) True=keep; returns 5-tuple like flash_attn 2.7.4
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    S = attention_mask.shape[1]
    unpadded = hidden_states[
        torch.div(indices, S, rounding_mode="floor"), indices % S]
    return unpadded, indices, cu_seqlens, max_seqlen, None


def pad_input(hidden_states, indices, batch, seqlen):
    # hidden_states: (total, H, D); returns (B, S, H, D)
    out = hidden_states.new_zeros(batch * seqlen, *hidden_states.shape[1:])
    out[indices] = hidden_states
    return rearrange(out, "(b s) ... -> b s ...", b=batch)
