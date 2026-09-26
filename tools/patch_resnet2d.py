"""Patch: add ResNet2D pair-scorer arm (Plan B) to decision_head.py.

Inserts a 2D-context scorer between PairRepresentation and the scalar head,
controlled by config.scorer == 'resnet2d'. Capacity-matched to the flat head
(same d_z=128); the 2D ResNet adds spatial context over the pairing matrix
(stem-stacking correlations), which the independent per-pair MLP lacks.

Usage (cluster): python patch_resnet2d.py <repo_root>
"""
import sys
from pathlib import Path

REPO = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/cunyuliu/rna-jepa")
TARGET = REPO / "src" / "rnajepa" / "decision_head.py"

src = TARGET.read_text()

if "class ResNet2DScorer" in src:
    print("already patched")
    raise SystemExit(0)

RESCORER = '''

# --------------------------------------------------------------------------- #
# Plan-B arm: 2D-context pair scorer (stem-stacking correlations)
# --------------------------------------------------------------------------- #
class ResNet2DBlock(nn.Module):
    """Bottleneck 2D residual block over the pairing matrix (B, L, L, C).

    Mirrors the semantics of structRFM/eFold-style 2D pair scorers: each pair's
    score sees its local neighbourhood in (i, j) space, so stacked pairs in a
    stem can reinforce each other -- the correlation the independent per-pair
    MLP cannot express.
    """

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(channels),
        )
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class ResNet2DScorer(nn.Module):
    """2D-context scorer: z (B, L, L, d_z) -> s (B, L, L).

    Pre-MLP_T positional embedding: relative-offset channel per (i-j) lets the
    convolutions condition on loop-span, which the per-pair representation also
    encodes implicitly. Last layer zero-initialised: at init the scorer outputs
    zeros, so an untrained Plan-B arm decodes exactly like the flat head with
    MLP_T=0 -- a clean control.
    """

    def __init__(self, d_z: int, n_blocks: int = 4, kernel_size: int = 3,
                 hidden: int = 64) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(d_z + 1, hidden, kernel_size=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResNet2DBlock(hidden, kernel_size) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.Conv2d(hidden, 1, kernel_size=1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, L, L2, C = z.shape
        assert L == L2
        idx = torch.arange(L, device=z.device, dtype=z.dtype)
        rel = (idx[:, None] - idx[None, :]).unsqueeze(0).unsqueeze(0)
        rel = rel.expand(B, 1, L, L)
        x = torch.cat([z.permute(0, 3, 1, 2), rel], dim=1)
        x = self.stem(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(x).squeeze(1)
'''

marker = "# --------------------------------------------------------------------------- #\n# 2b. Hierarchical decision cascade (our core architectural contribution)"
assert marker in src, "cascade marker not found"
src = src.replace(marker, RESCORER + "\n\n" + marker, 1)

# Wire into FlatDecisionHead: constructor switch + forward branch
old_init_sig = """        use_turner_prior: bool = True,
        chunk_size: int = 64,
        grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.min_loop = min_loop"""
new_init_sig = """        use_turner_prior: bool = True,
        chunk_size: int = 64,
        grad_checkpoint: bool = True,
        scorer: str = "mlp",
        resnet_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.min_loop = min_loop
        self.scorer = scorer"""
assert old_init_sig in src, "init sig not found"
src = src.replace(old_init_sig, new_init_sig, 1)

old_ctor = """        self.pair_repr = PairRepresentation(d_model, d_z)
        self.turner = TurnerResidual(d_z, hidden)"""
new_ctor = """        self.pair_repr = PairRepresentation(d_model, d_z)
        self.turner = TurnerResidual(d_z, hidden)
        if scorer == "resnet2d":
            self.resnet2d = ResNet2DScorer(d_z, n_blocks=resnet_blocks)
        else:
            self.resnet2d = None"""
assert old_ctor in src, "ctor marker not found"
src = src.replace(old_ctor, new_ctor, 1)

old_pair_scores = """    def pair_scores_and_types(self, h: torch.Tensor):
        \"\"\"``(scores, pair_types)`` for every ``(i, j)``, chunked along ``j``."""
new_pair_scores = """    def pair_scores_and_types(self, h: torch.Tensor):
        \"\"\"``(scores, pair_types)`` for every ``(i, j)``, chunked along ``j``.
        With ``scorer == 'resnet2d'`` the whole-matrix 2D path replaces the
        chunked per-pair MLP path (BatchNorm over the full matrix is what makes
        chunking wrong there); the flat path is untouched. \"\"\""""
assert old_pair_scores in src, "pair_scores docstring marker not found"
src = src.replace(old_pair_scores, new_pair_scores, 1)

old_body_head = """        B, L, _ = h.shape
        chunk = self.chunk_size if 0 < self.chunk_size < L else L
        #: Checkpointing is only meaningful when there is more than one chunk; with
        #: a single chunk it would just recompute the whole matrix for nothing.
        use_ckpt = self.grad_checkpoint and chunk < L"""
new_body_head = """        B, L, _ = h.shape
        if self.resnet2d is not None:
            z = self.pair_repr(h)
            s = self.resnet2d(z)
            t = self.type_head(self.type_head.cross(h, h))
            return s, self.type_head.symmetrize(t)
        chunk = self.chunk_size if 0 < self.chunk_size < L else L
        #: Checkpointing is only meaningful when there is more than one chunk; with
        #: a single chunk it would just recompute the whole matrix for nothing.
        use_ckpt = self.grad_checkpoint and chunk < L"""
assert old_body_head in src, "body head marker not found"
src = src.replace(old_body_head, new_body_head, 1)

TARGET.write_text(src)
print("patched:", TARGET)

# Now wire train_decision.py: config fields + build
TD = REPO / "src" / "rnajepa" / "train_decision.py"
tsrc = TD.read_text()
if "scorer: str" not in tsrc:
    old = """    d_z: int = 128
    hidden: int = 64"""
    new = """    d_z: int = 128
    hidden: int = 64
    #: Pair-scorer architecture: 'mlp' (per-pair, all existing arms) or
    #: 'resnet2d' (Plan B: 2D-context scorer over the pairing matrix).
    scorer: str = "mlp"
    resnet_blocks: int = 4"""
    assert old in tsrc, "train cfg marker not found"
    tsrc = tsrc.replace(old, new, 1)

    old_build = """        return FlatDecisionHead(d_model=d_model, d_z=config.d_z, hidden=config.hidden,"""
    new_build = """        return FlatDecisionHead(d_model=d_model, d_z=config.d_z, hidden=config.hidden,
            scorer=config.scorer, resnet_blocks=config.resnet_blocks,"""
    assert old_build in tsrc, "build marker not found"
    tsrc = tsrc.replace(old_build, new_build, 1)
    TD.write_text(tsrc)
    print("patched:", TD)

print("done")
