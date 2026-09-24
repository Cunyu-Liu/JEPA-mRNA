"""Architecture tests for the RNA secondary-structure decision model.

Covers the spec §5.3 encoder and the spec §5.4 / §5.0.2 decision head + the
hierarchical cascade.  Everything is CPU-only, tiny (``d = 32``, two layers,
``L <= 64``) and fast; no data or network access is used.

Required assertions (task deliverable 3):

1. ``s_ij = s_ji`` **exactly** (max abs difference ``== 0``) for random inputs;
2. zero-init equivalence -- with ``MLP_T`` zeroed the scores equal ``s^phys``
   exactly and the DP reproduces the same structure (self-consistency, *not* a
   ViennaRNA claim);
3. no autoregressive decoding -- exactly one forward pass produces the ``L x L``
   matrix (forward calls are counted);
4. the pair-type permutation constraint ``t_ji = Pi t_ij`` holds exactly;
5. cascade levels run independently; L0 shapes are as specified and
   sparsification really reduces the block-pair count;
6. the cascade preserves long-range pairs (span ``> L/2``) where banding cannot;
7. L0 helix recall is in ``[0, 1]`` and equals ``1.0`` on a crafted all-detected
   case;
8. illegal-structure rate is ``0`` when the head's scores go through
   ``harness.nussinov_map``.

Run:  python -m pytest tests/test_architecture.py -v
  or: python tests/test_architecture.py
"""

import math
import os
import random
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np  # noqa: E402

from rnajepa.decision_head import (  # noqa: E402
    TYPE_PERM, DecisionModel, FlatDecisionHead, HierarchicalCascade,
    LengthAdaptiveDispatcher, cascade_flops_estimate, compare_flops,
    flat_flops_estimate, group_helices, l0_helix_recall, pair_mask_from_ids,
    turner_phys_scores,
)
from rnajepa.encoder import (  # noqa: E402
    BASE_FEATURES, FEATURE_DIM, SIZE_PRESETS, RNAEncoder, TURNER_NN_DG,
    build_encoder, encode_sequence, nn_dg_grid,
)
from rnajepa.harness import MIN_LOOP, PAIRS, fold_system1, nussinov_map  # noqa: E402

D_MODEL = 32
D_Z = 16
HIDDEN = 16
N_LAYER = 2
N_HEAD = 4
MAX_LEN = 256

# A 24-nt sequence with plenty of candidate pairs.
SEQ24 = "GGGGAAAACCCCGGGGAAAACCCC"
# 64-nt crafted helix with a long-range outer pair: (0,63) ... (4,59).
SEQ_LR = "GCGC" + "A" * 55 + "U" + "GCGC"


def make_encoder(**kw):
    kw.setdefault("d_model", D_MODEL)
    kw.setdefault("n_layer", N_LAYER)
    kw.setdefault("n_head", N_HEAD)
    kw.setdefault("d_ff", 64)
    kw.setdefault("max_len", MAX_LEN)
    kw.setdefault("window", 64)
    kw.setdefault("global_stride", 32)
    return RNAEncoder(**kw)


def make_head(**kw):
    kw.setdefault("d_model", D_MODEL)
    kw.setdefault("d_z", D_Z)
    kw.setdefault("hidden", HIDDEN)
    kw.setdefault("calibration_edges", (0, 64, 128, 256))
    return FlatDecisionHead(**kw)


def make_cascade(**kw):
    kw.setdefault("d_model", D_MODEL)
    kw.setdefault("block_size", 8)
    kw.setdefault("top_k", 2)
    kw.setdefault("d_z", D_Z)
    kw.setdefault("hidden", HIDDEN)
    return HierarchicalCascade(**kw)


def ids_of(seq, batch=1):
    return encode_sequence(seq).unsqueeze(0).expand(batch, -1).contiguous()


def assert_legal(structure, seq):
    """Non-crossing, min-loop >= 3, pairs inside PAIRS, no repeated position."""
    used = set()
    for i, j in structure:
        assert 0 <= i < j < len(seq), f"bad pair {(i, j)}"
        assert j - i > MIN_LOOP, f"hairpin violation {(i, j)}"
        assert (seq[i], seq[j]) in PAIRS, f"non-pair {(i, j)}={seq[i]}{seq[j]}"
        assert i not in used and j not in used, f"position reused in {(i, j)}"
        used.add(i)
        used.add(j)
    for a in range(len(structure)):
        for b in range(a + 1, len(structure)):
            i, j = structure[a]
            k, l = structure[b]
            assert not (i < k < j < l), f"crossing {(i, j)} x {(k, l)}"
            assert not (k < i < l < j), f"crossing {(i, j)} x {(k, l)}"


# --------------------------------------------------------------------------- #
# 1. exact pair-score symmetry
# --------------------------------------------------------------------------- #
def test_pair_score_symmetry_is_exact():
    torch.manual_seed(0)
    head = make_head()
    ids = ids_of(SEQ24, batch=2)

    for h in (torch.randn(2, len(SEQ24), D_MODEL), make_encoder()(ids)):
        out = head(h, ids)
        s = out.scores
        assert s.shape == (2, len(SEQ24), len(SEQ24))
        # -inf on illegal pairs must be symmetric too, so compare the whole matrix.
        assert torch.equal(s, s.transpose(1, 2)), "s_ij != s_ji on the full matrix"
        finite = torch.isfinite(s)
        diff = (s - s.transpose(1, 2))[finite]
        assert float(diff.detach().abs().max()) == 0.0, "non-zero max abs asymmetry"

    # the same must hold for the permutation-invariant representation itself
    z = head.pair_repr(torch.randn(1, 12, D_MODEL))
    assert torch.equal(z, z.transpose(1, 2))


# --------------------------------------------------------------------------- #
# 2b. the Turner prior weight is learnable (and old checkpoints still load)
# --------------------------------------------------------------------------- #
def test_prior_weight_defaults_to_one_and_scales_the_prior_exactly():
    """`w = 1` reproduces the old behaviour; other `w` scales the prior exactly.

    The head used to add the prior at a hard-coded weight of 1, which made the
    weight a hyper-parameter training could not reach: `MLP_T` sees only `z_ij` and
    the temperature divides the sum, so `MLP_T / prior` is invariant.  A zero-training
    sweep (`tools/probe_prior_weight.py`) put the frozen-RiNALMo optimum at w = 0.5
    (+0.048 micro F1) and the from-scratch optimum at ~1, so the weight is
    arm-dependent and has to be learned rather than fixed.

    `calibrate=False` is used so the temperature does not divide the comparison.
    """
    torch.manual_seed(2)
    head = make_head()
    head.zero_residual()                      # MLP_T == 0, so scores == w * prior
    assert head.prior_weight is not None
    assert float(head.prior_weight) == 1.0

    ids = ids_of(SEQ24)
    h = torch.randn(1, len(SEQ24), D_MODEL)
    phys = turner_phys_scores(ids)
    mask = head(h, ids, calibrate=False).mask

    base = head(h, ids, calibrate=False).scores
    assert torch.equal(base[mask], phys[mask]), "w=1 must be exactly the old behaviour"

    for w in (0.0, 0.5, 2.0):
        with torch.no_grad():
            head.prior_weight.fill_(w)
        got = head(h, ids, calibrate=False).scores
        assert torch.equal(got[mask], (w * phys)[mask]), f"w={w} did not scale the prior"
        assert torch.equal(got[~mask], base[~mask]), "mask handling changed with w"


def test_prior_weight_receives_a_gradient():
    """It must be on the loss path, or `assert_gradient_coverage` would reject a run."""
    torch.manual_seed(3)
    head = make_head()
    ids = ids_of(SEQ24)
    h = torch.randn(1, len(SEQ24), D_MODEL, requires_grad=True)
    out = head(h, ids, lengths=torch.tensor([len(SEQ24)]))
    loss = out.scores[torch.isfinite(out.scores)].sum()
    loss.backward()
    assert head.prior_weight.grad is not None
    assert float(head.prior_weight.grad.abs().sum()) > 0.0


def test_prior_weight_absent_when_the_prior_is_disabled():
    """With no prior there is nothing to weight; a dangling Parameter would break the
    gradient-coverage assertion that every learnable block is on the loss path."""
    head = FlatDecisionHead(d_model=D_MODEL, use_turner_prior=False)
    assert head.prior_weight is None
    assert "prior_weight" not in dict(head.named_parameters())
    ids = ids_of(SEQ24)
    out = head(torch.randn(1, len(SEQ24), D_MODEL), ids, calibrate=False)
    assert torch.isfinite(out.scores[out.mask]).all()


def test_checkpoint_without_prior_weight_still_loads_identically():
    """Every checkpoint trained before this change must keep its exact behaviour.

    `strict=False` leaves a missing key at its initialisation, which is 1, so the
    forward pass is bit-identical.  Checked rather than assumed, because a silent
    change of behaviour here would invalidate every number already measured.
    """
    torch.manual_seed(4)
    head = make_head()
    with torch.no_grad():
        head.turner.net[-1].weight.normal_(0.0, 0.2)   # non-trivial MLP_T
    ids = ids_of(SEQ24)
    h = torch.randn(1, len(SEQ24), D_MODEL)
    reference = head(h, ids, lengths=torch.tensor([len(SEQ24)])).scores

    legacy = {k: v for k, v in head.state_dict().items() if k != "prior_weight"}
    assert "prior_weight" in head.state_dict(), "the parameter should exist to be dropped"

    fresh = make_head()
    missing, unexpected = fresh.load_state_dict(legacy, strict=False)
    assert list(missing) == ["prior_weight"], missing
    assert not unexpected
    assert float(fresh.prior_weight) == 1.0
    assert torch.equal(fresh(h, ids, lengths=torch.tensor([len(SEQ24)])).scores, reference)


# --------------------------------------------------------------------------- #
# 2. zero-init Turner residual equivalence (self-consistency, not ViennaRNA)
# --------------------------------------------------------------------------- #
def test_zero_init_residual_reproduces_physics_prior():
    torch.manual_seed(1)
    head = make_head()
    head.zero_residual()
    ids = ids_of(SEQ24)
    h = torch.randn(1, len(SEQ24), D_MODEL)

    out = head(h, ids, lengths=torch.tensor([len(SEQ24)]))
    phys = turner_phys_scores(ids)
    mask = out.mask
    assert bool(mask.any())
    # exact equality on every legal candidate pair
    assert torch.equal(out.scores[mask], phys[mask]), "MLP_T=0 is not exactly s^phys"

    scores = out.scores[0].detach().numpy()
    mask_np = mask[0].numpy()
    struct_from_head = nussinov_map(scores, mask_np)
    struct_from_phys = nussinov_map(phys[0].numpy(), mask_np)
    # self-consistency: identical to running the DP directly on the physics prior
    assert struct_from_head == struct_from_phys
    assert struct_from_head, "the Turner stacking prior should fold this sequence"
    assert_legal(struct_from_head, SEQ24)

    # and the equivalence is specific to the zeroed state: perturbing the residual
    # (which is zero-initialised by construction) breaks it.
    head2 = make_head()
    with torch.no_grad():
        head2.turner.net[-1].weight.normal_(0.0, 0.1)
    out2 = head2(h, ids)
    assert not torch.equal(out2.scores[mask], phys[mask])
    assert torch.equal(out2.scores[~mask], out.scores[~mask]), "mask handling changed"


# --------------------------------------------------------------------------- #
# 3. no autoregressive decoding: one forward -> the whole L x L matrix
# --------------------------------------------------------------------------- #
def test_single_forward_produces_full_score_matrix():
    torch.manual_seed(2)
    enc, head = make_encoder(), make_head()
    model = DecisionModel(enc, head)
    ids = ids_of(SEQ24, batch=3)

    calls = {"encoder": 0, "head": 0}
    enc_forward, head_forward = enc.forward, head.forward

    def counting_encoder(*a, **k):
        calls["encoder"] += 1
        return enc_forward(*a, **k)

    def counting_head(*a, **k):
        calls["head"] += 1
        return head_forward(*a, **k)

    enc.forward = counting_encoder
    head.forward = counting_head

    out = model(ids)
    assert calls == {"encoder": 1, "head": 1}, f"unexpected forward count {calls}"
    assert out.scores.shape == (3, len(SEQ24), len(SEQ24))
    # a single forward must cover *all* ordered candidate positions at once
    assert int(torch.isfinite(out.scores).sum()) == int(out.mask.sum())


# --------------------------------------------------------------------------- #
# 4. hard pair-type permutation constraint
# --------------------------------------------------------------------------- #
def test_pair_type_permutation_constraint_is_exact():
    torch.manual_seed(3)
    head = make_head()
    h = torch.randn(2, 18, D_MODEL)
    t = head.type_head(h)
    assert t.shape == (2, 18, 18, 6)
    # t_ji = Pi t_ij  <=>  t_ij = Pi t_ji
    assert torch.equal(t, t.transpose(1, 2)[..., list(TYPE_PERM)])
    # diagonal is unused / zero
    assert float(t.diagonal(dim1=1, dim2=2).detach().abs().max()) == 0.0


# --------------------------------------------------------------------------- #
# 5. cascade levels are independently runnable; L0 shapes; real sparsification
# --------------------------------------------------------------------------- #
def test_cascade_levels_and_sparsification():
    torch.manual_seed(4)
    enc, cascade = make_encoder(), make_cascade()
    ids = ids_of(SEQ24)
    h = enc(ids)
    L = len(SEQ24)
    w = cascade.w
    nb = math.ceil(L / w)

    block_logits, block_active, block_compat = cascade.l0(h, ids)
    assert block_logits.shape == (1, nb, nb)
    assert block_active.shape == (1, nb, nb) and block_active.dtype == torch.bool
    assert block_compat.shape == (1, nb, nb) and block_compat.dtype == torch.bool
    # L0 scores are finite (they are the coarse helix strength)
    assert bool(torch.isfinite(block_logits).all())
    # sparsification: at most top_k per row, strictly fewer than all block pairs
    assert int(block_active.sum()) <= nb * cascade.top_k
    assert int(block_active.sum()) < nb * (nb + 1) // 2

    helix_spans, helix_logits, helix_mask = cascade.l1(h, ids, block_logits, block_active)
    assert helix_spans.shape == (1, nb * nb, 2)
    assert helix_logits.shape == (1, nb * nb)
    assert helix_mask.shape == (1, nb * nb) and helix_mask.dtype == torch.bool
    assert torch.equal(helix_mask, block_active.reshape(1, -1))

    scores, types = cascade.l2(h, ids, block_active)
    assert scores.shape == (1, L, L)
    assert types.shape == (1, L, L, 6)

    out = cascade(h, ids)
    assert out.scores.shape == (1, L, L)
    assert out.flops is not None and out.flops > 0
    # every active block pair must actually carry at least one scored pair
    for b1, b2 in out.block_active[0].nonzero().tolist():
        assert bool(torch.isfinite(out.scores[0, b1 * w:(b1 + 1) * w,
                                            b2 * w:(b2 + 1) * w]).any())


# --------------------------------------------------------------------------- #
# 6. cascade preserves long-range pairs (the key difference from banding)
# --------------------------------------------------------------------------- #
def test_cascade_represents_long_range_pair():
    torch.manual_seed(5)
    enc, cascade = make_encoder(), make_cascade()
    ids = ids_of(SEQ_LR)
    L = len(SEQ_LR)
    h = enc(ids)

    out = cascade(h, ids)
    # the long-range block pair (0, 7) is a candidate and survived sparsification
    assert bool(out.block_active[0, 0, L // cascade.w - 1])
    s = out.scores[0]
    assert torch.isfinite(s[0, 63]) and float(s[0, 63].detach()) > 0.0
    assert torch.isfinite(s[1, 62])

    mask = pair_mask_from_ids(ids)[0].numpy()
    assert bool(mask[0, 63]) and bool(mask[1, 62])
    structure = nussinov_map(s.detach().numpy(), mask)
    assert_legal(structure, SEQ_LR)
    assert any(j - i > L // 2 for i, j in structure), "long-range pair was lost"

    # banding *cannot* represent it: every legal pair here has span >= 55
    assert fold_system1(s.detach().numpy(), mask, band=L // 4) == []


# --------------------------------------------------------------------------- #
# 7. L0 helix recall
# --------------------------------------------------------------------------- #
def test_l0_helix_recall():
    w, L = 8, 64
    nb = math.ceil(L / w)
    helix_pairs = [(0, 63), (1, 62), (2, 61)]
    assert group_helices(helix_pairs) == [helix_pairs]

    all_detected = torch.zeros(nb, nb, dtype=torch.bool)
    all_detected[0, 7] = True
    recall = l0_helix_recall(all_detected, helix_pairs, w)
    assert recall == 1.0

    none_detected = torch.zeros(nb, nb, dtype=torch.bool)
    assert l0_helix_recall(none_detected, helix_pairs, w) == 0.0

    two_helices = [(0, 63), (1, 62), (10, 40), (11, 39)]
    assert len(group_helices(two_helices)) == 2
    assert l0_helix_recall(all_detected, two_helices, w) == 0.5

    # the real cascade detects every helix of the crafted long-range sequence
    enc, cascade = make_encoder(), make_cascade()
    ids = ids_of(SEQ_LR)
    out = cascade(enc(ids), ids)
    real = l0_helix_recall(out.block_active[0], [(0, 63), (1, 62), (2, 61), (3, 60)], w)
    assert real == 1.0
    for value in (recall, real, l0_helix_recall(all_detected, two_helices, w)):
        assert 0.0 <= value <= 1.0


# --------------------------------------------------------------------------- #
# 8. illegal-structure rate is 0 through the deterministic harness
# --------------------------------------------------------------------------- #
def test_illegal_structure_rate_is_zero():
    torch.manual_seed(6)
    enc, head, cascade = make_encoder(), make_head(), make_cascade()
    rng = random.Random(7)
    checked = 0
    for _ in range(12):
        L = rng.choice([12, 16, 24, 32, 64])
        seq = "".join(rng.choice("ACGU") for _ in range(L))
        ids = ids_of(seq)
        h = enc(ids)
        mask = pair_mask_from_ids(ids)[0].numpy()

        struct_head = nussinov_map(head(h, ids).scores[0].detach().numpy(), mask)
        assert_legal(struct_head, seq)
        struct_cascade = nussinov_map(cascade(h, ids).scores[0].detach().numpy(), mask)
        assert_legal(struct_cascade, seq)
        checked += 2
    assert checked == 24


# --------------------------------------------------------------------------- #
# supporting checks: dispatcher, FLOPs hook, physics table, backbone stubs
# --------------------------------------------------------------------------- #
def test_length_adaptive_dispatcher_picks_path():
    torch.manual_seed(8)
    enc = make_encoder()
    flat, cascade = make_head(), make_cascade()
    disp = LengthAdaptiveDispatcher(flat, cascade, flat_max_len=32)
    short = ids_of(SEQ24)
    path, out = disp(enc(short), short)
    assert path == "flat" and out.scores.shape == (1, len(SEQ24), len(SEQ24))
    long_ids = ids_of(SEQ_LR)
    path, out = disp(enc(long_ids), long_ids)
    assert path == "cascade" and out.scores.shape == (1, len(SEQ_LR), len(SEQ_LR))


def test_flops_accounting_favours_cascade_on_long_sequences():
    small = compare_flops(64, 8, D_MODEL, 2, D_Z)
    large = compare_flops(1024, 8, D_MODEL, 2, D_Z)
    assert small["cascade"] > 0 and small["flat"] > 0
    assert cascade_flops_estimate(1024, 8, D_MODEL, 2, D_Z) < flat_flops_estimate(1024, D_Z)
    assert large["speedup"] > small["speedup"] > 0
    # the hook records a real number after a forward
    cascade = make_cascade()
    ids = ids_of(SEQ_LR)
    cascade(make_encoder()(ids), ids)
    assert cascade.last_flops == cascade_flops_estimate(len(SEQ_LR), 8, D_MODEL, 2, D_Z)


def test_base_features_and_size_presets():
    assert set(BASE_FEATURES) == {"A", "C", "G", "U"}
    assert FEATURE_DIM == 5
    for vec in BASE_FEATURES.values():
        assert vec.shape == (5,)
    # G stacks more strongly than A (Turner nearest-neighbour derived)
    assert BASE_FEATURES["G"][4] > BASE_FEATURES["A"][4]
    # the shared Turner table: stack of pair (0,3) on (1,2) of "GCGC" is GC/CG
    dg = nn_dg_grid("GCGC")
    assert abs(float(dg[0, 3]) - TURNER_NN_DG[("G", "C", "C", "G")]) < 1e-6
    for size in ("35M", "150M", "650M"):
        assert size in SIZE_PRESETS
    enc = build_encoder(size="35M", d_model=16, n_layer=1, n_head=2, max_len=64)
    assert enc(ids_of("ACGUACGUACGU", batch=2)).shape == (2, 12, 16)
    # off-the-shelf backbones are explicit, documented stubs
    for name in ("mrnabert", "rnafm", "rinalmo"):
        try:
            build_encoder(backbone=name)
        except NotImplementedError as exc:
            assert "TODO" in str(exc)
        else:  # pragma: no cover - only if weights suddenly appear
            raise AssertionError(f"{name} adapter should be a stub")


if __name__ == "__main__":
    import traceback

    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
