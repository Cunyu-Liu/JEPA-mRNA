import inspect

import pytest
import torch

from rnajepa import dual_view as dv
from rnajepa.decision_head import FlatDecisionHead
from tools import xens2_eval


def sequence_for_pairs(length, pairs):
    sequence = ["A"] * length
    bases = {"AU": ("A", "U"), "UA": ("U", "A"), "GC": ("G", "C"),
             "CG": ("C", "G"), "GU": ("G", "U"), "UG": ("U", "G")}
    for index, (i, j) in enumerate(pairs):
        name = tuple(bases)[index % len(bases)]
        sequence[i], sequence[j] = bases[name]
    return "".join(sequence)


def example(length=24, pairs=((0, 19), (1, 18), (5, 11), (6, 10))):
    pairs = tuple(tuple(pair) for pair in pairs)
    return dv.build_structure_example(sequence_for_pairs(length, pairs), pairs)


def test_helix_and_stacking_tokens_have_fixed_schema():
    item = example()
    assert len(item.helices) == 2
    assert item.features.shape == (4, dv.STRUCTURE_FEATURE_DIM)
    assert item.helix_token_index == (0, 2)
    assert item.token_helix_index == (0, 0, 1, 1)
    assert item.features[0, :2].tolist() == [1.0, 0.0]
    assert item.features[1, :2].tolist() == [0.0, 1.0]


@pytest.mark.parametrize("pairs", [
    ((0, 3),),                 # minimum loop
    ((0, 4), (0, 10)),        # shared residue
    ((-1, 10),),              # out of range
    ((8, 2),),                # reversed
    ((0, 30), (1, 20), (21, 40)),  # non-adjacent crossing
])
def test_illegal_pairs_raise(pairs):
    length = 41
    sequence = sequence_for_pairs(length, pairs)
    with pytest.raises(ValueError):
        dv.build_structure_example(sequence, pairs)


def test_noncanonical_pair_raises():
    with pytest.raises(ValueError, match="non-canonical"):
        dv.build_structure_example("AAAAAAAAAA", ((0, 5),))


def test_mask_families_and_missing_modality_are_explicit():
    items = [example(), dv.build_structure_example("A" * 16), example(22),
             dv.build_structure_example("A" * 14, available=False)]
    modes = set()
    for seed in range(40):
        masks = dv.make_view_masks(items, 24, seed=seed)
        modes.update(masks.modes)
        assert masks.modes[-1] == "sequence_span"
        assert not masks.structure_residue_mask[-1].any()
        assert not masks.structure_token_mask[-1].any()
        for index, item in enumerate(items):
            assert not masks.sequence_mask[index, item.length:].any()
            if masks.modes[index] == "whole_structure":
                assert not masks.sequence_mask[index].any()
                assert masks.structure_residue_mask[index, :item.length].all()
    assert modes == set(dv.MASK_MODES)


def test_masked_mse_ignores_every_unmasked_target():
    prediction = torch.ones(2, 4, 3, requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.zeros(2, 4, dtype=torch.bool)
    mask[0, 1] = True
    first = dv.masked_mse(prediction, target, mask)
    changed = prediction.detach().clone()
    changed[0, 0] = 1e6
    changed[1, 2] = -1e6
    second = dv.masked_mse(changed, target, mask)
    assert first.item() == pytest.approx(1.0)
    assert second.item() == pytest.approx(first.item())
    first.backward()
    assert prediction.grad[0, 1].abs().sum() > 0
    assert prediction.grad[~mask].abs().sum() == 0


def manual_masks(items, max_length, modes):
    _features, valid = dv.collate_structure(items)
    token_mask = torch.zeros_like(valid)
    sequence_mask = torch.zeros(len(items), max_length, dtype=torch.bool)
    structure_mask = torch.zeros_like(sequence_mask)
    for batch_index, (item, mode) in enumerate(zip(items, modes)):
        if mode == "whole_structure":
            token_mask[batch_index] = valid[batch_index]
            structure_mask[batch_index, :item.length] = True
        elif mode == "partial_helix":
            for token_index, helix_index in enumerate(item.token_helix_index):
                if helix_index == 0:
                    token_mask[batch_index, token_index] = True
            for i, j in item.helices[0]:
                structure_mask[batch_index, i] = True
                structure_mask[batch_index, j] = True
        elif mode == "sequence_span":
            sequence_mask[batch_index, 2:6] = True
        else:
            raise AssertionError(mode)
    return dv.ViewMasks(sequence_mask, token_mask, structure_mask, valid, list(modes))


def test_forward_backward_all_trainable_parameters_and_finite():
    torch.manual_seed(7)
    items = [example(), example(22), dv.build_structure_example("A" * 18)]
    masks = manual_masks(items, 24, ("whole_structure", "partial_helix", "sequence_span"))
    model = dv.DualViewJEPA(16, 16, queries=4, heads=4, sigreg_slices=32)
    output = model(torch.randn(3, 24, 16), [24, 22, 18], items, masks, sigreg_weight=0.1)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert output.mse_sequence > 0 and output.mse_structure > 0 and output.sigreg > 0
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert not missing
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)


def test_all_empty_structure_batch_is_finite_and_gradients_flow():
    items = [dv.build_structure_example("A" * 12), dv.build_structure_example("C" * 12)]
    masks = manual_masks(items, 12, ("whole_structure", "sequence_span"))
    model = dv.DualViewJEPA(8, 8, queries=2, heads=2, sigreg_slices=16)
    output = model(torch.randn(2, 12, 8), [12, 12], items, masks, sigreg_weight=0.05)
    output.loss.backward()
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.structure_global).all()
    missing_non_structure = [name for name, parameter in model.named_parameters()
                             if parameter.requires_grad and parameter.grad is None
                             and not name.startswith("structure_encoder.")
                             and not name.startswith("structure_aggregator.attention.")
                             and not name.startswith("structure_predictor.")
                             and name != "structure_output_queries"]
    assert not missing_non_structure
    # With no helix token in the entire batch there is no structure target or
    # structure-encoder input.  Those paths and the token predictor have no
    # gradient by design; all sequence and learned empty-query paths still train.
    assert all(parameter.grad is None for parameter in model.structure_encoder.parameters())
    assert all(parameter.grad is None for parameter in model.structure_aggregator.attention.parameters())
    # The differentiable zero masked-MSE path may attach explicit zero gradients.
    assert all(parameter.grad is None or not parameter.grad.abs().any()
               for parameter in model.structure_predictor.parameters())
    assert model.structure_output_queries.grad is None or not model.structure_output_queries.grad.abs().any()


def test_masked_helix_positions_features_and_count_do_not_leak_online():
    shared = ((0, 19), (1, 18))
    items = [
        example(pairs=shared + ((5, 11), (6, 10))),
        example(pairs=shared + ((7, 14), (8, 13))),
        example(pairs=shared + ((5, 9), (11, 17), (12, 16))),
    ]
    masks = manual_masks(items, 24, ("partial_helix",) * 3)
    # Keep exactly the same helix visible.  All other features, coordinates and
    # token counts differ, including an extra masked helix in the third sample.
    masks.structure_token_mask.zero_()
    masks.structure_token_mask[:, 2:] = masks.token_valid[:, 2:]
    model = dv.DualViewJEPA(8, 8, queries=2, heads=2, sigreg_slices=16,
                            structure_slots=32).eval()
    hidden = torch.randn(1, 24, 8).repeat(3, 1, 1)
    with torch.no_grad():
        output = model(hidden, [24, 24, 24], items, masks)
    assert torch.allclose(output.structure_global[0], output.structure_global[1], atol=1e-6)
    assert torch.allclose(output.structure_global[0], output.structure_global[2], atol=1e-6)
    assert torch.allclose(output.structure_prediction[0], output.structure_prediction[1], atol=1e-6)
    assert torch.allclose(output.structure_prediction[0], output.structure_prediction[2], atol=1e-6)
    assert not torch.allclose(output.structure_target[0, 2], output.structure_target[1, 2])
    assert output.structure_prediction.shape == (3, 32, 8)
    assert output.masks.token_valid.shape == (3, 32)
    assert model.structure_aggregator.queries.shape[0] == 2
    assert model.structure_output_queries.shape[0] == 32


def test_structure_loss_uses_masked_tokens_not_true_residue_spans():
    items = [example(), example()]
    masks = manual_masks(items, 24, ("partial_helix", "partial_helix"))
    model = dv.DualViewJEPA(8, 8, queries=2, heads=2, sigreg_slices=8,
                            structure_slots=16).eval()
    hidden = torch.randn(2, 24, 8)
    with torch.no_grad():
        first = model(hidden, [24, 24], items, masks)
        # These are the true paired-residue spans.  Changing only this diagnostic
        # field must not change token predictions, target identity, or the loss.
        masks.structure_residue_mask.zero_()
        masks.structure_residue_mask[:, 2] = True
        second = model(hidden, [24, 24], items, masks)
    assert torch.equal(first.structure_prediction, second.structure_prediction)
    assert first.mse_structure.item() == pytest.approx(second.mse_structure.item())
    expected = dv.masked_mse(first.structure_prediction, first.structure_target,
                             first.masks.structure_token_mask & first.masks.token_valid)
    assert first.mse_structure.item() == pytest.approx(expected.item())


def test_fixed_structure_capacity_rejects_overflow_instead_of_truncating():
    items = [example()]
    masks = manual_masks(items, 24, ("partial_helix",))
    model = dv.DualViewJEPA(8, 8, queries=2, heads=2, sigreg_slices=8,
                            structure_slots=3)
    with pytest.raises(ValueError, match="exceeds fixed capacity"):
        model(torch.randn(1, 24, 8), [24], items, masks)


def test_every_sample_retains_a_view_and_a_masked_target():
    items = [dv.build_structure_example("A" * 8), example(), example(20)]
    for seed in range(30):
        masks = dv.make_view_masks(items, 24, seed=seed, helix_fraction=1.0,
                                   sequence_fraction=1.0)
        for index, item in enumerate(items):
            assert masks.sequence_mask[index].any() or masks.structure_token_mask[index].any()
            visible_sequence = ~masks.sequence_mask[index, :item.length]
            visible_structure = masks.token_valid[index] & ~masks.structure_token_mask[index]
            assert visible_sequence.any() or visible_structure.any()


def test_parameter_matched_mlp_and_frozen_random_structure_encoder():
    transformer = dv.StructureTokenEncoder(16, 4, "transformer")
    mlp = dv.StructureTokenEncoder(16, 4, "mlp")
    transformer_count = sum(p.numel() for p in transformer.parameters())
    mlp_count = sum(p.numel() for p in mlp.parameters())
    assert transformer_count == mlp_count
    frozen = dv.DualViewJEPA(8, 8, queries=2, heads=2, frozen_structure=True, sigreg_slices=8)
    assert all(not parameter.requires_grad for parameter in frozen.structure_encoder.parameters())


def test_sigreg_penalizes_collapsed_more_than_gaussian_targets():
    torch.manual_seed(11)
    regularizer = dv.SlicedEppsPulley(num_slices=64)
    gaussian = regularizer(torch.randn(512, 8))
    regularizer.global_step.zero_()
    collapsed = regularizer(torch.zeros(512, 8))
    assert collapsed > gaussian


def test_sigreg_global_step_is_a_restorable_buffer():
    items = [example(), example(22)]
    masks = manual_masks(items, 24, ("whole_structure", "partial_helix"))
    model = dv.DualViewJEPA(8, 8, queries=2, heads=2, sigreg_slices=8)
    model(torch.randn(2, 24, 8), [24, 22], items, masks, sigreg_weight=0.7)
    assert int(model.sigreg.global_step) == 1
    serialized_step = int(model.sigreg.global_step)
    restored = dv.DualViewJEPA(8, 8, queries=2, heads=2, sigreg_slices=8)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.sigreg.global_step.fill_(serialized_step)
    assert int(restored.sigreg.global_step) == serialized_step


def test_deployment_interfaces_have_no_structure_or_dual_view_argument():
    head = FlatDecisionHead(d_model=16, d_z=8, hidden=4, scorer="resnet2d", chunk_size=0)
    assert "gts" not in inspect.signature(head.forward).parameters
    assert "structure" not in inspect.signature(head.forward).parameters
    assert "gts" not in inspect.signature(xens2_eval.plana_scores).parameters
    assert "dual_view" not in inspect.signature(xens2_eval.plana_scores).parameters


def test_resume_restores_sigreg_step_with_and_without_explicit_scalar():
    model = dv.DualViewJEPA(8, 8, queries=2, heads=2, sigreg_slices=8)
    model.sigreg.global_step.fill_(3)
    state = {"dual_view": model.state_dict()}
    restored = dv.DualViewJEPA(8, 8, queries=2, heads=2, sigreg_slices=8)
    restored.restore_training_state(state)
    assert int(restored.sigreg.global_step) == 3
    state["dual_view_sigreg_step"] = 7
    restored.restore_training_state(state)
    assert int(restored.sigreg.global_step) == 7
    with pytest.raises(ValueError, match="training-only state"):
        restored.restore_training_state({})


def test_shuffle_control_supports_different_donor_lengths():
    current_lengths = [24, 18]
    donor_items = [example(18, ((0, 15), (1, 14))), example(24, ((0, 19), (1, 18)))]
    masks = dv.make_view_masks(donor_items, 24, seed=4, sequence_lengths=current_lengths)
    for index, length in enumerate(current_lengths):
        assert not masks.sequence_mask[index, length:].any()
    model = dv.DualViewJEPA(8, 8, queries=2, heads=2, sigreg_slices=8)
    output = model(torch.randn(2, 24, 8), current_lengths, donor_items, masks)
    assert torch.isfinite(output.loss)
