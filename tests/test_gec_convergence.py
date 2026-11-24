import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from rnajepa import harness
from rnajepa.gec_context import (
    GECConvergenceStateV1,
    GECResidualV1,
    gec_destroy_key_v1,
    gec_geometry_v1,
    terminal_unfreeze_step_v1,
)
from rnajepa.gec_numeric_v1 import canonical_pair_mask_v1, compute_gec_context_v1
from rnajepa.raw_score_matrix import capped_unfreeze_blocks
from tools.train_plan_a_clean_head_p1 import (
    BatchStream,
    restore_batch_stream,
    restore_scheduler_state,
    restore_unfrozen_blocks,
)
from tools.train_plan_a_gec_convergence_v1 import (
    BATCH_NORM_POLICY_V2,
    MODEL_DEV_REGISTRY_SHA256,
    RUN_META_SCHEMA_V2,
    VALIDATION_DESTROY_STEP_V2,
    _append_jsonl,
    _atomic_json,
    _batch_norm_running_state,
    _finite_float_or_none,
    _read_previous_run_meta,
    _restore_batch_norm_running_state,
    load_gec_config,
    require_model_dev_registry,
    validate_checkpoint_identity,
    validate_converged_reentry,
    validate_run_identity,
    validate_smoke_artifact_identity,
)
from rnajepa.decision_head import FlatDecisionHead
from rnajepa.prior_alignment import sha256_file, tensor_state_sha256


def test_training_json_helpers_write_valid_json_and_jsonl(tmp_path):
    _append_jsonl({"step": 1, "loss": 1.25}, tmp_path / "train.jsonl")
    _append_jsonl({"step": 2, "loss": 1.0}, tmp_path / "train.jsonl")
    import json
    rows = [json.loads(line) for line in (tmp_path / "train.jsonl").read_text().splitlines()]
    assert rows == [{"step": 1, "loss": 1.25}, {"step": 2, "loss": 1.0}]
    _atomic_json({"best_validation_loss": None}, tmp_path / "meta.json")
    assert json.loads((tmp_path / "meta.json").read_text()) == {"best_validation_loss": None}
    assert _finite_float_or_none(1.5) == 1.5
    assert _finite_float_or_none(float("nan")) is None
    assert _finite_float_or_none(float("inf")) is None
    _append_jsonl({"step": 3, "loss": _finite_float_or_none(float("nan")),
                   "divergence_reason": "NON_FINITE_LOSS_OR_GRADIENT_NORM"}, tmp_path / "train.jsonl")
    assert json.loads((tmp_path / "train.jsonl").read_text().splitlines()[-1])["loss"] is None


def test_terminal_schedule_is_20_of_33_at_9600_and_never_exceeds_cap():
    assert terminal_unfreeze_step_v1() == 9600
    expected = {0: 0, 1600: 0, 1601: 0, 2400: 2, 8800: 18, 9600: 20,
                9601: 20, 10400: 20, 20000: 20}
    for step, count in expected.items():
        if step:
            assert capped_unfreeze_blocks(step, 1600, 800, 2, 33, 20) == count
    blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(33)])
    restore_unfrozen_blocks(blocks, 20)
    assert sum(block.weight.requires_grad for block in blocks) == 20
    assert [index for index, block in enumerate(blocks) if block.weight.requires_grad] == list(range(13, 33))


def test_preterminal_validation_cannot_select_or_stop():
    state = GECConvergenceStateV1()
    for step in range(500, 9600, 500):
        decision = state.observe(step, 0.0, capped_unfreeze_blocks(step, 1600, 800, 2, 33, 20))
        assert not decision["eligible"] and not decision["save_best"] and not decision["stop"]
    assert state.best_step is None and state.bad_evaluations == 0


def test_first_terminal_validation_resets_legacy_state_and_patience_selects_best():
    state = GECConvergenceStateV1()
    state.best_step = 8000
    state.best_validation_loss = -100.0
    state.bad_evaluations = 4
    first = state.observe(9600, 2.0, 20)
    assert first["eligible"] and first["save_best"] and not first["stop"]
    assert state.best_step == 9600 and state.best_validation_loss == 2.0
    assert not state.observe(10100, 2.0 - 1e-6, 20)["improved"]
    assert state.observe(10600, 1.9999, 20)["save_best"]
    for step in (11100, 11600, 12100, 12600):
        assert not state.observe(step, 2.1, 20)["stop"]
    assert state.observe(13100, 2.1, 20)["stop"]
    assert state.best_step == 10600


def test_convergence_resume_identity_and_nonfinite_guards():
    state = GECConvergenceStateV1()
    state.observe(9600, 1.0, 20)
    restored = GECConvergenceStateV1()
    restored.load_state_dict(copy.deepcopy(state.state_dict()))
    assert restored.state_dict() == state.state_dict()
    with pytest.raises(ValueError, match="identity"):
        GECConvergenceStateV1(patience=6).load_state_dict(state.state_dict())
    with pytest.raises(FloatingPointError):
        state.observe(10100, float("nan"), 20)


def test_batch_order_is_identical_across_arms_and_resume_safe():
    records = [(f"AUAUAUAU{index % 2}", [(0, 5)]) for index in range(12)]
    streams = [BatchStream(records, 4, 0) for _ in range(4)]
    first = [stream.next() for stream in streams]
    assert all(bucket == first[0] for bucket in first)
    saved = copy.deepcopy(streams[0].state)
    expected = [streams[0].next() for _ in range(3)]
    for stream in streams[1:]:
        restore_batch_stream(stream, copy.deepcopy(saved), 4)
        assert [stream.next() for _ in range(3)] == expected


def test_registry_pin_previous_meta_and_checkpoint_identity(tmp_path):
    config = load_gec_config(REPO / "configs/gec_four_arm_convergence_v1.json")
    registry_path, _ = require_model_dev_registry(config)
    assert sha256_file(registry_path) == MODEL_DEV_REGISTRY_SHA256
    assert MODEL_DEV_REGISTRY_SHA256 == "62ae08e06f1b3a291c2de2a9cf52d2e57ec55570396fc810cd3638612c774902"
    identity = {
        "arm": "matched_control", "gec_config_sha256": "config",
        "shared_init_sha256": "shared", "gec_init_sha256": "gec",
        "initial_residual_tensor_sha256": "residual",
        "train_record_ids": ["train"], "validation_record_ids": ["validation"],
        "model_dev_registry_sha256": MODEL_DEV_REGISTRY_SHA256,
    }
    previous = {"schema_version": RUN_META_SCHEMA_V2, "status": "interrupted",
                "run_identity": identity, **identity}
    path = tmp_path / "run_meta.json"
    path.write_text(json.dumps(previous))
    assert _read_previous_run_meta(path) == previous
    validate_run_identity(previous, copy.deepcopy(identity))
    validate_checkpoint_identity(
        {"format": "foldnar.gec_checkpoint.v2", "run_identity": identity}, identity,
    )
    changed = copy.deepcopy(identity)
    changed["validation_record_ids"] = ["stale"]
    with pytest.raises(ValueError, match="resume identity mismatch"):
        validate_run_identity(previous, changed)
    path.write_text("{not-json")
    with pytest.raises(json.JSONDecodeError):
        _read_previous_run_meta(path)
    assert path.read_text() == "{not-json"


def test_converged_reentry_requires_restored_best_eligible_checkpoint():
    identity = {"terminal_unfrozen_blocks": 20}
    previous = {
        "converged": True, "restored_best_step": 9600,
        "convergence": {"best_step": 9600},
    }
    common = {
        "format": "foldnar.gec_checkpoint.v2", "run_identity": identity,
        "step": 9600, "n_unfrozen": 20, "model_tensor_sha256": {"residual": "best"},
    }
    state = {**common, "checkpoint_phase": "restored_best_eligible", "terminal_status": "converged"}
    best = {**common, "checkpoint_phase": "best_eligible"}
    validate_converged_reentry(previous, state, best, identity)
    with pytest.raises(ValueError, match="restored best eligible"):
        validate_converged_reentry(previous, {**state, "step": 10600}, best, identity)
    with pytest.raises(ValueError, match="tensor identities"):
        validate_converged_reentry(
            previous, state, {**best, "model_tensor_sha256": {"residual": "other"}}, identity,
        )


def test_stale_smoke_identity_cannot_be_rebranded_as_current():
    expected = {"code_manifest_sha256": "current", "data_sha256": "data"}
    with pytest.raises(ValueError, match="stale or invalid"):
        validate_smoke_artifact_identity(
            {"code_manifest_sha256": "old", "data_sha256": "data"}, expected,
        )


def _small_resnet_head():
    torch.manual_seed(41)
    return FlatDecisionHead(
        d_model=4, d_z=4, hidden=4, scorer="resnet2d", chunk_size=0,
        resnet_blocks=1, resnet_hidden=4, use_turner_prior=False,
    )


def test_batchnorm_checkpoint_recomputation_preserves_single_running_update():
    head = _small_resnet_head()
    head.train()
    z = torch.randn(1, 8, 8, 4, requires_grad=True)
    saved = _batch_norm_running_state(head)
    scores = torch.utils.checkpoint.checkpoint(
        head.score_pair_matrix, z, None, False, None, None, use_reentrant=False,
    )
    scores.square().mean().backward()
    batch_norms = [module for module in head.resnet2d.modules()
                   if isinstance(module, torch.nn.BatchNorm2d)]
    assert batch_norms and all(module.num_batches_tracked == 2 for module in batch_norms)
    _restore_batch_norm_running_state(saved)
    assert all(module.num_batches_tracked == 0 for module in batch_norms)
    for module, running_mean, running_var, _ in saved:
        assert torch.equal(module.running_mean, running_mean)
        assert torch.equal(module.running_var, running_var)
    assert z.grad is not None and torch.isfinite(z.grad).all()


def test_padding_batch_dependence_is_explicit_and_eval_uses_frozen_stats():
    single = _small_resnet_head()
    batched = copy.deepcopy(single)
    single.train()
    batched.train()
    z = torch.randn(1, 8, 8, 4)
    padded = torch.zeros(2, 12, 12, 4)
    padded[0, :8, :8] = z[0]
    padded[1, :9, :9] = torch.randn(9, 9, 4)
    with torch.no_grad():
        single_score = single.score_pair_matrix(z)
        batch_score = batched.score_pair_matrix(padded)[0, :8, :8]
    assert not torch.allclose(single_score, batch_score)
    single.eval()
    with torch.no_grad():
        eval_single = single.score_pair_matrix(z)
        eval_batch = single.score_pair_matrix(padded)[0, :8, :8]
        eval_single_repeat = single.score_pair_matrix(z)
    assert torch.equal(eval_single, eval_single_repeat)
    assert torch.allclose(eval_single, eval_batch, atol=1e-6, rtol=1e-5)
    assert BATCH_NORM_POLICY_V2 == "inherited_padded_batch_train_once_plus_frozen_singleton_eval"


def test_validation_destroy_permutation_identity_is_frozen_across_steps():
    sequence = "AUAUAUAUAUAU"
    mask = canonical_pair_mask_v1(sequence)
    scores = torch.randn(len(sequence), len(sequence), dtype=torch.float64)
    scores = torch.triu(scores, 1)
    scores = scores + scores.T
    key = gec_destroy_key_v1(0, 0, VALIDATION_DESTROY_STEP_V2, sequence)
    first = compute_gec_context_v1(
        scores, mask, mode="destroy_interval_shuffle", shuffle_key=key,
    )
    later = compute_gec_context_v1(
        scores, mask, mode="destroy_interval_shuffle", shuffle_key=key,
    )
    changed = compute_gec_context_v1(
        scores, mask, mode="destroy_interval_shuffle",
        shuffle_key=gec_destroy_key_v1(0, 0, 500, sequence),
    )
    assert torch.equal(first.context, later.context)
    assert not torch.equal(first.context, changed.context)


def _assert_nested_equal(expected, actual, path=""):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(expected, actual), path
    elif isinstance(expected, dict):
        assert expected.keys() == actual.keys(), path
        for key in expected:
            _assert_nested_equal(expected[key], actual[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual), path
        for index, (left, right) in enumerate(zip(expected, actual)):
            _assert_nested_equal(left, right, f"{path}.{index}")
    elif isinstance(expected, np.ndarray):
        assert np.array_equal(expected, actual), path
    else:
        assert expected == actual, path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.skipif(not os.environ.get("GEC_ACTIVE_RESUME_ROOT"),
                    reason="set GEC_ACTIVE_RESUME_ROOT for the real active-trainer test")
def test_active_trainer_interrupted_resume_matches_uninterrupted_step_two():
    root = Path(os.environ["GEC_ACTIVE_RESUME_ROOT"])
    reference_dir = root / "active_resume_reference"
    resumed_dir = root / "active_resume_resumed"
    if reference_dir.exists() or resumed_dir.exists():
        raise RuntimeError("active resume regression requires fresh output directories")
    python = sys.executable
    config = REPO / "configs/gec_four_arm_convergence_v1.json"
    shared_init = json.loads(config.read_text())["assets"]["shared_init"]
    gec_init = root / "gec_shared_residual_init_seed0.pt"
    env = dict(os.environ, PYTHONPATH=f"{REPO / 'src'}:{REPO}", CUDA_VISIBLE_DEVICES="0",
               RN_CUDA_INSIDE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    if not gec_init.exists():
        subprocess.run([
            python, str(REPO / "tools/init_gec_residual_v1.py"), "--shared-init", shared_init,
            "--gec-config", str(config), "--out", str(gec_init), "--seed", "0",
        ], check=True, env=env)
    subprocess.run([
        python, str(REPO / "tools/verify_gec_smoke_v1.py"), "--root", str(root),
        "--repo", str(REPO), "--write-identity", "--config", str(config),
        "--shared-init", shared_init, "--gec-init", str(gec_init),
    ], check=True, env=env)

    def run(out, interrupt=False):
        command = [
            python, str(REPO / "tools/train_plan_a_gec_convergence_v1.py"),
            "--gec-config", str(config), "--arm", "matched_control", "--out", str(out),
            "--shared-init", shared_init, "--gec-init", str(gec_init),
            "--device", "cuda:0", "--engineering-smoke", "--steps", "2",
            "--train-record-limit", "8", "--validation-record-limit", "2",
            "--validation-every", "0", "--save-every", "1",
        ]
        if interrupt:
            command += ["--smoke-interrupt-after-step", "1"]
        subprocess.run(command, check=True, env=env)

    run(reference_dir)
    run(resumed_dir, interrupt=True)
    interrupted = json.loads((resumed_dir / "run_meta.json").read_text())
    interrupted_checkpoint = torch.load(resumed_dir / "resume.pt", map_location="cpu", weights_only=False)
    assert interrupted["status"] == "interrupted" and interrupted["steps_completed"] == 1
    assert interrupted_checkpoint["step"] == 1
    run(resumed_dir)
    reference_meta = json.loads((reference_dir / "run_meta.json").read_text())
    resumed_meta = json.loads((resumed_dir / "run_meta.json").read_text())
    assert resumed_meta["resume_history"][0]["from_step"] == 1
    reference = torch.load(reference_dir / "resume.pt", map_location="cpu", weights_only=False)
    resumed = torch.load(resumed_dir / "resume.pt", map_location="cpu", weights_only=False)
    for name in ("encoder", "head", "residual", "optimizer", "scheduler", "stream_state", "random_state"):
        _assert_nested_equal(reference[name], resumed[name], name)
    assert reference["model_tensor_sha256"] == resumed["model_tensor_sha256"]
    assert reference["step"] == resumed["step"] == 2
    assert reference["optimizer_steps"] == resumed["optimizer_steps"] == 2

    def ledger(directory, name):
        rows = [json.loads(line) for line in (directory / name).read_text().splitlines()]
        if name == "train_log.jsonl":
            for row in rows:
                row.pop("elapsed_seconds")
                row.pop("peak_cuda_bytes")
        return rows

    assert ledger(reference_dir, "batch_order.jsonl") == ledger(resumed_dir, "batch_order.jsonl")
    assert ledger(reference_dir, "train_log.jsonl") == ledger(resumed_dir, "train_log.jsonl")
    before = sha256_file(resumed_dir / "run_meta.json")
    run(resumed_dir)
    assert sha256_file(resumed_dir / "run_meta.json") == before
    evidence = {
        "schema_version": "foldnar.gec_active_resume_determinism.v2",
        "status": "PASS_EXACT_ACTIVE_TRAINER_RESUME",
        "device": "cuda:0", "cpu_fallback": False,
        "arm": "matched_control", "interrupted_step": 1, "final_step": 2,
        "reference_resume_sha256": sha256_file(reference_dir / "resume.pt"),
        "resumed_resume_sha256": sha256_file(resumed_dir / "resume.pt"),
        "model_tensor_sha256": resumed["model_tensor_sha256"],
        "reference_run_meta_sha256": sha256_file(reference_dir / "run_meta.json"),
        "resumed_run_meta_sha256": sha256_file(resumed_dir / "run_meta.json"),
        "reference_residual_tensor_sha256": reference_meta["final_residual_tensor_sha256"],
        "resumed_residual_tensor_sha256": resumed_meta["final_residual_tensor_sha256"],
        "completed_reentry_noop": True,
    }
    (root / "active_resume_determinism_v2.json").write_text(json.dumps(evidence, indent=2) + "\n")
