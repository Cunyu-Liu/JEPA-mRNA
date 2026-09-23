"""Tests for the pretraining execution harness (Task 18), the teacher-throughput
benchmark (Task 11) and the paper scaffolding / manuscript linter (Task 21).

Everything runs on CPU with tiny synthetic data and no network.  ViennaRNA *is*
installed out-of-tree at ``/mnt/cunyuliu/pylibs`` (the ``/home`` quota is full),
so the thermodynamic-teacher tests are written to hold in either environment:
the assertion is "a real tool is never silently replaced by the mock", not "this
host happens to lack the tool".

Run:  python -m pytest tests/test_training_and_paper.py -v
  or: python tests/test_training_and_paper.py
"""

import json
import math
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "eval"))
sys.path.insert(0, os.path.join(_ROOT, "paper"))

import check_manuscript as CM  # noqa: E402
import rnajepa.train_decision as TD  # noqa: E402
from rnajepa.distill import MockTeacher  # noqa: E402
from rnajepa.harness import inside_outside, valid_pair_mask  # noqa: E402
from rnajepa.rlcd import ObjectiveWeights  # noqa: E402

from ss import teacher_throughput as TP  # noqa: E402

PAPER_DIR = os.path.join(_ROOT, "paper")
STUB = os.path.join(PAPER_DIR, "manuscript_stub.md")
OBJECTIONS = os.path.join(PAPER_DIR, "reviewer_objections.md")


def test_loader_skips_teacher_probs_when_not_needed(tmp_path):
    """need_teacher_probs=False must not run the O(L^3) mock DP.

    Regression: the loader called probs_for() for every record unconditionally,
    and with no teacher dir that runs a numpy inside-outside DP per sequence --
    ~2.5e10 operations over a 10k corpus, which made a real run look hung at
    100% CPU with no GPU work.
    """
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps({"seq": "ACGUACGUACGU", "structure": "(((...)))..."}) + "\n",
        encoding="utf-8")
    store = TD.TeacherLabelStore.from_mock()

    lazy = TD.load_dataset_from_jsonl(str(path), store, need_teacher_probs=False)
    assert lazy[0].teacher_probs is None
    assert lazy.describe()["has_teacher_probs"] is False

    eager = TD.load_dataset_from_jsonl(str(path), store, need_teacher_probs=True)
    assert eager[0].teacher_probs is not None
    assert eager.describe()["has_teacher_probs"] is True


def test_objective_refuses_missing_teacher_labels():
    """lambda_distill != 0 with no labels must fail, not zero-fill."""
    dataset = TD.make_synthetic_dataset(4, 16, seed=0)
    dataset.examples = [
        TD.DecisionExample(seq=e.seq, gt_pairs=e.gt_pairs, teacher_probs=None)
        for e in dataset.examples]
    teacher = TD.TeacherLabelStore.from_mock()
    with tempfile.TemporaryDirectory() as out:
        config = TD.TrainConfig(tiny=True, steps=2, batch_size=2, out_dir=out,
                                allow_cpu=True, lambda_distill=1.0)
        try:
            TD.run_training(config, dataset, teacher)
        except TD.ConfigError as exc:
            assert "teacher probabilities" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("missing teacher labels must not be zero-filled")


def test_objective_accepts_absent_teacher_labels_when_distill_is_off():
    """The same dataset must train fine with lambda_distill == 0."""
    dataset = TD.make_synthetic_dataset(4, 16, seed=0)
    dataset.examples = [
        TD.DecisionExample(seq=e.seq, gt_pairs=e.gt_pairs, teacher_probs=None)
        for e in dataset.examples]
    teacher = TD.TeacherLabelStore.from_mock()
    with tempfile.TemporaryDirectory() as out:
        config = TD.TrainConfig(tiny=True, steps=2, batch_size=2, out_dir=out,
                                allow_cpu=True, lambda_distill=0.0,
                                lambda_rlcd=0.0, lambda_cal=0.0)
        result = TD.run_training(config, dataset, teacher)
        assert result["steps_completed"] == 2


def test_strict_teacher_lookup_refuses_the_mock_fallback():
    """A real teacher dir must not silently mix in mock labels."""
    store = TD.TeacherLabelStore({}, name="thermo:viennarna", version_lock="2.7.2",
                                 source="dir:/nowhere", n_labels=0)
    # non-strict keeps the documented mock fallback (tests, no-teacher runs)
    assert store.probs_for("ACGUACGU").shape == (8, 8)
    # strict refuses
    try:
        store.probs_for("ACGUACGU", strict=True)
    except TD.ConfigError as exc:
        assert "mock teacher" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("strict lookup must not fall back to the mock teacher")


def test_strict_teacher_lookup_returns_stored_labels():
    seq = "ACGUACGUACGU"
    probs = np.triu(np.ones((12, 12)) * 0.1, k=1)
    store = TD.TeacherLabelStore({seq: probs}, name="thermo:viennarna",
                                 version_lock="2.7.2", source="dir:x", n_labels=1)
    assert np.allclose(store.probs_for(seq, strict=True), probs)


def test_cpu_requires_an_explicit_opt_in():
    """A silent CPU run must be impossible: it would still yield a checkpoint."""
    dataset = TD.make_synthetic_dataset(4, 16, seed=0)
    teacher = TD.TeacherLabelStore.from_mock()
    with tempfile.TemporaryDirectory() as out:
        config = TD.TrainConfig(tiny=True, steps=2, batch_size=2, out_dir=out,
                                log_every=1)          # allow_cpu deliberately absent
        try:
            TD.run_training(config, dataset, teacher)
        except TD.ConfigError as exc:
            assert "--allow-cpu" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("device='cpu' without --allow-cpu must be refused")


def test_cuda_request_is_refused_when_unavailable():
    """Requesting cuda on a CPU-only host must fail, not fall back."""
    import torch

    dataset = TD.make_synthetic_dataset(4, 16, seed=0)
    teacher = TD.TeacherLabelStore.from_mock()
    with tempfile.TemporaryDirectory() as out:
        config = TD.TrainConfig(tiny=True, steps=2, batch_size=2, out_dir=out,
                                log_every=1, device="cuda", allow_cpu=False)
        try:
            has_cuda = bool(torch.cuda.is_available())
        except Exception:   # CPU-only torch builds raise instead of returning False
            has_cuda = False
        if has_cuda:
            print("[skip] this host has CUDA, so the refusal path is not reachable")
            return
        try:
            TD.run_training(config, dataset, teacher)
        except TD.ConfigError as exc:
            assert "cuda" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("device='cuda' without CUDA must be refused")


def test_run_meta_records_the_resolved_device():
    dataset = TD.make_synthetic_dataset(4, 16, seed=0)
    teacher = TD.TeacherLabelStore.from_mock()
    with tempfile.TemporaryDirectory() as out:
        result = TD.run_training(_tiny_config(out, steps=2), dataset, teacher)
        env = result["run_meta"]["environment"]
        assert env["device"] == "cpu"
        assert env["resolved_device"] == "cpu"
        assert "cuda_visible_devices" in env and "gpu_name" in env


# ---------------------------------------------------------------------------
# shared fixtures / helpers
# ---------------------------------------------------------------------------
def _tiny_config(out_dir, **overrides):
    # allow_cpu: the driver refuses device="cpu" unless this is set, so a silent
    # CPU training run cannot be mistaken for a real one. This fixture is the
    # CPU path by definition, so it opts in.
    kwargs = dict(tiny=True, steps=24, batch_size=2, lr=5e-3, seed=0,
                  log_every=1, warmup_steps=2, out_dir=out_dir, allow_cpu=True)
    kwargs.update(overrides)
    return TD.TrainConfig(**kwargs)


def _read_log(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_ledger(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ===========================================================================
# 1. the training driver runs end to end on CPU, loss is finite and decreasing
# ===========================================================================
def test_training_runs_on_cpu_loss_finite_and_decreasing():
    dataset = TD.make_synthetic_dataset(8, 24, seed=0)
    teacher = TD.TeacherLabelStore.from_mock()
    with tempfile.TemporaryDirectory() as out:
        config = _tiny_config(out)
        result = TD.run_training(config, dataset, teacher)

        assert result["dry_run"] is False
        assert result["steps_completed"] == config.steps

        rows = _read_log(os.path.join(out, "train_log.jsonl"))
        assert len(rows) == config.steps, "one log row per step was expected"
        losses = [row["loss"] for row in rows]
        assert all(math.isfinite(value) for value in losses), losses

        half = len(losses) // 2
        first_half = float(np.mean(losses[:half]))
        second_half = float(np.mean(losses[half:]))
        assert second_half < first_half, (
            f"loss did not decrease: first half {first_half:.4f} vs "
            f"second half {second_half:.4f} (curve {losses})")
        assert math.isfinite(result["final_loss"])

        # every per-term value is finite too (the four-term objective is wired up)
        for row in rows:
            assert set(row["terms"]) == {"nll", "distill", "rlcd", "cal"}
            assert all(math.isfinite(v) for v in row["terms"].values())

        # artefacts + ledger
        for name in ("run_meta.json", "train_log.jsonl", "ledger.jsonl", "resume.pt"):
            assert os.path.isfile(os.path.join(out, name)), name
        ledger = _read_ledger(os.path.join(out, "ledger.jsonl"))
        assert [row["status"] for row in ledger] == ["start", "completed"]

        meta = json.loads(open(os.path.join(out, "run_meta.json"), encoding="utf-8").read())
        assert meta["status"] == "completed"
        assert meta["config"]["lr"] == config.lr
        assert meta["data"]["n_examples"] == 8
        assert meta["teacher"]["is_mock"] is True
        # the honesty block is carried into every run record
        assert "CONTRAfold" in meta["honesty"]["gibbs_framework"]
        assert "RLCD-inspired" in meta["honesty"]["rlcd"]


def test_gradient_coverage_assertion_passes_and_covers_every_loss_path_block():
    dataset = TD.make_synthetic_dataset(4, 20, seed=1)
    teacher = TD.TeacherLabelStore.from_mock()
    with tempfile.TemporaryDirectory() as out:
        result = TD.run_training(_tiny_config(out, steps=4), dataset, teacher)
    coverage = result["run_meta"]["gradient_coverage"]
    assert coverage["missing"] == [], coverage["missing"]
    assert coverage["nonfinite"] == []
    assert len(coverage["covered"]) >= 20
    # the only excluded parameters are the two documented, off-loss-path blocks
    assert all(name.startswith(("head.type_head", "encoder.channel_proj"))
               for name in coverage["excluded"]), coverage["excluded"]
    assert any(name.startswith("head.type_head") for name in coverage["excluded"])
    assert any(name.startswith("encoder.channel_proj") for name in coverage["excluded"])


def test_gradient_coverage_assertion_is_not_vacuous():
    """A disconnected block must be caught, and the zero-init window is documented."""
    import torch.nn as nn

    class TwoHeads(nn.Module):
        def __init__(self):
            super().__init__()
            self.used = nn.Linear(4, 4)
            self.unused = nn.Linear(4, 4)          # never called -> grad is None

        def forward(self, x):
            return self.used(x)

    module = TwoHeads()
    module(torch.ones(2, 4)).sum().backward()
    report = TD.gradient_coverage(module)
    assert report["missing"] == ["unused.weight", "unused.bias"]
    with pytest.raises(TD.GradientCoverageError):
        TD.assert_gradient_coverage(module)

    # The zero-initialised MLP_T makes the *first* backward pass dead for the
    # encoder; the assertion is therefore evaluated after the first optimizer step.
    config = _tiny_config("", steps=1)
    dataset = TD.make_synthetic_dataset(4, 20, seed=2)
    model = TD.build_decision_model(config)
    batch = TD.collate([dataset[0], dataset[1]])
    weights = ObjectiveWeights(1.0, 1.0, 1.0, 1.0)
    loss, _terms = TD.objective_terms(model, batch, weights)
    loss.backward()
    with pytest.raises(TD.GradientCoverageError):
        TD.assert_gradient_coverage(model)                       # step-0 dead gradient
    report0 = TD.gradient_coverage(model)
    assert report0["zero"], "the zero-init dead-gradient window should be visible"
    assert report0["missing"] == []


def test_nan_guard_triggers_on_injected_nan():
    dataset = TD.make_synthetic_dataset(4, 20, seed=3)
    teacher = TD.TeacherLabelStore.from_mock()
    original = TD.objective_terms

    def poisoned(model, batch, weights, **kwargs):
        loss, terms = original(model, batch, weights, **kwargs)
        return loss * float("nan"), terms

    TD.objective_terms = poisoned
    try:
        with tempfile.TemporaryDirectory() as out:
            with pytest.raises(TD.DivergenceError):
                TD.run_training(_tiny_config(out, steps=3), dataset, teacher)
            # the failure is recorded, not swallowed
            ledger = _read_ledger(os.path.join(out, "ledger.jsonl"))
            assert ledger[-1]["status"] == "diverged"
            meta = json.loads(open(os.path.join(out, "run_meta.json"),
                                   encoding="utf-8").read())
            assert meta["status"] == "diverged"
            assert os.path.isfile(os.path.join(out, "resume.pt"))
    finally:
        TD.objective_terms = original


def test_training_resumes_from_checkpoint_and_appends_to_the_ledger():
    dataset = TD.make_synthetic_dataset(4, 20, seed=4)
    teacher = TD.TeacherLabelStore.from_mock()
    with tempfile.TemporaryDirectory() as out:
        first = TD.run_training(_tiny_config(out, steps=3), dataset, teacher)
        assert first["steps_completed"] == 3
        second = TD.run_training(_tiny_config(out, steps=6), dataset, teacher)
        assert second["steps_completed"] == 6
        ledger = _read_ledger(os.path.join(out, "ledger.jsonl"))
        assert [row["status"] for row in ledger] == ["start", "completed",
                                                     "start", "completed"]
        assert ledger[2]["resume_from"] == 3


# ===========================================================================
# 2. every loss weight is independently switchable
# ===========================================================================
def test_four_loss_weights_are_independently_switchable():
    config = _tiny_config("", steps=1)
    dataset = TD.make_synthetic_dataset(4, 20, seed=5)
    model = TD.build_decision_model(config)
    batch = TD.collate([dataset[0], dataset[1]])

    all_on = ObjectiveWeights(1.0, 1.0, 1.0, 1.0)
    total_all, terms = TD.objective_terms(model, batch, all_on)
    total_all = float(total_all.detach())
    assert math.isfinite(total_all)
    assert set(terms) == {"nll", "distill", "rlcd", "cal"}
    for name, value in terms.items():
        assert abs(value) > 0.0, f"term {name} is zero; the test would be vacuous"

    # the total is exactly the weighted sum of the terms
    expected = sum(terms.values())
    assert abs(total_all - expected) < 1e-9

    for name in ("nll", "distill", "rlcd", "cal"):
        off = _weights_without(name)
        total_off = float(TD.objective_terms(model, batch, off)[0].detach())
        assert abs(total_off - total_all) > 1e-9, f"switching off {name} changed nothing"
        assert abs(total_off - (total_all - terms[name])) < 1e-9

    # all four off is rejected by the config validator (nothing to optimise)
    zero = TD.TrainConfig(tiny=True, steps=1, out_dir="", allow_cpu=True,
                          lambda_nll=0.0,
                          lambda_distill=0.0, lambda_rlcd=0.0, lambda_cal=0.0)
    with pytest.raises(TD.ConfigError):
        zero.validate()

    # and the switch is visible in the dry-run plan
    dataset2 = TD.make_synthetic_dataset(2, 20, seed=6)
    teacher = TD.TeacherLabelStore.from_mock()
    plan = TD.plan(TD.TrainConfig(tiny=True, steps=1, out_dir="", allow_cpu=True,
                                  lambda_rlcd=0.0),
                   dataset2, teacher)
    assert plan["active_terms"] == ["nll", "distill", "cal"]
    assert plan["objective"]["rlcd"] == 0.0


def _weights_without(name):
    values = {"nll": 1.0, "distill": 1.0, "rlcd": 1.0, "cal": 1.0}
    values[name] = 0.0
    return ObjectiveWeights(lambda_nll=values["nll"], lambda_distill=values["distill"],
                            lambda_rlcd=values["rlcd"], lambda_cal=values["cal"])


# ===========================================================================
# 3. learning-rate calibration
# ===========================================================================
def test_lr_calibration_returns_a_value_and_records_it():
    dataset = TD.make_synthetic_dataset(6, 20, seed=7)
    config = _tiny_config("", steps=1)
    report = TD.calibrate_learning_rate(config, dataset,
                                        candidates=(1e-3, 5e-3), probe_steps=2)
    assert report["chosen"] in (1e-3, 5e-3)
    assert len(report["rows"]) == 2
    assert report["probe_steps"] == 2
    for row in report["rows"]:
        assert row["lr"] in (1e-3, 5e-3)
        assert row["diverged"] is False
        assert math.isfinite(row["final_loss"])

    teacher = TD.TeacherLabelStore.from_mock()
    with tempfile.TemporaryDirectory() as out:
        cfg = _tiny_config(out, steps=3, lr_calibrate=True,
                           lr_candidates=(1e-3, 5e-3), lr_probe_steps=2)
        result = TD.run_training(cfg, dataset, teacher)
        meta = result["run_meta"]
        assert meta["lr_calibration"]["chosen"] in (1e-3, 5e-3)
        # the measured value is what the run actually used and recorded
        assert meta["config"]["lr"] == meta["lr_calibration"]["chosen"]
        assert meta["lr_calibration"]["probe_steps"] == 2


def test_lr_calibration_records_diverged_candidates_and_falls_back():
    dataset = TD.make_synthetic_dataset(4, 20, seed=8)
    config = _tiny_config("", steps=1)
    original = TD.objective_terms

    def poisoned(model, batch, weights, **kwargs):
        loss, terms = original(model, batch, weights, **kwargs)
        return loss * float("nan"), terms

    TD.objective_terms = poisoned
    try:
        report = TD.calibrate_learning_rate(config, dataset,
                                            candidates=(5e-3, 1e-3), probe_steps=1)
    finally:
        TD.objective_terms = original

    assert len(report["rows"]) == 2
    assert all(row["diverged"] for row in report["rows"])
    assert all(row["final_loss"] is None for row in report["rows"])
    # with every candidate diverged, the most conservative one is chosen and said so
    assert report["chosen"] == 1e-3
    assert "diverged" in report["reason"]


# ===========================================================================
# 4. --dry-run performs no training
# ===========================================================================
def test_dry_run_performs_no_training():
    dataset = TD.make_synthetic_dataset(4, 20, seed=9)
    teacher = TD.TeacherLabelStore.from_mock()
    with tempfile.TemporaryDirectory() as parent:
        out = os.path.join(parent, "never_created")
        plan = TD.run_training(_tiny_config(out, steps=5), dataset, teacher,
                               dry_run=True)
        assert plan["dry_run"] is True
        assert plan["steps"] == 5
        assert plan["data"]["n_examples"] == 4
        assert not os.path.exists(out), "dry-run created the run directory"
        assert not os.path.exists(os.path.join(out, "resume.pt"))
        assert not os.path.exists(os.path.join(out, "train_log.jsonl"))

        # the CLI path too
        code = TD.main(["--synthetic", "4", "--length", "20", "--tiny",
                        "--steps", "3", "--out", out, "--dry-run"])
        assert code == 0
        assert not os.path.exists(out)

        # and a real run does create them (so the assertion above is meaningful)
        TD.run_training(_tiny_config(out, steps=2), dataset, teacher)
        assert os.path.isfile(os.path.join(out, "resume.pt"))


def test_invalid_configuration_is_rejected_before_anything_runs():
    dataset = TD.make_synthetic_dataset(2, 20, seed=10)
    teacher = TD.TeacherLabelStore.from_mock()
    with pytest.raises(TD.ConfigError):
        TD.plan(TD.TrainConfig(tiny=True, steps=0, out_dir="", allow_cpu=True),
                dataset, teacher)
    with pytest.raises(TD.ConfigError):
        TD.plan(TD.TrainConfig(tiny=True, steps=1, out_dir="", lr=0.0,
                               allow_cpu=True), dataset, teacher)
    with pytest.raises(TD.ConfigError):
        TD.plan(TD.TrainConfig(tiny=True, steps=1, out_dir="", allow_cpu=True,
                               distill_kind="nope"),
                dataset, teacher)


# ===========================================================================
# 5. teacher throughput benchmark (mock) + ensemble + self-consistency
# ===========================================================================
def test_teacher_throughput_mock_report_labels_extrapolation_as_mock():
    teacher = MockTeacher(seed=0)
    bench = TP.benchmark_teacher(teacher, buckets=(32, 64), n_per_bucket=2,
                                 repeats=1, warmup=0)
    assert bench["is_mock"] is True
    assert bench["teacher"] == "mock"
    assert len(bench["buckets"]) == 2
    for row in bench["buckets"]:
        assert math.isfinite(row["sec_per_seq"]) and row["sec_per_seq"] > 0.0
        assert math.isfinite(row["seq_per_sec"]) and row["seq_per_sec"] > 0.0
    assert math.isfinite(bench["overall_seq_per_sec"])

    extrapolation = TP.extrapolate_corpus(bench, corpus_size=1000)
    assert extrapolation["basis"] == TP.MOCK_BASIS
    assert extrapolation["basis"] == "mock_teacher_throughput"
    assert extrapolation["is_extrapolation"] is True
    assert extrapolation["corpus_size"] == 1000
    assert math.isfinite(extrapolation["hours"])
    # the extrapolation is labelled as mock-based, in words, in the report itself
    warning = extrapolation["warning"] or ""
    assert "MOCK" in warning and "DO NOT QUOTE" in warning
    assert "not a physical model" in warning

    rendered = TP.format_report({"benchmark": bench, "extrapolation": extrapolation})
    assert "MOCK" in rendered
    assert "mock_teacher_throughput" in rendered
    assert "DO NOT QUOTE" in rendered


def test_teacher_throughput_default_corpus_and_mean_length_choice():
    teacher = MockTeacher(seed=1)
    bench = TP.benchmark_teacher(teacher, buckets=(32, 64), n_per_bucket=2, warmup=0)
    default = TP.extrapolate_corpus(bench)
    assert default["corpus_size"] == TP.CORPUS_SIZE == 36_000_000
    assert default["rate_source"] == "overall rate across all buckets"

    # supplying a mean length selects the nearest bucket instead of the overall rate
    by_length = TP.extrapolate_corpus(bench, corpus_size=1000, mean_length=10.0)
    assert by_length["rate_source"] == "bucket 0-32 (length 32)"
    assert by_length["basis"] == TP.MOCK_BASIS
    assert math.isfinite(by_length["hours"]) and by_length["hours"] > 0.0


def test_teacher_ensemble_assembly_and_probability_self_consistency():
    seq = "".join(np.random.default_rng(0).choice(list("ACGU"), 24))
    mask = valid_pair_mask(seq)

    ensemble = TP.verify_ensemble_assembly(
        [MockTeacher(seed=k) for k in range(3)], seq, mask)
    assert ensemble["n_teachers"] == 3
    assert ensemble["matches_mean"] is True
    assert ensemble["max_abs_deviation"] <= 1e-12
    assert all(member["valid_probability_matrix"] for member in ensemble["members"])
    assert "NOT evidence" in ensemble["note"]

    rng = np.random.default_rng(1)
    scores = np.triu(rng.normal(0.0, 1.0, size=(len(seq), len(seq))), k=1)
    _log_z, probs = inside_outside(scores, mask)
    check = TP.probability_self_consistency(probs, mask, scores=scores)
    assert check["passes_structural"] is True
    assert check["identity_ok"] is True, check
    assert check["matches_supplied_probs"] is True
    # the finite-difference estimate of d logZ / dc reproduces the marginal sum
    assert check["expected_pairs_sum_marginals"] > 0.0
    relative = check["identity_abs_diff"] / max(1.0, check["expected_pairs_sum_marginals"])
    assert relative < 1e-3, check
    assert check["passes"] is True

    # a deliberately broken matrix is caught
    broken = probs.copy()
    broken[0, 1] = 5.0
    assert TP.probability_self_consistency(broken, mask)["passes"] is False


def test_teacher_throughput_never_degrades_a_real_tool_to_the_mock(monkeypatch):
    """A real teacher is used when importable, and raises when it is not.

    The earlier version of this test asserted that ViennaRNA was *absent*; that
    held only before it was installed out-of-tree at ``/mnt/cunyuliu/pylibs``.
    The contract is about behaviour, so it is checked in both directions here.
    """
    from rnajepa import distill as Distill

    try:
        import RNA  # noqa: F401  (optional dependency of the real teacher)
    except ImportError:  # pragma: no cover - depends on the host
        pytest.skip("ViennaRNA not importable in this interpreter")

    real = TP._resolve_teacher("viennarna", 0)
    assert real.tool == "viennarna"
    assert real.name == "thermo:viennarna"
    assert real.tool_version().startswith("ViennaRNA")

    def _absent():
        raise Distill.ThermodynamicUnavailableError("simulated: ViennaRNA absent")

    monkeypatch.setattr(Distill.ThermodynamicTeacher, "_import_rna",
                        staticmethod(_absent))
    with pytest.raises(TP.TeacherUnavailableError):
        TP._resolve_teacher("viennarna", 0)
    assert TP.main(["--teacher", "viennarna"]) == 3

    # --mock still runs, exits 0, and reports itself as a mock
    assert TP.main(["--mock", "--buckets", "32", "--n-per-bucket", "1",
                    "--repeats", "1", "--warmup", "0"]) == 0


# ===========================================================================
# 6. manuscript linter
# ===========================================================================
def _write(tmpdir, name, text):
    path = os.path.join(tmpdir, name)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return path


def test_shipped_manuscript_stub_passes_every_check():
    report = CM.lint_manuscript(STUB)
    assert report["checks_run"] == ["a", "b", "c", "d"]
    assert report["forbidden_claims"] == []
    assert report["non_claimed_devices_in_contribution_sentence"] == []
    assert report["generalization_wording"]["findings"] == []
    assert report["objections"]["missing"] == []
    assert report["ok"] is True, report["failed_checks"]


def test_shipped_objections_table_fails_the_landing_point_self_check():
    """The spec requires all twelve landing points; the shipped file leaves them blank."""
    check = CM.check_objections_landing_points(OBJECTIONS)
    assert check["found"] is True
    assert check["n_rows"] == 12
    assert check["missing"] == list(CM.OBJECTION_IDS)
    assert check["all_filled"] is False
    assert check["ok"] is False
    # and the linter reports it as a failure, not a pass
    report = CM.lint_manuscript(STUB, objections_path=OBJECTIONS, checks="c")
    assert report["c_ok"] is False
    assert report["failed_checks"] == ["c"]
    assert report["ok"] is False


def test_linter_flags_a_forbidden_claim():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "bad_a.md",
                      "# Draft\n\nWe are the first to propose the Gibbs framework "
                      "for RNA folding.\n")
        findings = CM.find_forbidden_claims(open(path, encoding="utf-8").read())
        assert any(f["id"] == "no_gibbs_framework_first" for f in findings)
        report = CM.lint_manuscript(path, checks="a")
        assert report["a_ok"] is False and report["ok"] is False

        # all six forbidden claims are recognised
        cases = {
            "no_gibbs_framework_first":
                "We are the first to propose the Gibbs framework for RNA folding.",
            "no_cotranscriptional_innovation":
                "A key contribution of this work is the cotranscriptional decision order.",
            "no_illegal_rate_contribution":
                "Our contribution is that the illegal-structure rate is exactly 0.",
            "no_mlp_t_equals_vienna":
                "Zeroing MLP_T is equivalent to ViennaRNA.",
            "no_reproduce_jev_rlcd":
                "We reproduce Jev's RLCD algorithm and improve on it.",
            "no_better_complexity_than_linearfold":
                "Our approach has better complexity than LinearFold.",
        }
        for claim_id, sentence in cases.items():
            ids = [f["id"] for f in CM.find_forbidden_claims(sentence)]
            assert claim_id in ids, f"{claim_id} not detected in {sentence!r}"

        # an honest disclaimer is not flagged
        for sentence in (
            "We do not claim to be the first to propose the Gibbs framework.",
            "The cotranscriptional order is not a contribution; it is an ablation.",
            "The illegal-structure rate being zero is not a contribution.",
            "MLP_T = 0 is not equivalent to ViennaRNA.",
            "We do not reproduce Jev's RLCD.",
            "We do not claim that our algorithmic complexity beats LinearFold's O(L).",
        ):
            assert CM.find_forbidden_claims(sentence) == [], sentence


def test_linter_flags_a_non_claimed_device_in_a_contribution_sentence():
    with tempfile.TemporaryDirectory() as tmp:
        text = ("# Draft\n\nOur main contribution is a novel Turner residual prior "
                "combined with constructive symmetry.\n")
        path = _write(tmp, "bad_b.md", text)
        findings = CM.find_non_claimed_devices_in_contribution_sentences(text)
        ids = {f["id"] for f in findings}
        assert "turner_residual" in ids
        assert "constructive_symmetry" in ids
        report = CM.lint_manuscript(path, checks="b")
        assert report["b_ok"] is False and report["ok"] is False

        # the same devices in a *method* sentence are fine (not a contribution sentence)
        method = ("# Draft\n\nMethod. We apply the Turner residual prior on top of the "
                  "physics baseline.\n")
        assert CM.find_non_claimed_devices_in_contribution_sentences(method) == []

        # a disclaimer inside a contribution sentence is not a violation
        ok = ("# Draft\n\nThe log-linear CRF framework is not our contribution.\n")
        assert CM.find_non_claimed_devices_in_contribution_sentences(ok) == []


def test_linter_flags_a_missing_landing_point():
    header = ("| # | Objection | Response | Evidence | Landing point in manuscript |\n"
              "|---|---|---|---|---|\n")

    def table(landings):
        return header + "".join(
            f"| **Q{i}** | x | y | z | {landings.get(i, '待填')} |\n"
            for i in range(1, 13))

    with tempfile.TemporaryDirectory() as tmp:
        # one filled, one explicitly 待填, the rest absent
        text = (header + "| **Q1** | x | y | z | §1 |\n"
                          "| **Q2** | x | y | z | 待填 |\n")
        path = _write(tmp, "bad_c.md", text)
        check = CM.check_objections_landing_points(path)
        assert check["found"] is True
        assert "Q1" not in check["missing"]
        assert "Q2" in check["missing"]
        assert set(check["missing"]) >= {"Q2", "Q3", "Q12"}
        report = CM.lint_manuscript(path, checks="c")
        assert report["c_ok"] is False and report["ok"] is False

        # all twelve filled -> passes
        good = _write(tmp, "good_c.md", table({i: f"§{i}" for i in range(1, 13)}))
        check_good = CM.check_objections_landing_points(good)
        assert check_good["missing"] == [] and check_good["all_filled"] is True
        assert CM.lint_manuscript(good, checks="c")["c_ok"] is True

        # one blank among the twelve is enough to fail
        eleven = _write(tmp, "eleven_c.md", table({i: f"§{i}" for i in range(1, 12)}))
        assert CM.check_objections_landing_points(eleven)["missing"] == ["Q12"]


def test_linter_flags_forbidden_ood_phrasing():
    with tempfile.TemporaryDirectory() as tmp:
        path = _write(tmp, "bad_d.md",
                      "# Draft\n\nOur model achieves higher OOD accuracy than the "
                      "baselines.\n")
        findings = CM.find_forbidden_ood_phrasing(open(path, encoding="utf-8").read())
        assert findings and "higher OOD accuracy" in findings[0]["match"]
        report = CM.lint_manuscript(path, checks="d")
        assert report["d_ok"] is False and report["ok"] is False

        # the sanctioned wording is accepted and reported as present
        good = _write(tmp, "good_d.md",
                      "# Draft\n\nOur model shows smaller OOD degradation than the "
                      "baselines and is more robust.\n")
        report_good = CM.lint_manuscript(good, checks="d")
        assert report_good["d_ok"] is True
        hits = {h["match"] for h in report_good["generalization_wording"]["allowed_hits"]}
        assert "smaller OOD degradation" in hits


def test_linter_cli_exit_codes_and_check_selection():
    assert CM.main(["--manuscript", STUB]) == 0
    assert CM.main(["--manuscript", STUB, "--checks", "c",
                    "--objections", OBJECTIONS]) == 1
    with tempfile.TemporaryDirectory() as tmp:
        bad = _write(tmp, "bad.md", "# Draft\n\nWe are the first to propose the "
                                    "Gibbs framework.\n")
        assert CM.main(["--manuscript", bad, "--checks", "a"]) == 1
        assert CM.main(["--manuscript", bad, "--checks", "d"]) == 0
        assert CM.main(["--manuscript", os.path.join(tmp, "missing.md")]) == 2
    with pytest.raises(ValueError):
        CM.lint_manuscript(STUB, checks="z")


# ===========================================================================
# 7. generalization-claim guard
# ===========================================================================
def test_generalization_guard_accepts_correct_and_rejects_forbidden_wording():
    accepted = [
        "Our model shows smaller OOD degradation than the baselines.",
        "The proposed model is more robust under distribution shift.",
        "Cross-family performance degrades less than the baselines.",
        "本方法在跨家族场景下衰减更小，更鲁棒。",
    ]
    for text in accepted:
        guard = CM.check_generalization_wording(text)
        assert guard["ok"] is True, text
        assert guard["findings"] == []

    rejected = [
        "Our model achieves higher OOD accuracy than the baselines.",
        "The method improves out-of-distribution accuracy.",
        "We report higher cross-family accuracy.",
        "Our model has superior cross-family performance.",
        "本方法 OOD 精度更高。",
        "本方法跨家族精度超越基线。",
    ]
    for text in rejected:
        guard = CM.check_generalization_wording(text)
        assert guard["ok"] is False, text
        assert guard["findings"], text

    # the guard states the rule it enforces
    assert "smaller OOD degradation" in CM.check_generalization_wording("")["rule"]

    # the paper's own rule document is consistent with the guard's patterns
    rule_doc = open(os.path.join(PAPER_DIR, "generalization_claim.md"),
                    encoding="utf-8").read()
    assert "smaller OOD degradation" in rule_doc
    assert "higher OOD accuracy" in rule_doc
    assert len(CM.FORBIDDEN_OOD_PATTERNS) >= 4


# ===========================================================================
# scaffolding presence (the paper directory is a deliverable, not an accident)
# ===========================================================================
def test_paper_scaffolding_files_present_and_cross_referenced():
    for name in ("outline.md", "contributions.md", "limitations.md",
                 "reviewer_objections.md", "reproducibility_checklist.md",
                 "generalization_claim.md", "check_manuscript.py",
                 "manuscript_stub.md"):
        assert os.path.isfile(os.path.join(PAPER_DIR, name)), name

    outline = open(os.path.join(PAPER_DIR, "outline.md"), encoding="utf-8").read()
    # the one-sentence positioning statement must be quoted verbatim
    assert "RNA 二级结构预测的瓶颈不是表征能力，而是" in outline
    assert "完全省去配分函数" in outline
    assert "计算自适应折叠" in outline

    contributions = open(os.path.join(PAPER_DIR, "contributions.md"),
                         encoding="utf-8").read()
    for cid in ("C1", "C2", "C3", "C4"):
        assert cid in contributions
    # the six forbidden claims are listed as an explicit checklist
    assert contributions.count("❌") >= 6
    # the five non-claimed devices are listed (spec §2.3)
    for device in ("Gibbs 框架", "Turner 残差先验", "构造性对称配对表示",
                   "隐式微分可微 DP", "多通道证据"):
        assert device in contributions, device

    limitations = open(os.path.join(PAPER_DIR, "limitations.md"),
                       encoding="utf-8").read()
    for phrase in ("无同行评审论文", "RLCD 算法未公开", "非本文首创",
                   "不必然优于 LinearFold", "dev == test"):
        assert phrase in limitations, phrase


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
