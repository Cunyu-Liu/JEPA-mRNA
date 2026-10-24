import json
import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import rnajepa.train_decision as TD  # noqa: E402


def _config(out_dir, **overrides):
    kwargs = dict(tiny=True, steps=2, batch_size=2, lr=5e-3, seed=0,
                  log_every=1, warmup_steps=1, out_dir=str(out_dir),
                  allow_cpu=True, lambda_distill=0.0, lambda_rlcd=0.0,
                  lambda_cal=0.0)
    kwargs.update(overrides)
    return TD.TrainConfig(**kwargs)


def _rows(path):
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_validation_subset_and_no_grad_eval_are_reproducible(tmp_path):
    dataset = TD.make_synthetic_dataset(5, 12, seed=11)
    first = TD.deterministic_validation_subset(dataset, 3, seed=17)
    second = TD.deterministic_validation_subset(dataset, 3, seed=17)
    assert len(first) == 3
    assert [e.seq for e in first.examples] == [e.seq for e in second.examples]
    assert first.version == dataset.version

    config = _config(tmp_path)
    model = TD.build_decision_model(config)
    model.train()
    loss1 = TD.evaluate_validation(model, first, config, step=0)
    loss2 = TD.evaluate_validation(model, first, config, step=0)
    assert loss1 == pytest.approx(loss2, rel=0.0, abs=1e-12)
    assert model.training
    assert all(p.grad is None for p in model.parameters())


def test_improvement_resets_patience_and_early_stop_restores_best(
        tmp_path, monkeypatch):
    train = TD.make_synthetic_dataset(4, 12, seed=1)
    validation = TD.make_synthetic_dataset(2, 12, seed=2)
    losses = iter([3.0, 3.5, 2.0, 1.95, 1.94])
    evaluated_steps = []

    def fake_validation(model, dataset, config, *, step=None):
        evaluated_steps.append(step)
        assert dataset is validation
        return next(losses)

    monkeypatch.setattr(TD, "evaluate_validation", fake_validation)
    config = _config(tmp_path, steps=10, validation_every=1, validation_patience=2,
                     validation_min_delta=0.1)
    result = TD.run_training(config, train, TD.TeacherLabelStore.from_mock(),
                             validation_dataset=validation)

    assert evaluated_steps == [1, 2, 3, 4, 5]
    assert result["steps_completed"] == 5
    assert result["stop_reason"] == "early_stopping"
    assert result["early_stopped"] is True
    assert result["converged"] is True
    assert result["best_step"] == 3
    assert result["best_validation_loss"] == 2.0

    validation_rows = _rows(tmp_path / "validation_log.jsonl")
    assert [row["bad_evaluations"] for row in validation_rows] == [0, 1, 0, 1, 2]
    assert [row["improved"] for row in validation_rows] == [True, False, True, False, False]
    train_rows = _rows(tmp_path / "train_log.jsonl")
    assert len(train_rows) == 5
    assert all("validation_loss" not in row for row in train_rows)

    best = torch.load(tmp_path / "best.pt", map_location="cpu")
    resumed = torch.load(tmp_path / "resume.pt", map_location="cpu")
    assert best["step"] == 3
    assert resumed["step"] == 5
    assert best["model"].keys() == resumed["model"].keys()
    assert all(torch.equal(best["model"][name], resumed["model"][name])
               for name in best["model"])
    assert resumed["validation_state"]["best_step"] == 3
    assert resumed["validation_state"]["bad_evaluations"] == 2

    meta = json.loads((tmp_path / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["stop_reason"] == "early_stopping"
    assert meta["early_stopped"] is True
    assert meta["best_step"] == 3
    assert meta["best_validation_loss"] == 2.0
    ledger = _rows(tmp_path / "ledger.jsonl")
    assert ledger[-1]["stop_reason"] == "early_stopping"
    assert ledger[-1]["early_stopped"] is True
    assert ledger[-1]["best_step"] == 3

    resumed_result = TD.run_training(config, train, TD.TeacherLabelStore.from_mock(),
                                     validation_dataset=validation)
    assert resumed_result["steps_completed"] == 5
    assert resumed_result["stop_reason"] == "early_stopping"
    assert resumed_result["early_stopped"] is True
    assert evaluated_steps == [1, 2, 3, 4, 5]
    assert len(_rows(tmp_path / "validation_log.jsonl")) == 5


def test_step_cap_is_not_reported_as_early_stopping_or_convergence(
        tmp_path, monkeypatch):
    train = TD.make_synthetic_dataset(4, 12, seed=3)
    validation = TD.make_synthetic_dataset(2, 12, seed=4)
    losses = iter([1.0, 1.0])
    monkeypatch.setattr(
        TD, "evaluate_validation",
        lambda model, dataset, config, *, step=None: next(losses))
    config = _config(tmp_path, steps=3, validation_every=2, validation_patience=1)
    result = TD.run_training(config, train, TD.TeacherLabelStore.from_mock(),
                             validation_dataset=validation)

    assert result["steps_completed"] == 3
    assert result["stop_reason"] == "step_cap"
    assert result["early_stopped"] is False
    assert result["converged"] is False
    assert result["best_step"] == 2
    assert [row["step"] for row in _rows(tmp_path / "validation_log.jsonl")] == [2, 3]
    assert result["run_meta"]["status"] == "completed"
    assert result["run_meta"].get("converged") is not True
    assert _rows(tmp_path / "ledger.jsonl")[-1]["stop_reason"] == "step_cap"


def test_default_behavior_and_resume_stay_on_the_legacy_path(tmp_path, monkeypatch):
    train = TD.make_synthetic_dataset(4, 12, seed=5)

    def unexpected_validation(*args, **kwargs):
        raise AssertionError("validation must be opt-in")

    monkeypatch.setattr(TD, "evaluate_validation", unexpected_validation)
    teacher = TD.TeacherLabelStore.from_mock()
    first = TD.run_training(_config(tmp_path, steps=2), train, teacher)
    assert first["steps_completed"] == 2
    assert first["stop_reason"] == "step_cap"
    assert first["best_step"] is None
    assert first["best_validation_loss"] is None
    assert first["early_stopped"] is False
    assert first["converged"] is False
    assert not (tmp_path / "best.pt").exists()
    assert not (tmp_path / "validation_log.jsonl").exists()

    checkpoint = torch.load(tmp_path / "resume.pt", map_location="cpu")
    assert "validation_state" not in checkpoint
    second = TD.run_training(_config(tmp_path, steps=3), train, teacher)
    assert second["steps_completed"] == 3
    rows = _rows(tmp_path / "train_log.jsonl")
    assert len(rows) == 3
    assert set(rows[0]) == {"ts", "step", "lr", "loss", "terms", "grad_norm"}


def test_validation_jsonl_cli_dry_run_wires_the_fixed_subset(tmp_path, capsys):
    path = tmp_path / "held-out.jsonl"
    records = [{"seq": seq, "pairs": []}
               for seq in ("ACGUACGUACGU", "ACGUACGUACGA", "ACGUACGUACGC")]
    path.write_text("".join(json.dumps(row) + "\n" for row in records),
                    encoding="utf-8")
    code = TD.main([
        "--synthetic", "2", "--length", "12", "--tiny", "--steps", "5",
        "--dry-run", "--validation-data", str(path),
        "--validation-max-examples", "1", "--validation-every", "3",
        "--validation-patience", "4", "--validation-min-delta", "0.25",
        "--lambda-distill", "0", "--lambda-rlcd", "0", "--lambda-cal", "0"])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    validation = result["validation"]
    assert validation["early_stopping"] is True
    assert validation["steps_are_safety_cap"] is True
    assert validation["data"]["n_examples"] == 1
    assert validation["every"] == 3
    assert validation["patience"] == 4
    assert validation["min_delta"] == 0.25
    assert any(path.endswith("best.pt") for path in result["would_write"])
    assert any(path.endswith("validation_log.jsonl") for path in result["would_write"])


def test_cli_allow_cpu_is_limited_to_synthetic_tests(tmp_path, capsys):
    real_code = TD.main(["--data", "real.jsonl", "--allow-cpu",
                         "--out", str(tmp_path / "real")])
    assert real_code == 2
    assert "require CUDA" in capsys.readouterr().err

    validation_code = TD.main([
        "--synthetic", "2", "--validation-data", "held-out.jsonl",
        "--allow-cpu", "--out", str(tmp_path / "validation")])
    assert validation_code == 2
    assert "require CUDA" in capsys.readouterr().err
