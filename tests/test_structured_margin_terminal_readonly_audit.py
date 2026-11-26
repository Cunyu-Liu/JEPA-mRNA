import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import pickle
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).parents[1] / "tools/audit_structured_margin_terminal_readonly_v1.py"
SPEC = importlib.util.spec_from_file_location("terminal_audit", MODULE_PATH)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def write_checkpoint(path, value):
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", pickle.dumps(value, protocol=2))


class TerminalAuditTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.root = self.base / "root"
        self.code = self.base / "code"
        self.nll_dir = self.base / "nll"
        self.output = self.base / "report.json"
        self.dev_ids = [f"dev-{index:05d}" for index in range(audit.EXPECTED_DEV)]
        dev_bytes = ("\n".join(self.dev_ids) + "\n").encode()
        (self.base / "dev_ids.txt").write_bytes(dev_bytes)
        config = {"assets": {"validation_ids": str(self.base / "dev_ids.txt")}}
        write_json(self.base / "config.json", config)
        trainer = '''
device = require_explicit_cuda(args.device)
CUDA structured objective left the requested device
for row in validation_records:
require_finite_validation_loss(loss.detach(), step, row, validation_metric(config))
require_finite_validation_loss(aggregate, step, metric=validation_metric(config))
not torch.isfinite(parameter.grad).all()
records.sort(key=lambda row: (len(row["sequence"]), row["id"]))
'''
        context = '''
resolved.type != "cuda"
torch.cuda.synchronize(resolved)
CPU fallback is forbidden
if n_unfrozen < self.terminal_unfrozen_blocks or step < self.terminal_step:
"eligible": False, "improved": False, "save_best": False, "stop": False
"save_best": improved
"stop": self.bad_evaluations >= self.patience
'''
        trainer_path = self.code / "tools/train_plan_a_gec_convergence_v1.py"
        context_path = self.code / "src/rnajepa/gec_context.py"
        trainer_path.parent.mkdir(parents=True)
        context_path.parent.mkdir(parents=True)
        trainer_path.write_text(trainer)
        context_path.write_text(context)
        manifest = {
            "tools/train_plan_a_gec_convergence_v1.py": {"sha256": hashlib.sha256(trainer.encode()).hexdigest()},
            "src/rnajepa/gec_context.py": {"sha256": hashlib.sha256(context.encode()).hexdigest()},
        }
        nll_summary = {
            "status": "PASS_NLL_V3_REQUALIFIED", "qualification_pass": True,
            "reference_use": {"scope": "structure evaluation reference only", "margin_superior_to_nll_claim_allowed": False},
        }
        write_json(self.nll_dir / "qualification_summary.json", nll_summary)
        summary_bytes = (self.nll_dir / "qualification_summary.json").read_bytes()
        write_json(self.nll_dir / "artifact_manifest.json", {
            "artifacts": {"qualification_summary.json": {
                "sha256": hashlib.sha256(summary_bytes).hexdigest(), "bytes": len(summary_bytes),
            }},
        })
        for arm, pid in ((audit.ARMS[0], 3977100), (audit.ARMS[1], 3977304)):
            directory = self.root / arm
            directory.mkdir(parents=True)
            identity = {"arm": arm, "code_manifest": manifest}
            meta = {
                "gec_config": str(self.base / "config.json"),
                "schedule": {"validation_record_limit": 0},
                "validation_ids_sha256": hashlib.sha256(dev_bytes).hexdigest(),
                "validation_record_ids": self.dev_ids,
                "validation_metric": audit.MARGIN_METRIC,
                "run_identity": identity,
                "status": "running", "steps_completed": 0, "converged": False,
                "environment": {"requested_device": "cuda:0", "resolved_device": "cuda:0"},
                "decoder": {"search_allowed": False},
            }
            write_json(directory / "run_meta.json", meta)
            write_json(directory / "launch_identity_v1.json", {"code_commit": audit.EXPECTED_LAUNCH_COMMIT, "pid": pid})
            resume = {
                "format": "foldnar.structured_margin_checkpoint.v1", "step": audit.STEP,
                "optimizer_steps": audit.STEP, "checkpoint_phase": "latest", "terminal_status": "running",
                "n_unfrozen": audit.TERMINAL_UNFROZEN, "run_identity": identity,
                "model_tensor_sha256": "same-model",
                "convergence": {"eligible": False, "best_step": None, "best_validation_loss": None},
            }
            write_checkpoint(directory / "resume.pt", resume)
            best = copy.deepcopy(resume)
            best.update({
                "checkpoint_phase": "best_eligible", "validation_loss": 0.25,
                "convergence": {"eligible": True, "best_step": audit.STEP, "best_validation_loss": 0.25, "bad_evaluations": 0},
            })
            write_checkpoint(directory / "best.pt", best)
            preterminal = {
                "step": 9500, "validation_loss": 0.3, "eligible": False, "save_best": False,
                "stop": False, "best_step": None,
            }
            terminal = {
                "step": audit.STEP, "validation_loss": 0.25, "n_unfrozen": audit.TERMINAL_UNFROZEN,
                "eligible": True, "improved": True, "save_best": True, "stop": False,
                "best_step": audit.STEP, "best_validation_loss": 0.25, "bad_evaluations": 0,
                "validation_metric": audit.MARGIN_METRIC,
            }
            (directory / "validation_log.jsonl").write_text(json.dumps(preterminal) + "\n" + json.dumps(terminal) + "\n")
            batches = []
            computes = []
            trains = []
            for step in range(1, audit.STEP + 1):
                ids = [f"train-{step:05d}-{index}" for index in range(4)]
                hashes = [hashlib.sha256(value.encode()).hexdigest() for value in ids]
                batches.append(json.dumps({"step": step, "record_ids": ids, "sequence_sha256": hashes}))
                computes.append(json.dumps({"step": step, "record_ids": ids}))
                trains.append(json.dumps({
                    "step": step, "loss": 0.1, "gradient_norm": 1.0, "residual_gradient_norm": 0.1,
                    "divergence_reason": "", "optimizer_step": True,
                }))
            (directory / "batch_order.jsonl").write_text("\n".join(batches) + "\n")
            (directory / "compute_ledger.jsonl").write_text("\n".join(computes) + "\n")
            (directory / "train_log.jsonl").write_text("\n".join(trains) + "\n")
        self.git_result = {
            "launch_commit": audit.EXPECTED_LAUNCH_COMMIT,
            "integration_head": audit.EXPECTED_INTEGRATION_HEAD,
            "audit_code_head": audit.EXPECTED_LAUNCH_COMMIT,
            "launch_commit_is_ancestor_of_integration_head": True,
            "working_tree_clean": True,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def run_audit(self):
        argv = [
            str(MODULE_PATH), "--root", str(self.root), "--code", str(self.code),
            "--nll-summary", str(self.nll_dir / "qualification_summary.json"), "--output", str(self.output),
        ]
        with mock.patch.object(audit, "git_identity", return_value=self.git_result), mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stdout(io.StringIO()):
                status = audit.main()
        return status, json.loads(self.output.read_text())

    def test_pass_is_deterministic(self):
        first_status, first = self.run_audit()
        first_bytes = self.output.read_bytes()
        second_status, second = self.run_audit()
        self.assertEqual((first_status, second_status), (0, 0))
        self.assertEqual(first, second)
        self.assertEqual(first_bytes, self.output.read_bytes())
        self.assertTrue(all(value for arm in first["arms"].values() for value in arm["checks"].values()))

    def test_missing_best_fails_closed(self):
        (self.root / audit.ARMS[0] / "best.pt").unlink()
        status, report = self.run_audit()
        self.assertEqual(status, 1)
        self.assertFalse(report["arms"][audit.ARMS[0]]["checks"]["best_checkpoint_present"])
        self.assertTrue(report["audit_status"].startswith("FAIL"))

    def test_wrong_save_best_fails_closed(self):
        path = self.root / audit.ARMS[0] / "validation_log.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["save_best"] = False
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        status, report = self.run_audit()
        self.assertEqual(status, 1)
        self.assertFalse(report["arms"][audit.ARMS[0]]["checks"]["terminal_validation_decision"])

    def test_nonfinite_margin_fails_closed_with_valid_json(self):
        path = self.root / audit.ARMS[0] / "validation_log.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[-1]["validation_loss"] = float("nan")
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        status, report = self.run_audit()
        self.assertEqual(status, 1)
        self.assertIsNone(report["arms"][audit.ARMS[0]]["validation"]["validation_loss"])
        self.assertFalse(report["arms"][audit.ARMS[0]]["checks"]["validation_margin_finite"])

    def test_wait_timeout_does_not_write_report(self):
        for arm in audit.ARMS:
            (self.root / arm / "validation_log.jsonl").unlink()
        with mock.patch.object(audit, "pid_alive", return_value=True):
            with self.assertRaises(TimeoutError):
                audit.wait_for_terminal(self.root, 0, 0.001)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
