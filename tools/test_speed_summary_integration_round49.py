import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from summarize_speed_round49 import REQUIRED


class SpeedSummaryIntegrationTests(unittest.TestCase):
    def test_all_protocols_and_vl0_not_fastest_seed_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data/structrfm_pretrain_clean/bprna_ts0.jsonl"
            data.parent.mkdir(parents=True)
            data.write_text('{"seq":"ACGU","pairs":[]}\n' * 79)
            audit = root / "audits/round48"
            audit.mkdir(parents=True)
            (audit / "complex_d1_decision.json").write_text(json.dumps({
                "complete": True, "allow_lineardesign": True,
                "next_base": {"run_prefix": "arm_s"}}))
            for seed, loss in [(0, 20.0), (1, 18.0), (2, 19.0)]:
                run = root / f"runs/arm_s{seed}"
                run.mkdir(parents=True)
                (run / "run_meta.json").write_text(json.dumps({
                    "status": "completed", "converged": True, "early_stopped": True,
                    "best_validation_loss": loss}))
            out = root / "eval_decision/speed_round49"
            out.mkdir(parents=True)
            for seed, value in [(0, 100.0), (1, 50.0), (2, 90.0)]:
                document = {"status": "PASS_STRUCTRFM_RAW_SPEED", "throughput_median_seq_per_s": value,
                            "protocol": {"n_raw": 79, "data_sha256": self.sha(data), "device": "cuda",
                                         "gpu_name": "NVIDIA GeForce RTX 4090", "batch_size": 1,
                                         "warmup_sequences_per_repeat": 1, "repeats": 3, "cached_embeddings": False},
                            "runs": [{"n": 79, "n_timed": 78}] * 3}
                (out / f"selected_s{seed}_clean_ts0.json").write_text(json.dumps(document))
            for name in REQUIRED:
                if name == "nucleicbert_ft":
                    document = {"status": "PASS_NUCLEICBERT_RAW_SPEED", "system": name,
                                "throughput_median_seq_per_s": 10.0, "data_sha256": self.sha(data),
                                "n_sequences": 79, "device": "cuda", "gpu_name": "NVIDIA GeForce RTX 4090",
                                "batch_size": 1, "warmup_sequences_per_repeat": 1,
                                "cached_embeddings": False, "cached_tokenization": False,
                                "runs": [{"n": 79, "n_timed": 78}] * 3}
                elif name in {"plana", "r2d", "xens2", "rinalmo_ft", "mxfold2", "rnaformer"}:
                    document = {"status": "PASS", "throughput_median_seq_per_s": 10.0,
                                "protocol": {"device_type": "cuda", "device": "NVIDIA GeForce RTX 4090",
                                             "data_sha256": self.sha(data), "n_sequences": 79, "batch_size": 1,
                                             "repeats": 3, "warmup_batches_per_repeat": 1, "cached_embeddings": False,
                                             "n_length_filtered": 0, "sample_scope": "full_source_file"},
                                "runs": [{"n_sequences": 79, "n_timed": 78}] * 3}
                else:
                    document = {"status": "PASS", "system": name, "clean_speed_protocol": True,
                                "data_sha256": self.sha(data), "device": "cuda", "gpu_name": "NVIDIA GeForce RTX 4090",
                                "n_sequences": 79, "batch_size": 1, "warmup_sequences_per_repeat": 1,
                                "cached_embeddings": False,
                                "repeats": [{"n_timed": 78, "sequences_per_second": 10.0}] * 3}
                (out / f"{name}_clean_ts0.json").write_text(json.dumps(document))
            cpu = {"status": "PASS", "clean_speed_protocol": True, "data_sha256": self.sha(data),
                   "n_sequences": 79, "device": "CPU native", "cross_device_gpu_rankable": False,
                   "repeats": [{"sequences_per_second": 1.0}] * 3}
            (out / "eternafold_clean_ts0.json").write_text(json.dumps(cpu))
            for suffix in ["json", "recheck.json"]:
                subprocess.run([sys.executable, "/home/rna_jev/tools/summarize_speed_round49.py",
                                "--root", str(root), "--out", str(out / f"summary.{suffix}")],
                               check=True, capture_output=True)
            first = (out / "summary.json").read_bytes()
            self.assertEqual(first, (out / "summary.recheck.json").read_bytes())
            result = json.loads(first)
            self.assertEqual(result["primary_seed"], 1)
            self.assertEqual(result["primary"]["throughput_seq_per_s"], 50.0)
            self.assertTrue(result["speed_sota_claim"])
            self.assertFalse(result["native_cpu_descriptive"]["cross_device_gpu_rankable"])
            modified = json.loads((out / "bpfold_clean_ts0.json").read_text())
            modified["repeats"] = [{"n_timed": 78, "sequences_per_second": 60.0}] * 3
            (out / "bpfold_clean_ts0.json").write_text(json.dumps(modified))
            subprocess.run([sys.executable, "/home/rna_jev/tools/summarize_speed_round49.py",
                            "--root", str(root), "--out", str(out / "summary.slower.json")],
                           check=True, capture_output=True)
            self.assertFalse(json.loads((out / "summary.slower.json").read_text())["speed_sota_claim"])

    @staticmethod
    def sha(path):
        import hashlib
        return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    unittest.main()
