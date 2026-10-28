#!/usr/bin/env python3
"""External engineering-only QA for Round-33 v2 cache smoke runs."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

TOOLS = Path("/home/rna_jev/tools")
sys.path.insert(0, str(TOOLS))
import nb_splice_dual_round33 as base
import rinalmo_ncrna_dual_round33 as ncrna
import rinalmo_splice_dual_round33 as splice

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
OUT = ROOT / "eval_decision"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def configs():
    for noise in ("clean", "bn200"):
        suffix = "" if noise == "clean" else "_bn200"
        yield {
            "name": f"ncrna_{noise}", "family": "ncrna", "noise": suffix,
            "result": OUT / f"rinalmo_ncrna_{noise}_round33_smoke_v2/result.json",
            "cache": ncrna.DATA / f"ncrna_{noise}_rinalmo_cls_v2_smoke8.npz",
        }
    for ss_type in ("acceptor", "donor"):
        yield {
            "name": f"splice_{ss_type}", "family": "splice", "ss_type": ss_type,
            "result": OUT / f"rinalmo_splice_{ss_type}_round33_smoke_v2/result.json",
            "cache": base.DATA / f"{ss_type}_rinalmo_cls_v2_smoke8.npz",
        }


def main():
    assert torch.cuda.is_available(), "CUDA unavailable; refusing CPU QA"
    for config in configs():
        result = json.loads(config["result"].read_text())
        protocol = result["protocol"]
        assert result["status"] == "PASS"
        assert protocol["sample_scope"] == "engineering_smoke_subset_not_scientific_result"
        assert protocol["device"] == "cuda" and protocol["seeds"] == [0, 1, 2, 3, 4]
        if config["family"] == "ncrna":
            df = ncrna.load_frame(config["noise"], 8)
            classes = {"official_head": ncrna.OfficialNcRNAHead, "jev_head": ncrna.JevNcRNAHead}
            script = TOOLS / "rinalmo_ncrna_dual_round33.py"
        else:
            df = base.load_frame(config["ss_type"], 8)
            classes = {"official_head": splice.OfficialSpliceHead, "jev_head": splice.JevSpliceHead}
            script = TOOLS / "rinalmo_splice_dual_round33.py"
        assert protocol["code_sha256"] == sha256(script)
        with np.load(config["cache"], allow_pickle=False) as cache:
            assert str(cache["sequence_sha256"]) == base.sequence_sha256(df["sequences"])
            assert str(cache["pretrained_sha256"]) == protocol["pretrained_sha256"]
            assert str(cache["extractor_sha256"]) == sha256(TOOLS / "rinalmo_splice_dual_round33.py")
            assert len(cache["embs"]) == len(df)
            assert np.array_equal(cache["labels"], df["labels"].to_numpy())
            embeddings = cache["embs"].copy()
        test_indices = df.index[df["split"] == "test"].to_numpy()
        truth = df["labels"].to_numpy()[test_indices]
        x = torch.as_tensor(embeddings[test_indices], dtype=torch.float32, device="cuda")
        for head_name, model_cls in classes.items():
            runs = result["heads"][head_name]["runs"]
            assert len(runs) == 5 and [run["seed"] for run in runs] == [0, 1, 2, 3, 4]
            for run in runs:
                assert run["device"] == "cuda" and run["early_stopped"] and run["independent_recheck"] == "PASS"
                assert len(run["validation_history"]) == run["epochs_run"] == run["best_epoch"] + 32
                assert sha256(run["prediction_file"]) == run["prediction_sha256"]
                assert sha256(run["checkpoint"]) == run["checkpoint_sha256"]
                checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=True)
                model = model_cls()
                model.load_state_dict(checkpoint["model_state_dict"], strict=True)
                model = model.cuda().eval()
                with torch.inference_mode(), np.load(run["prediction_file"], allow_pickle=False) as saved:
                    assert np.array_equal(saved["truth"], truth)
                    output = model(x)
                    if config["family"] == "ncrna":
                        reproduced = output.argmax(dim=-1).cpu().numpy()
                        assert np.array_equal(reproduced, saved["predictions"])
                    else:
                        reproduced = torch.sigmoid(output).cpu().numpy()
                        assert np.allclose(reproduced, saved["predictions"], atol=1e-7, rtol=0)
                del model, checkpoint
        del x, embeddings
        torch.cuda.empty_cache()
        print("V2_CACHE_AND_GPU_QA_PASS", config["name"], "runs=10",
              "cache_sha256=" + sha256(config["cache"]),
              "result_sha256=" + sha256(config["result"]), flush=True)


if __name__ == "__main__":
    main()
