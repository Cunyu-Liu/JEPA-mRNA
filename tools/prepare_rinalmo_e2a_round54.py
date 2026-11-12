import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from rnajepa.encoder import BASE_TO_INDEX, UNK_INDEX
from rnajepa.motif import FEATURE_NAMES, add_vienna_hairpin_table, candidate_motif_features, candidate_motif_features_torch, export_parameters, load_parameters, shuffle_positions

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024")
DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_tr1c.jsonl")
PARAMETERS = ROOT / "data/e2a_motif_round54/rna_turner2004.par"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "data/e2a_motif_round54"))
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    parameters = export_parameters(PARAMETERS, out / "parameters.npz")
    import RNA
    assert RNA.__version__ == "2.7.2"
    parameters = add_vienna_hairpin_table(parameters, out / "parameters.npz")
    RNA.params_load_RNA_Turner2004()
    references = []
    for sequence in ("GGGAAACCC", "GGGAAACCCAAA", "CAAAAUG", "GGGACGAAACGUCC", "GCACGAAAGUGC"):
        feature = candidate_motif_features(sequence, parameters)
        for i in range(len(sequence)):
            for j in range(i + 4, len(sequence)):
                if not feature[i, j, 2]:
                    continue
                structure = ["." for _ in sequence]
                structure[i], structure[j] = "(", ")"
                pt = [0] * (len(sequence) + 1)
                pt[i + 1], pt[j + 1] = j + 1, i + 1
                vienna = RNA.fold_compound(sequence).eval_loop_pt(i + 1, pt) / 100.0
                reference = float(feature[i, j, 4])
                assert abs(reference - vienna) < 1e-6, (sequence, i, j, reference, vienna)
                assert abs(RNA.fold_compound(sequence).eval_structure("".join(structure)) - vienna) < 1e-6
                references.append({"sequence": sequence, "i": i, "j": j, "feature": reference, "vienna": vienna})
    torch_features = candidate_motif_features_torch(
        torch.tensor([[BASE_TO_INDEX[c] for c in "GGGAAACCC"]]), parameters, ["GGGAAACCC"])
    assert torch.equal(torch_features[0].numpy(), candidate_motif_features("GGGAAACCC", parameters))
    sums = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    squares = np.zeros_like(sums)
    count = 0
    n_sequences = 0
    for line in DATA.open():
        sequence = json.loads(line)["seq"].upper().replace("T", "U")
        ids = torch.tensor([[BASE_TO_INDEX.get(c, UNK_INDEX) for c in sequence]])
        feature = candidate_motif_features_torch(ids, parameters, [sequence])[0]
        mask = feature[..., 2] != 0
        values = feature[mask].double().numpy()
        sums += values.sum(axis=0)
        squares += np.square(values).sum(axis=0)
        count += len(values)
        n_sequences += 1
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 0.0))
    assert np.all(std > 1e-6)
    normalization = {
        "feature_names": FEATURE_NAMES, "mean": mean.tolist(), "std": std.tolist(),
        "legal_pair_count": count, "train_sequences": n_sequences,
        "normalization_split": "TR1c_only", "ground_truth_used": False,
        "units": "negative_delta_G_37_over_RT", "RT_kcal_mol": 0.616,
        "vienna_version": "2.7.2", "parameter_set": "Turner2004",
        "parameter_sha256": sha256(PARAMETERS), "train_sha256": sha256(DATA),
        "candidate_scope": "legal pairs only; hypothetical local motifs, not observed structures",
    }
    (out / "normalization.json").write_text(json.dumps(normalization, indent=2) + "\n")
    shuffled = shuffle_positions(torch_features, ["GGGAAACCC"])
    reference = {
        "status": "PASS_E2A_PARAMETER_AND_SHUFFLE_REFERENCE",
        "vienna_version": RNA.__version__, "n_loop_references": len(references),
        "examples": references, "cpu_torch_equal": True,
        "shuffle_permutation": True,
        "shuffle_feature_sums_equal": bool(torch.equal(shuffled.sum((0, 1)), torch_features.sum((0, 1)))),
        "parameter_sha256": sha256(PARAMETERS), "parameters_npz_sha256": sha256(out / "parameters.npz"),
        "normalization_sha256": sha256(out / "normalization.json"),
    }
    assert reference["shuffle_feature_sums_equal"] and not torch.equal(shuffled, torch_features)
    (out / "reference_qa.json").write_text(json.dumps(reference, indent=2) + "\n")
    print(json.dumps({"status": reference["status"], "n_loop_references": len(references),
                      "n_sequences": n_sequences, "legal_pair_count": count,
                      "normalization_sha256": reference["normalization_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
