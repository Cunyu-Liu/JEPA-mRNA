import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rnajepa.dual_view import build_structure_example

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
OUT = ROOT
BASES = "ACGU"
EXPECTED = {"holdout": (83, 79, 119), "train": (376, 346, 428)}
VALID = {"holdout": 65, "train": 300}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(split):
    records = [json.loads(line) for line in (ROOT / f"{split}_contacts.jsonl").open()]
    assert len(records) == EXPECTED[split][0]
    valid = []
    excluded = []
    for record in records:
        invalid = [row["residue_index"] for row in record["residues"]
                   if row["residue_index"] < 0 or row["residue_index"] >= len(record["sequence"])]
        if invalid:
            excluded.append({"name": record["name"], "reason": "RESIDUE_INDEX_OUT_OF_RANGE",
                             "examples": invalid[:3]})
        else:
            valid.append(record)
    assert len(valid) == VALID[split]
    (OUT / f"{split}_baseline_excluded.json").write_text(json.dumps(excluded, indent=2) + "\n")
    return valid


def features(record, include_structure=True):
    sequence = record["sequence"]
    length = len(sequence)
    gc = (sequence.count("G") + sequence.count("C")) / length
    rows = []
    labels = []
    pair_target = torch.zeros(len(record["residues"]))
    if include_structure:
        for i, j in record.get("canonical_nested_pairs", []):
            if 0 <= i < length and 0 <= j < length:
                pair_target[i] = pair_target[j] = 1.0
    for index, residue in enumerate(record["residues"]):
        i = residue["residue_index"]
        vector = [float(sequence[i] == base) for base in BASES]
        vector += [float(i > 0 and sequence[i - 1] == base) for base in BASES]
        vector += [float(i + 1 < length and sequence[i + 1] == base) for base in BASES]
        local = sequence[max(0, i - 2):i + 3]
        vector += [i / max(1, length - 1), float(np.log(length) / 7), gc,
                   (local.count("G") + local.count("C")) / len(local)]
        rows.append(vector)
        labels.append(float(residue["interface"]))
    if include_structure:
        structure = build_structure_example(sequence, record.get("canonical_nested_pairs", []))
        structure_target = (structure.features.mean(dim=0) if structure.features.shape[0]
                            else torch.zeros(16))
    else:
        structure_target = torch.zeros(16)
    return (torch.tensor(rows, dtype=torch.float32), torch.tensor(labels),
            pair_target, structure_target)


class InterfaceMLP(nn.Module):
    def __init__(self, structure_aware):
        super().__init__()
        self.structure_aware = structure_aware
        self.core = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))
        if structure_aware:
            self.pair_head = nn.Linear(32, 1)
            self.structure_head = nn.Linear(32, 16)

    def forward(self, x):
        hidden = F.gelu(self.core[1](self.core[0](x)))
        output = self.core[2:](hidden)
        result = {"logit": output}
        if self.structure_aware:
            result["pair"] = self.pair_head(hidden).squeeze(-1)
            result["structure"] = self.structure_head(hidden.mean(dim=0, keepdim=True)).squeeze(0)
        return result


def metrics(records, model, structure_aware, threshold):
    probabilities = []
    labels = []
    per_sequence = []
    for record in records:
        x, y, _, _ = features(record, include_structure=False)
        with torch.no_grad():
            output = model(x.cuda())
        probabilities.extend(torch.sigmoid(output["logit"]).cpu().numpy())
        labels.extend(y.numpy())
        per_sequence.append({"name": record["name"], "group": record["group"],
                             "n": len(y), "positive": int(y.sum())})
    probabilities = np.asarray(probabilities).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    order = np.argsort(-probabilities)
    positives = labels[order]
    precision = np.cumsum(positives) / np.arange(1, len(positives) + 1)
    auprc = float((precision * positives).sum() / max(1, positives.sum()))
    prediction = probabilities >= threshold
    tp = int((prediction & (labels == 1)).sum())
    fp = int((prediction & (labels == 0)).sum())
    fn = int((~prediction & (labels == 1)).sum())
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return {"auprc": auprc, "macro_f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "threshold": threshold, "per_sequence": per_sequence}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["sequence_only", "structure_aware", "length", "gc", "random"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = load("train")
    holdout = load("holdout")
    train_groups = {record["group"] for record in train}
    holdout_groups = {record["group"] for record in holdout}
    train_pdbs = {record["pdb"] for record in train}
    holdout_pdbs = {record["pdb"] for record in holdout}
    if train_groups & holdout_groups or train_pdbs & holdout_pdbs:
        raise AssertionError("train/holdout group or PDB leakage")
    groups = sorted({record["pdb"] for record in train})
    validation_pdbs = set(groups[::5])
    training = [record for record in train if record["pdb"] not in validation_pdbs]
    validation = [record for record in train if record["pdb"] in validation_pdbs]
    if not training or not validation:
        raise AssertionError("empty train/validation split")
    structure_aware = args.mode == "structure_aware"
    model = InterfaceMLP(structure_aware).cuda()
    if args.mode in ("length", "gc"):
        model = nn.Linear(1, 1).cuda()
    if args.mode == "random":
        model = InterfaceMLP(False).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    destination = OUT / f"{args.mode}_seed{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_step = 0
    patience = 0
    history = []
    for step in range(1, args.max_steps + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for record in training:
            x, y, pair_target, structure_target = features(record)
            if args.mode == "random":
                y = y[torch.randperm(len(y))]
            if args.mode in ("length", "gc"):
                value = float(np.log(len(record["sequence"]))) if args.mode == "length" else (
                    (record["sequence"].count("G") + record["sequence"].count("C")) / len(record["sequence"]))
                loss = F.binary_cross_entropy_with_logits(model(torch.tensor([value], device="cuda")).reshape(1), y.mean().cuda().reshape(1))
            else:
                output = model(x.cuda())
                loss = F.binary_cross_entropy_with_logits(output["logit"].squeeze(-1), y.cuda())
                if structure_aware:
                    loss = loss + 0.1 * F.mse_loss(output["pair"], pair_target.cuda())
                    loss = loss + 0.1 * F.mse_loss(output["structure"], structure_target.cuda())
            total = total + loss
        (total / len(training)).backward()
        optimizer.step()
        if step % 10 == 0:
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                for record in validation:
                    x, y, pair_target, structure_target = features(record)
                    if args.mode in ("length", "gc"):
                        value = float(np.log(len(record["sequence"]))) if args.mode == "length" else (
                            (record["sequence"].count("G") + record["sequence"].count("C")) / len(record["sequence"]))
                        loss = F.binary_cross_entropy_with_logits(model(torch.tensor([value], device="cuda")).reshape(1), y.mean().cuda().reshape(1))
                    else:
                        output = model(x.cuda())
                        loss = F.binary_cross_entropy_with_logits(output["logit"].squeeze(-1), y.cuda())
                        if structure_aware:
                            loss = loss + 0.1 * F.mse_loss(output["pair"], pair_target.cuda())
                            loss = loss + 0.1 * F.mse_loss(output["structure"], structure_target.cuda())
                    validation_loss += float(loss)
            validation_loss /= len(validation)
            history.append({"step": step, "validation_loss": validation_loss})
            if validation_loss < best - 1e-4:
                best = validation_loss
                best_step = step
                patience = 0
                torch.save({"step": step, "model": model.state_dict()}, destination / "best.pt")
            else:
                patience += 1
            if patience >= 30:
                break
    if best_step == 0:
        raise RuntimeError("no improving validation checkpoint")
    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    thresholds = np.linspace(0.01, 0.99, 99)
    candidates = [metrics(validation, model, structure_aware, threshold) for threshold in thresholds]
    chosen = max(candidates, key=lambda result: result["macro_f1"])
    result = metrics(holdout, model, structure_aware, chosen["threshold"])
    result.update({
        "status": "PASS_INTERFACE_BASELINE", "mode": args.mode, "seed": args.seed,
        "best_step": best_step, "early_stopped": patience >= 30,
        "converged": patience >= 30, "max_steps": args.max_steps,
        "train_n": len(training), "validation_n": len(validation),
        "holdout_n": len(holdout), "train_holdout_group_overlap": 0,
        "train_holdout_pdb_overlap": 0, "history": history,
        "checkpoint_sha256": sha256(destination / "best.pt"),
        "holdout_structure_labels_used_for_inference": False,
    })
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "mode", "seed", "best_step", "auprc", "macro_f1", "tp", "fp", "fn")}, indent=2))


if __name__ == "__main__":
    main()
