import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path("/teams/HoulabData_1788502323/liucunyu/rna_jev_xens2_dualview_round56/artifacts/v2/interface")
MODES = ("rinalmo", "h0", "h2", "h0_structure", "h2_structure", "handcrafted", "random", "length", "gc")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def handcrafted(record):
    sequence = record["sequence"]
    gc = (sequence.count("G") + sequence.count("C")) / len(sequence)
    indices = [row["residue_index"] for row in record["residues"]]
    features = []
    for index in indices:
        vector = [float(sequence[index] == base) for base in "ACGU"]
        vector += [float(index > 0 and sequence[index - 1] == base) for base in "ACGU"]
        vector += [float(index + 1 < len(sequence) and sequence[index + 1] == base) for base in "ACGU"]
        local = sequence[max(0, index - 2):index + 3]
        vector += [index / max(1, len(sequence) - 1), math.log(len(sequence)) / 7, gc,
                   (local.count("G") + local.count("C")) / len(local)]
        features.append(vector)
    return torch.tensor(features, dtype=torch.float32)


def load_split(split, mode):
    records = [json.loads(line) for line in (ROOT / f"{split}_contacts.jsonl").open()]
    examples = []
    for record in records:
        if mode == "handcrafted":
            embedding = handcrafted(record)
            structure = torch.empty(len(embedding), 0)
        elif mode in ("length", "gc"):
            value = math.log(len(record["sequence"])) / 7 if mode == "length" else (
                (record["sequence"].count("G") + record["sequence"].count("C")) / len(record["sequence"]))
            embedding = torch.tensor([[value]], dtype=torch.float32).expand(len(record["residues"]), 1)
            structure = torch.empty(len(embedding), 0)
        else:
            source = ("rinalmo" if mode in ("rinalmo", "random") else
                      "h0" if mode == "h0" else "h2" if mode == "h2" else
                      "h0_structure" if mode == "h0_structure" else "h2_structure")
            path = ROOT / "features" / source / split / f"{record['name']}.npz"
            with np.load(path, allow_pickle=False) as saved:
                assert str(saved["sequence"]) == record["sequence"]
                embedding = torch.from_numpy(saved["embedding"].astype(np.float32))
                structure = torch.from_numpy(saved["structure_features"])
                assert np.array_equal(saved["residue_indices"], [row["residue_index"] for row in record["residues"]])
        labels = torch.tensor([row["interface"] for row in record["residues"]], dtype=torch.float32)
        assert embedding.shape[0] == len(labels) and structure.shape[0] == len(labels)
        assert torch.isfinite(embedding).all() and torch.isfinite(structure).all()
        examples.append({"name": record["name"], "pdb": record["pdb"], "group": record["group"],
                         "x": embedding, "structure": structure, "y": labels})
    return examples


class Probe(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.norm = nn.LayerNorm(dimension) if dimension > 1 else nn.Identity()
        self.net = nn.Sequential(nn.Linear(dimension, 64), nn.GELU(), nn.Linear(64, 32),
                                 nn.GELU(), nn.Linear(32, 1)) if dimension > 1 else nn.Linear(1, 1)

    def forward(self, example, use_structure):
        x = torch.cat([example["x"], example["structure"]], dim=1) if use_structure else example["x"]
        return self.net(self.norm(x.cuda())).squeeze(-1)


def loss(records, model, use_structure):
    total = 0.0
    count = 0
    for example in records:
        logits = model(example, use_structure)
        total = total + F.binary_cross_entropy_with_logits(logits, example["y"].cuda(), reduction="sum")
        count += len(logits)
    return total / count


@torch.no_grad()
def raw_predictions(records, model, use_structure):
    output = []
    for example in records:
        probabilities = torch.sigmoid(model(example, use_structure)).cpu().numpy()
        output.append({"name": example["name"], "pdb": example["pdb"], "group": example["group"],
                       "probability": probabilities, "label": example["y"].numpy()})
    return output


def binary_counts(labels, predictions):
    tp = int(np.sum(predictions & (labels == 1)))
    fp = int(np.sum(predictions & (labels == 0)))
    fn = int(np.sum(~predictions & (labels == 1)))
    tn = int(np.sum(~predictions & (labels == 0)))
    return tp, fp, fn, tn


def f1(tp, fp, fn):
    return 2 * tp / max(1, 2 * tp + fp + fn)


def metrics(predictions, threshold):
    all_probabilities = np.concatenate([row["probability"] for row in predictions])
    all_labels = np.concatenate([row["label"] for row in predictions])
    order = np.argsort(-all_probabilities, kind="stable")
    positives = all_labels[order]
    precision = np.cumsum(positives) / np.arange(1, len(positives) + 1)
    micro_auprc = float(np.sum(precision * positives) / max(1, positives.sum()))
    tp, fp, fn, tn = binary_counts(all_labels, all_probabilities >= threshold)
    positive_f1 = f1(tp, fp, fn)
    negative_f1 = f1(tn, fn, fp)
    chain_f1 = []
    chain_macro = []
    chain_auprc = []
    for row in predictions:
        ctp, cfp, cfn, ctn = binary_counts(row["label"], row["probability"] >= threshold)
        pf1 = f1(ctp, cfp, cfn)
        nf1 = f1(ctn, cfn, cfp)
        chain_f1.append(pf1)
        chain_macro.append((pf1 + nf1) / 2)
        if row["label"].sum() > 0:
            order = np.argsort(-row["probability"], kind="stable")
            positive = row["label"][order]
            chain_auprc.append(float(np.sum(np.cumsum(positive) / np.arange(1, len(positive) + 1) * positive) / positive.sum()))
    return {"micro_auprc": micro_auprc, "micro_f1": positive_f1,
            "macro_f1_binary": (positive_f1 + negative_f1) / 2,
            "macro_f1_per_chain": float(np.mean(chain_f1)),
            "macro_f1_binary_per_chain": float(np.mean(chain_macro)),
            "macro_auprc_positive_chains": float(np.mean(chain_auprc)) if chain_auprc else None,
            "n_auprc_chains": len(chain_auprc), "precision": tp / max(1, tp + fp),
            "interface_residue_recall": tp / max(1, tp + fn),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "threshold": float(threshold),
            "n_residues": len(all_labels), "n_chains": len(predictions)}


def save_checkpoint(path, model, optimizer, epoch, best, patience, history, args):
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "best": best, "patience": patience, "history": history,
                "mode": args.mode, "seed": args.seed}, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    qa = json.loads((ROOT / "label_qa_summary.json").read_text())
    assert qa["status"] == "PASS_INDEPENDENT_INTERFACE_LABEL_QA"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train = load_split("train", args.mode)
    holdout = load_split("holdout", args.mode)
    if {row["pdb"] for row in train} & {row["pdb"] for row in holdout}:
        raise AssertionError("PDB leakage")
    if {row["group"] for row in train} & {row["group"] for row in holdout}:
        raise AssertionError("sequence-cluster leakage")
    validation_groups = {group for i, group in enumerate(sorted({row["group"] for row in train})) if i % 5 == 0}
    validation = [row for row in train if row["group"] in validation_groups]
    training = [row for row in train if row["group"] not in validation_groups]
    if not training or not validation or {row["pdb"] for row in training} & {row["pdb"] for row in validation}:
        raise AssertionError("invalid internal PDB/group validation split")
    if args.mode == "random":
        rng = np.random.default_rng(args.seed)
        for example in training:
            example["y"] = torch.from_numpy(rng.permutation(example["y"].numpy()).astype(np.float32))
    use_structure = args.mode.endswith("_structure")
    dimension = training[0]["x"].shape[1] + (training[0]["structure"].shape[1] if use_structure else 0)
    model = Probe(dimension).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    destination = ROOT / "converged_baselines" / f"{args.mode}_seed{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    epoch = 0
    best = float("inf")
    patience = 0
    history = []
    if args.resume:
        state = torch.load(destination / "resume.pt", map_location="cpu", weights_only=False)
        assert state["mode"] == args.mode and state["seed"] == args.seed
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        epoch, best, patience, history = state["epoch"], state["best"], state["patience"], state["history"]
    torch.cuda.reset_peak_memory_stats()
    while patience < 30:
        epoch += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective = loss(training, model, use_structure)
        if not torch.isfinite(objective):
            raise FloatingPointError("non-finite training loss")
        objective.backward()
        for parameter in model.parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("missing or non-finite parameter gradient")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(loss(validation, model, use_structure))
        if not np.isfinite(validation_loss):
            raise FloatingPointError("non-finite validation loss")
        history.append({"epoch": epoch, "train_loss": float(objective), "validation_loss": validation_loss})
        if validation_loss < best - 1e-5:
            best = validation_loss
            patience = 0
            torch.save({"model": model.state_dict(), "epoch": epoch}, destination / "best.tmp.pt")
            os.replace(destination / "best.tmp.pt", destination / "best.pt")
        else:
            patience += 1
        save_checkpoint(destination / "resume.pt", model, optimizer, epoch, best, patience, history, args)
        print(json.dumps({"mode": args.mode, "seed": args.seed, "epoch": epoch,
                          "train_loss": float(objective), "validation_loss": validation_loss,
                          "patience": patience}), flush=True)
    checkpoint = torch.load(destination / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    validation_predictions = raw_predictions(validation, model, use_structure)
    validation_probabilities = np.concatenate([row["probability"] for row in validation_predictions])
    validation_labels = np.concatenate([row["label"] for row in validation_predictions])
    def validation_macro(threshold):
        tp, fp, fn, tn = binary_counts(validation_labels, validation_probabilities >= threshold)
        return (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2
    threshold = float(max(np.linspace(0.01, 0.99, 99), key=validation_macro))
    selected = metrics(validation_predictions, threshold)
    holdout_predictions = raw_predictions(holdout, model, use_structure)
    result = metrics(holdout_predictions, threshold)
    result.update({"status": "PASS_CONVERGED_INTERFACE_PROBE", "mode": args.mode, "seed": args.seed,
        "scientific_scope": "frozen representation interface probe; not dual-view secondary-structure accuracy rescue",
        "converged": True, "stop_reason": "internal_validation_plateau_30_epochs",
        "epochs": epoch, "best_epoch": checkpoint["epoch"], "best_validation_loss": best,
        "n_training_chains": len(training), "n_validation_chains": len(validation), "n_holdout_chains": len(holdout),
        "validation_group_overlap": 0, "holdout_group_overlap": 0, "holdout_pdb_overlap": 0,
        "threshold_selection": "internal train group validation; binary-class macro-F1; no holdout labels",
        "validation_metrics_at_threshold": selected,
        "inference_structure_labels_used": False,
        "structure_features": "frozen model predictions" if use_structure else None,
        "pretraining_exposure": "UNKNOWN; PDB/group isolation does not prove RiNALMo pretraining decontamination",
        "gpu": torch.cuda.get_device_name(0), "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
        "torch": torch.__version__, "label_qa_sha256": sha256(ROOT / "label_qa_summary.json"),
        "labels_sha256": {split: sha256(ROOT / f"{split}_contacts.jsonl") for split in ("train", "holdout")},
        "checkpoint_sha256": sha256(destination / "best.pt"), "source_sha256": sha256(__file__), "history": history})
    (destination / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    with (destination / "holdout_predictions.npz").open("wb") as handle:
        np.savez_compressed(handle,
            probabilities=np.concatenate([row["probability"] for row in holdout_predictions]),
            labels=np.concatenate([row["label"] for row in holdout_predictions]),
            lengths=np.asarray([len(row["probability"]) for row in holdout_predictions]),
            names=np.asarray([row["name"] for row in holdout_predictions]),
            pdbs=np.asarray([row["pdb"] for row in holdout_predictions]),
            groups=np.asarray([row["group"] for row in holdout_predictions]))
    export = [{**row, "probability": row["probability"].tolist(), "label": row["label"].tolist()}
              for row in holdout_predictions]
    (destination / "holdout_predictions.json").write_text(json.dumps(export) + "\n")
    print(json.dumps({key: result[key] for key in ("status", "mode", "seed", "best_epoch", "epochs",
        "micro_auprc", "macro_f1_binary", "macro_f1_per_chain", "interface_residue_recall")}, indent=2))


if __name__ == "__main__":
    main()
