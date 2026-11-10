import hashlib
import math

from qa_rinalmo_frozen_representation_round52 import recompute_split


labels = [0.0, 1.0] * 128
logits = [math.log((index + 1) / (257 - index)) for index in range(256)]
positives = sum(labels)
negatives = len(labels) - positives
weight = negatives / positives
nll = sum((max(logit, 0) - logit * label + math.log1p(math.exp(-abs(logit))))
          * (weight if label else 1.0) for logit, label in zip(logits, labels)) / len(labels)
brier = sum((1 / (1 + math.exp(-logit)) - label) ** 2
            for logit, label in zip(logits, labels)) / len(labels)
split = {"weighted_nll": nll, "brier": brier, "n_pairs": 256,
         "n_positive": int(positives), "n_negative": int(negatives),
         "records": [{"sequence_sha256": hashlib.sha256(b"test").hexdigest(), "length": 300,
                      "pairs": [[1, 0]] + [[index, index + 1] for index in range(1, 256)],
                      "labels": labels, "logits": logits}]}
result = recompute_split(split)
assert result["n_pairs"] == 256
split["weighted_nll"] += 1
try:
    recompute_split(split)
except AssertionError:
    pass
else:
    raise AssertionError("incorrect NLL must be rejected")
print("PASS frozen-representation arithmetic QA and failure injection")
