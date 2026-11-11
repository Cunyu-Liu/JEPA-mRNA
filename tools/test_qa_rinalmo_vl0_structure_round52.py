import math

from qa_rinalmo_vl0_structure_round52 import micro_metrics


metrics = micro_metrics(2, 1, 1)
assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 1, 1)
assert all(math.isclose(metrics[key], 2 / 3, rel_tol=0, abs_tol=1e-15)
           for key in ("f1", "precision", "recall"))
assert micro_metrics(0, 0, 0) == {"f1": 0.0, "precision": 0.0,
                                  "recall": 0.0, "tp": 0, "fp": 0, "fn": 0}
changed = micro_metrics(2, 1, 1)
changed["f1"] += 1e-6
assert not math.isclose(changed["f1"], 2 / 3, rel_tol=0, abs_tol=1e-12)
print("PASS VL0 structure micro-metric and zero-denominator arithmetic checks")
