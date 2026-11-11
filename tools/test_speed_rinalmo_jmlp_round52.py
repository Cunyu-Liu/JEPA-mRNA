from speed_rinalmo_jmlp_round52 import summarize_runs


runs = [{"wall_seconds": 2.0, "sequences_per_second": 5.0, "predictions_sha256": "abc",
         "micro": {"f1": 0.5}} for _ in range(3)]
summary = summarize_runs(runs)
assert summary["median_sequences_per_second"] == summary["mean_sequences_per_second"] == 5.0
assert summary["population_sd_sequences_per_second"] == 0.0
changed = [dict(row) for row in runs]
changed[1]["predictions_sha256"] = "different"
try:
    summarize_runs(changed)
except AssertionError:
    pass
else:
    raise AssertionError("nondeterministic predictions accepted")
print("PASS speed repeat statistics and prediction-identity failure injection")
