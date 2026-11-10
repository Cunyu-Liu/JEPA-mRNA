from qa_capacity_error_round51 import independently_summarize


rows = []
for index in range(196):
    rows.append({"sequence_sha256": f"{index:064d}", "length": 50 + index,
                 "n_gt_pairs": index % 7,
                 "losses": {"total": index / 10, "nll": index / 20}})
split = {"n_sequences": 196, "per_sequence": rows,
         "mean_losses": {"total": sum(row["losses"]["total"] for row in rows) / 196,
                         "nll": sum(row["losses"]["nll"] for row in rows) / 196}}
means, buckets = independently_summarize(split)
assert abs(means["total"] - 9.75) < 1e-12
assert abs(means["nll"] - 4.875) < 1e-12
assert buckets["<=100"]["n"] == 51
assert buckets["100-200"]["n"] == 100
assert buckets["200-400"]["n"] == 45
assert buckets[">400"]["n"] == 0
for mutate in (
    lambda value: value["per_sequence"][0]["losses"].update(total=999),
    lambda value: value["per_sequence"][1].update(
        sequence_sha256=value["per_sequence"][0]["sequence_sha256"]),
):
    invalid = {"n_sequences": split["n_sequences"],
               "per_sequence": [dict(row, losses=dict(row["losses"]))
                                for row in split["per_sequence"]],
               "mean_losses": dict(split["mean_losses"])}
    mutate(invalid)
    try:
        independently_summarize(invalid)
    except AssertionError:
        pass
    else:
        raise AssertionError("tampered or duplicate diagnostic data must be rejected")
print("PASS independent capacity-error aggregation and failure injection")
