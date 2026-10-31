from tools.profile_speed_round36 import batch_statistics


def test_batch_statistics_matches_exact_length_bucket_contract():
    records = [{"seq": "A" * 2}, {"seq": "A" * 2}, {"seq": "A" * 3}, {"seq": "A" * 5}]
    assert batch_statistics(records, 2) == {
        "n_sequences": 4,
        "n_unique_lengths": 3,
        "nominal_batch_size": 2,
        "n_batches": 3,
        "mean_actual_batch_size": 4 / 3,
    }


def test_batch_statistics_nominal_batch_does_not_overstate_occupancy():
    records = [{"seq": "A" * length} for length in range(1, 33)]
    result = batch_statistics(records, 256)
    assert result["n_batches"] == 32
    assert result["mean_actual_batch_size"] == 1
