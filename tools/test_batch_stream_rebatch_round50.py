from tools.train_plan_a import BatchStream, restore_batch_stream


def test_rebatch_preserves_remaining_sequence_order_epoch_and_step_contract():
    records = [(chr(65 + index) * (index + 1), []) for index in range(10)]
    stream = BatchStream(records, 4, 42)
    saved = {"epoch": 7, "queue": [[8, 9, 2, 3], [4, 5, 6, 7], [0, 1]]}
    restore_batch_stream(stream, saved, 1)
    assert stream.state["epoch"] == 7 and stream.state["batch_size"] == 1
    assert [stream.next() for _ in range(10)] == [[i] for i in [8, 9, 2, 3, 4, 5, 6, 7, 0, 1]]


def test_rebatch_is_identity_when_batch_size_is_unchanged_and_supports_empty_queue():
    records = [("AAAA", []) for _ in range(4)]
    stream = BatchStream(records, 2, 2026)
    saved = {"epoch": 3, "queue": [[2, 3], [0, 1]]}
    restore_batch_stream(stream, saved, 2)
    assert stream.state == {"epoch": 3, "queue": [[2, 3], [0, 1]], "batch_size": 2}
    restore_batch_stream(stream, {"epoch": 4, "queue": []}, 1)
    assert stream.state == {"epoch": 4, "queue": [], "batch_size": 1}
