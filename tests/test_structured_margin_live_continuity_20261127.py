import pytest

from tools import audit_structured_margin_live_continuity_20261127 as audit


def test_post_resume_validation_requires_exact_500_cadence():
    rows = [{"step": step} for step in (9500, 10000, 10500, 11000)]
    assert audit.validation_steps_after(rows, 9500) == [10000, 10500, 11000]
    rows.insert(2, {"step": 10500})
    with pytest.raises(ValueError, match="duplicate/missing"):
        audit.validation_steps_after(rows, 9500)


def test_finite_losses_cover_post_resume_train_and_validation():
    train = [{"step": 9500, "loss": 1.0}, {"step": 9501, "loss": 0.5}]
    validation = [{"step": 9500, "validation_loss": 1.0},
                  {"step": 10000, "validation_loss": 0.25}]
    audit.require_finite_losses(train, validation, 9500)
    train[1]["loss"] = float("nan")
    with pytest.raises(ValueError, match="non-finite training loss"):
        audit.require_finite_losses(train, validation, 9500)


def test_best_resume_identity_supports_newer_resume_without_relaxing_same_step():
    hashes = {"encoder": "a", "head": "b", "residual": "c"}
    assert audit.best_resume_relationship(16000, 16000, hashes, hashes) == "SAME_STEP_IDENTICAL_TENSORS"
    assert audit.best_resume_relationship(16500, 16000, hashes, dict(hashes)) == (
        "NEWER_RESUME_AFTER_BEST_WITH_INDEPENDENTLY_VERIFIED_TENSORS")
    with pytest.raises(ValueError, match="same-step"):
        audit.best_resume_relationship(16000, 16000, hashes, {**hashes, "head": "different"})


def test_next_batch_must_match_saved_stream():
    checkpoint = {"step": 16000, "stream_state": {"queue": [[2, 0, 3, 1]]}}
    meta = {"train_record_ids": ["a", "b", "c", "d"]}
    batch_order = [None] * 16000 + [{"record_ids": ["c", "a", "d", "b"]}]
    ids, witness = audit.next_record_ids(checkpoint, meta, batch_order)
    assert ids == ["c", "a", "d", "b"] and witness is True
    batch_order[-1]["record_ids"] = ["wrong"]
    with pytest.raises(ValueError, match="next batch differs"):
        audit.next_record_ids(checkpoint, meta, batch_order)
