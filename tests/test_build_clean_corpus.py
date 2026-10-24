import hashlib
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import build_clean_corpus as BC  # noqa: E402

S1 = "ACGTACGU"
S2 = "GGCUAAUC"
S3 = "CCGUAGGC"
S4 = "GACUUCGA"


def _norm(seq):
    return seq.upper().replace("T", "U")


def _record(seq, pairs=(), name=""):
    structure = ["."] * len(seq)
    for i, j in pairs:
        structure[i] = "("
        structure[j] = ")"
    row = {"seq": seq, "structure": "".join(structure),
           "pairs": [list(pair) for pair in pairs]}
    if name:
        row["name"] = name
    return row


def _soft(seq, value=0.2):
    length = len(seq)
    matrix = np.zeros((length, length), dtype=np.float64)
    matrix[np.triu_indices(length, 1)] = value
    return matrix


def _one_hot(row):
    seq = _norm(row["seq"])
    matrix = np.zeros((len(seq), len(seq)), dtype=np.float64)
    for i, j in row["pairs"]:
        matrix[i, j] = 1.0
    return matrix


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _object_array(items):
    result = np.empty(len(items), dtype=object)
    for index, item in enumerate(items):
        result[index] = item
    return result


def _write_teacher(directory, rows, matrices, shard_size=2):
    directory.mkdir(parents=True)
    sequences = [_norm(row["seq"]) for row in rows]
    shards = []
    for start in range(0, len(rows), shard_size):
        index = len(shards)
        shard_sequences = sequences[start:start + shard_size]
        shard_matrices = matrices[start:start + shard_size]
        filename = f"shard_{index:05d}.npz"
        path = directory / filename
        np.savez(path, sequences=np.array(shard_sequences, dtype=object),
                 probs=_object_array(shard_matrices))
        shards.append({
            "index": index,
            "file": filename,
            "n": len(shard_sequences),
            "sha256": _sha256(path),
            "seq_sha256": hashlib.sha256(
                "\n".join(shard_sequences).encode("utf-8")).hexdigest(),
        })
    manifest = {"teacher": "thermo:test", "version_lock": "test-v1",
                "shard_size": shard_size, "n_sequences": len(rows), "shards": shards}
    (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _mfe_input(tmp_path, name, rows, matrices=None, shard_size=2):
    jsonl = _write_jsonl(tmp_path / f"{name}.jsonl", rows)
    matrices = matrices or [_soft(row["seq"]) for row in rows]
    teacher = _write_teacher(tmp_path / f"{name}_teacher", rows, matrices, shard_size)
    return BC.InputSpec(str(jsonl), "mfe", str(tmp_path / f"{name}_teacher")), teacher


def _build(tmp_path, specs, name="mixed_train", **kwargs):
    out_jsonl = tmp_path / f"{name}.jsonl"
    out_teacher = tmp_path / f"{name}_teacher"
    manifest = BC.build_corpus(specs, out_jsonl=out_jsonl, out_teacher_dir=out_teacher,
                               **kwargs)
    return manifest, out_jsonl, out_teacher


def _rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _read_teacher(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    sequences = []
    matrices = []
    for shard in manifest["shards"]:
        with np.load(directory / shard["file"], allow_pickle=True) as data:
            sequences.extend(str(seq) for seq in data["sequences"])
            matrices.extend(np.asarray(matrix) for matrix in data["probs"])
    return manifest, sequences, matrices


def test_same_name_source_shards_do_not_overwrite_and_build_is_reproducible(tmp_path):
    first_rows = [_record(S1, [(0, 5)], "first"), _record(S2, name="second")]
    second_rows = [_record(S3, [(1, 6)], "third")]
    first, _ = _mfe_input(tmp_path, "first", first_rows, shard_size=2)
    second_jsonl = _write_jsonl(tmp_path / "second.jsonl", second_rows)
    _write_teacher(tmp_path / "second_teacher", second_rows,
                   [_one_hot(second_rows[0])], shard_size=1)
    second = BC.InputSpec(str(second_jsonl), "gt", str(tmp_path / "second_teacher"))
    source_one = tmp_path / "first_teacher" / "shard_00000.npz"
    source_two = tmp_path / "second_teacher" / "shard_00000.npz"
    hashes_before = (_sha256(source_one), _sha256(source_two))

    manifest, jsonl, teacher = _build(tmp_path, [first, second], shard_size=2)
    assert (_sha256(source_one), _sha256(source_two)) == hashes_before
    assert [shard["file"] for shard in manifest["shards"]] == [
        "shard_00000.npz", "shard_00001.npz"]
    assert jsonl.name == "mixed_train.jsonl"
    assert manifest["teacher"] == "mixed:mfe+gt"
    assert [row["name"] for row in _rows(jsonl)] == ["first", "second", "third"]
    _, sequences, matrices = _read_teacher(teacher)
    assert sequences == [_norm(S1), _norm(S2), _norm(S3)]
    assert np.array_equal(matrices[2], _one_hot(second_rows[0]))

    replay, replay_jsonl, replay_teacher = _build(
        tmp_path, [first, second], name="replay", shard_size=2)
    assert replay["output"]["sha256"] == manifest["output"]["sha256"]
    assert [shard["sha256"] for shard in replay["shards"]] == [
        shard["sha256"] for shard in manifest["shards"]]
    assert _rows(replay_jsonl) == _rows(jsonl)
    assert _read_teacher(replay_teacher)[1] == sequences


def test_teacher_order_mismatch_fails_without_final_output(tmp_path):
    rows = [_record(S1), _record(S2)]
    spec, _ = _mfe_input(tmp_path, "misordered", rows,
                         matrices=[_soft(S2), _soft(S1)])
    manifest_path = tmp_path / "misordered_teacher" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_path = tmp_path / "misordered_teacher" / "shard_00000.npz"
    np.savez(shard_path, sequences=np.array([_norm(S2), _norm(S1)], dtype=object),
             probs=_object_array([_soft(S2), _soft(S1)]))
    shard = manifest["shards"][0]
    shard["sha256"] = _sha256(shard_path)
    shard["seq_sha256"] = hashlib.sha256(
        "\n".join([_norm(S2), _norm(S1)]).encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(BC.CorpusBuildError, match="!= teacher sequence"):
        _build(tmp_path, [spec])
    assert not (tmp_path / "mixed_train.jsonl").exists()
    assert not (tmp_path / "mixed_train_teacher" / "manifest.json").exists()


def test_manifest_shard_hash_mismatch_fails(tmp_path):
    rows = [_record(S1)]
    spec, _ = _mfe_input(tmp_path, "hash", rows, shard_size=1)
    shard = tmp_path / "hash_teacher" / "shard_00000.npz"
    with open(shard, "ab") as handle:
        handle.write(b"changed-after-manifest")
    with pytest.raises(BC.CorpusBuildError, match="manifest sha256"):
        _build(tmp_path, [spec])
    assert not (tmp_path / "mixed_train_teacher").exists()


def test_input_hash_change_between_pre_and_post_read_fails(tmp_path, monkeypatch):
    rows = [_record(S1), _record(S2)]
    spec, _ = _mfe_input(tmp_path, "changing", rows)
    input_path = tmp_path / "changing.jsonl"
    real_hash = BC._sha256_file
    calls = 0

    def changing_hash(path, *args, **kwargs):
        nonlocal calls
        actual = real_hash(path, *args, **kwargs)
        if path == input_path:
            calls += 1
            if calls == 2:
                return "0" * 64
        return actual

    monkeypatch.setattr(BC, "_sha256_file", changing_hash)
    with pytest.raises(BC.CorpusBuildError, match="input hash changed while reading"):
        _build(tmp_path, [spec])
    assert calls == 2
    assert not (tmp_path / "mixed_train.jsonl").exists()


def test_global_dedup_counts_conflicts_and_keeps_first(tmp_path):
    first_rows = [_record(S1, [(0, 5)], "kept")]
    second_rows = [_record(S1, [(0, 5)], "same-label-duplicate"),
                   _record(S1, [(1, 6)], "conflicting-duplicate")]
    first, _ = _mfe_input(tmp_path, "dedup_first", first_rows,
                          matrices=[_soft(S1, 0.2)], shard_size=1)
    second, _ = _mfe_input(tmp_path, "dedup_second", second_rows,
                           matrices=[_soft(S1, 0.2), _soft(S1, 0.4)], shard_size=2)
    manifest, jsonl, teacher = _build(tmp_path, [first, second], shard_size=1)

    assert manifest["stats"]["total_input_records"] == 3
    assert manifest["stats"]["n_records_written"] == 1
    assert manifest["stats"]["dropped_exact_duplicate_sequence"] == 2
    assert manifest["stats"]["conflicting_labels_kept_first"] == 1
    assert manifest["stats"]["conflicting_structure_kept_first"] == 1
    assert manifest["stats"]["conflicting_teacher_kept_first"] == 1
    assert manifest["inputs"][1]["n_in"] == 2
    assert manifest["inputs"][1]["n_kept"] == 0
    output = _rows(jsonl)
    assert output[0]["name"] == "kept"
    assert output[0]["pairs"] == [[0, 5]]
    _, _, matrices = _read_teacher(teacher)
    assert np.array_equal(matrices[0], _soft(S1, 0.2))


def test_blocklist_is_filtered_before_dedup_and_output_overlap_is_zero(tmp_path):
    rows = [_record(S1), _record(S2), _record(S3)]
    spec, _ = _mfe_input(tmp_path, "block", rows)
    blocklist = _write_jsonl(tmp_path / "held_out.jsonl",
                             [{"seq": S2.lower()}, {"seq": "ACGTAAAA"}])
    second_blocklist = _write_jsonl(tmp_path / "test.jsonl", [{"seq": S3}])
    manifest, jsonl, teacher = _build(
        tmp_path, [spec], blocklists=[blocklist, second_blocklist], shard_size=2)
    output_sequences = {_norm(row["seq"]) for row in _rows(jsonl)}
    blocked = {_norm(S2), _norm(S3), _norm("ACGTAAAA")}
    assert output_sequences == {_norm(S1)}
    assert not output_sequences & blocked
    assert manifest["stats"]["filtered_blocklist"] == 2
    assert manifest["stats"]["output_blocklist_overlap"] == 0
    assert manifest["output"]["blocklist_overlap"] == 0
    assert len(manifest["blocklists"]) == 2
    assert manifest["blocklists"][0]["sha256"] == _sha256(blocklist)
    assert _read_teacher(teacher)[1] == [_norm(S1)]


def test_gt_existing_teacher_must_equal_pairs_one_hot(tmp_path):
    rows = [_record(S1, [(0, 5)])]
    jsonl = _write_jsonl(tmp_path / "gt_bad.jsonl", rows)
    _write_teacher(tmp_path / "gt_bad_teacher", rows, [_soft(S1)], shard_size=1)
    spec = BC.InputSpec(str(jsonl), "gt", str(tmp_path / "gt_bad_teacher"))
    with pytest.raises(BC.CorpusBuildError, match="existing GT teacher differs"):
        _build(tmp_path, [spec])
    assert not (tmp_path / "mixed_train_teacher" / "manifest.json").exists()


def test_validation_generates_gt_without_old_teacher_and_preserves_order(tmp_path):
    rows = [_record(S2, [(1, 6)], "v0"), _record(S1, name="v1"),
            _record(S3, [(0, 5)], "v2")]
    jsonl = _write_jsonl(tmp_path / "held_out.jsonl", rows)
    spec = BC.InputSpec(str(jsonl), "gt")
    manifest, out_jsonl, out_teacher = _build(
        tmp_path, [spec], name="chanrg_train_validation", role="validation", shard_size=2)
    output = _rows(out_jsonl)
    assert [row["name"] for row in output] == ["v0", "v1", "v2"]
    assert output[1]["seq"] == _norm(S1)
    assert manifest["stats"]["dropped_exact_duplicate_sequence"] == 0
    assert manifest["stats"]["filtered_blocklist"] == 0
    assert manifest["blocklist_applied"] is False
    _, sequences, matrices = _read_teacher(out_teacher)
    assert sequences == [_norm(row["seq"]) for row in rows]
    for actual, row in zip(matrices, rows):
        assert np.array_equal(actual, _one_hot(row))
        assert np.array_equal(actual, np.triu(actual, k=1))


def test_validation_rejects_duplicate_instead_of_deduplicating(tmp_path):
    rows = [_record(S1, name="first"), _record(S1.lower(), name="duplicate")]
    jsonl = _write_jsonl(tmp_path / "validation_source.jsonl", rows)
    with pytest.raises(BC.CorpusBuildError, match="duplicate validation sequence"):
        _build(tmp_path, [BC.InputSpec(str(jsonl), "gt")],
               name="validation_duplicate", role="validation")
    assert not (tmp_path / "validation_duplicate.jsonl").exists()
    assert not (tmp_path / "validation_duplicate_teacher").exists()


def test_independent_output_reread_detects_post_write_tamper(tmp_path, monkeypatch):
    rows = [_record(S1), _record(S2)]
    spec, _ = _mfe_input(tmp_path, "tamper", rows)
    original_finish = BC.BuildConsumer.finish

    def tamper_after_hash(self):
        result = original_finish(self)
        with open(self.jsonl_path, "a", encoding="utf-8") as handle:
            handle.write("\n")
        return result

    monkeypatch.setattr(BC.BuildConsumer, "finish", tamper_after_hash)
    with pytest.raises(BC.CorpusBuildError,
                       match="hash changed before independent verification"):
        _build(tmp_path, [spec])
    assert not (tmp_path / "mixed_train.jsonl").exists()
    assert not (tmp_path / "mixed_train_teacher").exists()


def test_existing_final_output_is_never_overwritten(tmp_path):
    rows = [_record(S1)]
    spec, _ = _mfe_input(tmp_path, "existing", rows)
    final_teacher = tmp_path / "mixed_train_teacher"
    final_teacher.mkdir()
    sentinel = final_teacher / "manifest.json"
    sentinel.write_text("do-not-overwrite", encoding="utf-8")
    with pytest.raises(BC.CorpusBuildError, match="refusing to overwrite"):
        _build(tmp_path, [spec])
    assert sentinel.read_text(encoding="utf-8") == "do-not-overwrite"
    assert not (tmp_path / "mixed_train.jsonl").exists()


@pytest.mark.parametrize("role", ["train", "validation"])
@pytest.mark.parametrize("bad_kind,expected", [
    ("shape", "teacher shape"),
    ("nan", "NaN or Inf"),
    ("range", "outside"),
])
def test_probability_shape_finite_and_range_are_checked_for_both_roles(
        tmp_path, role, bad_kind, expected):
    rows = [_record(S1)]
    matrix = _soft(S1)
    if bad_kind == "shape":
        matrix = np.zeros((len(S1), len(S1) + 1))
    elif bad_kind == "nan":
        matrix[0, 1] = np.nan
    else:
        matrix[0, 1] = 1.01
    spec, _ = _mfe_input(tmp_path, f"bad_{role}_{bad_kind}", rows, matrices=[matrix])
    with pytest.raises(BC.CorpusBuildError, match=expected):
        _build(tmp_path, [spec], name=f"out_bad_{role}_{bad_kind}", role=role)


@pytest.mark.parametrize("row,expected", [
    ({"seq": S1, "structure": "........", "pairs": [[0, 5]]},
     "pairs and structure disagree"),
    ({"seq": "ACGXACGU", "pairs": []}, "non-ACGU"),
])
def test_sequence_and_structure_validation_fail_closed(tmp_path, row, expected):
    jsonl = _write_jsonl(tmp_path / "invalid.jsonl", [row])
    with pytest.raises(BC.CorpusBuildError, match=expected):
        _build(tmp_path, [BC.InputSpec(str(jsonl), "gt")], name="clean_invalid")
    assert not (tmp_path / "clean_invalid_teacher").exists()


def test_empty_cleaned_corpus_is_not_a_valid_final_output(tmp_path):
    rows = [_record(S1)]
    spec, _ = _mfe_input(tmp_path, "empty_after_filter", rows)
    blocklist = _write_jsonl(tmp_path / "block_everything.jsonl", [{"seq": S1}])
    with pytest.raises(BC.CorpusBuildError, match="zero output records"):
        _build(tmp_path, [spec], name="empty_output", blocklists=[blocklist])
    assert not (tmp_path / "empty_output.jsonl").exists()
    assert not (tmp_path / "empty_output_teacher").exists()


def test_cli_train_requires_an_explicit_blocklist(tmp_path, capsys):
    jsonl = _write_jsonl(tmp_path / "cli_input.jsonl", [_record(S1)])
    code = BC.main(["--role", "train", "--input", str(jsonl), "gt",
                    "--out-jsonl", str(tmp_path / "cli_out.jsonl"),
                    "--out-teacher-dir", str(tmp_path / "cli_teacher")])
    assert code == 2
    assert "requires at least one explicit --blocklist" in capsys.readouterr().err
    assert not (tmp_path / "cli_out.jsonl").exists()
    assert not (tmp_path / "cli_teacher").exists()


def test_legacy_empty_join_sequence_hash_is_accepted_and_recorded(tmp_path):
    rows = [_record(S1), _record(S2)]
    spec, _ = _mfe_input(tmp_path, "legacy_hash", rows, shard_size=2)
    manifest_path = tmp_path / "legacy_hash_teacher" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][0]["seq_sha256"] = hashlib.sha256(
        "".join([_norm(S1), _norm(S2)]).encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    result, _, _ = _build(tmp_path, [spec], name="legacy_output")
    assert result["inputs"][0]["seq_hash_algorithm"] == "empty_join"
    assert result["shards"][0]["seq_sha256"] == hashlib.sha256(
        "\n".join([_norm(S1), _norm(S2)]).encode("utf-8")).hexdigest()


def test_mixed_sequence_hash_conventions_fail_closed(tmp_path):
    rows = [_record(S1), _record(S2), _record(S3)]
    spec, _ = _mfe_input(tmp_path, "mixed_hash", rows, shard_size=2)
    manifest_path = tmp_path / "mixed_hash_teacher" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["shards"][1]["seq_sha256"] = hashlib.sha256(
        (_norm(S2) + _norm(S3)).encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(BC.CorpusBuildError, match="mixed hash conventions"):
        _build(tmp_path, [spec], name="mixed_hash_output")
    assert not (tmp_path / "mixed_hash_output.jsonl").exists()
    assert not (tmp_path / "mixed_hash_output_teacher").exists()
