import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from tools import ernierna_bprna_new_batch as batch


COMMIT = batch.OFFICIAL_COMMIT
PREDICT = "'bpRNA-new': 'ERNIE-RNA_attn-map_frozen_ss_prediction_bpRNA-1m_checkpoint.pt'"
UTILS = "rna_models[0]\nmlm_pretrained_model_path.split(os.pathsep)"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_fixture(tmp_path, n=2):
    official = tmp_path / "assets/ERNIE-RNA"
    (official / "src").mkdir(parents=True)
    (official / "predict_ss_rna.py").write_text(PREDICT)
    (official / "src/utils.py").write_text(UTILS)
    weights = []
    for asset_id, filename in (("pretrain", "pretrain.pt"), ("head", "head.pt")):
        path = official.parent / "weights" / filename
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((asset_id * 10).encode())
        weights.append({"asset_id": asset_id, "official_url": "https://drive.google.com/file/d/official",
                        "local_relative_path": f"weights/{filename}", "sha256": batch.sha256(path)})
    provenance = {
        "official_source": {"commit": COMMIT, "required_file_sha256": {
            "predict_ss_rna.py": batch.sha256(official / "predict_ss_rna.py"),
            "src/utils.py": batch.sha256(official / "src/utils.py")}},
        "weights": weights,
        "model_identity": {"model_form": "single_model_not_ensemble"},
    }
    provenance_path = tmp_path / "provenance.json"
    write_json(provenance_path, provenance)
    rows = [{"name": f"original/id-{i}", "seq": seq, "pairs": []}
            for i, seq in enumerate(("ACGU", "GGCAAU")[:n])]
    data = tmp_path / "data.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "batch"
    args = Namespace(official_root=official, provenance=provenance_path, output_root=output,
                     data=data, data_sha256=batch.sha256(data))
    batch.prepare(args)
    return official, provenance_path, output, rows


def ct_text(title, sequence, pairs=()):
    partners = {index: 0 for index in range(1, len(sequence) + 1)}
    for left, right in pairs:
        partners[left + 1] = right + 1
        partners[right + 1] = left + 1
    lines = [f"{len(sequence)}\t{title}"]
    for index, base in enumerate(sequence, 1):
        lines.append(f"{index}\t{base}\t{index - 1 if index > 1 else 0}\t"
                     f"{index + 1 if index < len(sequence) else 0}\t{partners[index]}\t{index}")
    return "\n".join(lines) + "\n"


def add_ct(output, records, kind="finetune", pairs=None):
    ct_root = output / "official_ct"
    ct_root.mkdir(exist_ok=True)
    for index, row in enumerate(records):
        title = f"bprna_new_{index:06d}_{kind}_prediction"
        sequence = row["seq"]
        selected = ((0, len(sequence) - 1),) if pairs is None and index == 0 else pairs
        selected = selected or ()
        (ct_root / f"{title}.ct").write_text(ct_text(title, sequence, selected))


def test_prepare_reproducible_and_label_free(tmp_path):
    official, provenance, output, rows = make_fixture(tmp_path)
    first = (output / "bprna_new.official_input.fasta").read_bytes()
    contract = json.loads((output / "batch_contract.json").read_text())
    args = Namespace(official_root=official, provenance=provenance, output_root=output,
                     data=tmp_path / "data.jsonl", data_sha256=batch.sha256(tmp_path / "data.jsonl"))
    batch.prepare(args)
    assert first == (output / "bprna_new.official_input.fasta").read_bytes()
    assert contract["eligible_to_launch"] is True
    assert contract["execution_status"] == "NOT_RUN_NO_INFERENCE_AUTHORIZED"
    assert contract["output_contract"]["target_prediction"] == "finetune"
    assert all("pairs" not in row and "sequence" not in row for row in contract["records"])
    batch.load_contract(output / "batch_contract.json")


def test_prepare_rejects_duplicate_original_names(tmp_path):
    official, provenance, output, rows = make_fixture(tmp_path)
    data = tmp_path / "data.jsonl"
    duplicate = [rows[0], rows[0]]
    data.write_text("".join(json.dumps(row) + "\n" for row in duplicate))
    args = Namespace(official_root=official, provenance=provenance, output_root=output,
                     data=data, data_sha256=batch.sha256(data))
    with pytest.raises(AssertionError, match="duplicate"):
        batch.prepare(args)


def test_prepare_records_missing_weights_without_launch_eligibility(tmp_path):
    official, provenance_path, output, _ = make_fixture(tmp_path)
    document = json.loads(provenance_path.read_text())
    document["weights"][0]["sha256"] = None
    write_json(provenance_path, document)
    data = tmp_path / "data.jsonl"
    args = Namespace(official_root=official, provenance=provenance_path, output_root=output,
                     data=data, data_sha256=batch.sha256(data))
    batch.prepare(args)
    contract = json.loads((output / "batch_contract.json").read_text())
    assert contract["eligible_to_launch"] is False
    assert contract["asset_audit"]["ready"] is False


def test_parse_ct_preserves_crossing_and_zero_pairs(tmp_path):
    sequence = "ACGUAU"
    path = tmp_path / "x.ct"
    path.write_text(ct_text("x", sequence, ((0, 4), (1, 5))))
    assert batch.parse_ct(path, "x", sequence) == [(0, 4), (1, 5)]
    path.write_text(ct_text("x", sequence))
    assert batch.parse_ct(path, "x", sequence) == []


@pytest.mark.parametrize("mutation", ["base", "symmetry", "title", "index"])
def test_parse_ct_rejects_invalid_identity(tmp_path, mutation):
    sequence = "ACGU"
    path = tmp_path / "x.ct"
    text = ct_text("x", sequence, ((0, 3),))
    if mutation == "base":
        text = text.replace("1\tA\t", "1\tC\t")
    elif mutation == "symmetry":
        text = text.replace("4\tU\t3\t0\t1\t4", "4\tU\t3\t0\t0\t4")
    elif mutation == "title":
        text = text.replace("4\tx", "4\tother")
    else:
        text = text.replace("2\tC\t1", "3\tC\t1")
    path.write_text(text)
    with pytest.raises(AssertionError):
        batch.parse_ct(path, "x", sequence)


def test_audit_assets_ready_is_not_full_prediction_ready(tmp_path):
    official, provenance, output, _ = make_fixture(tmp_path)
    args = Namespace(official_root=official, provenance=provenance, output_root=output,
                     contract=output / "batch_contract.json")
    batch.audit(args)
    result = json.loads((output / "readiness_audit.json").read_text())
    assert result["status"] == "ASSETS_READY_PREDICTIONS_PENDING"
    assert result["asset_ready"] is True
    assert result["full_prediction_ready"] is False


def test_export_full_and_separate_zero_shot(tmp_path):
    official, provenance, output, rows = make_fixture(tmp_path)
    add_ct(output, rows, "finetune")
    add_ct(output, rows, "zeroshot", ())
    args = Namespace(official_root=official, provenance=provenance, output_root=output,
                     contract=output / "batch_contract.json")
    batch.export_predictions(args)
    manifest = json.loads((output / "export_manifest.json").read_text())
    assert manifest["coverage"] == 1.0
    assert manifest["target_prediction"] == "finetune"
    fine = [json.loads(line) for line in (output / "finetune_predictions.jsonl").read_text().splitlines()]
    zero = [json.loads(line) for line in (output / "zeroshot_predictions.jsonl").read_text().splitlines()]
    assert [row["name"] for row in fine] == [row["name"] for row in rows]
    assert fine[0]["pairs"] == [[0, 3]] and fine[1]["pairs"] == []
    assert all(row["pairs"] == [] for row in zero)
    batch.audit(args)
    assert json.loads((output / "readiness_audit.json").read_text())["full_prediction_ready"] is True


def test_export_rejects_demo_missing_and_extra(tmp_path):
    official, provenance, output, rows = make_fixture(tmp_path)
    add_ct(output, rows, "finetune")
    add_ct(output, rows, "zeroshot")
    (output / "official_ct/bprna_new_000001_zeroshot_prediction.ct").unlink()
    args = Namespace(official_root=official, provenance=provenance, output_root=output,
                     contract=output / "batch_contract.json")
    with pytest.raises(AssertionError) as error:
        batch.export_predictions(args)
    assert "missing_ct" in str(error.value)
    add_ct(output, rows, "zeroshot")
    (output / "official_ct/demo_finetune_prediction.ct").write_text("not a prediction")
    with pytest.raises(AssertionError) as error:
        batch.export_predictions(args)
    assert "extra_ct" in str(error.value)


def test_export_rejects_modified_checkpoint(tmp_path):
    official, provenance, output, rows = make_fixture(tmp_path)
    add_ct(output, rows, "finetune")
    add_ct(output, rows, "zeroshot")
    (official.parent / "weights/head.pt").write_bytes(b"modified")
    args = Namespace(official_root=official, provenance=provenance, output_root=output,
                     contract=output / "batch_contract.json")
    with pytest.raises(AssertionError, match="identity"):
        batch.export_predictions(args)


def test_cli_does_not_execute_inference(tmp_path):
    script = Path(batch.__file__)
    result = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "prepare" in result.stdout and "export" in result.stdout and "audit" in result.stdout
    assert "torch" not in script.read_text()
