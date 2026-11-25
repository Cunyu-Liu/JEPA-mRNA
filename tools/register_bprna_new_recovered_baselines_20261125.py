#!/usr/bin/env python3
"""Register twenty recovered bpRNA-new prediction assets without protocol mixing."""
import argparse
import hashlib
import json
import math
import tarfile
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFESTS = REPO / "manifests/foldnar_v1"
ROOT = Path("/teams/HoulabData_1788502323/liucunyu/foldnar")
EXT = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark")
RAW_DATA = Path("/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl")
RAW_DATA_SHA = "271efd8d2d143c1ff6d5d2210775a5099727a31d76fbab9525740347ae1368f9"
CLEAN_DATA = EXT / "clean_splits/tr1c/bprna_new.jsonl"
CLEAN_DATA_SHA = "231ae0079c183d481e4f5d4518f77a41fa52f6a6544b2feb0a7d60f8f9b23595"
OFFICIAL_DATA = Path("/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/source_corpora/BPfold_data.tar.gz")
OFFICIAL_DATA_SHA = "819acdea53c6b093df14cab35bf079c4678325d2d5ea4296b904068d35959403"
BASE_SHA = {
    "data_registry.json": "1cbb76f5248bac7850168ecb6975b4f668517428e34bffa0dd23be9cd8129f2e",
    "prediction_registry.json": "da2b6fadaf994496dc004375be96cda9fdaa024dba4ce0e465738f048010d1c9",
    "baseline_registry.json": "33c91e028765c6ab6b4a0b91f8d54aa63a030c5077571a02dfa8a8eea427e930",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def expected(n, tp, fp, fn, precision, recall, micro, macro):
    return {"n_dataset_sequences": n, "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": precision, "micro_recall": recall,
            "micro_f1": micro, "macro_f1": macro}


def asset(asset_id, baseline_id, prediction_id, path, digest, source, source_digest,
          metrics, model_form="single_model", label_protocol="PROJECT_DATABASE_RAW_5388",
          origin="project_replay", training="see recovered provenance", postprocess="frozen native decode"):
    return {"asset_id": asset_id, "baseline_id": baseline_id, "prediction_id": prediction_id,
            "prediction_path": str(path), "prediction_sha256": digest,
            "source_path": str(source), "source_sha256": source_digest,
            "expected_metrics": metrics, "model_form": model_form,
            "label_protocol": label_protocol, "prediction_origin": origin,
            "training_identity": training, "postprocess_identity": postprocess,
            "raw_clean_scope": "clean_tr1c" if "CLEAN" in label_protocol else "raw"}


ASSETS = [
    asset("bpfold_project_six_checkpoint_raw", "bpfold", "bpfold/bprna_new",
          EXT / "bpfold/raw/bprna_new.json", "9832e97bcb5e6879e57a317ef560dd1ca364b11a48575b3f89cd9c94c773255a",
          EXT / "bpfold/raw/bprna_new.json", "9832e97bcb5e6879e57a317ef560dd1ca364b11a48575b3f89cd9c94c773255a",
          expected(5388, 106998, 48176, 28186, 0.6895356180803485, 0.7914989939637826, 0.7370074184282851, 0.7447389175263477),
          "six_checkpoint_ensemble", training="BPfold project replay; six released checkpoints", postprocess="project model_predict six-checkpoint ensemble; not a single model"),
    asset("ufold_converted_single_raw", "ufold", "ufold/bprna_new",
          EXT / "ufold_mm_corrected/raw/bprna_new.json", "f92b9ae16591efa055c84fd3e722128dd8a66e5e59fc7dddfae9c1fa08afe2eb",
          EXT / "ufold_mm_corrected/raw/bprna_new.json", "f92b9ae16591efa055c84fd3e722128dd8a66e5e59fc7dddfae9c1fa08afe2eb",
          expected(5388, 80572, 84827, 54612, 0.48713716527911294, 0.5960172801514972, 0.5361048362681855, 0.5589247295596474),
          training="MultiMolecule-converted UFold checkpoint; not original-author distribution", postprocess="threshold=0.5; frozen project decode"),
    asset("bpfold_author_release_official_reference", "bpfold", "bpfold_official_release/bprna_new_official_reference",
          ROOT / "baseline_recovery_v2/predictions/bpfold_official_release/bprna_new_all_official_v2.jsonl", "bc41c0bc196e9508274625c44b794d29eca9878d698f88570c89841fb6b618ad",
          ROOT / "baseline_recovery_v2/official_assets/bpfold/BPfold_test_results_v0.2.tar.gz", "e8a3420e3458f7aad4c86b500f5d56d0e935b7da82b936675fefda64c5e03073",
          expected(5401, 86933, 59441, 41092, 0.5939101206498422, 0.6790314391720367, 0.633624758107719, 0.6470223791166072),
          "six_checkpoint_ensemble", "BPFOLD_OFFICIAL_REFERENCE_5401", "author_release",
          "BPfold v0.2 author-released predictions", "author release; official BPfold reference labels; project-label audit is separate"),
    asset("efold_0_1_3_single_raw", "efold_0_1_3", "efold_0_1_3/bprna_new",
          ROOT / "baseline_recovery_v3_efold_family/predictions/efold_project_replay_bprna_new_frozen_v3.json", "17e2652ea8ea622574efc2ff8d4e277509d8d9d61e04040fb9c1c8a244a15e3e",
          EXT / "efold/raw/bprna_new.json", "17e2652ea8ea622574efc2ff8d4e277509d8d9d61e04040fb9c1c8a244a15e3e",
          expected(5388, 100583, 73955, 34601, 0.5762813828507259, 0.7440451532725766, 0.6495050400036162, 0.6556973548256184),
          training="eFold 0.1.3 official weights; project replay", postprocess="official constraints + UFold postprocess + Hungarian decode"),
    asset("rinalmo_author_bprna_ft_single_raw", "rinalmo_ft", "rinalmo_ft/bprna_new",
          ROOT / "baseline_recovery_v3_foundation/predictions/rinalmo_author_bprna_ft/bprna_new_v3.jsonl", "77a114b87cd3378cae0cd837f93c49055cf18c6cd0391eb64f76014db13a54f0",
          EXT / "rinalmo_ft/raw/bprna_new.json", "6bf0080d656a3859e36492c7497e64f50fbcc5d7abf091e3546c8ce3c0e6baee",
          expected(5388, 51448, 42537, 83736, 0.5474065010373996, 0.38057758314593443, 0.4489961556755058, 0.44123738936817275),
          training="author RiNALMo-giga bpRNA secondary-structure fine-tune; deterministic repeats are not an ensemble"),
    asset("nucleicbert_author_secstr_full_ft_single_raw", "nucleicbert_frozen_official", "nucleicbert_frozen_official/bprna_new",
          ROOT / "baseline_recovery_v3_foundation/predictions/nucleicbert_author_secstr_full_ft/bprna_new_v3.jsonl", "e763174fde88d566c9c3b2ef879f6bfe89a81a8852704c0ac9d74f7da7deaea3",
          EXT / "nucleicbert_frozen/raw/bprna_new.json", "e1a6d1f4f355648926505a3fa1e5bf5d7204b2eb037c47bfc13479bc46077a1c",
          expected(5388, 15926, 15754, 119258, 0.5027146464646465, 0.11780979997632857, 0.19088599098667178, 0.18515300147695582),
          training="author secstr.ckpt epoch99/global-step258000; official frozen=False full fine-tune",
          postprocess="B=2 first-598-nt protocol; threshold=0.5; historical frozen_official filename is not the training identity"),
    asset("nucleicbert_project_frozen_encoder_tr1c_single_raw", "nucleicbert_ft", "nucleicbert_ft/bprna_new",
          ROOT / "baseline_recovery_v3_foundation/predictions/nucleicbert_project_frozen_encoder_tr1c/bprna_new_v3.jsonl", "89f4dc81d2d4172cb27d23b0cc20dca65f9b7db6c6e77f3249e74ad7b409bf29",
          EXT / "nucleicbert_ft/raw/bprna_new.json", "eb6dbea04a07db9140408d355e686629e7279c5005aed84a4fe5abcf3395cd34",
          expected(5388, 60809, 156240, 74375, 0.2801625439416906, 0.4498239436619718, 0.34527713189848763, 0.3611124978084714),
          training="official pretrained encoder frozen; project SecStruct2DPredictionHead trained on TR1c; not author full fine-tune",
          postprocess="B=4 first-598-nt protocol; VL0-selected threshold=0.55"),
    asset("structrfm_author_bprna1m_single_raw", "structrfm", "structrfm/bprna_new",
          ROOT / "baseline_recovery_v3_foundation/predictions/structrfm_author_bprna1m_ssp/bprna_new_v3.jsonl", "5753f4beacdd2d71c0ebb0edfa40fe5bc152dacff708e8008c19ed9e8611e7cd",
          Path("/root/rna_jepa_data/eval_decision_ref/structrfm_bprna_new.json"), "3243b141eed79a8c419bc737dece47943fd7708f2db5cd881523dcc9df1d6519",
          expected(5388, 73864, 62616, 61320, 0.5412075029308323, 0.5463960231980116, 0.5437893868896873, 0.5338105702966888),
          training="author structRFM_SSP_bpRNA1m downstream fine-tune; local checkpoint bytes remain unverified",
          postprocess="frozen LM embeddings + MixedFold scorer + constrained DP; max_len=512"),
    asset("rnafm_project_frozen_encoder_crf_single_raw", "rnafm_project_frozen_encoder_crf", "rnafm_project_frozen_encoder_crf/bprna_new",
          ROOT / "baseline_recovery_v3_foundation/predictions/rnafm_project_frozen_encoder_crf/bprna_new_v3.jsonl", "e56c25d0f23c4798a83ae24670c5a849a03dc96c9fb1436a7cc26cf56c39ee18",
          Path("/root/rna_jepa_data/eval_decision_ref/ow_rnafm_ff_b4_s0_step20000_bprna_new/result.json"), "758b0087ac4337420fbc6120913476fda98db9c63cf204ab69d6cb6720d60a8f",
          expected(5388, 37568, 77317, 97616, 0.3270052661357009, 0.27790271037992664, 0.3004610727439227, 0.3201747208074672),
          training="project frozen RNA-FM encoder + TR0 CRF/decision head; not official RNA-FM bpRNA ResNet checkpoint",
          postprocess="learned Turner prior + exact max-product/Nussinov decode"),
    asset("spot_rna_registry_raw_single", "spot_rna", "spot_rna/bprna_new",
          ROOT / "baseline_recovery_v3_spot_rnaformer/predictions/spot_rna_registry_raw_bprna_new_v3.jsonl", "f1cc23a4febb650f02fc3a83472f3acc2f01d2bd7910bfeae183e3575c7ee3f3",
          EXT / "spotrna/raw/bprna_new.json", "613ba7a4c8ceae0a8197657a6b96e8a01c3c54cd923cfa81974b9ddbb16af6cc",
          expected(5388, 0, 0, 135184, None, 0.0, 0.0, 0.0001855976243504083),
          training="MultiMolecule-converted SPOT-RNA checkpoint; not original-author distribution",
          postprocess="threshold=0.335; registry raw arm has 5388 empty predictions; not the postprocessed arm"),
    asset("spot_rna_postprocessed_single_raw", "spot_rna", "spot_rna_postprocessed/bprna_new",
          ROOT / "baseline_recovery_v3_spot_rnaformer/predictions/spot_rna_postprocessed_bprna_new_v3.jsonl", "f5303b71f5f1ca5e0786b418e13c49c077a2b49fe0e7027492f063fb84d5cdca",
          EXT / "spotrna/raw_postprocessed/bprna_new.json", "c76f473574fa0032213f49bc8e03e70406bf7cb2720d4a78a57d149b71ff5ec8",
          expected(5388, 77326, 52486, 57858, 0.5956768249468463, 0.5720055627884957, 0.58360126190584, 0.5682797332188521),
          training="same converted SPOT-RNA checkpoint as registry raw; separate inference identity",
          postprocess="threshold=0.335 plus valid-pair postprocessing; never substituted for registry raw"),
    asset("rnaformer_bprna_single_raw", "rnaformer_bprna", "rnaformer_bprna/bprna_new",
          ROOT / "baseline_recovery_v3_spot_rnaformer/predictions/rnaformer_bprna_bprna_new_v3.jsonl", "b7d2b2db977d0d9017875694b1b7e3455c222e599e7606afea88646656024a2f",
          EXT / "rnaformer/raw/bprna_new.json", "1559e6c4fbfcbb5e36debfe9c41b9ed0af6575a01100beac4b2c07780397c856",
          expected(5388, 51372, 21605, 83812, 0.7039478191759048, 0.38001538643626465, 0.49357948895326215, 0.45885354923978383),
          training="official RNAformer 32M bpRNA checkpoint; single model"),
    asset("kirigami_standard_single_raw", "kirigami", "kirigami/bprna_new",
          ROOT / "baseline_recovery_v4_kirigami/predictions/kirigami_project_replay_bprna_new_frozen_v4.jsonl", "7993d498db5ad1b8ca5a33a5c23984bdd6df121a09a73c574cac36396c9749cd",
          EXT / "kirigami/raw/bprna_new.json", "5139f66259f1af194008741daa36105ddf5be9f5a66124abfc04430bb443bb01",
          expected(5388, 64985, 24551, 70199, 0.7257974446032881, 0.480715173393301, 0.5783641865432538, 0.552667340300811),
          training="official Kirigami weights; project replay", postprocess="standard decoder; optimized C decode is pair-identical evidence, not another asset"),
]

THERMO = [
    ("vienna_mfe", "vienna_mfe/bprna_new/recovered_v4", "de411b04baad43e1c5cfff5627d0fa7b3bebe2e8cf2bf33e6ffb406ba09e81dd", "vienna/raw_strict/bprna_new.json", "615c47042b927f06d5870537ecc2d71ac3cbdabac03c14957d9e5c19ca42d86b", expected(5388, 99112, 76430, 36072, 0.5646056214467193, 0.7331636880104154, 0.6379382478453686, 0.6580838104925161), "ViennaRNA 2.7.2 MFE"),
    ("vienna_centroid", "vienna_centroid/bprna_new/recovered_v4", "c266eb84d708f6add59f36d2911ea55711fcaba13f51f9a40007a4212fbb0425", "vienna/raw_strict/bprna_new.json", "615c47042b927f06d5870537ecc2d71ac3cbdabac03c14957d9e5c19ca42d86b", expected(5388, 97991, 56297, 37193, 0.6351174427045525, 0.7248712865427861, 0.6770326663718771, 0.6820727922460756), "ViennaRNA 2.7.2 centroid/BPP"),
    ("vienna_mea", "vienna_mea/bprna_new/recovered_v4", "f11252da4f4b2fc643806d874e87f43f623cca86474b16dc9fffb55e91b49ad4", "vienna/raw_strict/bprna_new.json", "615c47042b927f06d5870537ecc2d71ac3cbdabac03c14957d9e5c19ca42d86b", expected(5388, 100390, 67724, 34794, 0.5971543119549829, 0.7426174695230204, 0.6619891987418315, 0.6764248352692245), "ViennaRNA 2.7.2 MEA/BPP"),
    ("rnastructure_fold", "rnastructure_fold/bprna_new", "cd24502d12b8213c51dac6b5b5dda6bbc41abfa6cab2758a3612b1eb3883d814", "rnastructure/raw/bprna_new.json", "ac78069c6fc49765c29e3c0cd376bb746b7bdbf0a64ab2b116c2c1bcec09ae42", expected(5388, 97675, 76320, 37509, 0.5613667059398258, 0.7225337318025802, 0.6318346330119445, 0.6496591555816604), "RNAstructure 6.6 Fold MFE"),
    ("rnastructure_maxexpect", "rnastructure_maxexpect/bprna_new", "fa78a043a545f9d0699ac4a6e564016d63e04ede716728c58416496d2f1fdcfe", "rnastructure/raw/bprna_new.json", "ac78069c6fc49765c29e3c0cd376bb746b7bdbf0a64ab2b116c2c1bcec09ae42", expected(5388, 99991, 66506, 35193, 0.6005573674000132, 0.7396659367972541, 0.6628922603677394, 0.6746697017140534), "RNAstructure 6.6 MaxExpect"),
    ("contrafold", "contrafold/bprna_new", "26430a1cb844fadcd23e9807aa86820ee169944c5e3b37ff738196fb26be3aab", "contrafold/raw/bprna_new.json", "6964a40fa3174f1b43bc13955b8e2a41b7f83014dabf91a1036c46b625884234", expected(5388, 104254, 59814, 30930, 0.6354316502913426, 0.7712007338146526, 0.6967639314022964, 0.7108879112448466), "CONTRAfold native posterior MEA"),
]
for system, prediction_id, digest, source, source_digest, metrics, protocol in THERMO:
    baseline_id = "contrafold" if system == "contrafold" else system
    asset_id = system + ("_mea" if system == "contrafold" else "") + "_single_raw"
    prediction_system = "contrafold_mea" if system == "contrafold" else system
    ASSETS.append(asset(asset_id, baseline_id, prediction_id,
                        ROOT / f"baseline_recovery_v4_thermodynamic/predictions/{prediction_system}/bprna_new_raw_v4.jsonl",
                        digest, EXT / source, source_digest, metrics, "deterministic_single_predictor",
                        training="not sequence-fit learned model; frozen parameters", postprocess=protocol))
ASSETS.append(asset("contrafold_mea_single_clean_tr1c", "contrafold", "contrafold/bprna_new/clean_tr1c",
                    ROOT / "baseline_recovery_v4_thermodynamic/predictions/contrafold_mea/bprna_new_clean_tr1c_v4.jsonl", "5066dcae384ec724059372e4c099ddabf9bf5dc072743d2ef514d44954db14e2",
                    EXT / "contrafold/clean_tr1c/bprna_new.json", "642059d4a0f4c48f4a8672a24a0daed2177a248133df238c9da281dc079d0597",
                    expected(5386, 104221, 59802, 30920, 0.6354047907915353, 0.7712019298362451, 0.6967482718508912, 0.7108660734957),
                    "deterministic_single_predictor", "PROJECT_TR1C_CLEAN_5386", training="same CONTRAfold predictor as raw; distinct label subset only",
                    postprocess="native MEA; project TR1c exclusion is not verified CONTRAfold-own-train cleaning"))


def pair_set(values, length):
    pairs = set()
    partners = set()
    for value in values:
        assert len(value) == 2
        i, j = map(int, value)
        assert 0 <= i < j < length and i not in partners and j not in partners
        partners.update((i, j))
        pairs.add((i, j))
    return pairs


def load_predictions(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        document = json.loads(path.read_text())
        source = document.get("per_sequence", document.get("predictions"))
        key = "pred_pairs" if "per_sequence" in document else "pairs"
        rows = [{"name": row["name"], "pairs": row[key]} for row in source]
    assert len({row["name"] for row in rows}) == len(rows)
    return rows


def load_jsonl(path, digest):
    assert sha256(path) == digest
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert len({row["name"] for row in rows}) == len(rows)
    return rows


def parse_bpseq(payload):
    sequence, directed = [], []
    for line in payload.decode().splitlines():
        if not line.strip():
            continue
        fields = line.split()
        index = int(fields[0]) - 1
        assert index == len(sequence)
        sequence.append(fields[1].upper().replace("T", "U"))
        partner = int(fields[2]) - 1
        if partner >= 0:
            directed.append((index, partner))
    pairs = {(min(i, j), max(i, j)) for i, j in directed}
    return "".join(sequence), pair_set(pairs, len(sequence))


def load_official_labels():
    assert sha256(OFFICIAL_DATA) == OFFICIAL_DATA_SHA
    rows = []
    with tarfile.open(OFFICIAL_DATA, "r:gz") as handle:
        members = [member for member in handle.getmembers() if member.isfile() and
                   member.name.startswith("BPfold_data/test_data/bpRNAnew/") and member.name.endswith(".bpseq")]
        assert len(members) == 5401
        for member in members:
            sequence, pairs = parse_bpseq(handle.extractfile(member).read())
            rows.append({"name": Path(member.name).name, "seq": sequence, "pairs": sorted(map(list, pairs))})
    assert len({row["name"] for row in rows}) == 5401
    return rows


def score(labels, predictions):
    label_by_name = {row["name"]: row for row in labels}
    prediction_by_name = {row["name"]: row for row in predictions}
    missing = sorted(set(label_by_name) - set(prediction_by_name))
    extra = sorted(set(prediction_by_name) - set(label_by_name))
    assert not missing and not extra, (missing[:3], extra[:3])
    tp = fp = fn = 0
    f1_values = []
    ordered_predictions = []
    for label in labels:
        name = label["name"]
        prediction = prediction_by_name[name]
        if prediction.get("seq") is not None:
            assert prediction["seq"].upper().replace("T", "U") == label["seq"].upper().replace("T", "U"), name
        truth = pair_set(label["pairs"], len(label["seq"]))
        predicted = pair_set(prediction["pairs"], len(label["seq"]))
        a, b, c = len(truth & predicted), len(predicted - truth), len(truth - predicted)
        tp, fp, fn = tp + a, fp + b, fn + c
        denominator = 2 * a + b + c
        f1_values.append(2 * a / denominator if denominator else 1.0)
        ordered_predictions.append([name, sorted(map(list, predicted))])
    n = len(labels)
    return {"n_dataset_sequences": n, "n_with_prediction": n, "coverage": 1.0,
            "missing_names": missing, "extra_names": extra, "tp": tp, "fp": fp, "fn": fn,
            "micro_precision": tp / (tp + fp) if tp + fp else None,
            "micro_recall": tp / (tp + fn) if tp + fn else None,
            "micro_f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "macro_f1": math.fsum(f1_values) / n,
            "zero_pair_f1_convention": "F1=1 when truth and prediction both have zero pairs",
            "order_matches_labels": [row["name"] for row in predictions] == [row["name"] for row in labels],
            "prediction_identity_sha256": hashlib.sha256(json.dumps(ordered_predictions, separators=(",", ":")).encode()).hexdigest()}


def check_expected(actual, expected_metrics):
    for key, value in expected_metrics.items():
        if isinstance(value, float) or value is None:
            if value is None:
                assert actual[key] is None, (key, actual[key])
            else:
                assert math.isclose(actual[key], value, rel_tol=0.0, abs_tol=1e-12), (key, actual[key], value)
        else:
            assert actual[key] == value, (key, actual[key], value)


def label_info(spec):
    if spec["label_protocol"] == "PROJECT_DATABASE_RAW_5388":
        return load_jsonl(RAW_DATA, RAW_DATA_SHA), str(RAW_DATA), RAW_DATA_SHA
    if spec["label_protocol"] == "PROJECT_TR1C_CLEAN_5386":
        return load_jsonl(CLEAN_DATA, CLEAN_DATA_SHA), str(CLEAN_DATA), CLEAN_DATA_SHA
    assert spec["label_protocol"] == "BPFOLD_OFFICIAL_REFERENCE_5401"
    return load_official_labels(), str(OFFICIAL_DATA), OFFICIAL_DATA_SHA


def build_rows():
    assert len(ASSETS) == 20
    assert len({row["asset_id"] for row in ASSETS}) == len({row["prediction_id"] for row in ASSETS}) == 20
    rows = []
    for spec in ASSETS:
        assert sha256(spec["prediction_path"]) == spec["prediction_sha256"], spec["asset_id"]
        assert sha256(spec["source_path"]) == spec["source_sha256"], spec["asset_id"]
        labels, label_path, label_sha = label_info(spec)
        result = score(labels, load_predictions(spec["prediction_path"]))
        check_expected(result, spec["expected_metrics"])
        row = {key: value for key, value in spec.items() if key != "expected_metrics"}
        row.update({"status": "PASS_SHA_ID_METRICS", "dataset_id": "bprna_new" if spec["label_protocol"] != "BPFOLD_OFFICIAL_REFERENCE_5401" else "bpfold_official_bprna_new_reference",
                    "label_path": label_path, "label_sha256": label_sha,
                    "n_sequences": result["n_dataset_sequences"], "metrics": {key: value for key, value in result.items() if key not in {"order_matches_labels", "prediction_identity_sha256"}},
                    "order_matches_labels": result["order_matches_labels"],
                    "prediction_identity_sha256": result["prediction_identity_sha256"],
                    "sota_use": ("official BPfold protocol only; not project raw or clean SOTA ranking" if row["label_protocol"] == "BPFOLD_OFFICIAL_REFERENCE_5401" else "project raw ranking only; not clean SOTA" if row["raw_clean_scope"] == "raw" else "clean subset only; not raw ranking")})
        rows.append(row)
    return rows


def strongest(rows, single_only=False):
    eligible = [row for row in rows if row["label_protocol"] == "PROJECT_DATABASE_RAW_5388" and
                (not single_only or row["model_form"] != "six_checkpoint_ensemble")]
    return max(eligible, key=lambda row: row["metrics"]["micro_f1"])


def public_strongest(row):
    return {"asset_id": row["asset_id"], "baseline_id": row["baseline_id"], "model_form": row["model_form"],
            "micro_f1": row["metrics"]["micro_f1"], "macro_f1": row["metrics"]["macro_f1"],
            "coverage": row["metrics"]["coverage"], "n_sequences": row["n_sequences"]}


def recovered_document(rows):
    best = strongest(rows)
    single = strongest(rows, single_only=True)
    return {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PASS_20_EXISTING_ASSETS_SHA_ID_METRICS", "n_assets": len(rows),
            "metric_contract": "strict zero-based inclusive exact pair sets; micro pools TP/FP/FN; macro mean per-sequence F1; empty truth+prediction F1=1",
            "scope_contract": {"project_raw_assets": 18, "bpfold_official_reference_assets": 1, "project_clean_assets": 1,
                               "chanrg_empty_name_records_processed": False,
                               "historical_artifacts_modified": False,
                               "raw_is_not_clean": True},
            "strongest_project_raw": public_strongest(best),
            "strongest_project_raw_single_predictor": public_strongest(single),
            "xens2_project_raw_reference": {"micro_f1": 0.553142073575729, "macro_f1": 0.5422090871138705,
                                             "gap_to_strongest_ensemble_micro": best["metrics"]["micro_f1"] - 0.553142073575729,
                                             "gap_to_strongest_ensemble_macro": best["metrics"]["macro_f1"] - 0.5422090871138705},
            "remaining_blocked": ["ContextFold, E2Efold, DEBFold and ERNIE-RNA have no eligible full bpRNA-new prediction asset",
                                  "raw assets are not homology-decontaminated; exact training-union and exposure blocks remain system-specific",
                                  "structRFM local checkpoint bytes and several license/training provenance fields remain blocked",
                                  "only CONTRAfold has this particular TR1c-clean asset; no cross-system clean SOTA ranking is supported",
                                  "CHANRG records with empty names were not processed"],
            "rows": rows}


def prediction_registry_row(asset_row):
    return {"prediction_id": asset_row["prediction_id"], "model": asset_row["baseline_id"],
            "split": "bprna_new" if asset_row["dataset_id"] == "bprna_new" else asset_row["dataset_id"],
            "repeat": None, "variant": asset_row["raw_clean_scope"], "path": asset_row["prediction_path"],
            "exists": True, "bytes": Path(asset_row["prediction_path"]).stat().st_size,
            "sha256": asset_row["prediction_sha256"], "hash_scope": "full_content",
            "n_sequences": asset_row["n_sequences"], "data_path": asset_row["label_path"],
            "prediction_schema": "normalized_name_pairs_jsonl" if asset_row["prediction_path"].endswith(".jsonl") else "frozen_json",
            "count_consistent": True, "status": "FROZEN", "e0_scope": "BPRNA_NEW_RECOVERY_20261125",
            "baseline_id": asset_row["baseline_id"], "dataset_id": asset_row["dataset_id"],
            "recovery_asset_id": asset_row["asset_id"], "model_form": asset_row["model_form"],
            "label_protocol": asset_row["label_protocol"], "prediction_origin": asset_row["prediction_origin"],
            "training_identity": asset_row["training_identity"], "postprocess_identity": asset_row["postprocess_identity"],
            "source_artifact": {"path": asset_row["source_path"], "sha256": asset_row["source_sha256"]},
            "prediction_identity_sha256": asset_row["prediction_identity_sha256"], "metrics": asset_row["metrics"],
            "comparability": "raw is not clean; label protocols and model forms are separate ranking columns"}


def update_registries(rows):
    documents = {}
    for name, digest in BASE_SHA.items():
        path = MANIFESTS / name
        assert sha256(path) == digest, f"unexpected base registry: {name}"
        documents[name] = json.loads(path.read_text())
    prediction_doc = documents["prediction_registry.json"]
    prediction_by_id = {row["prediction_id"]: row for row in prediction_doc["rows"]}
    for asset_row in rows:
        recovered = prediction_registry_row(asset_row)
        if asset_row["prediction_id"] in prediction_by_id:
            existing = prediction_by_id[asset_row["prediction_id"]]
            assert existing["sha256"] == asset_row["source_sha256"] or existing["sha256"] == asset_row["prediction_sha256"]
            existing["recovered_bprna_new"] = recovered
        else:
            prediction_doc["rows"].append(recovered)
            prediction_by_id[asset_row["prediction_id"]] = recovered
    prediction_doc["n_rows"] = len(prediction_doc["rows"])
    prediction_doc["recovery_overlay"] = "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json"

    data_doc = documents["data_registry.json"]
    raw_data = next(row for row in data_doc["rows"] if row["dataset_id"] == "bprna_new")
    official_data = {"dataset_id": "bpfold_official_bprna_new_reference", "role": "external_protocol_reference",
                     "description": "BPfold v0.2 official bpRNAnew reference labels; not project DATABASE labels",
                     "scope": "EXTERNAL_PROTOCOL_ONLY", "label_source": "BPFOLD_OFFICIAL_REFERENCE",
                     "path": str(OFFICIAL_DATA), "exists": True, "sha256": OFFICIAL_DATA_SHA,
                     "hash_scope": "full_content", "n_sequences": 5401, "status": "FROZEN",
                     "prediction_ids": ["bpfold_official_release/bprna_new_official_reference"],
                     "comparability": "Do not mix with project raw 5388 labels; 13 official-only records retained"}
    data_doc["rows"] = [row for row in data_doc["rows"] if row["dataset_id"] != official_data["dataset_id"]] + [official_data]
    new_raw_ids = [row["prediction_id"] for row in rows if row["dataset_id"] == "bprna_new" and row["prediction_id"] not in raw_data["prediction_ids"]]
    raw_data["prediction_ids"].extend(new_raw_ids)
    data_doc["n_rows"] = len(data_doc["rows"])

    baseline_doc = documents["baseline_registry.json"]
    baseline_by_id = {row["baseline_id"]: row for row in baseline_doc["rows"]}
    if "rnafm_project_frozen_encoder_crf" not in baseline_by_id:
        source = next(row for row in rows if row["baseline_id"] == "rnafm_project_frozen_encoder_crf")
        baseline_doc["rows"].append({"baseline_id": "rnafm_project_frozen_encoder_crf", "model": "RNA-FM frozen encoder + project CRF head",
            "category": "FOUNDATION_MODEL", "row_type": "PREDICTOR", "role": "baseline", "source_cohorts": [], "cohort_memberships": [],
            "version_protocol": source["training_identity"], "official_weights": {"status": "PROJECT_DOWNSTREAM_ASSET; official pretrained identity retained in recovery provenance"},
            "training_data": {"status": "PROJECT_TR0_FROZEN_ENCODER_CRF", "project_training_dataset_ids": ["ref_tr_bprna"], "development_dataset_ids": ["bprna_vl0"]},
            "license": {"status": "BLOCKED_PROVENANCE", "spdx": None}, "inference": {"status": "FROZEN_RAW_PROTOCOL"},
            "comparability": {"status": "PROJECT_RAW_GT_ONLY; not official RNA-FM ResNet", "clean_subset_policy": "not applied"},
            "provenance_status": "BLOCKED_PROVENANCE", "evidence": {"source_type": "local_file", "path": source["source_path"], "exists": True, "sha256": source["source_sha256"], "hash_scope": "full_content"},
            "evidence_additional": [], "status": "RAW_BPRNA_NEW_ONLY", "inclusion_reason": "recovered project backbone-swap predictor; not the official downstream model",
            "exclusion_reason": None, "prediction_registry_scope": "FROZEN_RAW", "prediction_ids": [], "n_frozen_prediction_rows": 0,
            "n_summary_only_rows": 0, "n_blocked_prediction_rows": 0, "unknown_fields_are_not_zero": True})
        baseline_by_id = {row["baseline_id"]: row for row in baseline_doc["rows"]}
    for baseline_id, group in __import__("itertools").groupby(sorted(rows, key=lambda row: row["baseline_id"]), key=lambda row: row["baseline_id"]):
        group = list(group)
        baseline = baseline_by_id[baseline_id]
        baseline["bprna_new_recovery"] = {"registry": "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json",
                                            "asset_ids": [row["asset_id"] for row in group],
                                            "status": "PASS_20_ASSET_OVERLAY_MEMBER", "raw_clean_sota_claim": False}
        for row in group:
            if row["prediction_id"] not in baseline["prediction_ids"]:
                baseline["prediction_ids"].append(row["prediction_id"])
        baseline["n_frozen_prediction_rows"] = sum(prediction_by_id[pid]["status"] == "FROZEN" for pid in baseline["prediction_ids"])
        if baseline["prediction_registry_scope"] in {"SUMMARY_ONLY", "NO_CURRENT_PREDICTION", "BLOCKED"}:
            baseline["prediction_registry_scope"] = "FROZEN_RAW"
        if baseline_id.startswith("vienna_") or baseline_id == "structrfm":
            baseline["status"] = "RAW_BPRNA_NEW_RECOVERED"
            baseline["inference"]["status"] = "FROZEN_RAW_PROTOCOL"
            baseline["comparability"]["status"] = "PROJECT_RAW_GT_ONLY; raw is not clean"
    author = baseline_by_id["nucleicbert_frozen_official"]
    author["model"] = "NucleicBERT author secstr full fine-tune"
    author["version_protocol"] = "official secstr.ckpt epoch99/global-step258000; official frozen=False; full fine-tune"
    author["training_data"] = {"status": "AUTHOR_FULL_FINE_TUNE", "external_training_corpora": ["RNAStrAlign/TR0-style secondary-structure data"], "project_training_dataset_ids": []}
    author["identity_correction"] = "Historical frozen_official filename does not mean frozen-encoder training."
    project = baseline_by_id["nucleicbert_ft"]
    project["model"] = "NucleicBERT frozen encoder + project TR1c head"
    project["version_protocol"] = "official pretrained encoder frozen; project SecStruct2DPredictionHead on TR1c; threshold=0.55"
    project["training_data"] = {"status": "PROJECT_FROZEN_ENCODER_DOWNSTREAM_HEAD", "project_training_dataset_ids": ["bprna_tr1c"], "development_dataset_ids": ["bprna_vl0"]}
    project["identity_correction"] = "Historical nucleicbert_ft is a project frozen-encoder head, not the author full-fine-tuned checkpoint."
    baseline_by_id["nucleicbert_frozen_project_head"]["identity_disposition"] = "Legacy cohort alias of nucleicbert_ft; no additional prediction asset"
    baseline_doc["n_rows"] = len(baseline_doc["rows"])
    baseline_doc["supplemental_row_count"] = len([row for row in baseline_doc["rows"] if not row["source_cohorts"]])
    baseline_doc["four_category_counts"] = {category: sum(row["category"] == category for row in baseline_doc["rows"])
                                              for category in ("STRUCTURED_DECODER", "DIRECT_NEURAL", "FOUNDATION_MODEL", "PROJECT")}
    baseline_doc["recovery_overlay"] = "manifests/foldnar_v1/bprna_new_recovered_prediction_registry.json"
    return documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=ROOT / "baseline_registry_update_20261125")
    args = parser.parse_args()
    rows = build_rows()
    document = recovered_document(rows)
    documents = update_registries(rows)
    write_json(MANIFESTS / "bprna_new_recovered_prediction_registry.json", document)
    for name, value in documents.items():
        write_json(MANIFESTS / name, value)
    args.artifact_root.mkdir(parents=True, exist_ok=True)
    write_json(args.artifact_root / "registration_recomputation.json", document)
    print(json.dumps({"status": document["status"], "n_assets": len(rows),
                      "strongest_project_raw": document["strongest_project_raw"],
                      "strongest_single": document["strongest_project_raw_single_predictor"]}, indent=2))


if __name__ == "__main__":
    main()
