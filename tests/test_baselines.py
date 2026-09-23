"""Tests for the baseline reproduction matrix, the data inventory/fetch
scaffolding, and the CDPFold calibration check (spec §7.2 / §3 / §0.9.4).

Everything runs offline: the inventory/fetch tools are exercised in offline /
dry-run mode, and the CDPFold verdict logic is exercised on crafted synthetic
probability matrices (the real CDPFold model is not available here).

Run:  python -m pytest tests/test_baselines.py -v
"""

import json
import os
import sys
from dataclasses import replace

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "data", "ss"))
sys.path.insert(0, os.path.join(_ROOT, "eval", "ss"))
sys.path.insert(0, os.path.join(_ROOT, "src"))

import baselines as B  # noqa: E402
import cdpfold_check as C  # noqa: E402
import fetch as F  # noqa: E402
import inventory as I  # noqa: E402


# ===========================================================================
# 1. inventory
# ===========================================================================
def test_inventory_every_source_has_a_four_category_status():
    inv = I.build_inventory(offline=True)
    assert inv["entries"], "inventory is empty"
    for e in inv["entries"]:
        assert e["status"] in I.STATUSES, e["key"]
        assert e["evidence"] in I.EVIDENCES, e["key"]
    # exactly four categories, no extras
    assert set(I.STATUSES) == {"已落盘", "待获取", "网络受限", "不可得"}


def test_inventory_offline_never_marks_unprobed_as_verified():
    inv = I.build_inventory(offline=True)  # no probe cache -> nothing probed
    for e in inv["entries"]:
        assert e["verified"] is False, e["key"]
        assert e["evidence"] != I.EVIDENCE_LIVE, e["key"]


def test_inventory_no_assumed_available_assertion_holds():
    inv = I.build_inventory(offline=True)
    assert I.assert_no_assumed_available(inv) is True


def test_inventory_no_assumed_available_assertion_catches_bad_status():
    inv = I.build_inventory(offline=True)
    inv["entries"][0]["status"] = "assumed_available"
    with pytest.raises(AssertionError):
        I.assert_no_assumed_available(inv)


def test_inventory_required_sources_present():
    keys = {e["key"] for e in I.build_inventory(offline=True)["entries"]}
    required = {
        "bprna_1m", "bprna_new", "archiveii", "rnastralign", "pdb_derived",
        "pseudobase_pp", "rna_puzzles_casp", "probing_data", "rnacentral", "rfam",
        "ensembl_gencode", "viennarna", "rnastructure", "linearpartition",
    }
    assert required <= keys
    # the seven Zenodo 17786045 archives + the 20 GB corpus + mRNABERT are re-reported
    assert "zenodo_17786045::full_length" in keys
    assert "zenodo_17786045::3UTR" in keys
    assert "zenodo_12516160" in keys
    assert "mrnabert_weights" in keys


def test_inventory_landed_assets_carry_recorded_md5s_and_zero_ss_annotations():
    inv = I.build_inventory(offline=True)
    by_key = {e["key"]: e for e in inv["entries"]}
    assert by_key["zenodo_17786045::full_length"]["expected_md5"] == "3652178c257341010800e2d241a9c258"
    assert by_key["zenodo_17786045::full_length"]["expected_size_bytes"] == 281158
    assert by_key["zenodo_12516160"]["expected_md5"] == "bf8bc5c946a0bd3b07716b1c7f785d54"
    assert by_key["zenodo_12516160"]["expected_size_bytes"] == 20021485711
    # the prominent warning: landed assets carry no SS annotations
    assert "不含任何二级结构标注" in inv["ss_annotation_warning"]
    for e in inv["entries"]:
        if e["kind"] == "landed_asset":
            assert e["ss_annotations"] == I.SS_NO


def test_inventory_probe_cache_promotes_only_cached_entries(tmp_path):
    cache = tmp_path / "probe_cache.json"
    cache.write_text(json.dumps({
        "bprna_1m": {"status": I.STATUS_LANDED, "evidence": I.EVIDENCE_LIVE, "detail": "probed"}
    }))
    inv = I.build_inventory(offline=True, cache_path=str(cache))
    by_key = {e["key"]: e for e in inv["entries"]}
    assert by_key["bprna_1m"]["verified"] is True
    assert by_key["bprna_1m"]["evidence"] == I.EVIDENCE_LIVE
    # every other entry stays unprobed
    assert by_key["archiveii"]["verified"] is False


def test_inventory_probe_tool_reports_unavailable_fast():
    src = I.Source(key="x", name="x", kind="teacher", probe=I.PROBE_TOOL,
                   tool_import="definitely_not_installed_module_xyz",
                   recorded_status=I.STATUS_TO_FETCH)
    out = I.probe_source(src)
    assert out["status"] in I.STATUSES
    assert out["status"] == I.STATUS_TO_FETCH


def test_inventory_probe_local_detects_present_file(tmp_path):
    f = tmp_path / "artefact.bin"
    f.write_bytes(b"12345")
    src = I.Source(key="y", name="y", kind="landed_asset", probe=I.PROBE_LOCAL,
                   local_path=str(f), expected_size_bytes=5,
                   recorded_status=I.STATUS_LANDED)
    out = I.probe_source(src)
    assert out["status"] == I.STATUS_LANDED


def test_inventory_render_markdown_has_the_warning_banner():
    md = I.render_markdown(I.build_inventory(offline=True))
    assert "已落盘资产含二级结构标注数 = 0" in md
    assert "recorded-fact" in md or "recorded_fact" in md


# ===========================================================================
# 2. fetch
# ===========================================================================
def test_fetch_dry_run_makes_no_network_or_subprocess(monkeypatch, capsys):
    def _boom(*a, **k):
        raise AssertionError("dry-run must not touch the network / spawn processes")
    monkeypatch.setattr(F.subprocess, "run", _boom)
    rc = F.main(["--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out or "获取计划" in out


def test_fetch_plan_actions_are_fetch_or_blocked():
    plan = F.build_plan(F.structure_specs(), "/tmp/ss")
    assert plan
    for e in plan:
        assert e["action"] in ("fetch", "blocked")
        if e["action"] == "blocked":
            assert e["blocked_reason"]


def test_fetch_unknown_hashes_are_marked_and_never_invented():
    for spec in F.structure_specs():
        # unknown hash -> blank + 待核验, never a fabricated value
        assert spec.expected_md5 == ""
        assert spec.hash_status == F.HASH_UNKNOWN


def test_fetch_blocked_specs_raise_actionable_error():
    spec = F.FetchSpec(key="z", name="z", target_name="z.tgz", urls=[],
                       expected_size_bytes=None)
    with pytest.raises(F.FetchBlockedError):
        F.fetch_spec(spec, "/tmp/ss")
    spec2 = F.FetchSpec(key="z2", name="z2", target_name="z2.tgz",
                        urls=["https://example.invalid/x"], expected_size_bytes=None)
    with pytest.raises(F.FetchBlockedError):
        F.fetch_spec(spec2, "/tmp/ss")


def test_fetch_manifest_write_and_roundtrip(tmp_path):
    m = F.new_manifest()
    F.upsert_entry(m, {"key": "a", "status": "ok", "actual_md5": "aa"})
    F.upsert_entry(m, {"key": "b", "status": "missing"})
    F.upsert_entry(m, {"key": "a", "status": "updated"})  # replace, not duplicate
    assert len(m["entries"]) == 2
    assert {e["key"] for e in m["entries"]} == {"a", "b"}
    assert next(e for e in m["entries"] if e["key"] == "a")["status"] == "updated"

    path = tmp_path / "manifest.json"
    F.write_manifest(str(path), m)
    m2 = F.load_manifest(str(path))
    assert "generated_at_utc" in m2
    assert len(m2["entries"]) == 2
    assert F.load_manifest(str(tmp_path / "nope.json"))["entries"] == []


def test_fetch_verify_artefact(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello")
    import hashlib
    good = hashlib.md5(b"hello").hexdigest()
    assert F.verify_artefact(str(f), good)["verified"] is True
    assert F.verify_artefact(str(f), "deadbeef")["verified"] is False
    # unknown expected hash -> unverifiable, never silently "ok"
    res = F.verify_artefact(str(f), "")
    assert res["status"] == "unverifiable" and res["verified"] is False


# ===========================================================================
# 3. baselines
# ===========================================================================
SPEC_7_2_2 = {
    "rnafold", "rnastructure", "contrafold",
    "linearfold", "linearpartition",
    "ufold", "spotrna", "spotrna2", "mxfold2", "e2efold",
    "rnafm", "rinalmo", "mrnabert", "rnamsm",
    "autoreg_ss_decoder",
    "rhofold", "trrosetta", "alphafold3",
}
CLOSEST_PRIOR = {"contrafold", "cdpfold", "linearpartition", "e2efold"}


def test_baselines_all_routes_registered():
    reg = B.registry()
    keys = {b.key for b in reg}
    assert SPEC_7_2_2 <= keys, SPEC_7_2_2 - keys
    routes = {b.route for b in reg}
    for r in B.ROUTES:
        assert r in routes, r


def test_baselines_closest_prior_set_present_and_flagged():
    for key in CLOSEST_PRIOR:
        b = B.get_baseline(key)
        assert b.closest_prior is True, key
    flagged = {b.key for b in B.list_baselines(closest_only=True)}
    assert CLOSEST_PRIOR <= flagged


def test_baselines_autoregressive_control_present():
    assert any(b.route == B.ROUTE_AUTOREG for b in B.registry())


def test_baselines_linearpartition_speedup_reported_separately():
    assert B.get_baseline("linearpartition").report_speedup_separately is True
    plan = B.plan_baseline("linearpartition")
    assert plan["report_speedup_separately"] is True


def test_baselines_provenance_gate_fails_on_missing_entry():
    # the registry is honestly incomplete (nothing installed, no network)
    rep = B.reproducibility_report()
    assert rep["complete"] is False
    assert len(rep["missing"]) == rep["n_entries"]
    with pytest.raises(B.ReproducibilityGateError):
        B.assert_reproducible()

    # a fully-filled registry passes the gate
    complete = [replace(b, commit="c0ffee", weight_hash="w0rth") for b in B.registry()]
    assert B.reproducibility_report(complete)["complete"] is True
    B.assert_reproducible(complete)  # must not raise

    # drop one commit -> the gate fails and names that entry
    broken = complete[:-1] + [replace(complete[-1], commit="")]
    with pytest.raises(B.ReproducibilityGateError) as ei:
        B.assert_reproducible(broken)
    assert broken[-1].key in str(ei.value)


def _crafted_labels(L=8):
    labels = np.zeros((L, L))
    for i, j in [(0, 3), (1, 5), (2, 7)]:
        labels[i, j] = labels[j, i] = 1.0
    return labels


def test_baselines_calibration_metrics_emitted_alongside_f1_inf():
    labels = _crafted_labels()
    probs = np.where(labels > 0.5, 0.95, 0.05)
    np.fill_diagonal(probs, 0.0)
    res = B.run_baseline("contrafold", probs=probs, labels=labels)
    for metric in B.REQUIRED_METRICS:
        assert metric in res["metrics"], metric
    assert res["metrics"]["f1"] > 0.0
    assert "inf" in res["metrics"]
    assert 0.0 <= res["metrics"]["ece"] <= 1.0


def test_baselines_dry_run_returns_plan_without_metrics():
    plan = B.run_baseline("ufold", dry_run=True)
    assert plan["dry_run"] is True
    assert plan["metrics"] is None
    assert plan["version_lock"]
    assert plan["install_hint"]


def test_baselines_unavailable_tool_raises_actionable_error():
    with pytest.raises(B.ToolUnavailableError) as ei:
        B.run_baseline("ufold")
    err = ei.value
    assert "UFold" in err.tool or "UFold" in str(err)
    assert err.version_lock
    assert err.install_hint


def test_baselines_pair_metrics_known_values():
    m = B.pair_metrics([(0, 3), (1, 4), (2, 5)], [(0, 3), (1, 4)])
    assert m["tp"] == 2 and m["fp"] == 1 and m["fn"] == 0
    assert abs(m["f1"] - (2 * 2 / (2 * 2 + 1 + 0))) < 1e-12
    assert abs(m["inf"] - (2 / np.sqrt(3 * 2))) < 1e-12


# ===========================================================================
# 4. CDPFold calibration check (the project-blocking item)
# ===========================================================================
def test_cdpfold_ece_exact_on_constant_probability_matrix():
    L = 6
    labels = np.zeros((L, L))
    for i, j in [(0, 3), (1, 4), (2, 5)]:
        labels[i, j] = labels[j, i] = 1.0
    probs = np.full((L, L), 0.5)
    np.fill_diagonal(probs, 0.0)
    # 15 upper-triangle pairs, 3 positive -> base rate 0.2, all in one bin
    m = C.calibration_metrics(probs, labels)
    assert abs(m["ece"] - 0.3) < 1e-9
    assert abs(m["marginal_calibration"] - 0.3) < 1e-9


def test_cdpfold_ece_zero_for_perfectly_calibrated_matrix():
    labels = _crafted_labels()
    probs = labels.copy()  # p == y exactly
    m = C.calibration_metrics(probs, labels)
    assert abs(m["ece"]) < 1e-12
    assert abs(m["brier"]) < 1e-12
    assert m["reliability"], "reliability diagram must not be empty"


def test_cdpfold_verdict_c1_holds_vs_fails_on_crafted_cases():
    expectations = {
        "c1_holds": True,
        "c1_fails_within_tol": False,
        "c1_fails_cdp_better": False,
    }
    for kind, expected in expectations.items():
        case = C.synthetic_case(kind)
        res = C.compare_cdpfold(case["cdpfold"], case["ours"], case["exact"],
                                case["labels"], mask=case["mask"])
        assert res["verdict"]["c1_holds"] is expected, kind
        assert res["verdict"]["fallback"] == ("C2 + C1" if expected else "C2 + C4")


def test_cdpfold_verdict_threshold_boundary():
    # exactly at the tolerance -> "within" -> C1 does NOT hold
    v = C.c1_verdict(0.02 + 0.02, 0.02, 0.02)   # delta 0.02
    assert v["within_tol"] is True and v["c1_holds"] is False
    # just outside -> C1 holds
    v2 = C.c1_verdict(0.02 + 0.0201, 0.02, 0.02)
    assert v2["c1_holds"] is True
    # CDPFold better than ours -> C1 does NOT hold
    v3 = C.c1_verdict(0.005, 0.10, 0.02)
    assert v3["c1_holds"] is False


def test_cdpfold_decision_record_written(tmp_path):
    case = C.synthetic_case("c1_fails_within_tol")
    res = C.compare_cdpfold(case["cdpfold"], case["ours"], case["exact"], case["labels"])
    path = tmp_path / "rec.md"
    C.write_decision_record(str(path), res, synthetic=True)
    text = path.read_text()
    assert "CDPFold" in text
    assert "C1 不成立" in text
    assert "C2 + C4" in text            # fallback decision documented
    assert "SYNTHETIC" in text          # honestly marked as a placeholder

    # the C1-holds case documents the other decision
    case2 = C.synthetic_case("c1_holds")
    res2 = C.compare_cdpfold(case2["cdpfold"], case2["ours"], case2["exact"], case2["labels"])
    path2 = tmp_path / "rec2.md"
    C.write_decision_record(str(path2), res2, synthetic=True)
    assert "C2 + C1" in path2.read_text()


def test_cdpfold_load_prob_matrix_roundtrip(tmp_path):
    m = np.array([[0.0, 0.5], [0.5, 0.0]])
    npy = tmp_path / "m.npy"
    np.save(str(npy), m)
    assert np.allclose(C.load_prob_matrix(str(npy)), m)
    txt = tmp_path / "m.txt"
    txt.write_text("0.0 0.5\n0.5 0.0\n")
    assert np.allclose(C.load_prob_matrix(str(txt)), m)
