import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manuscript", default="交接文档/paper/rna_evidence_v2_manuscript_draft.md")
    parser.add_argument("--evidence-dir", default=".round56_work")
    parser.add_argument("--deck", default="交接文档/PRISM_组会汇报_20261107_完整版_副本.pptx")
    args = parser.parse_args()
    root = Path(args.evidence_dir)
    text = Path(args.manuscript).read_text()
    read = lambda name: json.loads((root / name).read_text())
    sha = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
    pilot = read("pilot_v1_decision.json")
    assert pilot["status"] == "STOP_PILOT_NEGATIVE_DEV_ONLY" and not pilot["p3_started"]
    for arm, result in pilot["results"].items():
        assert f"{result['macro_f1']:.6f}" in text
    assert "0.0071958679" in text and "three-seed formal training and frozen external evaluation were not started" in text
    init = read("initialization_report.json")
    assert f"{init['arms']['H0']['native_rinalmo_macro_f1']:.6f}" in text
    assert f"{init['arms']['H2']['native_rinalmo_macro_f1']:.6f}" in text
    assert f"{init['native_h2_minus_native_h0']:.6f}" in text and "step 500 without convergence" in text
    motif = read("bpfold_decision.json")
    assert motif["status"] == "STOP_BPFOLD_DEV_NEGATIVE" and not motif["external_evaluation_started"]
    for value in motif["results_macro_f1"].values():
        assert f"{value:.10f}" in text
    assert "0.0000492302" in text and "not presented as a complete BPfold reproduction" in text
    interface = read("converged_baseline_summary.json")
    labels = {"rinalmo": "RiNALMo", "h0": "H0 host", "handcrafted": "Handcrafted local sequence features",
              "random": "Within-chain label shuffle", "length": "Length", "gc": "GC content"}
    for mode, label in labels.items():
        result = interface["modes"][mode]
        row = f"| {label} | {result['micro_auprc_mean']:.4f} ± {result['micro_auprc_sd']:.4f} | {result['macro_f1_binary_mean']:.4f} ± {result['macro_f1_binary_sd']:.4f} |"
        assert row in text, row
    strata = read("interface_stratified_summary.json")["modes"]["h0"]["mean"]
    assert round(strata["length:100-499"]["micro_auprc"], 4) == 0.6233
    assert strata["length:100-499"]["n_chains"] == 23
    assert round(strata["length:>=1500"]["micro_auprc"], 4) == 0.7725
    assert strata["length:>=1500"]["n_chains"] == 88
    assert "0.6233" in text and "0.7725" in text
    metric_qa = read("independent_metric_qa.json")
    assert metric_qa["n_runs"] == 18 and metric_qa["max_abs_metric_difference"] < 1e-12
    assert "2.22×10⁻16" in text
    exact = read("exact_union_summary.json")
    homology = read("homology_union_summary.json")
    for key in ("bprna_tr1c", "bprna_vl0", "bprna_new", "bprna_ts0", "rfam_fam", "ncrna_test.csv", "interface_holdout"):
        exact_row = exact["queries"][key]
        homology_row = homology["queries"][key]
        assert f"{exact_row['exact_overlap_unique_any_known_source']:,} / {exact_row['unique_sequences']:,}" in text
        assert f"{homology_row['detected_by_any_source']:,} / {homology_row['n_queries']:,}" in text
    assert homology["detected_by_any_source"] == 162489 and homology["not_detected_by_any_source"] == 3365
    assert "162,489/165,854" in text and "3,365" in text
    families = read("rfam14_9_exact_sequence_family_mapping_summary.json")
    for dataset in ("rfam_fam", "bprna_new", "bprna_tr1c"):
        report = families["datasets"][dataset]
        assert report["counts"]["PASS_UNIQUE_EXACT_SEQUENCE_FAMILY"] == report["mapped_unique_sequences"]
        assert f"{report['mapped_unique_sequences']:,}" in text
        assert f"{report['n_unique_mapped_families']:,}" in text
    assert len(families["mapped_family_overlap_with_tr1c"]["rfam_fam"]) == 120
    assert len(families["mapped_family_overlap_with_tr1c"]["bprna_new"]) == 0
    assert "120 mapped families" in text and "intersection was zero only for the mapped downstream labels" in text
    identity = read("h2_encoder_identity.json")
    assert identity["encoder_exact_equal"] and identity["warmup_head_steps"] == 1600
    assert "1,600 steps" in text and "H0 alias" in text
    profile = read("full_host_profile.json")
    assert f"{profile['forward_backward_profiler_flops']:,}" in text
    assert "−0.052939" in text and "one batch" in text
    nt = read("nt_snapshot_assessment.json")
    assert nt["status"] == "BLOCKED_HISTORICAL_NT_SNAPSHOT_UNPINNED"
    assert not nt["current_nt"]["downloaded"]
    assert "Current nt was not downloaded as a substitute" in text
    assert "Canonical-nested labels do not model complete three-dimensional structures" in text
    assert "original full-pair ResNet2D ran out of memory at 7,224 nt" in text
    assert "not submission-ready" in text
    assert len(text.split()) > 1800
    hashes = {"exact_overlap.json": "6cd88fc5e3b6385ac15f179c225bf575f79c1cec308c1328ce1af31baf558877",
              "exact_union_summary.json": "d441239311605a2572b3f561357531689b0ea24a19068e1ac417dca75148baa4",
              "homology_union_summary.json": "69a0b13b2ef0cb061807e10f50efca3a6da7889ee006cd032e2e72bca2197131",
              "rfam14_9_exact_sequence_family_mapping_summary.json": "bdef1812946c4c00c36801b1cbddfb4433a3d3c433b8ecf71a8a6256f79c8a99",
              "converged_baseline_summary.json": "2b465cd3816407cb92c886d2b9fe9e3f00d48d81834f34113fab9ccda7f24870",
              "nt_snapshot_assessment.json": "5f4282bdcda743cc14de112a4b6e6c8d5436b9404dff3180553f27a04c336232"}
    for filename, expected in hashes.items():
        assert sha(filename) == expected
    assert "Calibration and pair dependence" in text and "no structure-level posterior calibration" in text
    for number in range(1, 13):
        assert f"**Q{number}**" in text
    assert hashlib.sha256(Path(args.deck).read_bytes()).hexdigest() == "abc4f31ecd043ec407bf8ad6e748eb93362e7f96371e63214717dbf35afd472e"
    print(json.dumps({"status": "PASS_EVIDENCE_MANUSCRIPT_NUMBERS_AND_LIMITS",
                      "manuscript": args.manuscript, "words": len(text.split()),
                      "checked_arms": len(pilot["results"]), "checked_interface_modes": len(labels),
                      "checked_query_sets": 7, "checked_hashes": len(hashes) + 1}, indent=2))


if __name__ == "__main__":
    main()
