import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request

API = "https://search.rcsb.org/rcsbsearch/v2/query"


def query(cutoff, return_type, method=None):
    nodes = [
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "rcsb_accession_info.initial_release_date",
            "operator": "greater_or_equal", "value": cutoff + "T00:00:00Z"}},
        {"type": "terminal", "service": "text", "parameters": {
            "attribute": "entity_poly.rcsb_entity_polymer_type",
            "operator": "exact_match", "value": "RNA"}},
    ]
    if method is not None:
        nodes.append({"type": "terminal", "service": "text", "parameters": {
            "attribute": "exptl.method", "operator": "exact_match", "value": method}})
    payload = {
        "query": {"type": "group", "logical_operator": "and", "nodes": nodes},
        "return_type": return_type,
        "request_options": {"paginate": {"start": 0, "rows": 25},
                            "results_content_type": ["experimental"]},
        "request_info": {"src": "ui"},
    }
    request = urllib.request.Request(API, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"RCSB rejected query: {error.code}: {error.read().decode(errors='replace')}") from error
    result = json.loads(raw)
    assert "total_count" in result and isinstance(result["total_count"], int)
    assert len(result.get("result_set", [])) <= 25
    return {"request": payload, "response": result, "response_sha256": hashlib.sha256(raw).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default="2025-07-01")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    datetime.date.fromisoformat(args.cutoff)
    entry = query(args.cutoff, "entry")
    entity = query(args.cutoff, "polymer_entity")
    methods = {method: query(args.cutoff, "polymer_entity", method)["response"]["total_count"]
               for method in ["X-RAY DIFFRACTION", "ELECTRON MICROSCOPY", "SOLUTION NMR"]}
    report = {
        "status": "TEMPORAL_ASSET_AVAILABILITY_ONLY_QA_NOT_PASSED",
        "source": API,
        "cutoff_initial_release_date": args.cutoff,
        "retrieved_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "candidate_rna_entries": entry["response"]["total_count"],
        "candidate_rna_entities": entity["response"]["total_count"],
        "candidate_rna_entities_by_method": methods,
        "evidence": {"entry": entry, "entity": entity},
        "scope": [
            "Counts are pre-QA candidates, not a dataset or an independent test set",
            "A release after 2025-07-01 is a conservative temporal filter, not proof of the unknown historical pretraining cutoff",
            "No model performance, labels or outcomes were used to select the cutoff or candidates",
            "Entity/polymer sequence mapping, coordinates, modified residues, missing atoms, pairing provenance, duplicates and homology remain unverified",
            "No training, model selection, threshold choice or external accuracy evaluation has been started",
        ],
        "required_next_gates": [
            "freeze full metadata and content hashes without replacing any existing structure asset",
            "derive polymer/canonical-nested labels with an independent sequence-coordinate and pairing QA",
            "remove exact and approximate sequence redundancy against downstream training and known pretraining sources",
            "record experimental method, resolution where applicable, missing residues and exclusion reasons",
            "freeze dataset and evaluation protocol before any model inference",
        ],
    }
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refuse to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "evidence"}, ensure_ascii=False, indent=2))
    print("artifact_sha256", hashlib.sha256(output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
