#!/usr/bin/env bash
set -euo pipefail
cd /home/rna_jev
export PYTHONPATH=/home/rna_jev:/home/rna_jev/src:/home/rna_jev/tools:/home/rna_jev/eval
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python=/root/miniconda3/envs/lucaone/bin/python
root=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024/eval_decision/round39_clean_benchmark
data=$root/clean_splits/tr1c/archiveii_embok_clean.jsonl
audit=$root/clean_splits/tr1c/archiveii_embok_clean.audit.json
out=$root/xens2/clean_tr1c/archiveii_embok_clean.json
short=$root/xens2/diagnostics/archiveii_embok_clean_len600.json
"$python" - "$data" "$audit" <<'PY'
import hashlib,json,sys
from pathlib import Path
data,audit=map(Path,sys.argv[1:])
a=json.loads(audit.read_text())
h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=[json.loads(line) for line in data.read_text().splitlines() if line.strip()]
assert a["status"]=="PASS" and a["clean_n"]==len(rows)==1107
assert a["clean_sha256"]==h(data)
assert a["raw_n"]==len(rows)+a["removed_exact"]+a["removed_homology_only"]
print("XENS2_FULL_ARCHIVE_DATA_PASS",len(rows),max(len(row["seq"]) for row in rows),flush=True)
PY
if [[ -s "$out" ]] && "$python" - "$out" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));p=x["protocol"]
assert x["status"]=="PASS" and len(x["runs"])==3
assert p["n_sequences"]==1107 and p["n_source_records"]==1107 and p["n_length_filtered"]==0
PY
then
    echo XENS2_FULL_ARCHIVE_ALREADY_COMPLETE
    exit 0
fi
"$python" - "$out" "$short" <<'PY'
import json,shutil,sys
from pathlib import Path
out,short=map(Path,sys.argv[1:])
x=json.loads(out.read_text());p=x["protocol"]
assert x["status"]=="PASS" and len(x["runs"])==3
assert p["n_sequences"]==1068 and p["n_source_records"]==1107 and p["n_length_filtered"]==39
short.parent.mkdir(parents=True,exist_ok=True)
if short.exists():
    assert short.read_bytes()==out.read_bytes()
else:
    shutil.copy2(out,short)
print("XENS2_LEN600_RESULT_PRESERVED",short,flush=True)
PY
"$python" tools/speed_end_to_end_4090.py --model xens2 \
    --batch-size 1 --repeats 3 --resume-runs --allow-long \
    --data "$data" --clean-audit "$audit" --out "$out" \
    > logs/round39_clean_benchmark/xens2_full_archiveii.log 2>&1
"$python" - "$out" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]));p=x["protocol"]
assert x["status"]=="PASS" and len(x["runs"])==3
assert p["device_type"]=="cuda" and p["cached_embeddings"] is False
assert p["n_sequences"]==p["n_source_records"]==1107 and p["n_length_filtered"]==0
assert len(x["predictions"])==1107 and len(set(x["f1_runs"]))==1
print("XENS2_FULL_ARCHIVE_3_REPEAT_PASS",x["f1_runs"],flush=True)
PY
