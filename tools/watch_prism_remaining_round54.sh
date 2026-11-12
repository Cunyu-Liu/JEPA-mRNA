#!/usr/bin/env bash
set -euo pipefail
ROOT=/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024
PY=/root/miniconda3/envs/lucaone/bin/python
while true; do
  "$PY" - <<'PY'
import json
from datetime import datetime
from pathlib import Path
root=Path('/teams/HoulabData_1788502323/liucunyu/rna_ssp_20261024')
print('\nSTATUS',datetime.now().isoformat(),flush=True)
for arm in ('cl2','cl3','cl4','cl5','e2a_motif','e2a_shuffled','e2b','e2c'):
    states=[]
    for seed in (42,2026,3407):
        p=root/'runs'/f'round54_{arm}_s{seed}/run_meta.json'
        if p.exists():
            d=json.loads(p.read_text())
            states.append(f'{seed}:{d.get("status")}@{d.get("best_step")}/{d.get("best_validation_loss")}')
    if states: print(arm,' | '.join(states),flush=True)
p=root/'logs/round54/prism_remaining.log'
if p.exists(): print('PIPELINE',p.read_text().splitlines()[-1],flush=True)
PY
  sleep 60
done
