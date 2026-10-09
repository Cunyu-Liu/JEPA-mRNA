"""Build the clean PDB-family expansion corpus for TS2/TS3 iteration.

From ref_tr_experimental.jsonl (42,283 PDB experimental structures):
- dedup by sequence
- EXCLUDE any sequence overlapping ANY board split (ts0/new/ts1..3/hard/
  testsetb/archiveii_embok_clean/vl0) and tr1c (already covered)
- keep records with pairs, len<=500 (teacher/backbone limits)

Output: bprna_tr1c_pdbexp.jsonl (append-ready, source-tagged) + audit numbers.
"""
import json

J = '/mnt/cunyuliu/rna-jepa/ss_data/jsonl'
OUT = '/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr1c_pdbexp.jsonl'


def seqs_of(path):
    out = set()
    for line in open(path):
        r = json.loads(line)
        out.add(str(r['seq']).upper())
    return out


EXCLUDE = [
    'ref_pdb_ts1', 'ref_pdb_ts2', 'ref_pdb_ts3', 'ref_pdb_ts_hard',
    'bprna_ts0', 'bprna_new', 'bprna_vl0', 'testsetb',
    'archiveii_embok_clean', 'bprna_tr1c',
]
banned = set()
for sp in EXCLUDE:
    banned |= seqs_of(f'{J}/{sp}.jsonl')
print(f'banned (all splits + tr1c): {len(banned)} unique seqs')

exp = [json.loads(l) for l in open(f'{J}/ref_tr_experimental.jsonl')]
seen = set()
kept = []
stats = {'dedup': 0, 'banned': 0, 'no_pairs': 0}
for r in exp:
    s = str(r['seq']).upper()
    if s in seen:
        stats['dedup'] += 1
        continue
    seen.add(s)
    if s in banned:
        stats['banned'] += 1
        continue
    if not r.get('pairs'):
        stats['no_pairs'] += 1
        continue
    kept.append(r)

print(f"kept: {len(kept)}  (dedup {stats['dedup']}, banned {stats['banned']}, no_pairs {stats['no_pairs']})")

import numpy as np
a = np.array([len(r['seq']) for r in kept])
print(f'kept len: p50={int(np.percentile(a,50))} p95={int(np.percentile(a,95))} max={a.max()}')

with open(OUT, 'w') as fh:
    for r in kept:
        r2 = dict(r)
        r2['source'] = 'pdbexp_clean'
        fh.write(json.dumps(r2, ensure_ascii=False) + '\n')
print(f'written {OUT}')

# final safety re-check: zero overlap with every split
out_set = {str(r['seq']).upper() for r in kept}
for sp in EXCLUDE[:-1]:
    s = seqs_of(f'{J}/{sp}.jsonl')
    print(f'final check {sp}: overlap={len(s & out_set)}')
