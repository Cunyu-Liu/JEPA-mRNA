#!/usr/bin/env python3
"""Fill the two blank vienna_mfe cells of the draft's 4.3 source x length grid.

The grid left RFAM 200-400 nt and RFAM >400 nt blank for vienna_mfe
('not run ... left blank rather than estimated'). The length-bucket runs exist
(baselines_bprna_ts0_gt200_le400.json / _gt400.json) but hold only pooled
metrics, so the source crossing is recomputed here from scratch with the same
predictor and pooling code as run_baselines.py.

Built-in verification: vienna_centroid is recomputed on the same cells and
must reproduce the draft's published values (0.4345 and 0.4010) to 4 dp -
if it does not, the mfe numbers are NOT protocol-consistent and must not be
used. Sequence counts must match the grid's n (126 and 15).
"""
import json
import sys

sys.path.insert(0, '/mnt/cunyuliu/pylibs')
sys.path.insert(0, '/home/cunyuliu/rna-jepa/src')
sys.path.insert(0, '/home/cunyuliu/rna-jepa/eval')

import RNA  # noqa: E402  (ViennaRNA bindings, /mnt/cunyuliu/pylibs)
from rnajepa.clean.c3_structure import parse_pairs  # noqa: E402
from rnajepa.decision_head import valid_pair_mask  # noqa: E402
from ss.metrics import PairLevelMetrics  # noqa: E402


def vienna(seq, kind):
    if kind == 'vienna_mfe':
        structure, _ = RNA.fold(seq)
    else:
        fc = RNA.fold_compound(seq)
        fc.pf()
        structure, _ = fc.centroid()
    return sorted(parse_pairs(structure))


def pooled_f1(records, kind):
    tp = fp = fn = 0
    for seq, gt in records:
        mask = valid_pair_mask(seq)
        pred = vienna(seq, kind)
        m = PairLevelMetrics.from_pairs(pred, gt, L=len(seq), mask=mask)
        tp += m.tp; fp += m.fp; fn += m.fn
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * p * r / (p + r) if (p + r) else 0.0), p, r


rows = [json.loads(l) for l in open('/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_ts0.jsonl') if l.strip()]

cells = {
    'RFAM_200_400': (201, 400, 126, 0.4345),
    'RFAM_gt400': (401, 10**9, 15, 0.4010),
}
out = {}
for name, (lo, hi, n_expect, centroid_expect) in cells.items():
    recs = []
    for row in rows:
        toks = row['name'].split('_')
        src = toks[1] if len(toks) > 2 and toks[0] == 'bpRNA' else ''
        if src != 'RFAM':
            continue
        L = len(row['seq'])
        if lo <= L <= hi:
            pairs = row["pairs"] if isinstance(row["pairs"], list) else json.loads(row["pairs"]); recs.append((row["seq"], [tuple(p) for p in pairs]))
    n_ok = len(recs) == n_expect
    f1_c, p_c, r_c = pooled_f1(recs, 'vienna_centroid')
    f1_m, p_m, r_m = pooled_f1(recs, 'vienna_mfe')
    centroid_ok = abs(f1_c - centroid_expect) < 5e-5
    out[name] = {'n': len(recs), 'n_expected': n_expect, 'n_ok': n_ok,
                 'centroid_f1': round(f1_c, 4), 'centroid_expected': centroid_expect,
                 'centroid_reproduced': centroid_ok,
                 'mfe_f1': round(f1_m, 4), 'mfe_precision': round(p_m, 4),
                 'mfe_recall': round(r_m, 4),
                 'TRUSTWORTHY': bool(n_ok and centroid_ok)}
    print(name, '->', out[name])

with open('/mnt/cunyuliu/rna-jepa/eval_decision/stratified_mfe_cells.json', 'w') as fh:
    json.dump({'ts': out, 'vienna_version': f'ViennaRNA {RNA.__version__}',
               'note': 'fills the two blank vienna_mfe cells of draft 4.3 grid; centroid recomputed on identical cells as the protocol check'}, fh, indent=1)
print('saved eval_decision/stratified_mfe_cells.json')
