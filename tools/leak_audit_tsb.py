"""Leakage decomposition of RiNALMo-ft TestSetB 0.8711.

The Zenodo ckpt was fine-tuned on bpRNA TR0; our audit (15.x) found
TR0-cap-TestSetB = 247 sequences. Split the per_seq results into
leaked (seq in TR0) vs clean (not in TR0) and compute strict micro F1
for each subset. Our own numbers get the same decomposition for a fair
same-subset comparison.
"""
import json

E = '/mnt/cunyuliu/rna-jepa/eval_decision'
J = '/mnt/cunyuliu/rna-jepa/ss_data/jsonl'


def seqs_of(path):
    out = set()
    for line in open(path):
        r = json.loads(line)
        out.add(str(r['seq']).upper())
    return out


def micro(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return 2 * p * r / max(1e-9, p + r)


tr0 = seqs_of(f'{J}/bprna_tr0.jsonl')
tsb = [json.loads(l) for l in open(f'{J}/testsetb.jsonl')]
leaked = set()
for i, r in enumerate(tsb):
    if str(r['seq']).upper() in tr0:
        leaked.add(i)
print(f'TestSetB n={len(tsb)}, leaked-in-TR0={len(leaked)}, clean={len(tsb)-len(leaked)}')

ft = json.load(open(f'{E}/rinalmo_ft_testsetb.json'))
per = ft['per_seq']
print(f"ft per_seq n={len(per)}, ids sample: {[p['id'] for p in per[:3]]}")

# map per_seq id back to index: per_seq ids come from r.get('id') or r.get('name')
# testsetb.jsonl records have 'name' like 'testsetb_0.bpseq'
idx_of_name = {r['name']: i for i, r in enumerate(tsb)}
groups = {'leak': [0, 0, 0], 'clean': [0, 0, 0]}
n_unmatched = 0
for p in per:
    pid = p['id']
    if pid in idx_of_name:
        i = idx_of_name[pid]
    else:
        try:
            i = int(str(pid).split('_')[1].split('.')[0])
        except Exception:
            n_unmatched += 1
            continue
    g = 'leak' if i in leaked else 'clean'
    groups[g][0] += p['strict_tp']
    groups[g][1] += p['strict_fp']
    groups[g][2] += p['strict_fn']
print(f'unmatched ids: {n_unmatched}')
for g, (tp, fp, fn) in groups.items():
    print(f'RiNALMo-ft {g}: n_pairs tp={tp} fp={fp} fn={fn} strict_micro={micro(tp, fp, fn):.4f}')

# whole-set check
tp = sum(p['strict_tp'] for p in per)
fp = sum(p['strict_fp'] for p in per)
fn = sum(p['strict_fn'] for p in per)
print(f'RiNALMo-ft all: strict_micro={micro(tp, fp, fn):.4f} (should be ~0.8711)')
