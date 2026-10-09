"""Clean-subset head-to-head on TestSetB: ours (xens3/xens2/plana2same)
vs RiNALMo-ft, both decomposed into TR0-leaked (247) vs clean (181)."""
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
name2is_leak = {}
for i, r in enumerate(tsb):
    name2is_leak[r['name']] = str(r['seq']).upper() in tr0


def decomp_ours(result_path, label):
    d = json.load(open(result_path))
    groups = {'leak': [0, 0, 0], 'clean': [0, 0, 0]}
    for p in d['per_sequence']:
        g = 'leak' if name2is_leak.get(p['name'], False) else 'clean'
        ps = {(i, j) for i, j in p['pred_pairs'] if i < j}
        gs = set()
        # find gt pairs by name
        r = next(x for x in tsb if x['name'] == p['name'])
        gs = {(i, j) for i, j in r['pairs'] if i < j}
        a = len(ps & gs)
        groups[g][0] += a
        groups[g][1] += len(ps) - a
        groups[g][2] += len(gs) - a
    print(f'{label}: all={d["pair_level"]["micro"]["f1"]:.4f}', end='  ')
    for g in ('leak', 'clean'):
        tp, fp, fn = groups[g]
        print(f'{g}={micro(tp, fp, fn):.4f}(tp{tp}/fp{fp}/fn{fn})', end='  ')
    print()


ft = json.load(open(f'{E}/rinalmo_ft_testsetb.json'))
groups = {'leak': [0, 0, 0], 'clean': [0, 0, 0]}
idx_of_name = {r['name']: i for i, r in enumerate(tsb)}
for p in ft['per_seq']:
    g = 'leak' if name2is_leak.get(p['id'], False) else 'clean'
    groups[g][0] += p['strict_tp']
    groups[g][1] += p['strict_fp']
    groups[g][2] += p['strict_fn']
print('RiNALMo-ft (Zenodo, trained on TR0):', end='  ')
for g in ('leak', 'clean'):
    tp, fp, fn = groups[g]
    print(f'{g}={micro(tp, fp, fn):.4f}', end='  ')
print()

for path, label in [
    (f'{E}/xens3_testsetb_w0.7/result.json', 'xens3 (ours)'),
    (f'{E}/xens2_testsetb_w0.7/result.json', 'xens2 (ours)'),
    (f'{E}/plana2same_testsetb/result.json', 'plana2same (ours)'),
]:
    decomp_ours(path, label)
