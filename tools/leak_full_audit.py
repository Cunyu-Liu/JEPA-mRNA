"""Full leakage re-audit across ALL reference systems x ALL splits.

Reference training corpora (from papers/ledger):
- RiNALMo-ft (Zenodo): bpRNA TR0 fine-tune  -> overlap vs TS0/new/TSB (bpRNA family)
- RNAformer (bprna ckpt): bpRNA TR0         -> same
- RNAformer (inter-fam ckpt): inter-family corpus (ref_tr_inter-like) -> PDB family
- UFold (ufold_train_alldata): bpRNA TR0 + ArchiveII -> new/TSB/ArchII
- EternaFold: params trained on 26M bpRNA sequences (thermodynamic, flagged)
- Vienna/MXfold2/Nussinov: physics only, no data leakage
- our arms: bprna_tr1c (audited zero-overlap) + tr1c_pdbexp (this round)

Step 1: TR0 overlap counts for every split (the family membership map).
Step 2: RiNALMo-ft per-seq decomposition on all 8 splits (we have per_seq).
"""
import json

J = '/mnt/cunyuliu/rna-jepa/ss_data/jsonl'
E = '/mnt/cunyuliu/rna-jepa/eval_decision'


def seqs_of(path):
    out = set()
    for line in open(path):
        r = json.loads(line)
        out.add(str(r['seq']).upper())
    return out


SPLITS = ['bprna_ts0', 'bprna_new', 'ref_pdb_ts1', 'ref_pdb_ts2',
          'ref_pdb_ts3', 'ref_pdb_ts_hard', 'archiveii_embok_clean', 'testsetb']


def micro(tp, fp, fn):
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return 2 * p * r / max(1e-9, p + r)


print('=== Step 1: training-corpus membership map (overlap n / split n) ===')
corpora = {
    'TR0 (RiNALMo-ft / RNAformer-bprna / UFold-pt1)': 'bprna_tr0',
    'TR1 (leaky corpus, retracted)': 'bprna_tr1',
    'tr1c (ours, audited clean)': 'bprna_tr1c',
}
corp_sets = {k: seqs_of(f'{J}/{v}.jsonl') for k, v in corpora.items()}
for sp in SPLITS:
    s = seqs_of(f'{J}/{sp}.jsonl')
    parts = '  '.join(f'{k.split(" (")[0]}: {len(s & cs)}/{len(s)}' for k, cs in corp_sets.items())
    print(f'{sp:24s} {parts}')

print()
print('=== Step 2: RiNALMo-ft per-seq decomposition, all splits ===')
for sp in SPLITS:
    try:
        d = json.load(open(f'{E}/rinalmo_ft_{sp}.json'))
    except FileNotFoundError:
        print(f'{sp}: NO rinalmo_ft result')
        continue
    per = d['per_seq']
    recs = [json.loads(l) for l in open(f'{J}/{sp}.jsonl')]
    seq_by_name = {r['name']: str(r['seq']).upper() for r in recs}
    tr0 = corp_sets['TR0 (RiNALMo-ft / RNAformer-bprna / UFold-pt1)']
    groups = {'leak': [0, 0, 0], 'clean': [0, 0, 0]}
    for p in per:
        name = p['id']
        s = seq_by_name.get(name)
        if s is None:
            try:
                s = seq_by_name.get(f'{sp}_{name}') or str(recs[int(str(name).split("_")[1].split(".")[0])]['seq']).upper()
            except Exception:
                continue
        g = 'leak' if s in tr0 else 'clean'
        groups[g][0] += p['strict_tp']
        groups[g][1] += p['strict_fp']
        groups[g][2] += p['strict_fn']
    full = micro(sum(p['strict_tp'] for p in per), sum(p['strict_fp'] for p in per), sum(p['strict_fn'] for p in per))
    lk = micro(*groups['leak'])
    cl = micro(*groups['clean'])
    n_leak = sum(1 for p in per if seq_by_name.get(p['id'], '') in tr0)
    print(f'{sp:24s} full={full:.4f}  leak(n={n_leak})={lk:.4f}  clean(n={len(per)-n_leak})={cl:.4f}')
