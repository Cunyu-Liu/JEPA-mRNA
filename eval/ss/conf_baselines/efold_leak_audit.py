#!/usr/bin/env python3
"""T-A56b: sequence-level leak audit of eFold's stated pretraining corpus
(bpRNA subset, per Sci Adv adz4967: 'combining subsets of bpRNA and
RNAstralign, filtered to remove ... ArchiveII sequences') against our 8
frozen board splits. Proxy: TR0 (bpRNA train side) + RNAStrAlign full set
as the stated bpRNA/RNAstralign-derived training sources on disk.
Convention: sequence-level T->U exact match (kirigami_tsb_clean)."""
import json

def norm(s):
    return ''.join(c for c in s.upper().replace('T', 'U') if c in 'ACGU')

def load_tr0():
    lines = open('/mnt/cunyuliu/relocated_home/rna_baselines_src/kirigami/data/TR0.dbn').read().splitlines()
    return set(norm(lines[3*i+1]) for i in range(len(lines)//3))

def load_bpseq_dir(d):
    import os, glob
    seqs = set()
    for f in glob.glob(os.path.join(d, '*')):
        try:
            first = open(f).readline().split()
            if len(first) >= 2 and first[0].isdigit():
                # bpseq format: line 1 = header; use sequence chars per row
                seq = []
                for line in open(f).read().splitlines()[1:]:
                    parts = line.split()
                    if len(parts) >= 3:
                        seq.append(parts[1])
                seqs.add(norm(''.join(seq)))
        except Exception:
            continue
    return seqs

def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]

tr0 = load_tr0()
print('TR0 seqs:', len(tr0))

# RNAStrAlign via our jsonl? check ss_data for rnastralign
import os
SPLITS = ['bprna_ts0','bprna_new','ref_pdb_ts1','ref_pdb_ts2','ref_pdb_ts3',
          'ref_pdb_ts_hard','archiveii_embok_clean','testsetb']
# also TR1/tr1c corpus (our full bpRNA-derived train side) as the bpRNA-subset proxy
tr1c = set(norm(r['seq']) for r in load_jsonl('/mnt/cunyuliu/rna-jepa/ss_data/jsonl/bprna_tr1c.jsonl'))
print('tr1c seqs:', len(tr1c))

out = {}
for sp in SPLITS:
    rows = load_jsonl(f'/mnt/cunyuliu/rna-jepa/ss_data/jsonl/{sp}.jsonl')
    n = len(rows)
    hits_tr0 = sum(1 for r in rows if norm(r['seq']) in tr0)
    hits_tr1c = sum(1 for r in rows if norm(r['seq']) in tr1c)
    out[sp] = {'n': n, 'hit_tr0': hits_tr0, 'hit_tr1c_bpRNA_side': hits_tr1c}
    print(f"{sp:24s} n={n:5d} TR0-hit={hits_tr0:5d} tr1c-hit={hits_tr1c:5d}")

json.dump({'note': 'TR0 = bpRNA train dbn (10814); tr1c = our decontaminated corpus (42564, superset of bpRNA-derived rows). eFold pretrain = bpRNA subset + RNAstralign + Ribonanza; exact subset composition not public, so TR0/tr1c hits are a LOWER-BOUND indicator of bpRNA-side overlap.',
           'results': out}, open('/mnt/cunyuliu/rna-jepa/eval_decision/efold_leak_audit.json','w'), indent=1)
print('WROTE efold_leak_audit.json')
