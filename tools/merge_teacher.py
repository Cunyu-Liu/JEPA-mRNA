"""Build a merged teacher directory for bprna_tr1c_plus (tr1c 167 shards +
pdbexp shards) by symlinking both manifests' shards into one directory with
a fresh manifest — same merge contract as parallel_teacher_labels.merge_parts.
"""
import json
import os

TR1C = '/mnt/cunyuliu/rna-jepa/ss_data/teacher/bprna_tr1c'
PDB = '/mnt/cunyuliu/rna-jepa/ss_data/teacher/bprna_tr1c_pdbexp'
OUT = '/mnt/cunyuliu/rna-jepa/ss_data/teacher/bprna_tr1c_plus'

os.makedirs(OUT, exist_ok=True)
merged = []
meta = {}
index = 0
for src_dir in (TR1C, PDB):
    m = json.load(open(f'{src_dir}/manifest.json'))
    meta.update({k: v for k, v in m.items() if k not in ('shards', 'n_sequences') and (k not in meta or meta[k] is None)})
    for shard in m['shards']:
        dst_name = f'shard_{index:05d}.npz'
        dst = os.path.join(OUT, dst_name)
        if os.path.islink(dst) or os.path.exists(dst):
            os.remove(dst)
        os.symlink(os.path.join(src_dir, str(shard['file'])), dst)
        s2 = dict(shard)
        s2['index'] = index
        s2['file'] = dst_name
        merged.append(s2)
        index += 1

n_seq = sum(s.get('n_sequences', 0) for s in merged)
manifest = {
    'n_sequences': n_seq,
    'n_shards': len(merged),
    'merged_from': ['bprna_tr1c', 'bprna_tr1c_pdbexp'],
    'shards': merged,
}
manifest.update({k: v for k, v in meta.items() if k in ('teacher', 'teacher_version_lock', 'version_lock', 'tool', 'params')})
with open(f'{OUT}/manifest.json', 'w') as fh:
    json.dump(manifest, fh, indent=1)
print(f'merged teacher dir: {len(merged)} shards, {n_seq} sequences -> {OUT}')
