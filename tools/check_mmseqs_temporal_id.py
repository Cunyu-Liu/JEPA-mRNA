from pathlib import Path
import json
import random
import subprocess

root = Path('.round56_work/temporal_homology_id_smoke_v6')
root.mkdir(exist_ok=False)
sequence = ''.join(random.Random(42).choices('ACGU', k=120))
other = 'U' * 120
query = root / 'q.fa'
target = root / 't.fa'
query.write_text(f'>q0|exact_detected=0\n{sequence}\n>q1|exact_detected=0\n{other}\n')
target.write_text(f'>target\n{sequence}\n')
command = ['/home/rna_pretrain_audit_env/bin/mmseqs', 'easy-search', str(query), str(target),
           str(root / 'matches.tsv'), str(root / 'tmp'), '--search-type', '3', '--min-seq-id', '0.8',
           '-c', '0.8', '--cov-mode', '0', '--max-seqs', '300', '--threads', '1', '--format-output',
           'query,target,pident,alnlen,qlen,tlen,qstart,qend,tstart,tend,evalue,bits']
completed = subprocess.run(command, capture_output=True, text=True)
(root / 'mmseqs.log').write_text(completed.stdout + completed.stderr)
assert completed.returncode == 0, completed.stdout + completed.stderr
rows = [line.split('\t') for line in (root / 'matches.tsv').read_text().splitlines()]
print(rows)
assert len(rows) == 1 and rows[0][0] == 'q0|exact_detected=0', rows
(root / 'qa.json').write_text(json.dumps({'status': 'PASS_ACTUAL_MMSEQS_ID_SMOKE', 'query_id': rows[0][0],
                                           'n_hits': len(rows), 'scope': 'identifier parsing only; not a scientific search result',
                                           'prior_empty_hit_directories': ['temporal_homology_id_smoke', 'temporal_homology_id_smoke_v3']}) + '\n')
