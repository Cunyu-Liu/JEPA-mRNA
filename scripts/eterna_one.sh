#!/bin/bash
# one contrafold prediction: $1 = per-sequence .fa path
f="$1"
b="${f%.fa}"
cd /mnt/cunyuliu/rna_baselines_src/EternaFold || exit 1
./src/contrafold predict "$f" --params parameters/EternaFoldParams.v1 > "$b.dbn" 2>&1
