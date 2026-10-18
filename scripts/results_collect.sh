#!/bin/bash
# Final collector: after speed v3 completes, dump every result JSON in a
# compact form for the ledger + mark the whole chain done.
set -u
PY=/root/miniconda3/envs/lucaone/bin/python
D=/root/rna_jepa_data
EVAL_DIR=$D/eval_decision/ow_rinalmo_r2dtr1c_b4_s1_step20000

while ! grep -q SPEED_V3_DONE $D/eval_decision/speed_v3.log 2>/dev/null; do sleep 300; done
{
echo "=== 8-split auto-eval (s1) ==="
for sp in bprna_ts0 bprna_new ref_pdb_ts1 ref_pdb_ts2 ref_pdb_ts3 ref_pdb_ts_hard archiveii_embok_clean testsetb; do
  f=$EVAL_DIR/$sp/result.json
  [ -f "$f" ] && $PY -c "
import json; d = json.load(open('$f'))
m = d.get('metrics', d)
mi = m.get('micro', m)
print('$sp', 'f1=%.4f P=%.4f R=%.4f' % (mi.get('f1',0), mi.get('precision',0), mi.get('recall',0)))" 2>/dev/null
done
echo "=== xens2 repro + TS2 resweep ==="
for f in $EVAL_DIR/xens2_board_ts0_repro.json $EVAL_DIR/xens2_ts2_w*_r2ds1.json; do
  [ -f "$f" ] && $PY -c "
import json; d = json.load(open('$f'))
print('$(basename $f)', 'f1=%.4f' % d.get('micro_f1', d.get('f1', 0)))" 2>/dev/null
done
echo "=== bias sweeps ==="
for f in $EVAL_DIR/bias_gate_vl0.json $EVAL_DIR/bias_sweep_ts2.json; do
  [ -f "$f" ] && echo "--- $(basename $f)" && cat "$f" | head -40
done
echo "=== speed v3 ==="
cat $D/eval_decision/speed_v3_4090_*_b64.json 2>/dev/null
} > /root/rna_jepa_data/chain_results_summary.txt 2>&1
echo CHAIN_ALL_DONE >> /root/rna_jepa_data/chain_results_summary.txt
