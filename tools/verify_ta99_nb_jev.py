#!/usr/bin/env python3
"""Round-25 T-A99: verify nb_jev (NucleicBERT + our r2d decision head) results.

Lays out, side by side, for all 8 splits:
  - NB-probe baseline  : baselines_nucleicbert_<split>.json  (frozen encoder +
                         official SecStruct head; eval_decision_ref/)
  - NB-JEV (T-A99)    : eval_decision/nb_jev_r2dtr1c_s0_step20000/<split>/result.json
                         pair_level.micro.f1  (same board metric as verify_board_r25)

Pre-registered verdict rule (spec round 25, unchanged since before training):
  gain vs NB-probe on >=3/8 splits with delta >= +0.02  ->  architecture gain
  This script only ARITHMETICALLY checks and REPORTS; the verdict is written
  by the human/ledger pass, never auto-declared here.

Refuses to run on missing artifacts (no partial/interim numbers as results).
"""
import json, os, sys

REF_DIR = '/root/rna_jepa_data/eval_decision_ref'
JEV_DIR = '/teams/HoulabData_1788502323/liucunyu/rna_jev_data/eval_decision/nb_jev_r2dtr1c_s0_step20000'
SPLITS = ['bprna_ts0', 'bprna_new', 'ref_pdb_ts1', 'ref_pdb_ts2', 'ref_pdb_ts3',
          'ref_pdb_ts_hard', 'archiveii_embok_clean', 'testsetb']
TOL = 5e-4  # only for the board-ref re-read self-check


def main():
    # --- arm A: NB-probe reference, re-read + self-check against frozen board copy
    board_probe = {  # from baselines_nucleicbert_*.json, verified 2026-10-22 03:50
        'bprna_ts0': 0.15092876097162686, 'bprna_new': 0.09554979046098583,
        'ref_pdb_ts1': 0.20287914295279544, 'ref_pdb_ts2': 0.15408560311284047,
        'ref_pdb_ts3': 0.14389989572471323, 'ref_pdb_ts_hard': 0.13430127041742287,
        'archiveii_embok_clean': 0.23719343607598178, 'testsetb': 0.12097459011039258,
    }
    probe = {}
    for s in SPLITS:
        f = f'{REF_DIR}/baselines_nucleicbert_{s}.json'
        if not os.path.exists(f):
            print(f'MISSING probe ref: {f}'); return 1
        d = json.load(open(f))
        v = d['baselines']['nucleicbert']['micro_f1']
        if abs(v - board_probe[s]) > TOL:
            print(f'DRIFT probe {s}: file={v} board={board_probe[s]}'); return 1
        probe[s] = v

    # --- arm B: T-A99 results (all 8 must exist; refuse partial)
    jev = {}
    for s in SPLITS:
        f = f'{JEV_DIR}/{s}/result.json'
        if not os.path.exists(f):
            print(f'NOT-READY: {f} (T-A99 stage-3 not finished for this split)'); return 2
        d = json.load(open(f))
        jev[s] = d['pair_level']['micro']['f1']
        # provenance: confirm the checkpoint actually is the nb_jev arm
        if 'nb_jev' not in d.get('checkpoint', ''):
            print(f'SUSPECT checkpoint for {s}: {d.get("checkpoint")}'); return 1

    # --- report
    print(f'{"split":<22}{"NB-probe":>10}{"NB-JEV":>10}{"delta":>9}')
    wins = 0
    for s in SPLITS:
        dl = jev[s] - probe[s]
        if dl >= 0.02:
            wins += 1
        print(f'{s:<22}{probe[s]:>10.4f}{jev[s]:>10.4f}{dl:>+9.4f}')
    print(f'\nsplits with delta >= +0.02: {wins}/8   (pre-registered rule: >=3/8)')
    print('NOTE: this is the arithmetic read-out only; the scientific verdict is'
          ' written by the ledger/PPT pass after this check passes.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
