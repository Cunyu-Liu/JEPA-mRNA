#!/usr/bin/env python3
"""Round-25 task 1: re-verify every baseline board cell against result.json (pass 1 of 3).

Reads board values from tables/sota_vs_ours.md (hard-coded copy below, will be
diff-checked in pass 2 via grep) and recomputes micro F1 from the stored
result.json artifacts on the new cluster.
"""
import json, glob, sys

E = '/root/rna_jepa_data/eval_decision_ref'

# board values as printed in tables/sota_vs_ours.md (Published baselines section)
BOARD = {
 'vienna_mfe':         {'bprna_ts0':0.5056,'bprna_new':0.6379,'ref_pdb_ts1':0.7209,'ref_pdb_ts2':0.8980,'ref_pdb_ts3':0.8206,'ref_pdb_ts_hard':0.7663,'archiveii_embok_clean':0.5752,'testsetb':0.5225},
 'vienna_centroid':    {'bprna_ts0':0.5393,'bprna_new':0.6770,'ref_pdb_ts1':0.7299,'ref_pdb_ts2':0.8930,'ref_pdb_ts3':0.8235,'ref_pdb_ts_hard':0.7958,'archiveii_embok_clean':0.6181,'testsetb':0.5482},
 'vienna_mea':         {'bprna_ts0':0.5241,'bprna_new':0.6620,'ref_pdb_ts1':0.7192,'ref_pdb_ts2':0.8938,'ref_pdb_ts3':0.8237,'ref_pdb_ts_hard':0.7864,'archiveii_embok_clean':0.6054,'testsetb':0.5393},
 'nussinov_turner':    {'bprna_ts0':0.2126,'bprna_new':0.3015,'ref_pdb_ts1':0.3730,'ref_pdb_ts2':0.4187,'ref_pdb_ts3':0.4131,'ref_pdb_ts_hard':0.4067,'archiveii_embok_clean':0.2000,'testsetb':0.2164},
 'MXfold2':            {'bprna_ts0':0.5651,'bprna_new':0.6688,'ref_pdb_ts1':0.7967,'ref_pdb_ts2':0.8991,'ref_pdb_ts3':0.8127,'ref_pdb_ts_hard':0.8028,'archiveii_embok_clean':0.7400,'testsetb':0.6504},
 'EternaFold':         {'bprna_ts0':0.5400,'bprna_new':0.6854,'ref_pdb_ts1':0.7426,'ref_pdb_ts2':0.8635,'ref_pdb_ts3':0.8005,'ref_pdb_ts_hard':0.7658,'archiveii_embok_clean':0.6157,'testsetb':0.5850},
 'UFold':              {'bprna_ts0':0.6598,'bprna_new':0.6106,'ref_pdb_ts1':0.6455,'ref_pdb_ts2':0.8792,'ref_pdb_ts3':0.6326,'ref_pdb_ts_hard':0.6388,'archiveii_embok_clean':0.4111,'testsetb':0.5858},
 'RiNALMo-ft':         {'bprna_ts0':0.7214,'bprna_new':0.4489,'ref_pdb_ts1':0.7139,'ref_pdb_ts2':0.6702,'ref_pdb_ts3':0.6055,'ref_pdb_ts_hard':0.5556,'archiveii_embok_clean':0.7613,'testsetb':0.8711},
 'eFold':              {'bprna_ts0':0.5612,'bprna_new':0.6640,'ref_pdb_ts1':0.7682,'ref_pdb_ts2':0.8862,'ref_pdb_ts3':0.7354,'ref_pdb_ts_hard':0.7632,'archiveii_embok_clean':0.6618,'testsetb':0.5706},
}
NAME = {'vienna_mfe':'vienna_mfe','vienna_centroid':'vienna_centroid','vienna_mea':'vienna_mea','nussinov_turner':'nussinov_turner','MXfold2':'mxfold2','EternaFold':'eternafold','UFold':'ufold','RiNALMo-ft':'rinalmo_ft','eFold':'efold'}


def find_f1(d):
    if isinstance(d, dict):
        for k in ('micro_f1', 'micro_F1'):
            if k in d and isinstance(d[k], (int, float)):
                return d[k]
        for v in d.values():
            r = find_f1(v)
            if r is not None:
                return r
    return None


def main():
    mismatch, checked, missing = 0, 0, []
    for sysname, splits in BOARD.items():
        for sp, val in splits.items():
            pats = [
                f'{E}/baselines_{NAME[sysname]}_{sp}.json',
                f'{E}/{NAME[sysname]}_{sp}.json',
                f'{E}/{NAME[sysname]}_{sp}_*.json',
                f'{E}/{NAME[sysname]}*{sp}*.json',
            ]
            cands = []
            for p in pats:
                cands += sorted(glob.glob(p))
            if not cands:
                missing.append((sysname, sp))
                continue
            got = None
            for c in cands:
                try:
                    got = find_f1(json.load(open(c)))
                    if got is not None:
                        break
                except Exception:
                    continue
            if got is None:
                print(f'NOF1 {sysname} {sp}')
                continue
            checked += 1
            if abs(got - val) > 5e-4:
                mismatch += 1
                print(f'MISMATCH {sysname} {sp}: board={val} json={round(got, 6)}')
            else:
                print(f'ok {sysname:16s} {sp:22s} board={val} json={round(got, 4)}')
    print(f'\n=== checked={checked} mismatch={mismatch} missing={missing}')
    sys.exit(1 if (mismatch or missing) else 0)


if __name__ == '__main__':
    main()
