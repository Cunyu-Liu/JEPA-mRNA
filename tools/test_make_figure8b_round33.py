#!/usr/bin/env python3
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

SCRIPT = Path(os.environ.get('FIGURE8B_SCRIPT', '/private/tmp/rnajev_audit/round33/make_figure8b_round33.py'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mrl_payload():
    heads = {}
    official = np.array([0.40, 0.41, 0.42, 0.43, 0.44])
    jev = official + np.array([0.01, 0.02, 0.015, 0.025, 0.02])
    for head, values in [('official_head', official), ('jev_head', jev)]:
        runs = [{'seed': seed, 'device': 'cuda', 'converged': True, 'early_stopped': True,
                 'independent_recheck': 'PASS', 'test_r2': float(value)} for seed, value in enumerate(values)]
        heads[head] = {'runs': runs, 'r2_mean': float(values.mean()), 'r2_sample_std': float(values.std(ddof=1))}
    test = stats.ttest_rel(jev, official, alternative='greater')
    comparison = {'primary_metric': 'r2', 'primary_delta_mean': float((jev - official).mean()),
                  'primary_one_sided_p_value': float(test.pvalue), 'significant_positive': bool(test.pvalue < 0.05)}
    return {'status': 'PASS', 'protocol': {'task': 'mrl', 'device': 'cuda',
            'sample_scope': 'full_official_protocol', 'seeds': list(range(5))},
            'heads': heads, 'comparison': comparison}


def cv_payload(task, cell, zero_variance=False):
    fold_results = []
    for fold in range(10):
        heads = {}
        for head, base in [('official_head', 0.4), ('jev_head', 0.4 if zero_variance else 0.42)]:
            runs = []
            for seed in range(5):
                delta = 0.0 if zero_variance else seed * 0.002
                runs.append({'seed': seed, 'spearman': base + fold * 0.005 + seed * 0.003 + delta})
            heads[head] = {'runs': runs, 'spearman_seed_mean': float(np.mean([r['spearman'] for r in runs]))}
        fold_results.append({'fold': fold, 'heads': heads})
    output_heads = {}
    for head in ['official_head', 'jev_head']:
        seed_cv = [float(np.mean([fold['heads'][head]['runs'][seed]['spearman'] for fold in fold_results]))
                   for seed in range(5)]
        fold_means = [fold['heads'][head]['spearman_seed_mean'] for fold in fold_results]
        output_heads[head] = {'seed_cv_spearman': seed_cv,
                              'cv_spearman_mean': float(np.mean(fold_means)),
                              'cv_spearman_fold_sd': float(np.std(fold_means, ddof=1))}
    official = output_heads['official_head']['seed_cv_spearman']
    jev = output_heads['jev_head']['seed_cv_spearman']
    delta = np.array(jev) - np.array(official)
    test = stats.ttest_rel(jev, official, alternative='greater')
    p = float(test.pvalue) if np.isfinite(test.pvalue) else None
    return {'status': 'PASS', 'scientific_eligibility': True,
            'protocol': {'task': task, 'cell_line': cell, 'folds': list(range(10)),
                         'n_fold_results': 10, 'n_runs_per_head': 50},
            'heads': output_heads,
            'comparison': {'seed_cv_delta_jev_minus_official': delta.tolist(),
                           'one_sided_ttest_rel_greater_p_value': p,
                           'significant_positive': bool(p is not None and p < 0.05 and delta.mean() > 0)},
            'fold_results': fold_results}


with tempfile.TemporaryDirectory(prefix='rnajev_fig8b_test_') as temporary:
    root = Path(temporary)
    mrl = root / 'mrl.json'
    mrl.write_text(json.dumps(mrl_payload()))
    env = os.environ.copy()
    env['MRL_RESULT'] = str(mrl)
    for task in ['te', 'el']:
        for cell in ['Muscle', 'pc3', 'HEK']:
            path = root / f'{task}_{cell}.json'
            path.write_text(json.dumps(cv_payload(task, cell, zero_variance=(task == 'te' and cell == 'HEK'))))
            env[f'{task.upper()}_{cell.upper()}_RESULT'] = str(path)
    out = root / 'out'
    env['FIGURE_OUT_DIR'] = str(out)
    completed = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr
    assert 'FIG08B_REGRESSION_WORKING_ASSET_PASS' in completed.stdout
    payload = json.loads((out / 'fig08b_rinalmo_regression_working_source.json').read_text())
    assert len(payload['tasks']) == 7 and payload['tasks']['te_HEK']['p_value'] is None
    assert payload['tasks']['mrl']['result_sha256'] == sha(mrl)
    assert (out / 'fig08b_rinalmo_regression_working_3840x1920.png').stat().st_size > 100000
    assert (out / 'fig08b_rinalmo_regression_working_3840x1920.pdf').stat().st_size > 5000
    if os.environ.get('FIGURE8B_TEST_ARTIFACT'):
        shutil.copytree(out, os.environ['FIGURE8B_TEST_ARTIFACT'], dirs_exist_ok=True)
    missing = root / 'el_HEK.json'
    missing.rename(root / 'el_HEK_missing.json')
    failed = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
    assert failed.returncode != 0 and 'el_HEK.json' in failed.stderr
    duplicated = json.loads((root / 'te_Muscle.json').read_text())
    duplicated['fold_results'][1] = duplicated['fold_results'][0]
    (root / 'te_Muscle.json').write_text(json.dumps(duplicated))
    failed = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
    assert failed.returncode != 0 and 'fold_results must contain each fold exactly once' in failed.stderr
    nonfinite = cv_payload('te', 'Muscle')
    nonfinite['fold_results'][0]['heads']['official_head']['runs'][0]['spearman'] = float('nan')
    (root / 'te_Muscle.json').write_text(json.dumps(nonfinite))
    failed = subprocess.run([sys.executable, str(SCRIPT)], env=env, text=True, capture_output=True)
    assert failed.returncode != 0 and 'non-finite Spearman' in failed.stderr
    print('FIG08B_SYNTHETIC_DATA_GATE_LAYOUT_DUPLICATE_FOLD_AND_NONFINITE_REJECTION_PASS')
