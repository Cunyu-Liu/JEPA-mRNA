#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

MODULE_PATH = Path(os.environ.get('AGGREGATOR_MODULE_PATH', '/private/tmp/rnajev_audit/round33/code/aggregate_te_el_cv_round33.py'))
spec = importlib.util.spec_from_file_location('aggregate', MODULE_PATH)
aggregate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aggregate)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_aggregator(root, out):
    return subprocess.run([
        sys.executable, str(MODULE_PATH), '--root', str(root), '--task', 'te',
        '--cell-line', 'Muscle', '--out', str(out)], text=True, capture_output=True)


with tempfile.TemporaryDirectory(prefix='rnajev_te_el_aggregate_test_') as temporary:
    root = Path(temporary)
    source = root / 'data/rinalmo/te_el_official/Muscle_sequence.csv'
    source.parent.mkdir(parents=True)
    frame = pd.DataFrame({'utr': ['ACGU' * 25] * 30,
                          'te_log': np.linspace(-1, 1, 30),
                          'rnaseq_log': np.linspace(1, 3, 30)})
    frame.to_csv(source, index=False)
    source_sha = sha(source)
    rng = np.random.default_rng(20261028)
    for fold in range(10):
        name = f'rinalmo_te_Muscle_fold{fold}_round33_full_v4'
        out_dir = root / 'eval_decision' / name
        out_dir.mkdir(parents=True)
        counts = aggregate.split_counts(frame, fold)
        heads = {}
        for head_index, head in enumerate(aggregate.HEADS):
            runs = []
            for seed in range(5):
                truth = np.linspace(-1, 1, counts['n_test'])
                predictions = truth + rng.normal(0, 0.2 + 0.02 * seed + 0.01 * head_index, len(truth))
                spearman = float(stats.spearmanr(predictions, truth)[0])
                r2 = float(1 - np.sum((truth - predictions) ** 2) / np.sum((truth - truth.mean()) ** 2))
                pcc = float(stats.pearsonr(predictions, truth)[0])
                checkpoint = out_dir / f'{head}_s{seed}_best.pt'
                checkpoint.write_bytes(f'{head}:{fold}:{seed}'.encode())
                prediction = out_dir / f'{head}_s{seed}_predictions.npz'
                np.savez_compressed(prediction, predictions=predictions, truth=truth)
                history = [{'epoch': epoch, 'validation_loss': 0.5 if epoch == 1 else 1.0}
                           for epoch in range(1, 34)]
                runs.append({'seed': seed, 'device': 'cuda', 'converged': True,
                             'early_stopped': True, 'independent_recheck': 'PASS',
                             'best_epoch': 1, 'epochs_run': 33, 'best_validation_loss': 0.5,
                             'validation_history': history, **counts,
                             'test_spearman': spearman, 'test_r2': r2, 'test_pcc': pcc,
                             'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                             'prediction_file': str(prediction), 'prediction_sha256': sha(prediction)})
            heads[head] = {'runs': runs}
        protocol = {'task': 'te', 'cell_line': 'Muscle', 'fold': fold, 'device': 'cuda',
                    'sample_scope': 'full_official_protocol', 'seeds': list(range(5)),
                    'source_data': str(source), 'source_data_sha256': source_sha,
                    'pretrained_sha256': 'weights', 'pretrained_config_sha256': 'config',
                    'preflight_sha256': 'preflight', 'embedding_cache_meta': '/fake/cache.json',
                    'embedding_cache_meta_sha256': 'cache', 'code_sha256': 'code'}
        result_path = out_dir / 'result.json'
        result_path.write_text(json.dumps({'status': 'PASS', 'protocol': protocol, 'heads': heads}))
        result_sha = sha(result_path)
        launch_log = root / 'eval_decision' / f'{name}.launch.log'
        launch_log.write_text('RINALMO_REGRESSION_DUAL_PASS training output only; no external QA marker\n')
        qa_watch = 'round34_qa_watch' if fold in (0, 1) else 'round34_qa_watch_folds2_9'
        qa_log = root / 'audits' / qa_watch / f'{name}.qa.log'
        qa_log.parent.mkdir(parents=True, exist_ok=True)
        qa_log.write_text(f'REGRESSION_RESULT_EXTERNAL_QA_PASS {{"scope": "full", "runs": 10, "result_sha256": "{result_sha}", "cuda_device": "NVIDIA GeForce RTX 4090"}}\n')
    aggregate_path = root / 'aggregate.json'
    completed = run_aggregator(root, aggregate_path)
    assert completed.returncode == 0, completed.stderr
    assert 'TE_EL_CV_AGGREGATE_PASS' in completed.stdout
    payload = json.loads(aggregate_path.read_text())
    assert payload['scientific_eligibility'] is True and len(payload['fold_results']) == 10
    assert payload['protocol']['n_runs_per_head'] == 50
    official = payload['heads']['official_head']['seed_cv_spearman']
    jev = payload['heads']['jev_head']['seed_cv_spearman']
    expected = np.asarray(jev) - np.asarray(official)
    assert np.allclose(payload['comparison']['seed_cv_delta_jev_minus_official'], expected)
    assert np.isclose(payload['comparison']['paired_delta_mean'], expected.mean())
    changed = root / 'eval_decision/rinalmo_te_Muscle_fold0_round33_full_v4/result.json'
    changed.write_text(changed.read_text() + '\n')
    failed = run_aggregator(root, root / 'stale.json')
    assert failed.returncode != 0 and 'no current full GPU-QA record' in failed.stderr
    print('TE_EL_AGGREGATOR_SYNTHETIC_TEST_PASS')
