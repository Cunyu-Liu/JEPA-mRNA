import argparse
import hashlib
import json
import math
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

DATA = Path('/root/rna_jepa_data/ss_data/jsonl/bprna_new.jsonl')
XENS2 = Path('/root/rna_jepa_data/eval_decision/xens2_board_4090/bprna_new.json')
BASELINE_ROOT = Path('/teams/HoulabData_1788502323/liucunyu/foldnar/baseline_recovery_v1')
VERIFIED = BASELINE_ROOT / 'phase0c_verification.json'
XENS2_SHA256 = '22287fe819384e740b9c2f9920cbe23c579783d66360c699905c86a7cf79914a'
PROMPT_XENS2_MICRO_F1 = 0.5531420736
N_SEQUENCES = 5388


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pair_set(pairs, length, context):
    result = set()
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f'{context}: malformed pair {pair!r}')
        i, j = pair
        if isinstance(i, bool) or isinstance(j, bool) or not isinstance(i, int) or not isinstance(j, int):
            raise ValueError(f'{context}: non-integer pair {pair!r}')
        if not 0 <= i < j < length:
            raise ValueError(f'{context}: non-canonical zero-based pair {pair!r}, length={length}')
        if (i, j) in result:
            raise ValueError(f'{context}: duplicate pair {pair!r}')
        result.add((i, j))
    return result


def score_pairs(truth_sets, prediction_sets):
    if len(truth_sets) != len(prediction_sets) or not truth_sets:
        raise ValueError('truth/prediction coverage mismatch')
    tp = fp = fn = 0
    per_sequence = []
    for truth, prediction in zip(truth_sets, prediction_sets):
        seq_tp = len(truth & prediction)
        seq_fp = len(prediction - truth)
        seq_fn = len(truth - prediction)
        tp += seq_tp
        fp += seq_fp
        fn += seq_fn
        denominator = 2 * seq_tp + seq_fp + seq_fn
        per_sequence.append(Fraction(2 * seq_tp, denominator) if denominator else Fraction(0))
    micro = Fraction(2 * tp, 2 * tp + fp + fn)
    macro = sum(per_sequence, Fraction(0)) / len(per_sequence)
    return {
        'n_sequences': len(truth_sets),
        'n_with_prediction': len(prediction_sets),
        'coverage': len(prediction_sets) / len(truth_sets),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'micro_precision': float(Fraction(tp, tp + fp)),
        'micro_recall': float(Fraction(tp, tp + fn)),
        'micro_f1': float(micro),
        'macro_f1': float(macro),
        '_micro_exact': micro,
        '_macro_exact': macro,
        '_per_sequence_exact': per_sequence,
    }


def align_rows(rows, names, pair_key, lengths):
    if not isinstance(rows, list):
        raise ValueError('prediction rows must be a list')
    row_names = [row['name'] for row in rows]
    duplicates = sorted({name for name in row_names if row_names.count(name) > 1})
    row_set = set(row_names)
    name_set = set(names)
    identity = {
        'unique_names': len(row_set),
        'duplicate_names': duplicates,
        'missing_names': sorted(name_set - row_set),
        'extra_names': sorted(row_set - name_set),
        'order_matches_frozen_data': row_names == names,
    }
    if duplicates or row_set != name_set or len(rows) != len(names):
        raise ValueError(f'prediction identity failure: {identity}')
    by_name = {row['name']: row for row in rows}
    predictions = [pair_set(by_name[name][pair_key], lengths[name], name) for name in names]
    return predictions, identity


def decimal_ratio(value):
    with localcontext() as context:
        context.prec = 40
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def public_metrics(score):
    return {key: value for key, value in score.items() if not key.startswith('_')}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=BASELINE_ROOT / 'xens2_absolute_gap_audit_20261124.json')
    args = parser.parse_args()

    verified = json.loads(VERIFIED.read_text())
    records = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    if len(records) != N_SEQUENCES or verified['dataset']['n_sequences'] != N_SEQUENCES:
        raise ValueError('frozen sequence count mismatch')
    names = [row['name'] for row in records]
    if len(set(names)) != len(names):
        raise ValueError('duplicate frozen sequence names')
    lengths = {row['name']: len(row['seq']) for row in records}
    truth = [pair_set(row['pairs'], len(row['seq']), row['name']) for row in records]
    for row, pairs in zip(records, truth):
        if row['n_pairs'] != len(pairs):
            raise ValueError(f"{row['name']}: n_pairs mismatch")

    xens2_obj = json.loads(XENS2.read_text())
    xens2_rows = xens2_obj['per_sequence']
    xens2_pred, xens2_identity = align_rows(xens2_rows, names, 'pred_pairs', lengths)
    xens2_score = score_pairs(truth, xens2_pred)
    if sha256(XENS2) != XENS2_SHA256:
        raise ValueError('xens2 prediction hash mismatch')
    if sha256(DATA) != verified['dataset']['sha256']:
        raise ValueError('frozen data hash mismatch')

    systems = {'xens2': public_metrics(xens2_score)}
    identities = {'xens2': xens2_identity}
    scores = {'xens2': xens2_score}
    source_protocols = {'xens2': xens2_obj['mode']}
    max_f1_difference = {}
    for system in ('mxfold2', 'eternafold'):
        path = BASELINE_ROOT / 'predictions' / system / 'bprna_new.json'
        obj = json.loads(path.read_text())
        pair_key = 'pairs' if system == 'mxfold2' else 'pred_pairs'
        rows = obj['predictions'] if system == 'mxfold2' else obj['per_sequence']
        predictions, identity = align_rows(rows, names, pair_key, lengths)
        score = score_pairs(truth, predictions)
        expected = verified['systems'][system]
        if sha256(path) != expected['prediction_sha256']:
            raise ValueError(f'{system}: prediction hash mismatch')
        for key, value in public_metrics(score).items():
            if key not in expected['metrics']:
                continue
            reference = expected['metrics'][key]
            if isinstance(value, float):
                if not math.isclose(value, reference, rel_tol=0, abs_tol=1e-15):
                    raise ValueError(f'{system}: {key} differs from verified manifest')
            elif value != reference:
                raise ValueError(f'{system}: {key} differs from verified manifest')
        systems[system] = public_metrics(score)
        identities[system] = identity
        scores[system] = score
        source_protocols[system] = obj['protocol']

    for system, rows in (('xens2', xens2_rows), ('eternafold', json.loads((BASELINE_ROOT / 'predictions/eternafold/bprna_new.json').read_text())['per_sequence'])):
        differences = [abs(float(exact) - row['f1']) for exact, row in zip(scores[system]['_per_sequence_exact'], rows)]
        max_f1_difference[system] = max(differences)
        if max_f1_difference[system] > 1e-15:
            raise ValueError(f'{system}: stored per-sequence F1 mismatch')

    stored_micro = xens2_obj['pair_level']['micro']
    for key in ('f1', 'precision', 'recall', 'tp', 'fp', 'fn'):
        metric_key = 'micro_' + key if key in ('f1', 'precision', 'recall') else key
        value = xens2_score[metric_key]
        reference = stored_micro[key]
        if isinstance(value, float):
            if not math.isclose(value, reference, rel_tol=0, abs_tol=1e-15):
                raise ValueError(f'xens2 stored {key} mismatch')
        elif value != reference:
            raise ValueError(f'xens2 stored {key} mismatch')
    if not math.isclose(xens2_score['macro_f1'], xens2_obj['pair_level']['macro']['f1'], rel_tol=0, abs_tol=1e-15):
        raise ValueError('xens2 stored macro F1 mismatch')

    gaps = {}
    for baseline in ('mxfold2', 'eternafold'):
        micro_gap = scores[baseline]['_micro_exact'] - xens2_score['_micro_exact']
        macro_gap = scores[baseline]['_macro_exact'] - xens2_score['_macro_exact']
        gaps[baseline + '_minus_xens2'] = {
            'direction': 'baseline minus xens2; positive means xens2 is lower',
            'micro_f1_absolute_gap': float(micro_gap),
            'micro_f1_absolute_gap_decimal_40_digits': decimal_ratio(micro_gap),
            'micro_f1_absolute_gap_percentage_points': float(100 * micro_gap),
            'macro_f1_absolute_gap': float(macro_gap),
            'macro_f1_absolute_gap_decimal_40_digits': decimal_ratio(macro_gap),
            'macro_f1_absolute_gap_percentage_points': float(100 * macro_gap),
        }

    report = {
        'schema_version': 1,
        'audit_id': 'xens2_absolute_gap_audit_20261124',
        'status': 'PASS',
        'scope': 'independent frozen-prediction pair-set recomputation; no model execution, training, or test-set development',
        'comparison_label': 'raw_vs_raw_frozen_bprna_new_full_5388_exact_zero_based_pairs',
        'dataset': {
            'dataset_id': 'bprna_new',
            'path': str(DATA),
            'sha256': sha256(DATA),
            'n_sequences': len(records),
            'pair_convention': 'zero-based inclusive, i<j, strict exact-pair matching; no pair cleaning or canonicalization',
        },
        'inputs': {
            'xens2_prediction': {'path': str(XENS2), 'sha256': sha256(XENS2)},
            'mxfold2_prediction': {'path': str(BASELINE_ROOT / 'predictions/mxfold2/bprna_new.json'), 'sha256': sha256(BASELINE_ROOT / 'predictions/mxfold2/bprna_new.json')},
            'eternafold_prediction': {'path': str(BASELINE_ROOT / 'predictions/eternafold/bprna_new.json'), 'sha256': sha256(BASELINE_ROOT / 'predictions/eternafold/bprna_new.json')},
            'verified_baseline_manifest': {'path': str(VERIFIED), 'sha256': sha256(VERIFIED)},
            'audit_script': {'path': str(Path(__file__).resolve()), 'sha256': sha256(Path(__file__).resolve())},
        },
        'systems': systems,
        'prediction_identity': identities,
        'source_protocols': source_protocols,
        'prompt_xens2_micro_f1_claim': {
            'claimed_rounded_value': PROMPT_XENS2_MICRO_F1,
            'recomputed_value': xens2_score['micro_f1'],
            'absolute_difference_from_rounded_claim': abs(xens2_score['micro_f1'] - PROMPT_XENS2_MICRO_F1),
            'matches_to_10_decimal_places': round(xens2_score['micro_f1'], 10) == PROMPT_XENS2_MICRO_F1,
            'verdict': 'CONFIRMED',
        },
        'absolute_gaps': gaps,
        'cross_checks': {
            'xens2_stored_summary_matches_recomputation': True,
            'baseline_verified_summaries_match_recomputation': True,
            'max_per_sequence_f1_absolute_difference': max_f1_difference,
            'full_coverage_all_systems': all(score['coverage'] == 1.0 for score in systems.values()),
        },
        'raw_vs_clean': {
            'label': 'raw/raw full-coverage comparison only',
            'xens2': 'raw bprna_new predictions; all 5,388 frozen sequences; n_skipped=0',
            'mxfold2': 'raw-sequence end-to-end predictions on the same all-5,388 sequences',
            'eternafold': 'raw-sequence end-to-end predictions on the same all-5,388 sequences',
            'not_an_own_clean_or_matched_clean_comparison': True,
            'clean_scores_recomputed_or_substituted': False,
        },
        'provenance_limitations': [
            'This audit verifies frozen prediction assets and their numerical summaries; it is not fresh model execution and does not independently establish training provenance.',
            'The MXfold2 official-release audit covers reference BPSEQ only, not official released predictions; its 13 official-only records are outside the frozen project split and receive no imputed coverage or score.',
            'Raw full-coverage values must not be relabeled as own-clean or matched-clean values.',
            'bpRNA-new was read only for this frozen-prediction audit, not for development, calibration, model selection, or training.',
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({
        'status': report['status'],
        'output': str(args.output),
        'xens2': systems['xens2'],
        'absolute_gaps': gaps,
        'cross_checks': report['cross_checks'],
    }, indent=2))


if __name__ == '__main__':
    main()
