"""Rebuild the six-work completion ledger from committed archives (stdlib only).

This is an accounting/claim-level check, NOT an experiment, network proof,
fresh artifact-availability search, or a scientific/paper-acceptance judgment.
The dated scope classifications are reviewed metadata, not inferred from PASS.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'docs/baseline_readiness_20260922.json'
SOURCES = {
    'dual_train': 'docs/dual_rs_training_landing_archive_20260922.json',
    'dual': 'docs/dual_rs_certification_archive_20260922_r1.json',
    'meta': 'docs/metamoe_paired_smoke_review_20260922_r2.json',
    'meta_capacity': 'docs/metamoe_hz_intake_diagnostic_20260922_r1.json',
    'robust': 'docs/robust_experts_landing_20260922_r1/review.json',
    'robust_semantics': 'docs/robust_experts_landing_20260922_r1/source_semantics.json',
    'rome': 'docs/rome_multinorm_archive_20260922_r1.json',
    'rome_interpretation': 'docs/rome_multinorm_interpretation_20260922_r1.json',
    'deployment': 'docs/recent_moe_deployment_20260921_r2.json',
}
CONFIGS = {
    'dual': 'configs/recent_moe/dual_rs_certification_execution_r1.json',
    'meta': 'configs/recent_moe/metamoe_paired_smoke_r2.json',
    'robust': 'configs/recent_moe/robust_experts_paper_training_execution_r1.json',
    'rome': 'configs/recent_moe/rome_multinorm_batch_r1.json',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: '+key)
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('nonfinite JSON number: '+value)

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=invalid)
    def finite(obj):
        if isinstance(obj, float):
            require(math.isfinite(obj), 'nonfinite JSON number')
        elif isinstance(obj, dict):
            for child in obj.values():
                finite(child)
        elif isinstance(obj, list):
            for child in obj:
                finite(child)
    finite(value)
    return value


def rebuild(root=ROOT):
    root = Path(root)
    inventory = {}
    def read(name):
        data = (root/name).read_bytes()
        inventory[name] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
        return strict_json(data)
    data = {key: read(name) for key, name in SOURCES.items()}
    configs = {}
    for key, name in CONFIGS.items():
        configs[key] = read(name)
        require(inventory[name]['sha256'] == data[key]['config_sha256'], 'config binding: '+key)

    train, dual = data['dual_train'], data['dual']
    require(train['status'] == 'TRAINING_LANDED' and train['epochs_checked'] == 90,
            'Dual RS training scope')
    require(train['final_epoch']['file_sha256'] ==
            configs['dual']['files'][configs['dual']['selector_checkpoint']],
            'Dual RS final checkpoint binding')
    require(dual['audit']['status'] == 'COUNT_IDENTITY_AUDIT_PASS', 'Dual RS count audit')
    require(dual['grade'] == 'PROBABILISTIC_RS_NATIVE_NUMERICAL' and
            dual['deterministic_formal_SAFE'] is False and dual['paper_accuracy_estimate'] is False,
            'Dual RS guarantee upgrade')
    dual_rows = dual['audit']['rows']
    require([r['index'] for r in dual_rows] == [0, 1], 'Dual RS pilot denominator')
    require(all(r['grade'] == dual['grade'] and r['deterministic_formal_SAFE'] is False and
                r['radius_l2'] >= 0 and r['correct'] == (r['label'] == r['prediction'])
                for r in dual_rows), 'Dual RS row consistency')

    meta = data['meta']
    require(meta['audit'] == 'PASS' and meta['execution_control_pass'] is False and
            meta['numerical_guarantees_equated'] is False, 'Meta audit is not an execution gate')
    expected = {('cifar10_0', 'act'): 'UNSAFE_REPLAYED',
                ('cifar10_0', 'author'): 'UNSAFE_REPLAYED',
                ('mnist_0', 'act'): 'ERROR', ('mnist_0', 'author'): 'BACKEND_POSITIVE'}
    require(len(meta['rows']) == 4 and
            {(r['id'], r['arm']): r['status'] for r in meta['rows']} == expected,
            'Meta smoke roster/states')
    cap = data['meta_capacity']['result']
    require(cap['config_sha256'] == meta['config_sha256'], 'Meta diagnostic config binding')
    require(cap['new_solver_queries'] == 0 and
            next(e for e in cap['events'] if e['kind'] == 'RELU')['drop'] == 'sparse_relu_size_limit',
            'Meta recorded representation blocker')

    robust, semantics = data['robust'], data['robust_semantics']
    require(robust['status'] == 'INDEPENDENT_SAVED_TRAINING_REVIEW_PASS' and
            robust['formal_SAFE'] is False and robust['prediction_replay'] is False,
            'Robust saved audit is not attack replay or a formal proof')
    require(semantics['top2_effective_ste'] is False and
            semantics['config_sha256'] == robust['config_sha256'], 'Robust effective source binding')
    formula = semantics['uniform_two_by_four_control']
    require(formula['source_entropy'] != formula['paper_negative_entropy_of_batch_mean'] and
            formula['source_column_entropy'] == formula['paper_negative_entropy_of_batch_mean'],
            'Robust objective distinction')
    require([r['arm'] for r in robust['arms']] == ['dense', 'convmoe'], 'Robust arm roster')
    robust_metrics = {}
    for arm in robust['arms']:
        require(arm['epochs'] == 200 and arm['updates'] == 12600 and
                arm['epoch_checkpoint_hashes_checked'] == 200, 'Robust training denominator')
        require(set(arm['evaluations']) == {'clean', 'PGD20', 'APGD20'}, 'Robust evaluation roster')
        for ev in arm['evaluations'].values():
            require(ev['examples'] == 10000 and type(ev['correct_from_aggregate']) is int and
                    0 <= ev['correct_from_aggregate'] <= ev['examples'] and
                    ev['accuracy_percent'] == ev['correct_from_aggregate']/100 and
                    abs(ev['recorded_accuracy']*10000-ev['correct_from_aggregate']) < .001,
                    'Robust accuracy denominator/aggregate')
        robust_metrics[arm['arm']] = {
            kind: ev['accuracy_percent'] for kind, ev in arm['evaluations'].items()}

    rome, interpretation = data['rome'], data['rome_interpretation']
    require(rome['formal_SAFE'] is False and rome['paper_scale_reproduction'] is False and
            rome['ACT_fair_comparison_completed'] is False, 'RoME scope upgrade')
    require(rome['denominator_inputs'] == 4 and rome['denominator_norm_requests'] == 12 and
            len(rome['rows']) == 12 and rome['counts'] ==
            {'ATTACK_EVALUATION_COMPLETED': 5, 'TIMEOUT': 7}, 'RoME terminal denominator')
    require(interpretation['parent_archive_sha256'] == inventory[SOURCES['rome']]['sha256'] and
            interpretation['counts'] == {'PREEXISTING_CLEAN_ERROR': 3,
                                         'PERTURBATION_BREAK_REPLAYED': 2,
                                         'INCOMPLETE_NOT_ROBUST': 7}, 'RoME interpretation binding')
    roster = {(index, norm) for index in (6044, 2890, 9399, 1917) for norm in ('Linf', 'L1', 'L2')}
    raw_rows = {(r['index'], r['norm']): r for r in rome['rows']}
    explained = {(r['index'], r['norm']): r for r in interpretation['rows']}
    require(len(interpretation['rows']) == 12 and set(raw_rows) == set(explained) == roster,
            'RoME unique request roster')
    require(Counter(r['status'] for r in rome['rows']) == rome['counts'] and
            Counter(r['interpretation'] for r in interpretation['rows']) == interpretation['counts'],
            'RoME row recount')
    for key, original in raw_rows.items():
        meaning = ('INCOMPLETE_NOT_ROBUST' if original['status'] == 'TIMEOUT' else
                   'PREEXISTING_CLEAN_ERROR' if original['clean_prediction'] != original['label'] else
                   'PERTURBATION_BREAK_REPLAYED' if original['adversarial_prediction'] != original['label'] else
                   'NO_BREAK_RECORDED')
        require(explained[key]['parent_status'] == original['status'] and
                explained[key]['interpretation'] == meaning, 'RoME per-row interpretation')
    repositories = data['deployment']['repositories']
    require(repositories['j_tlat']['tracked_files'] == 1 and
            repositories['j_tlat']['python_files'] == 0 and
            repositories['feature_noise']['not_a_claim_of_global_absence'] is True,
            'archived artifact search scope')

    def row(name, deployment, executed, comparison, grade, metrics, next_gate):
        return dict(name=name, deployment=deployment, executed_scope=executed,
                    entire_paper_reproduction_completed=False,
                    act_comparison=comparison, result_grade=grade, metrics=metrics,
                    next_execution_frozen=False, next_gate=next_gate)
    rows = [
        row('Dual RS', 'RUNNABLE_NAMED_VARIANT', '90_EPOCHS_AND_TWO_INPUT_CERTIFICATION',
            'DIFFERENT_FUNCTION_NO_SAME_OBJECT_RESULT', dual['grade'],
            {'indices': [r['index'] for r in dual_rows],
             'radii_l2': [r['radius_l2'] for r in dual_rows]},
            'Freeze native certification extension separately; g is not deterministic F.'),
        row('MetaMoE', 'RUNNABLE_NATIVE_AUTHOR_FRONTEND', 'TWO_INPUT_SAME_OBJECT_SMOKE',
            'SMOKE_ONLY_FORMAL_COHORT_BLOCKED', 'MIXED_REPLAY_AND_NUMERICAL_FILTER',
            {'states': expected_to_rows(expected), 'act_representation_blocked': True},
            'Separate CSR admission/storage controls, then new smoke; no new 20-input launch yet.'),
        row('Robust Experts', 'RUNNABLE_SOURCE_ENTROPY_VARIANT', 'TWO_200_EPOCH_FULL_10K_EVALUATIONS',
            'TRAINED_NATIVE_TOP2_UNSUPPORTED', 'EMPIRICAL_AGGREGATES_NOT_REPLAYED', robust_metrics,
            'Native k2 definedness and joint-HZ nonlinear support precede trained-model comparison.'),
        row('RoME', 'RUNNABLE_SOURCE_DEFAULT_CHECKPOINT', 'TWELVE_MULTINORM_TERMINALS',
            'CONTINUOUS_MULTILAYER_UNSUPPORTED', rome['evidence_grade'], interpretation['counts'],
            'Separate full evaluation freeze and original continuous-mixture semantics; no top-k substitution.'),
        row('J-TLAT', 'EXECUTABLE_ARTIFACT_NOT_OBTAINED', 'NO_TRAINING_OR_EVALUATION',
            'NO_SAME_OBJECT_PROTOCOL', 'ACCESS_BLOCKED_NOT_ZERO_SCORE', None,
            'Obtain legitimate supplemental artifact via PI; no automatic author contact.'),
        row('Feature Noise', 'FULL_AUTHOR_ARTIFACT_NOT_IDENTIFIED', 'NO_TRAINING_OR_EVALUATION',
            'THREAT_MODEL_NOT_ALIGNED', 'ARTIFACT_AND_PROTOCOL_MISSING_NOT_ZERO_SCORE', None,
            'Obtain implementation and define actual feature/text threat; pixel Linf is not equivalent.'),
    ]
    return {'schema': 'BASELINE_COMPLETION_ACCOUNTING_V1', 'as_of': '2026-09-22',
            'scope': 'committed archive accounting; no fresh execution, web search or proof checking',
            'sources': inventory, 'rows': rows,
            'summary': {'works': 6, 'works_with_real_execution': 4,
                        'entire_paper_reproductions_completed': 0,
                        'formal_cohort_act_comparisons_completed': 0,
                        'same_object_act_smoke_works': 1},
            'limitations': ['Classifications/next gates are reviewed metadata, not scientific proofs.',
                            'Archived search is not proof that an author artifact does not exist.',
                            'No cross-grade score ranking or CCF-A acceptance/readiness inference.',
                            'No reconstruction of uncommitted raw checkpoints or prediction replay.']}


def expected_to_rows(expected):
    return [{'input': key[0], 'arm': key[1], 'status': value} for key, value in expected.items()]


def check(root=ROOT, report=None):
    report = strict_json((Path(root)/OUTPUT).read_bytes()) if report is None else report
    require(report == rebuild(root), 'stale or upgraded readiness record: regenerate in a new reviewed stage')
    return {'status': 'ARCHIVED_ACCOUNTING_PASS', 'summary': report['summary'],
            'not_submission_ready_judgment': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output', type=Path, help='new derived JSON only; refuses overwrite')
    args = parser.parse_args()
    require(not (args.check and args.output), 'choose check or output')
    report = check() if args.check else rebuild()
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)+'\n'
    if args.output:
        with args.output.open('x') as out:
            out.write(text)
        print(json.dumps(check(report=report)))
    else:
        print(text)
