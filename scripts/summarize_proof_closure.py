"""Recount a fixed archived proof frontier, NOT reprove any network or bound.

Standard-library, saved compact records only: no raw paths in archives are
followed, no model/solver/producer is imported, and no experiment is launched.
The pinned hashes identify the inputs; they are not mathematical authority.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'docs/proof_closure_20260925_r1.json'
SOURCES = {
    'scoped': ('docs/scoped_proof_execution_archive_20260924_r1.json',
               '29e4b76fd75a08be441398688bbcd8bad564d4f0fdce3a4eff3dcf7c2f8a66c3'),
    'parse': ('docs/scoped_parse_proof_execution_archive_20260924_r1.json',
              '2a8eb3d0ea2956a0d3bb40fcaa827f8753f0dfb99f5fd147122caead9004dd41'),
    'frontier': ('docs/frontier_proof_execution_archive_20260924_r1.json',
                 '7f5ef460c79f5d9268926b7042eef3ae1c41e12fbc030db261df391251b12fbf'),
    'residual': ('docs/residual_proof_execution_archive_20260925_r1.json',
                 '05e97994d9c74cfac8e2ae0c1d23b9847173805e4c4f6afcf3fd6e45da7a6e16'),
    'upstream': ('docs/readonly_upstream_audit_20260925_r1.json',
                 '122fab635492e64d755d9cd1ce8ef112405caf8b3242ee9c8994b1ad680a295a'),
}
REAL = {
    'scoped': (4088, ('exhaustive',)),
    'parse': (4096, ('uncached', 'cached')),
    'frontier': (4098, ('exhaustive', 'checked_frontier')),
    'residual': (4099, ('pairwise', 'shared')),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            'missing/nonfinite/negative cost')
    return value


def load_sources(root=ROOT):
    result = {}
    for key, (name, digest) in SOURCES.items():
        path = root / name
        require(not path.is_symlink(), 'linked archive')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, 'archive identity: ' + name)
        result[key] = strict_json(raw)
    return result


def real_rows(data):
    result = []
    for family, (index, ids) in REAL.items():
        archive = data[family]
        require(archive['audit'] == 'PASS' and archive['issues'] == 0, 'archive audit')
        require(archive['dataset_index'] == index, 'request selection')
        required = archive['required_output_obligations'] if family == 'scoped' else archive['required_per_call']
        require(required == 28 * 9, 'original obligation count')
        if family == 'scoped':
            calls = [dict(id='exhaustive', status=archive['effective_status'],
                          cost=archive['cost'], evidence=archive['evidence'], stages=archive['stages'])]
        else:
            calls = archive['calls']
        require(tuple(c['id'] for c in calls) == ids, 'missing/duplicate/reordered call')
        for call in calls:
            e, cost, stages = call['evidence'], call['cost'], call['stages']
            status = call['status']
            require(status in ('TIMEOUT', 'RESOURCE_LIMIT') and status == cost['status'], 'terminal status')
            require(cost['budget_seconds'] == 300, 'budget changed')
            require(cost['complete_output_positive_proof'] is False, 'cost proof upgrade')
            seconds = finite(cost['end_to_end_seconds'])
            phase_seconds = math.fsum(finite(s['seconds']) for s in stages)
            require(abs(phase_seconds - finite(cost['stage_seconds'])) < 1e-8, 'stage sum')
            require(abs(phase_seconds + finite(cost['overhead_seconds']) - seconds) < 1e-8, 'cost closure')
            caller = call.get('return_seconds')  # Older single-call archive has no caller clock.
            if caller is not None:
                require(finite(caller) >= seconds, 'caller cost precedes ledger')
            names = [s['phase'] for s in stages]
            require(len(names) == len(set(names)) and names[0] == 'intake', 'phase identity')
            require(stages[-1]['status'] == status and all(s['status'] == 'COMPLETED' for s in stages[:-1]),
                    'inconsistent cutoff stages')
            require(not set(names) & {'source_check', 'propose', 'aggregate'}, 'unexpected output-phase evidence')
            for field in ('complete_output_positive_proof', 'native_float_proof', 'route_changing_established',
                          'construction_published'):
                require(e[field] is False, 'unsupported proof/construction upgrade: ' + field)
            require(e['source_captured'] is True and e['native_lp_calls_started'] == 0, 'source/query inventory')
            if family in ('scoped', 'parse'):
                require(e['source_independently_checked'] is False and e['candidate_files'] == [], 'source/candidates')
                checked, positive, excluded, offline = e['exact_bounds_checked'], e['positive_bounds'], 0, None
                missing = e['obligations_without_checked_bound']
                accepted = False
            else:
                require(e['source_check_receipt_published'] is False and e['output_candidate_files'] == [] and
                        e['final_aggregation_published'] is False, 'unreached output evidence')
                checked, positive = e['independently_checked_output_bounds'], e['positive_output_bounds']
                accepted = e['route_receipt_published']
                completed = 'route_check' in names and stages[names.index('route_check')]['status'] == 'COMPLETED'
                require(type(accepted) is bool and accepted == e['route_check_phase_completed'] == completed,
                        'offline check cannot become online receipt')
                excluded = e['route_discharged_duties']
                require(excluded == e['checked_excluded_pairs'] * 9 and (accepted or excluded == 0), 'route discharge')
                offline_row, = [r for r in archive['frozen_batch_audit']['rows'] if r['id'] == call['id']]
                offline = offline_row['audit']['checked_excluded_pairs']
                if accepted:
                    pairs = e['retained_pair_list']
                    require(len(pairs) == len({tuple(p) for p in pairs}) == 28 - e['checked_excluded_pairs'], 'pair coverage')
                    require(all(len(p) == 2 and 0 <= p[0] < p[1] < 8 for p in pairs), 'illegal pair')
                missing = e['duties_without_positive_checked_evidence']
            require(checked == positive == 0 and missing == required - excluded, 'output accounting')
            prefix = '' if family == 'scoped' else call['id'] + '/'
            require(prefix + 'construction.json' not in archive['files'], 'unexpected saved construction')
            source_hash = archive['files'][prefix + 'source.json']['sha256']
            result.append(dict(study=family, dataset_index=index, arm=call['id'], status=status,
                source_sha256=source_hash, request_sha256=cost['request_sha256'], original_duties=required,
                online_route_receipt=accepted, online_discharged_duties=excluded,
                offline_checked_excluded_pairs=offline, output_queries_started=0,
                output_bounds_checked=checked, positive_output_bounds=positive, unclosed_duties=missing,
                stopping_phase=names[-1], ledger_seconds=seconds, caller_seconds=caller,
                phase_costs=[dict(phase=s['phase'], seconds=s['seconds'], status=s['status']) for s in stages],
                later_phases='NOT_STARTED_NOT_ZERO_COST_ESTIMATES', construction_published=False,
                same_source_positive_request=False, route_changing_established=False,
                cost_exclusions=cost['excludes']))
        hashes = {r['source_sha256'] for r in result if r['study'] == family}
        require(len(hashes) == 1, 'paired source identity')
    return result


def synthetic_rows(archive):
    require(archive['audit'] == 'PASS' and archive['issues'] == 0 and archive['calls'] == archive['completed'] == 4,
            'synthetic inventory')
    require(archive['real_requests'] == archive['native_solver_queries'] == archive['new_real_certificates'] == 0 and
            archive['complete_output_positive_proof'] is False, 'synthetic guarantee')
    ids = ('repair_small_direct', 'repair_small_readonly', 'repair_medium_readonly', 'repair_medium_direct')
    require(tuple(r['call']['id'] for r in archive['rows']) == ids, 'synthetic call identity')
    result = []
    for row in archive['rows']:
        p, s, call = row['profile'], row['profile']['source_check'], row['call']
        fixture = call['spec']['fixture']
        n = math.comb(fixture['experts'], 2) * (fixture['classes'] - 1)
        require(row['status'] == row['returned']['status'] == 'COMPLETED', 'synthetic status')
        require(s['original_output_obligations'] == n == s['excluded_output_obligations'] + s['output_obligations'],
                'synthetic duties')
        require(p['native_solver_calls'] == s['lower_bounds_checked'] == 0 and
                p['complete_output_positive_proof'] is s['complete_output_positive_proof'] is False and
                s['route_changing_established'] is s['native_float_proof'] is False, 'construction is not output proof')
        require(s['source_sha256'] == p['source_sha256'] == call['spec']['source_sha256'], 'synthetic source binding')
        result.append(dict(id=call['id'], source_sha256=s['source_sha256'], original_duties=n,
            checked_excluded_duties=s['excluded_output_obligations'], retained_constructions=s['output_obligations'],
            positive_output_bounds=0, complete_output_positive_proof=False,
            whole_seconds=finite(row['returned']['seconds_including_terminal'])))
    return result


def build(data):
    real, synthetic = real_rows(data), synthetic_rows(data['upstream'])
    ops = data['residual']['calls'][1]['operations'][-1]
    constructed, = [r for r in ops['completed_operations'] if r['operation'] == 'checked_lazy_source_construction']
    pending = ops['open_operation_at_stop']
    require(constructed['status'] == 'EXIT' and pending['operation'] == 'serialize_construction' and
            finite(constructed['finished_seconds']) <= finite(pending['entered_seconds']) < 298, 'construction trace')
    pairs = []
    for name in ('small', 'medium'):
        off, = [r for r in synthetic if r['id'] == 'repair_' + name + '_direct']
        on, = [r for r in synthetic if r['id'] == 'repair_' + name + '_readonly']
        require(off['source_sha256'] == on['source_sha256'], 'pair identity')
        pairs.append(dict(fixture=name, readonly_minus_direct_seconds=on['whole_seconds'] - off['whole_seconds'],
                          observations_per_arm=1, extrapolation_to_real_requests=False))
    return dict(schema='PROOF_CLOSURE_ACCOUNTING_R1', evidence_grade='ARCHIVED_ACCOUNTING_NOT_REPROOF',
        source_records=[dict(key=k, path=v[0], sha256=v[1]) for k, v in SOURCES.items()],
        real_attempts=real, real_summary=dict(calls=len(real), distinct_inputs=len({r['dataset_index'] for r in real}),
            timeouts=sum(r['status'] == 'TIMEOUT' for r in real), resource_limits=sum(r['status'] == 'RESOURCE_LIMIT' for r in real),
            complete_positive_requests=0, output_queries_started=0, general_success_rate_estimate=False),
        synthetic_construction_controls=synthetic, synthetic_cost_pairs=pairs,
        last_real_frontier=dict(input=4099, arm='shared', original_duties=252, online_discharged=225,
            unclosed_duties=27, retained_potential_pairs=data['residual']['calls'][1]['evidence']['retained_pair_list'],
            in_memory_construction_return_seconds=constructed['finished_seconds'],
            time_before_work_deadline_at_return=298 - constructed['finished_seconds'],
            outstanding=['construction_publication', 'source_check', 'output_proposals', 'all_bound_checks_and_aggregation'],
            retained_does_not_establish_reachability=True),
        decision='STOP_TIMING_LINE_NO_AUTOMATIC_REAL_FREEZE',
        reasons=['real attempts did not reach output solving: no output relaxation diagnosis',
                 'synthetic construction correctness is not a complete output proof',
                 'one win/one loss in synthetic total cost supplies no real closure forecast',
                 'no new isolated, data-supported complete-obligation intervention identified by this analysis'],
        next_actions=['integrate these limitations into the manuscript and independent-human-review brief',
                      'PI-managed reviewer/access and separately scoped clean empirical reproduction remain open',
                      'new real proof research requires a specific hypothesis, controls and separately authorized freeze'],
        new_model_calls=0, new_solver_calls=0, new_propagations=0, new_certificates=0,
        strict_main_table_upgrade=False, human_review_completed=False, experiment_launched=False)


def check(root=ROOT):
    result = build(load_sources(root))
    require(strict_json((root / OUTPUT).read_bytes()) == result, 'derived report drift')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    a = p.parse_args()
    if a.check:
        result = check()
    else:
        result = build(load_sources())
        with (ROOT / OUTPUT).open('x') as stream:
            json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
    print(json.dumps(dict(status='PASS', **result['real_summary'], decision=result['decision'])))


if __name__ == '__main__':
    main()
