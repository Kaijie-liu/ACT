"""Derived timing review, no verification queries or changes to frozen traces."""
from collections import Counter
import json
from pathlib import Path

from scripts.conv_three_arm_contract import ROOT, read
from scripts.run_conv_f0_timing import DEFAULT
from scripts.audit_conv_f0_timing import audit
from act.pipeline.moe.conv_training import atomic_json
from act.pipeline.moe.experiment1 import _sha256


def summarize(checked):
    trace = checked['trace']; spans = trace['spans']; by_id = {s['id']:s for s in spans}
    def named(suffix):
        return [s for s in spans if s['name'].endswith(suffix)]
    def seconds(items):
        return sum(s['seconds'] for s in items if s['end'] is not None)
    def under(span, suffix):
        while span['parent'] is not None:
            span = by_id[span['parent']]
            if span['name'].endswith(suffix): return True
        return False
    solves = named('.solve_monolithic_weighted_top2_f0')
    property_table = []
    for solve in solves:
        children = [s for s in spans if s['parent']==solve['id']]
        builds = [s for s in children if s['name'].endswith('._build_disjunction')]
        calls = [s for s in children if s['name']=='scipy.optimize.milp']
        assert len(builds)==len(calls)==1
        native = calls[0]
        remaining = max(0., 300-native['start'])
        property_table.append({
            'property_index':solve['arguments']['property_index'],
            'encoding_count':solve['arguments']['encoding_count'],
            'covered_pairs':checked['detail']['facts']['pairs'],
            'solve_span_id':solve['id'], 'native_span_id':native['id'],
            'granted_seconds':solve['arguments']['time_limit'],
            'disjunction_build_seconds':builds[0]['seconds'],
            'native_start':native['start'], 'remaining_at_native_entry':remaining,
            'native_requested_limit':native['arguments']['options']['time_limit'],
            'native_seconds':native.get('seconds'),
            'native_result':native.get('result'),
            'native_exposure_to_terminal':native.get('exposure_to_cutoff_seconds'),
            'native_right_censored':native['right_censored'],
            'native_limit_excess_over_request_remaining':max(0., native['arguments']['options']['time_limit']-remaining),
            'decision':solve.get('result'),
            'matrix':{k:native['arguments'][k] for k in
                      ('objective_size','integral_entries','matrix_shape','matrix_nnz')}})
    supports = [s for s in named('._guarded_support_query') if under(s, '.run_monolithic')]
    support_summary = {}
    for label, relaxed in [('LP', True), ('MILP', False)]:
        selected = [s for s in supports if s['arguments']['relax_binaries'] is relaxed]
        results = [s['result'] for s in spans if s['parent'] in {x['id'] for x in selected}
                   and s['name'].endswith('.hz_support_bounds') and s['end'] is not None]
        assert len(results)==len(selected) and all('exact' in r for r in results)
        support_summary[label] = {'calls':len(selected), 'seconds':seconds(selected),
            'exact_completed_calls':sum(r['exact'] is True for r in results),
            'status_source':'nested hz_support_bounds return, not tuple-valued dispatch wrapper'}
    native_property = [s for s in named('.milp') if s['parent'] in {x['id'] for x in solves}]
    completed = [s for s in native_property if s['end'] is not None]
    open_calls = [s for s in native_property if s['end'] is None]
    return {
        'pre_f0_outer_seconds':named('.run_monolithic')[0]['start'],
        'pair_propagation_seconds':seconds(named('.shared_input_pair_propagation')),
        'pair_propagations':[{'pair':s['arguments']['pair'], 'seconds':s['seconds'], 'span_id':s['id']}
                             for s in named('.shared_input_pair_propagation')],
        'guarded_support_within_pair_propagation':support_summary,
        'gate_range_seconds':seconds(named('.compute_weighted_top2_gate_range')),
        'encoding_seconds':seconds(named('.build_weighted_top2_f0')),
        'difference_support_within_encoding_seconds':seconds(named('.compute_weighted_top2_difference_range')),
        'disjunction_build_seconds':seconds(named('._build_disjunction')),
        'property_native_completed_seconds':seconds(completed),
        'property_native_completed_calls':len(completed),
        'property_native_status_counts':dict(Counter(str(s['result'].get('status')) for s in completed)),
        'property_native_censored_calls':len(open_calls),
        'property_native_censored_exposure_to_terminal_seconds':sum(s['exposure_to_cutoff_seconds'] for s in open_calls),
        'property_table':property_table,
        'trace_event_count':trace['event_count'],
        'logging_serialization_write_fsync_observed_seconds':trace['logging_seconds_observed'],
        'trace_tail_bytes':trace['partial_tail_bytes'],
        'context_caveat':'Raw generic caller pair/index locals can outlive their loop: pair propagation index8 is not an active property; combined solve caller pair[1,3] is NOT its coverage. Use entry pair for propagation and encoding_count plus checked route inventory for union solves, as in this derived table.',
        'numerical_caveat':'All eight completed property-native calls report status1 and no recorded primal/dual/gap/nodes; omitted fields are unavailable, not zero. Last call has no return. No SAFE or UNSAFE is inferred.',
        'timing_caveat':'Inclusive nested timings overlap; support is within propagation and difference is within encoding. Native interval includes setup/presolve/search. Censored exposure is not a completed duration. Logging and metadata perturbation are charged and not subtracted.'}


def main():
    output = ROOT/'act/pipeline/moe/results/conv_f0_timing_review_20260915_r1.json'
    if output.exists(): raise ValueError('refuse to replace frozen derived review')
    checked = audit(DEFAULT)
    if checked != read(DEFAULT/'audit.final.json') or checked != read(DEFAULT/'audit.review.json'):
        raise ValueError('automatic/separate/fresh audits differ')
    old = read(ROOT/'act/pipeline/moe/results/conv_three_arm_smoke_review_20260915_r1.json')
    for item in old['artifact_inventory']:
        path = Path(old['raw_root'])/item['path']
        if path.stat().st_size != item['bytes'] or _sha256(path) != item['sha256']:
            raise ValueError('old frozen smoke artifact changed')
    runtime = read(DEFAULT/'runtime.json')
    result = {'schema':'conv_f0_timing_review_r1', 'execution_head':runtime['git_head'],
        'raw_root':str(DEFAULT), 'status':'PASS', 'issues':[],
        'independent_reaudit_exact_match':True, 'old_smoke_artifacts_unchanged':len(old['artifact_inventory']),
        'old_smoke_gate':'FAIL_UNCHANGED', 'full_started':False,
        'row':checked['row'], 'detail':checked['detail'], 'timing':summarize(checked),
        'execution_sources':runtime['sources'],
        'artifact_inventory':[{'path':str(p.relative_to(DEFAULT)), 'bytes':p.stat().st_size, 'sha256':_sha256(p)}
                              for p in sorted(DEFAULT.rglob('*')) if p.is_file()],
        'next_decision':'No automatic rerun. Evidence supports a separately scoped remaining-budget/partial-terminal engineering repair; not a promise of stronger bounds or new certificates.'}
    result['archival_note'] = ('First unpublished derivation read exact from the tuple-valued dispatch wrapper '
        'and counted missing as false. Preserved locally as archive.first_draft.json; corrected from nested '
        'hz_support_bounds results. No timing, terminal, trace or numerical result changed.')
    atomic_json(output, result)
    print(json.dumps(result['timing'], indent=2))


if __name__=='__main__':
    main()
