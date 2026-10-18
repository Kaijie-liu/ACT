"""Static counted-Work reconciliation of the implemented native CPU join.

Only source/JSON/git-object reads, AST comparison and integer arithmetic. Never
imports production code, allocates a source, launches an actor or sets a quota.
The output is conditional complete-traversal demand, not time/RSS or admission.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'docs/hz_native_counted_envelope_20261018.json'
PARENT = 'docs/hz_shared_execution_envelope_20261015.json'
OLD = '82411e0b4045a7015b732bd549321ccffab3544d'
BASE = 'd370e98f8ada51429c0bf55aa707344590c16a1a'
UNCHANGED = tuple('scoped_source/'+n+'.py' for n in (
    'shared_source', 'shared_build', 'shared_format', 'check_shared_source',
    'check_direct_source', 'check_source_dag', 'check_scaled_references',
    'check_affine_references', 'affine_references', 'scaled_references',
    'source_dag', 'tensor_rows', 'direct_source_format', 'hz_store_source',
    'hz_request_store', 'bounded_runtime', 'pair_workspace', 'forest_endpoints',
    'wire_forest', 'shared_archive', 'wire_archive', 'wire_package'))
WRAPPERS = tuple('scoped_source/'+n+'.py' for n in (
    'native_production', 'native_package', 'native_result', 'native_protocol',
    'native_supervised', 'native_loader', 'native_method', 'native_portable',
    'result_stream', 'check_result_stream', 'wire_archive', 'wire_package',
    'shared_archive', 'direct_supervised'))
FILES = tuple(sorted(set(UNCHANGED+WRAPPERS+(
    PARENT, 'docs/hz_model_scaled_resources_20261013.json',
    'docs/hz_native_full_shape_decision_20261018.json',
    'docs/hz_paged_result_controls_20261015.json',
    'scoped_source/native_source.py', 'scoped_source/native_lifecycle.py',
    'scoped_source/direct_candidates.py', 'scoped_source/direct_lifecycle.py',
    'scoped_source/scaled_endpoints.py',
    'act/back_end/moe/scaled_forest_endpoints.py',
    'act/back_end/moe/source_row_endpoints.py',
    'scripts/assess_native_counted_envelope.py'))))


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT/name).read_bytes())


def counters(rows, coefficients, arithmetic=0):
    return dict(row_visits=rows, coefficient_visits=coefficients, arithmetic=arithmetic)


def summed(*parts):
    return {k:sum(p[k] for p in parts) for k in counters(0, 0)}


def named(tree, path):
    node = tree
    for part in path.split('.'):
        node = next(n for n in node.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == part)
    return ast.dump(node, include_attributes=False)


def consumer_equivalence():
    # Git is read-only here. Finish/sink changes are reviewed separately below;
    # all actual preparation, pullback and exact consume bodies must match.
    spec = {
        'scoped_source/direct_candidates.py': ('Tile', 'Driver.__init__', 'Driver.prepare',
            'Driver.receive', 'Driver._consume_raw', 'Driver._release', 'Driver.abort'),
        'scoped_source/scaled_endpoints.py': ('pullback', 'weighted_norm', 'contract', '_plan',
            'Session.__init__', 'Session.prepare', 'Session._check', 'Session._consume',
            'Session._consume_block', 'Session.accept_direct', 'Session.abort')}
    result = {}
    for file, names in spec.items():
        raw = subprocess.check_output(['git', 'show', OLD+':'+file], cwd=ROOT)
        old, new = ast.parse(raw), ast.parse((ROOT/file).read_bytes())
        for name in names:
            require(named(old, name) == named(new, name), 'inherited consumer changed: '+file+':'+name)
        result[file] = list(names)
    finish = ast.parse((ROOT/'scoped_source/scaled_endpoints.py').read_bytes())
    cls = next(n for n in finish.body if isinstance(n, ast.ClassDef) and n.name == 'Session')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'finish')
    calls = [ast.unparse(n) for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'visit']
    require(calls == ["work.visit('row_visits', 9)", "work.visit('row_visits')", "work.visit('row_visits')"],
            'finish counted visits changed')
    return result


def analyze():
    old = load(PARENT); v = load('docs/hz_model_scaled_resources_20261013.json')['inventory']
    require(v == old['inventory'], 'registered inventory')
    for name in UNCHANGED:
        require(sha(name) == old['anchors'][name], 'inherited source/consumer drift: '+name)
    equivalent = consumer_equivalence()
    for name in WRAPPERS:
        tree = ast.parse((ROOT/name).read_bytes())
        require(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr in ('visit','exact') for n in ast.walk(tree)),
            'new wrapper counter requires reconciliation: '+name)
    # This AST observation is not a proof about transitive calls. Source,
    # Driver/Session, native operator, lease, frame and intake are counted below.
    source = old['source_counted_work']['build_and_two_checks_upper']
    consumer = old['consumers']['pair_scoped']
    require(source == 216424736 and consumer['consumer_core_upper'] == 82442004780,
            'reviewed inherited conditional upper')
    n,a,u = (v[k] for k in ('inputs','affine_outputs','relu_units'))
    N,F,L,R = (v[k] for k in ('nodes_upper','factors_upper','local_entries_upper','references_upper'))
    E,T,M = (v[k] for k in ('endpoints_upper','pair_local_tiles_upper','pair_dual_rows_upper'))
    W1 = v['first_weight_positions']; S = v['router_widths'][1]+v['experts']*v['expert_widths'][1]
    B,J = 1+v['experts'],2; Q = n+2*a+12*u
    require((E,T,M,S,B,J)==(504,84,2700,2176,9,2) and Q+W1==L, 'registered native roster/depth')
    lease = counters(6*E,22*E)
    frame = counters(2*T+2*E,T+E+4*E*M)
    replay_frame = counters(2*T+E,T)
    require(sum(replay_frame.values()) == consumer['replay_frame_upper'], 'replay frame inherited')
    bytes_upper = v['raw_binary64_tensor_bytes']+4*2**20
    members_upper = 4096-8
    # Every successful read makes positive byte progress, but may be short.
    intake = counters(members_upper,bytes_upper)
    arms = {}
    common = source+2*consumer['consumer_core_upper']+sum(replay_frame.values())
    for name, blocks in (('native_coo_reference',False),('native_source_row_blocks',True)):
        setup = counters(N+n+S if blocks else N,
                         Q+R+F+2*W1 if blocks else L+R+F+W1)
        optimize = counters(E+T*(M+129+257*(J+(B if blocks else 0))),4*E+T*M)
        counted = summed(setup,optimize,lease,frame)
        post = common+sum(counted.values())
        require(post == consumer['postcapture_counted_work_upper']-consumer['fixed_proposer_upper']
                -consumer['production_frame_upper']+sum(counted.values()), 'replace, never double-add')
        arms[name] = dict(setup_upper=setup,optimize_upper=optimize,root_lease_upper=lease,
            owned_frame_upper=frame,native_join_upper=counted,native_join_total=sum(counted.values()),
            post_acquisition_counted_upper=post,
            declared_complete_traversal_counted_upper=post+sum(intake.values()))
    totals = [r['declared_complete_traversal_counted_upper'] for r in arms.values()]
    require(totals == [165180064388,165180263928], 'reviewed totals')
    return dict(schema='NATIVE_COUNTED_ENVELOPE_STATIC_V1',based_on=BASE,
        anchors={f:sha(f) for f in FILES},inherited_exact_ast_comparisons=equivalent,
        inherited_unchanged_files=list(UNCHANGED),wrapper_direct_counter_scan=list(WRAPPERS),
        scope='well-formed already-materialized registered seed1/2 declared artifact; complete underlying traversal, not current-cap admission',
        inventory=v,source_core_upper=source,consumer_core_per_pass_upper=consumer['consumer_core_upper'],
        consumer_phases_per_pass=consumer['phases'],replay_frame_upper=replay_frame,
        source_seals_already_in_consumer=True,source_chunks_per_seal_upper=consumer['source_chunks_upper'],
        declared_intake=dict(source_bytes_upper=bytes_upper,source_files_upper=members_upper,
            counted_upper=intake,short_reads_may_return_one_byte=True,
            full_64KiB_read_assumption=False,checkpoint_capture_included=False),
        arms=arms,wrapper_direct_counted_increment=0,wrapper_time_or_memory_cost_is_zero=False,
        work_counter_decimal_digits_upper=len(str(max(totals))),
        removed=dict(fixed_producer_upper=consumer['fixed_proposer_upper'],
            generic_production_frame_upper=consumer['production_frame_upper'],
            legacy_native_string_extraction=2*E+E*M,generic_advisory_puts=2*E),
        mathematical_loop_derivations_machine_proved=False,
        inherited_conditions=old['source_counted_work']['assumptions']+
            [s for s in consumer['assumptions'] if not s.startswith('fixed standard-library')],
        open=dict(full_success_archive_byte_upper=None,full_self_owned_live_object_upper=None,
            full_RSS_upper=None,checkpoint_capture_counted_upper=None,completion_time_upper=None,
            missing=['new outer metadata and repeated census schemas',
                'future profile/source paths and encoded-field contracts',
                'success cleanup record including possible zombies',
                'Python/container/JSON/rational transient lifetimes',
                'checkpoint/preprocessing/eval capture correspondence and costs']),
        policy=dict(current_work_cap=8000000,current_artifact_cap=128*2**20,request_seconds=300,
            changed=False,new_quota=None,absolute_ceiling=None,execution_approval=False),
        source_builds=0,native_calls=0,model_runs=0,gpu_runs=0,
        decision='COUNTED_DECLARED_PATH_RECONCILED_FULL_RESOURCE_CONTRACT_STILL_OPEN')


if __name__ == '__main__':
    result = analyze()
    if sys.argv[1:] == ['--check']:
        require(load(REPORT) == result, 'native counted-envelope record drift')
        print('PASS: source bindings, consumer AST and integer reconciliation; no runtime execution')
    else:
        require(not sys.argv[1:], 'only --check is supported')
        print(json.dumps(result,indent=2,sort_keys=True))
