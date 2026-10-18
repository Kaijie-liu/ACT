"""Static AST/schema arithmetic only; never imports or executes the runtime.

This is not a heap proof, control freeze, performance test or execution permit.
Unknown full capacities stay null. Raw source and prior audit hashes are bound.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = 'cbe6bc07dbb76d742a41c62d64031d50d9f8ba6d'
DOC = 'docs/hz_native_phase_contract_20261018.md'
REPORT = 'docs/hz_native_phase_contract_20261018.json'
PRIOR = 'docs/hz_native_resource_freeze_readiness_20261018.json'
SUP = 'scoped_source/native_resource_supervised.py'
OWN = 'scoped_proof/owned_resource_v2.py'
RES = 'scoped_source/native_resources.py'
H = '0'*64


def require(ok, why):
    if not ok:
        raise ValueError(why)


def raw(name): return (ROOT/name).read_bytes()
def sha(name): return hashlib.sha256(raw(name)).hexdigest()
def load(name): return json.loads(raw(name))
def size(value):
    return len(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii'))


def dict_schema(name, schema):
    found = [n for n in ast.walk(ast.parse(raw(name))) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name) and n.func.id == 'dict'
        and any(k.arg == 'schema' and isinstance(k.value, ast.Constant)
                and k.value.value == schema for k in n.keywords)]
    # Normal return has the guarantee field; its exception fallback is smaller.
    if schema == 'NATIVE_RESOURCE_RETURN_V2':
        found = [n for n in found if any(k.arg == 'storage_complete'
            and not isinstance(k.value, ast.Constant) for k in n.keywords)]
    require(len(found) == 1, 'unique source schema '+schema)
    return found[0]


def constant(call, dynamic, extra=()):
    value = {}
    for key in call.keywords:
        require(key.arg is not None, 'closed kwargs')
        if key.arg in dynamic:
            value[key.arg] = None
        elif key.arg == 'bindings':
            value[key.arg] = {k:H for k in ('envelope','registry','common','layout')}
        else:
            value[key.arg] = ast.literal_eval(key.value)
    for key in extra:
        require(key not in value, 'new final return field')
        value[key] = None
    require(set(dynamic) <= set(value), 'complete dynamic field inventory')
    # Remove precisely the placeholder lengths, retaining punctuation and keys.
    return size(value)-4*len(dynamic)


def analyze():
    prior = load(PRIOR)
    for name, expected in prior['anchors'].items():
        require(sha(name) == expected, 'frozen source changed: '+name)
    require(load('docs/hz_native_resource_execution_approvals.json') ==
        {'schema':'NATIVE_RESOURCE_APPROVALS_V2','envelopes':[]}, 'no V2 execution approval')
    terminal = constant(dict_schema(SUP,'NATIVE_RESOURCE_TERMINAL_V2'),
        ('invocation','pending','observed_at'))
    caller = constant(dict_schema(SUP,'NATIVE_RESOURCE_CALLER_V2'),
        ('invocation','terminal','error','pending_eligible','storage_before_self',
         'started','observed_before_publication','resources'))
    ret = constant(dict_schema(SUP,'NATIVE_RESOURCE_RETURN_V2'),
        ('invocation','terminal_ref','caller_ref','error','storage','storage_complete',
         'result','work','owned_execution','online_report_received','online_positive',
         'started','observed_at','seconds','overrun_seconds','status'),
        extra=('observed_at','seconds','overrun_seconds','status'))
    owned = constant(dict_schema(OWN,'OWNED_RESOURCE_V2'),
        ('status','error','pid','returncode','leader_reaped','cleanup_status','remaining_group',
         'seconds','execution_seconds','cleanup_seconds','sampled_peak_rss','failed_observation',
         'exit_observed_at','cleanup_observed_at','run_deadline','cleanup_deadline'))
    tree = ast.parse(raw(SUP))
    pending_call = next(n.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == 'pending' for t in n.targets))
    pending = constant(pending_call,('status','eligible','error','result','work','storage','owned_execution'))
    receipt = dict(binding=H,role='caller',formula=H,scope='NAMED_REQUESTED_RESERVATIONS_NOT_RSS_OR_HEAP_PROOF',
        requested_live=None,requested_peak=None,acquired=None,released=None)
    q_constant = size(receipt)-16
    constants = dict(terminal=terminal,caller=caller,returned=ret,owned_run=owned,
                    pending=pending,caller_resources=q_constant)
    require(constants == dict(terminal=456,caller=492,returned=658,owned_run=342,
        pending=127,caller_resources=292), 'schema constants')
    summary = dict(status='CHECKED_POSITIVE_CAPTURED_REAL_SOURCE',required=252,positive=252,
        required_endpoints=504,checked_endpoints=504,missing_endpoints=504,
        native_float_proof=False,checkpoint_correspondence_checked=False)
    # Scalar length envelope, not a valid result: using all maxima bounds length.
    require(size(summary) == 222, 'summary scalar envelope')
    require(size(dict(row_visits=8000000,coefficient_visits=8000000,arithmetic=8000000)) == 72,
            'validated Work scalar envelope')
    require(size(dict(bytes=2**20,sha256=H)) == 93, 'publication reference envelope')
    owned_fixed = owned + size('CLEANUP_INCOMPLETE') + 10 + 20 + 11 + 5 + size('LEADER_REAPED_NO_LIVE_GROUP') + 24 + 185
    require(owned_fixed == 646, 'owned run excludes O and eight time lengths')
    t_fixed = terminal+66+pending+20+5+10+222+72+owned_fixed
    c_fixed = caller+66+93+10+5
    a_fixed = ret+66+2*93+10+5+222+72+owned_fixed+10+20
    require((t_fixed,c_fixed,a_fixed) == (1624,666,1895), 'late failure keeps all summaries')
    success_tc = 1491+1600+t_fixed+c_fixed+11*24
    success_tca = success_tc+1705+a_fixed+12*24
    require((success_tc,success_tca) == (5645,9533), 'success names/float-clock substitution')
    # Fixed source topology, not invoking factories or evaluating weight arrays.
    topology = {}
    for case,n,a,u,v,classes,pair_u in (
        ('mixed',2,24,15,19,2,10),('clipped',2,33,9,28,2,7),('guarded',1,18,0,13,5,0)):
        nodes,factors,membership = n+a+4*u+6,n+2*a+7*u,9*n+a+4*u+6
        topology[case] = dict(nodes=nodes,factors=factors,membership=membership,views=v,
            endpoints=6*(classes-1),pair_rows=3*pair_u+2,
            visible_entries=v*factors,planner_maps=[factors,nodes,factors])
    anchors = set(prior['anchors']) | {PRIOR,DOC,'scripts/assess_native_phase_contract.py',
        SUP,OWN,RES,'scoped_source/result_stream.py','scoped_source/scaled_endpoints.py',
        'scoped_source/direct_candidates.py','scoped_source/check_source_dag.py',
        'scoped_source/pair_workspace.py','scoped_source/sparse_build.py',
        'act/back_end/moe/block_support.py'}
    return dict(schema='NATIVE_PHASE_CONTRACT_STATIC_V2',base_head=BASE,
        anchors={n:sha(n) for n in sorted(anchors)},schema_constants=constants,
        encoded=dict(tau='max(24,J(t0),J(t0+299),J(t0+299.5),J(t0+300))',
            owned='O+646+8*tau',terminal='C13+O+1624+9*tau',caller='C14+Q+666+2*tau',
            returned='C15+O+1895+12*tau',summary=222,validated_work=72,reference=93,
            q='292+digits(s+p+c)+digits(s+p+max(m,o+n,t,c))+digits(s+p+6*m+o+n+t+c)+digits(6*m+o+n+t)',
            q_scope='caller finite acquisition chain only, observer leases outside loops',
            metadata_remaining=1703936,success_tc_constant=success_tc,success_tca_constant=success_tca,
            conditions=['actual E/I and slots remain unbound','failure census uses failure names',
                        'return publication is only a contract option','Session construction can raise before try']),
        topology_scope='successfully checked fixed factory topology, not arbitrary malformed intake',
        topology=topology,
        ownership=dict(replay_distinct_profile_trees=5,complete_heap_bytes=None,concrete_slots=None,
            return_persisted=False,transfer_enforced_by_generic_helper=False,
            failure_join='retained failure frames plus new diagnostic objects',
            early_bad_frame_limit=4*2**20,possible_owner_frame_seal_overlap=8*2**20,
            prior_arena_is_not_complete_heap=True),
        decision=dict(status='STOP_CONTROL_FREEZE_AND_RESOURCE_WRAPPER_EXPANSION',
            static_algorithm_research_may_continue=True,full_shape_admitted=False,
            two_parameter_check_floor=13922736,work_cap=8000000,membership_lower=126632,membership_cap=65536,
            next='one explicit relation/support algorithm delta or a separately authorized resource policy decision',
            controls_run=0,native_calls=0,models_run=0,gpu_runs=0,
            caps_changed=False,mathematical_checks_changed=False,scientific_upgrade=False,goals='G1-G6 OPEN'))


if __name__ == '__main__':
    result = analyze()
    if sys.argv[1:] == ['--check']:
        require(result == load(REPORT), 'phase contract record drift')
        print('PASS: static phase/schema contract; execution and full heap remain NOT READY')
    else:
        require(not sys.argv[1:], 'only --check supported')
        print(json.dumps(result,sort_keys=True,indent=2))
