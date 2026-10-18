"""Static-only interface proposal and current-source audit, not a runner.

Imports no production modules, evaluates no profiles, and reads no source/model
artifacts. A passing check means this non-executable draft matches its anchors.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = '61c6782131ef7380eeea606f1b0ef5b517312c6f'
PREVIOUS = 'docs/hz_native_joint_contract_20261018.json'
REPORT = 'docs/hz_native_resource_interface_proposal_20261018.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()


def tree(path):
    return ast.parse((ROOT/path).read_text())


def function(path, name):
    return next(n for n in tree(path).body if isinstance(n, ast.FunctionDef) and n.name == name)


def unbounded_reads(node):
    return sorted((dict(line=n.lineno, expression=ast.unparse(n)) for n in ast.walk(node)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
        and n.func.attr in ('read_bytes','read_text') and not n.args and not n.keywords),
        key=lambda x:x['line'])


def original_policy():
    """Read this anchored literal policy without importing production code."""
    assignment = next(n for n in tree('scoped_source/hz_request_store.py').body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == 'POLICY' for t in n.targets))
    outer = assignment.value
    require(isinstance(outer, ast.Call) and isinstance(outer.func, ast.Name)
            and outer.func.id == 'MappingProxyType' and len(outer.args) == 1
            and not outer.keywords, 'literal immutable policy')
    value = outer.args[0]
    require(isinstance(value, ast.Call) and isinstance(value.func, ast.Name)
            and value.func.id == 'dict' and not value.args, 'literal policy fields')
    def integer(n):
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Mult, ast.Pow)):
            left, right = integer(n.left), integer(n.right)
            if isinstance(n.op, ast.Pow):
                require(left == 2 and 0 <= right <= 30, 'known policy power')
                return left ** right
            return left * right
        raise ValueError('nonliteral policy expression')
    require(all(k.arg is not None for k in value.keywords), 'no policy unpacking')
    policy = {k.arg: integer(k.value) for k in value.keywords}
    require(len(policy) == len(value.keywords) == 15, 'complete original policy')
    return policy


def analyze():
    previous = json.loads((ROOT/PREVIOUS).read_bytes())
    for p,h in previous['anchors'].items():
        require(sha(p)==h, 'reviewed implementation drift: '+p)
    for p,h in previous['environment_anchors'].items():
        require(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h, 'bound environment drift: '+p)
    require(previous['complete_self_owned_heap_upper'] is None
            and previous['complete_archive_failure_upper'] is None, 'do not invent closure')
    protocol = 'scoped_source/native_protocol.py'
    supervisor = 'scoped_source/native_supervised.py'
    owned = 'scoped_proof/owned_bounded.py'
    reads = {}
    for p,n,count,cap in ((protocol,'approval_registry',1,65536),
        (protocol,'closure',1,131072),(protocol,'verify_code',1,131072),
        (owned,'observe',2,None),(owned,'parent_rss',1,None)):
        sites=unbounded_reads(function(p,n)); require(len(sites)==count,'inspected read sites: '+n)
        reads[p+':'+n]=dict(sites=sites,existing_pre_stat_or_related_cap=cap,
            claim='no byte-count argument at this read; pre-stat or later check does not bind read extent under change')
    command=function(supervisor,'command')
    bootstrap=next(n.value.value for n in ast.walk(command) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='bootstrap' for t in n.targets)
        and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))
    sites=unbounded_reads(ast.parse(bootstrap))
    require(len(sites)==2, 'bootstrap invocation/code read sites')
    reads[supervisor+':command.bootstrap']=dict(sites=sites,
        line_scope='embedded bootstrap string',existing_caps=dict(invocation=2**20,code=131072),
        claim='AS covers actor, not a bounded read extent; preserve hash and stable-file checks')
    # Full-path reference remains static. Do not instantiate a TensorRows/forest.
    counted=json.loads((ROOT/'docs/hz_native_counted_envelope_20261018.json').read_bytes())
    inv=counted['inventory']; P=inv['parameter_values']
    policy=original_policy()
    fixed=dict(seconds=300,work=8000000,artifact_bytes=2**27,stored_bits=4096,
        work_bits=8192,expanded_memberships=65536,references=65536,local_nnz=1048576,
        active_factors=16384,tile_bytes=4*2**20,candidate_stream_bytes=16*2**20-8192,
        candidate_width=8,checker_width=2)
    require(policy['visits']==fixed['work'] and policy['total_bytes']==fixed['artifact_bytes']
            and policy['stored_bits']==fixed['stored_bits'] and policy['work_bits']==fixed['work_bits'],
            'common policy and fixed reference agree')
    require(2*P+inv['affine_outputs']==13926024 and 12*P==83536416,'unchanged refusal arithmetic')
    shape=dict(first_snapshot_work_floor=2*P+inv['affine_outputs'],two_parameter_pass_floor=2*P,
        full_mandatory_loop_floor=12*P,expanded_membership_lower=inv['mandatory_roots'],
        expanded_membership_upper=inv['roots_upper'],candidate_stream_upper=22868160,
        unique_roots_not_equal_to_expanded_memberships=True,full_shape_admitted=False)
    require(shape['expanded_membership_lower']>fixed['expanded_memberships']
            and shape['first_snapshot_work_floor']>fixed['work'], 'known refusal must remain')
    objects={
        'common_contract': ['schema/version','original_policy_and_counter_semantics',
            'mathematical_gate_and_obligation_contract','source/ownership_formula_refs',
            'unchanged_300_second_schedule','fixed_algorithm_and_two_arm_difference'],
        'source_layout': ['manifest_raw_length/hash','canonical_source_identity/length',
            'request_identity/encoding','tensor_and_chunk_roster_ref','actual_chunk_count/rank',
            'JSON_lexical_census_with_optional_trust','online_verification_and_cost_scope'],
        'frozen_envelope': ['common_contract_hash','source_layout_hash','method_hash',
            'profile/root/run_id/version/property_bytes_and_hashes','implementation/environment/ABI',
            'no_registry_back_reference'],
        'authority_registry': ['approved_envelope_hashes','separately_approved_scope','revision/hash'],
        'invocation': ['envelope_hash','registry_hash','common_contract_hash','nonce/root',
            'one_t0_and_absolute_deadlines','per_request_identity'],
        'ownership_ledger': ['invocation/source/layout/method','actor/stage',
            'typed_backing_owner/generation','wrapper_and_lease_counts','formula_and_requested_bytes',
            'live/peak/release_state','not_RSS_or_a_mathematical_proof'],
        'observation': ['invocation/pgid/phase','complete/reason','live/zombies/rss',
            'entries_and_bytes_seen','timestamp','incomplete_values_are_not_empty_defaults'],
        'terminal': ['execution_status','original_evidence_status','online_report_received',
            'online_positive','timely_partial_or_absent_artifact_state','cleanup_completeness',
            'cumulative_work_and_cost','outside_caller_receipt'],
    }
    gates=['frozen_authority_and_identity','all_original_source_and_exact_checks',
        'complete_result_record_roster','fresh_replay_and_reception','cumulative_resources',
        'run_status_COMPLETED','leader_reaped','complete_timely_cleanup_observation',
        'observed_live_group_empty','durable_terminal_and_caller_receipt','timely_final_observation']
    blockers=['complete_self_owned_phase_sums','complete_success_and_failure_archive_binding',
        'first_construction/parse_reservation_implementation','bounded_process_observation_implementation',
        'versioned_common_policy_absolute_values_if_later_changed',
        'separate_implementation_and_control_approval','separate_full_capacity/real/GPU_approval']
    # Dependency direction is from an object to its prerequisites; authority is
    # external. A registry must not be embedded in the envelope it approves.
    dependencies={'common_contract':[], 'source_layout':['common_contract'],
        'frozen_envelope':['common_contract','source_layout'],
        'authority_registry':['frozen_envelope'],
        'invocation':['common_contract','frozen_envelope','authority_registry'],
        'ownership_ledger':['invocation'],'observation':['invocation'],
        'terminal':['invocation','ownership_ledger','observation']}
    seen=set()
    def visit(k, stack):
        require(k not in stack,'self-authorizing/cyclic design binding')
        if k in seen:return
        for child in dependencies[k]:visit(child,stack|{k})
        seen.add(k)
    for k in dependencies:visit(k,set())
    require(set(objects)==seen, 'complete protocol object dependency graph')
    extra={PREVIOUS,'docs/hz_native_full_shape_decision_20261018.json',
        'docs/hz_native_counted_envelope_20261018.json','scoped_source/check_result_stream.py',
        'docs/hz_native_resource_interface_proposal_20261018.md',
        'scripts/run_native_controls.py','scripts/check_native_resource_interface_proposal.py'}
    return dict(schema='NATIVE_RESOURCE_INTERFACE_PROPOSAL_V1',based_on=BASE,
        status='DRAFT_NOT_EXECUTABLE',anchors={p:sha(p) for p in sorted(set(previous['anchors'])|extra)},
        current_read_findings=reads,
        current_status_semantics='complete execution/report may contain checked UNKNOWN; never equate with positive proof',
        interface_objects=objects,dependency_graph=dependencies,
        common_original_policy=policy,common_fixed_reference=fixed,
        common_algorithm=dict(device='cpu',dtype='float64',threads=1,updates=128,evaluations=129,
            candidate_profile=0,roles=['proposed','zero'],production='pair_scoped',replay='pair_scoped'),
        arms=[dict(representation=r,common_contract='SAME_RESOLVED_HASH_REQUIRED',
            separately_computed_and_charged=True) for r in ('native_coo_reference','native_source_row_blocks')],
        acceptance=dict(complete_report_requires=gates,
            positive_additionally_requires='original complete positive-bound aggregation and declared guarantee scope',
            cleanup=dict(original_gate='LEADER_REAPED_NO_LIVE_GROUP',
                complete_observation_alone_sufficient=False,
                zombies='fully recorded; orphan zombies are not claimed reaped'),
            complete_unknown_allowed=True,missing_transport_record_allowed=False,
            failure_latched=True,retry_allowed=False,late_upgrade_allowed=False),
        cost=dict(start='before bounded envelope intake',finish='after caller receipt and final deadline observation',
            admission_wait='bounded and separate',request_specific_precomputation_free=False,
            setup_and_all_failed_paths_charged=True,library_unknowns_not_zero=True,
            artifact_cap_is_not_heap_cap=True,project_objects_not_native_scratch=True),
        full_shape=shape,blocking_requirements=blockers,
        future_control_categories=['identity/alias/envelope','changing_files/bootstrap/registry',
            'encoding_and_partial_write','incomplete_process_observation',
            'actor_deadline/exception/partial_evidence','terminal/caller_cost'],
        control_roster=None,control_attempts_authorized=0,
        resolved_bindings=dict(profile=None,source_layout=None,implementation=None,resource_policy=None,
            own_heap_limit=None,OS_limit_authority=None,run_identity=None),
        permissions=dict(implementation=False,quota_change=False,controls=False,
            native_execution=False,models=False,GPU=False),
        execution_approval=None,full_self_owned_heap_upper=None,
        source_builds=0,native_calls=0,model_runs=0,gpu_runs=0,
        decision='ONE_JOINT_INTERFACE_SCOPE_REQUIRES_APPROVAL_NO_FULL_SHAPE_OR_SCIENTIFIC_UPGRADE')


if __name__=='__main__':
    result=analyze()
    if sys.argv[1:]==['--check']:
        require(result==json.loads((ROOT/REPORT).read_bytes()),'interface draft/anchor drift')
        print('PASS: non-executable interface draft, read-site audit and binding DAG; no runtime controls')
    else:
        require(not sys.argv[1:],'only --check is supported')
        print(json.dumps(result,sort_keys=True,indent=2))
