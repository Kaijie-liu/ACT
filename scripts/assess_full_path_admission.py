"""Read-only cross-contract admission assessment; no runtime module imports.

AST anchors are change detectors for a reviewed derivation, not a Python proof.
No model, source, candidate, solver, supervisor, allocation probe or GPU is run.
"""
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'docs/hz_full_path_decision_20261016.json'
SHAPE = 'docs/hz_model_scaled_resources_20261013.json'
WORK = 'docs/hz_shared_execution_envelope_20261015.json'
PAGED = 'docs/hz_paged_execution_resources_20261015.json'
PROPOSAL = 'docs/hz_staged_memory_approval_20261015.json'
NATIVE = 'act/back_end/moe/scaled_forest_endpoints.py'
FILES = (SHAPE, WORK, PAGED, PROPOSAL, NATIVE,
    'scoped_source/direct_supervised.py', 'scoped_source/direct_method.py',
    'scoped_source/direct_lifecycle.py', 'scoped_source/direct_lifecycle_worker.py',
    'scoped_source/direct_lifecycle_execution.py', 'scoped_source/shared_archive.py',
    'scoped_source/shared_source.py', 'scoped_source/shared_build.py',
    'scoped_source/check_scaled_references.py', 'scoped_source/direct_candidates.py',
    'scoped_source/scaled_endpoints.py', 'scoped_source/tensor_rows.py',
    'scoped_source/capture.py', 'act/back_end/moe/factory.py',
    'docs/hz_source_lifetime_protocol_20261014.json',
    'scripts/assess_full_path_admission.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(name):
    return json.loads((ROOT / name).read_bytes())


def function(name, fn, cls=None):
    tree = ast.parse((ROOT / name).read_text())
    if cls:
        tree = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)


def expressions(node):
    return {ast.unparse(n) for n in ast.walk(node)}


def analyze():
    shape, work, paged, proposal = map(load, (SHAPE, WORK, PAGED, PROPOSAL))
    v = shape['inventory']
    require(work['inventory'] == v, 'same registered shape')
    require(work['unchanged'] == paged['unchanged'], 'same policy')
    require(proposal['active_as_bytes'] is None and not proposal['approval_granted'],
            'no AS activation or implementation approval')
    require((v['pairs'], v['property_duties'], v['endpoints_upper'], v['candidates_upper']) ==
            (28, 252, 504, 1008), 'full conservative roster')
    for row in work['consumers'].values():
        require(row['native_proposer_upper'] is None, 'native upper not established')
    spec = expressions(function('scoped_source/direct_supervised.py', 'spec'))
    require("case in ('half', 'guarded')" in spec, 'fixture-only source')
    method = expressions(function('scoped_source/direct_method.py', 'method'))
    require("dict(kind='FIXED_SCALAR_STUB_V1', profile=profile)" in method, 'fixed producer')
    life = expressions(function('scoped_source/direct_lifecycle.py', '__init__', 'Lifecycle'))
    require('FixedProposer(config, fault)' in life, 'no native dispatch in reviewed lifecycle')
    native = expressions(function(NATIVE, '__init__', 'Operator'))
    for term in ('rr = []', 'cc = []', 'vv = []', 'rr.append(n)', 'cc.append(j)',
                 'vv.append(float(q))', 'self.ix([rr, cc])', 'self.val(vv)',
                 "torch.sparse_coo_tensor(self.ix([rr, cc]), self.val(vv), self.shape, device='cpu')"):
        require(term in native, 'native construction changed: ' + term)
    require(sys.implementation.name == 'cpython' and struct.calcsize('P') == 8,
            'reviewed 64-bit CPython pointer-slot assumption')

    # Conditional prospective preparation, NOT observed checkpoint sparsity.
    # rr/cc/vv list slots and independent int64/float64 tensor inputs coexist
    # at the sparse_coo_tensor call. Exclude all Python value/header/allocator,
    # checked graph, library, coalesce and transpose costs from this lower bound.
    local_min = v['inputs'] + v['first_weight_positions']
    list_slots = 3 * 8 * local_min
    tensor_inputs = (2 * 8 + 8) * local_min
    native_lower = list_slots + tensor_inputs
    cap = proposal['proposed_as_bytes']
    slot = work['dense_owned_arena']['slot_bytes']
    dense_lower = 2 * 8 * slot * (v['inputs'] + v['affine_outputs'] + v['guards'] + v['inputs'])
    require(dense_lower == work['dense_owned_arena']['complete_width8_vector_lower'], 'dense lower')
    require(native_lower == 321011712 and dense_lower == 322734720, 'independent arithmetic')
    require(native_lower > cap and dense_lower > cap, 'no full-size inheritance of proposed AS')
    source_floor = 2 * v['parameter_values']
    require(source_floor == 13922736 and source_floor > work['unchanged']['work'], 'source floor')
    return dict(schema='FULL_PATH_CROSS_CONTRACT_DECISION_V1',
        base_head='e6b01e8db0e3db51ed1284f17363a90c7cdf01e0',
        anchors={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in FILES},
        roster={k:v[k] for k in ('pairs','property_duties','endpoints_upper','candidates_upper')},
        reviewed_entry_facts=dict(cases=['half','guarded'], producer='FIXED_SCALAR_STUB_V1',
            shared_forest_accepted_by_native_type_gate=True,
            native_direct_supervisor_dispatch=False, native_produce_uses_old_json_session=True,
            direct_supervised_trained_artifact_source_adapter=False,
            reviewed_native_operator_gpu_path=False,
            type_compatibility_is_not_executed_integration=True),
        proposed_as=dict(bytes=cap, active=False, inherited_by_full_model=False),
        native_construction=dict(condition='all first-affine weights nonzero; 64-bit CPython; existing CPU Operator; torch.tensor copies inputs',
            minimum_local_entries=local_min, python_list_pointer_bytes_lower=list_slots,
            tensor_input_payload_bytes_lower=tensor_inputs, simultaneous_payload_lower=native_lower,
            excess_over_proposed_as=native_lower-cap, actual_checkpoint_nnz=None,
            assumes_other_admission_failures_already_removed=True,
            native_calls_executed=0, full_peak_upper=None,
            additional_shared_setup_source_visits=v['first_weight_positions']),
        dense_consumption=dict(condition='existing width-8 complete all-present buffer schedule',
            payload_lower=dense_lower, excess_over_proposed_as=dense_lower-cap,
            actual_full_peak=None, hypothetical_ownership_only_lower=212116800,
            hypothetical_ownership_only_implemented=False),
        unchanged_refusals=dict(two_source_checks_lower=source_floor,
            work_cap=work['unchanged']['work'], memberships_lower=v['mandatory_roots'],
            memberships_cap=work['unchanged']['memberships'], arena_cap=work['unchanged']['arena']),
        inherited_postcapture_fixed_proposer_upper=paged['counted_work']['postcapture_upper'],
        inherited_archive_upper=paged['conditional_success_archive'],
        full_request=dict(capture_work_upper=None, native_proposer_work_upper=None,
            full_heap_upper=None, full_rss_upper=None, new_resource_policy=None,
            checkpoint_correspondence_checked=False, native_float_proof=False),
        decision='STOP_STUB_SUPERVISOR_AS_DEFAULT_NEXT_STEP_FULL_SOURCE_NATIVE_POLICY_DECISION_REQUIRED',
        runtime_changes=0, source_builds=0, solver_calls=0, worker_runs=0, gpu_runs=0,
        full_capacity_admitted=False, scientific_gates_closed=[])


if __name__ == '__main__':
    value = analyze()
    if sys.argv[1:] == ['--check']:
        require(load(REPORT) == value, 'decision record drift')
        print('PASS: static cross-contract identities/arithmetic; no runtime execution')
    else:
        require(not sys.argv[1:], 'only --check supported')
        print(json.dumps(value, indent=2, sort_keys=True))
