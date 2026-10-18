"""Read-only structural storage/lifetime accounting, not a capacity experiment.

Uses source AST/text, small frozen JSON records and integer arithmetic only.
No production imports, source construction, native tensors, actor or GPU calls.
Numerical values bound named payloads, NOT the complete Python/native heap.
"""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'docs/hz_native_storage_lifetimes_20261018.json'
BASE = '91b661ec55741a1bd07a87a4295d927b2d801f55'
COUNTED = 'docs/hz_native_counted_envelope_20261018.json'
ARENA = 'docs/hz_pair_scoped_capacity_20261015.json'
PAGED = 'docs/hz_paged_execution_resources_20261015.json'
ARENA_BASE = 'd2c7a6f68184c7773ba3f2496db03398a9156ff4'
FILES = (COUNTED, ARENA, PAGED,
    'scoped_source/native_supervised.py', 'scoped_source/native_protocol.py',
    'scoped_source/native_production.py', 'scoped_source/native_package.py',
    'scoped_source/native_lifecycle.py', 'scoped_source/native_source.py',
    'scoped_source/native_result.py', 'scoped_source/shared_source.py',
    'scoped_source/shared_build.py', 'scoped_source/shared_format.py',
    'scoped_source/shared_archive.py', 'scoped_source/check_shared_source.py',
    'scoped_source/check_direct_source.py', 'scoped_source/tensor_rows.py',
    'scoped_source/direct_supervised.py', 'scoped_source/wire_archive.py',
    'scoped_source/pair_workspace.py', 'scoped_source/bounded_runtime.py',
    'scoped_source/direct_candidates.py', 'scoped_source/scaled_endpoints.py',
    'scoped_source/result_stream.py', 'scoped_source/check_result_stream.py',
    'scoped_source/direct_lifecycle_package.py', 'scoped_source/direct_package.py',
    'scoped_proof/owned_bounded.py', 'scripts/run_native_controls.py',
    'scripts/audit_native_controls.py',
    'act/back_end/moe/scaled_forest_endpoints.py',
    'act/back_end/moe/source_row_endpoints.py',
    'scripts/assess_native_storage_lifetimes.py')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(name):
    return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT/name).read_bytes())


def assignment(file, name):
    tree = ast.parse((ROOT/file).read_bytes())
    return next(n.value for n in tree.body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))


def code_files():
    # Read the actual closed list expressions; never import packaged modules.
    a = assignment('scoped_source/direct_package.py', 'CODE_FILES')
    gen = a.left.args[0]
    require(ast.unparse(gen.elt) == "'scoped_source/' + name + '.py'", 'closed code names')
    base = tuple('scoped_source/'+n+'.py' for n in ast.literal_eval(gen.generators[0].iter))
    base += ast.literal_eval(a.right)
    f = 'scoped_source/direct_lifecycle_package.py'
    base += ast.literal_eval(assignment(f, 'CODE_FILES').right)
    shared = ast.literal_eval(assignment(f, 'SHARED_CODE'))
    paged = ast.literal_eval(assignment(f, 'PAGED_CODE'))
    expr = assignment('scoped_source/native_package.py', 'CODE_FILES')
    require(ast.unparse(expr).startswith('tuple(sorted(set(BASE_CODE + SHARED_CODE + PAGED_CODE + '),
            'native code roster expression')
    extra = ast.literal_eval(expr.args[0].args[0].args[0].right)
    result = tuple(sorted(set(base+shared+paged+extra)))
    require(len(result) == 32 and all(p.isascii() for p in result), '32 fixed ASCII checker files')
    return result


def compact(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def census_length(files):
    n = len(files)
    require(n > 0 and all(k.isascii() and '"' not in k and '\\' not in k for k in files),
            'nonempty generated safe ASCII roster')
    return (31+len(str(sum(files.values())))+2*sum(map(len, files))
            +sum(len(str(v)) for v in files.values())+74*n)


def arena_equivalence():
    spec = {'scoped_source/bounded_runtime.py': ('scalar_size','Owner','Block','ExactMap'),
        'scoped_source/pair_workspace.py': ('Scope','ScopedMap','SparseRoots',
            'PairWorkspace.__init__','PairWorkspace.mapping','PairWorkspace.pullback',
            'PairWorkspace.clear_block','PairWorkspace.clear_tile','PairWorkspace.close')}
    def item(tree, name):
        for part in name.split('.'):
            tree = next(n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))
                        and n.name == part)
        return ast.dump(tree,include_attributes=False)
    for file, names in spec.items():
        before = ast.parse(subprocess.check_output(['git','show',ARENA_BASE+':'+file],cwd=ROOT))
        after = ast.parse((ROOT/file).read_bytes())
        for name in names:
            require(item(before,name) == item(after,name), 'arena component changed: '+file+':'+name)
    # RootTable now interns shared membership buffers. R=16N+4H still bounds
    # its payload (no more than one buffer per old roster). The extra Python
    # dictionary is NOT part of that payload and remains in the heap ledger.
    # shared_objective returns the same <=4-root SparseRoots descriptor.
    return {file: list(names) for file, names in spec.items()}


def analyze():
    old, arena = load(COUNTED), load(ARENA)
    v = old['inventory']
    require(v == arena['inventory'], 'same registered topology')
    equivalent = arena_equivalence()
    for n, h in old['anchors'].items():
        require(sha(n) == h, 'counted current implementation drift: '+n)
    code = code_files()
    code_names = ['artifacts/package/code/'+n for n in code]
    require(sum(map(len, code_names)) == 1758 and max(map(len, code_names)) == 64,
            'current checker relative-name lengths')
    # Small record algebra differential only; no source or evidence generated.
    for sizes in ({'x':0}, {'a/b':17, 'test.json':128}, {'zero':0, 'z':99999, 'q/r':2**27}):
        value = dict(bytes=sum(sizes.values()), files=sizes,
                     hashes={k:'0'*64 for k in sizes})
        require(census_length(sizes) == len(compact(value)), 'exact census length formula')
    n, N, F, L, R, W1, P, H, M = (v[k] for k in (
        'inputs','nodes_upper','factors_upper','local_entries_upper','references_upper',
        'first_weight_positions','parameter_values','roots_upper','pair_dual_rows_upper'))
    S = v['router_widths'][1]+v['experts']*v['expert_widths'][1]
    Q = L-W1
    frame = 128+8*(2152+16*M)
    roots = 16*N+4*H
    scope = 4*(N+F+7592+9400)
    descriptors = 8*4*(4+2064)
    exact = roots+scope+descriptors+frame+2*2065*(7592+2*9400)
    require(exact == 110559088 and frame == 362944, 'unchanged conditional exact arena')
    best_dual = 8*8*M
    base = roots+scope+descriptors
    iteration = 16*M+64*(4*N+3*F+5*M+2*R+4)+8*(N+F+M+1)
    operators = {}
    for arm, local, bank in (('native_coo_reference', L, 0),
                            ('native_source_row_blocks', Q, 8*W1+8*S)):
        payload = 48*local+24*R+8*F+bank
        operators[arm] = dict(local_coo_entries_upper=local,
            rr_cc_vv_list_length_each_upper=local,
            local_and_transpose_payload_upper=48*local,
            reference_indices_and_values_upper=24*R,scales_upper=8*F,
            source_bank_and_node_indices_upper=bank,
            retained_named_tensor_payload_upper=payload,
            during_exact_named_payload_subtotal=payload+exact+best_dual,
            subtotal_including_retained_source_row_bytes=payload+exact+best_dual+8*(W1+S),
            full_setup_heap_upper=None,full_iteration_heap_upper=None)
        block_arm = arm == 'native_source_row_blocks'
        named_setup = (72*local+24*R+8*F+bank+(8*n if block_arm else 0)
                       +local+R+F+(256*n+n if block_arm else 0))
        named_iteration = iteration+(64*(n+256) if block_arm else 0)
        operators[arm].update(setup_named_tensors_upper=named_setup,
            optimizer_named_arrays_additional_upper=named_iteration,
            during_optimizer_named_payload_subtotal=payload+base+named_iteration,
            current_row_transcode_tensor_upper=8*n if block_arm else 0,
            candidate_seal_named_payload_subtotal=payload+base+2*frame+best_dual)
    require(operators['native_source_row_blocks']['retained_named_tensor_payload_upper'] == 62712192,
            'include source block node-index payload')
    cap = 2**27
    W = max(r['declared_complete_traversal_counted_upper'] for r in old['arms'].values())
    widths = {}
    for name, w in (('current_success_cap',8000000), ('conditional_traversal_not_admitted',W)):
        widths[name] = dict(work=w, work_digits=len(str(w)),
            owner_cumulative_bytes_upper=cap*w, owner_cumulative_bytes_digits=len(str(cap*w)),
            owner_allocations_upper=w, owner_live_peak_upper=cap,
            progress_all_three_actors_upper=(v['pair_local_tiles_upper']+20)*(137+3*len(str(w))),
            production_phases_json_upper=1+(v['pair_local_tiles_upper']+15)*(140+3*len(str(w))))
    return dict(schema='NATIVE_STORAGE_LIFETIME_STATIC_V1',based_on=BASE,
        anchors={p:sha(p) for p in FILES},code_files=list(code),
        arena_unchanged_ast_components=equivalent,
        arena_reviewed_differences=['RootTable shared membership interning only reduces buffer count; Python dictionary extra',
            'shared objective uses same <=4-root SparseRoots, not a new dense vector'],
        scope='registered well-formed declared-source success traversal; not actual heap or current admission',
        inventory=v,
        source_payload=dict(whole_snapshot_parameter_bytes=8*P,
            postcheck_retained_first_rows_bytes=8*(W1+S),
            forest_binary_payload_upper=36+32*N+12*(Q+R)+4*H+8*F,
            forest_pages_payload_given_admitted_header_upper=36+8*2**20+32*N+12*(Q+R)+4*H+8*F,
            snapshots_simultaneously_required=1,source_actor_execution_sequential=True,
            checked_forest_fraction_positions_upper=Q+R+2*N+F,
            checked_forest_work_bit_norm_positions_upper=N,
            additional_checker_local_norm_positions_upper=N,
            remaining_fraction_positions_binary64_bits_upper=1075,
            checker_visibility_memberships_upper=v['views']*F,
            plan_pullback_fraction_positions_upper=N+2*F,
            distinct_scale_groups_upper=n+1,
            pinned_manifest_parts_count_upper=4*2**20,
            pinned_manifest_join_payload_overlap_upper=8*2**20,
            input_literal_parser_is_not_covered_by_work_bit_gate=True,
            retained_endpoint_receipt_records_upper=504,
            retained_endpoint_final_values_upper=504,
            per_row_fraction_values_upper=2*(n+1)),
        operators=operators,exact_arena_payload_upper=exact,best_dual_payload_upper=best_dual,
        replay_exact_plus_external_frame_upper=exact+frame,
        archived_file_topology=dict(symbols={'Ns':'source members including manifest',
            'Nf':'forest binary pages','Nt':'candidate tiles <=84','Nr':'result pages <=32'},
            package_files='nP=Ns+Nf+Nt+Nr+37',
            package_root_relative_name_sum='LP=1915+42*Ns+33*Nf+32*Nt+34*Nr',
            code_root_relative_name_sum=1758,package_code_files=32,
            request_files='nP+13',with_external_caller_receipt='nP+14',
            census_exact='C=31+digits(total)+2*sum(name lengths)+sum(digits(file bytes))+74*n; n>0',
            census_under_original_byte_field_width='C<=40+2*L+83*n',
            persisted_census_counts=['nP+10','nP+11','nP+12'],
            persisted_census_name_sums=['LP+144','LP+158','LP+173'],
            three_disk_censuses_byte_upper='3809+6*LP+249*nP',
            returned_caller_census_count='nP+13',returned_caller_census_name_sum='LP+186',
            caller_receipt_write_time_in_old_batch=True,
            old_request_byte_table_includes_external_caller_receipt=False,
            disk_terminal_is_not_complete_online_acceptance_receipt=True),
        counters_and_closed_metadata=widths,
        invocation_json_bound='362+Pprofile+Qroot; Qroot includes JSON quotes/escapes; four finite clocks <=24 chars',
        same_actor_retained_overlap=[
            'bootstrap raw invocation and decoded profile + actor invocation decode',
            'build Snapshot + builder graph + encoded pages; builder dies before independent check',
            'checker Snapshot + decoded graph + Packet pages; only first rows escape check',
            'native setup Python rr/cc/vv and edges + constructed tensors',
            'exact workspace + native operator + best DualTile (optimizer stack has returned)',
            'owned source census + initial census + Archive entries/prefixes/events/attempted',
            'read_json raw bytes + decoded objects + canonical encoding strings/bytes',
            'pinned manifest short-read parts list plus joined raw; each positive read may be one byte',
            'replay previous/new input frame overlap; no native operator in replay'],
        next_contract_parameters=['exact immutable profile/root encoding lengths',
            'source metadata/layout and forest-header demand binding (not old 95-chunk assumption)',
            'cleanup PID/zombie count and integer widths, not live-empty implies zombie-empty',
            'CPython container/allocation/temporary lifetime contract for named slots',
            'complete generated report schema lengths and external caller receipt scope'],
        unknown=dict(full_success_archive_bytes=None,full_self_owned_heap_bytes=None,
            native_internal_workspace_bytes=None,full_RSS_bytes=None,completion_seconds=None,
            arbitrary_failure_heap_bytes=None,live_checkpoint_capture_bytes=None),
        policy=dict(work=8000000,artifact_bytes=cap,seconds=300,changed=False,
            new_quota=None,execution_approval=False),
        source_builds=0,native_calls=0,model_runs=0,gpu_runs=0,
        decision='NAMED_STORAGE_AND_OVERLAP_RECONCILED_JOINT_METADATA_HEAP_CONTRACT_REQUIRED')


if __name__ == '__main__':
    result = analyze()
    if sys.argv[1:] == ['--check']:
        require(load(REPORT) == result, 'storage/lifetime record drift')
        print('PASS: source identities, named storage and census algebra; no runtime execution')
    else:
        require(not sys.argv[1:], 'only --check is supported')
        print(json.dumps(result,indent=2,sort_keys=True))
