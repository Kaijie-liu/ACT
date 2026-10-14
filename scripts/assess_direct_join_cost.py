"""Read-only static accounting for the direct join at 8b22bcd9bd.

No runtime source import, tensor load, candidate generation or large allocation.
Successful-path bounds are conditional; existing gates refuse earlier. Source
sub-bounds must not be used as a complete Work/RSS or new quota contract.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
FILES = (
    'scoped_source/direct_source_format.py', 'scoped_source/direct_source_build.py',
    'scoped_source/check_direct_source.py', 'scoped_source/direct_execution.py',
    'scoped_source/direct_package.py', 'scoped_source/direct_candidates.py',
    'scoped_source/source_dag.py', 'scoped_source/affine_references.py',
    'scoped_source/scaled_references.py', 'scoped_source/check_source_dag.py',
    'scoped_source/check_affine_references.py', 'scoped_source/check_scaled_references.py',
    'scoped_source/scaled_endpoints.py', 'scoped_source/forest_endpoints.py',
    'scoped_source/bounded_runtime.py', 'scoped_source/pair_workspace.py',
    'scoped_source/wire_archive.py', 'scoped_source/wire_package.py',
    'scoped_source/wire_forest.py', 'scoped_source/tensor_rows.py',
    'scoped_source/factored_io.py', 'scoped_source/hz_request_store.py',
    'docs/hz_full_resource_ledger_20261014.json',
    'docs/hz_pair_scoped_capacity_20261015.json',
    'docs/hz_direct_join_controls_20261015.json',
)


def sizes():
    result = {}
    for node in ast.parse((ROOT / FILES[0]).read_bytes()).body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if not (isinstance(call.func, ast.Attribute) and call.func.attr == 'Struct'
                and isinstance(call.func.value, ast.Name) and call.func.value.id == 'struct'):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result[target.id] = struct.calcsize(ast.literal_eval(call.args[0]))
    assert result == dict(PREFIX=36, NODE=20, ENTRY=12, ROOT=12, INDEX=4, SCALAR=8)
    return result


def analyze():
    parent = json.loads((ROOT / 'docs/hz_full_resource_ledger_20261014.json').read_bytes())
    capacity = json.loads((ROOT / 'docs/hz_pair_scoped_capacity_20261015.json').read_bytes())
    v = parent['inventory']
    assert (v['inputs'], v['experts'], v['classes'], v['pairs']) == (3072, 8, 10, 28)
    assert v == capacity['inventory']
    n, f, h = v['nodes_upper'], v['factors_upper'], v['roots_upper']
    l, r = v['local_entries_upper'], v['references_upper']
    w1 = v['first_weight_positions']
    layout = sizes()

    def extent(header, nodes, local, refs, uses, factors):
        assert min(header, nodes, local, refs, uses, factors) >= 0
        return 36 + header + 32*nodes + 12*(local+refs) + 4*uses + 8*factors

    # Existing math stays unchanged; direct encoder/intake add scans.
    phases = parent['methods']['proposed']['phases']
    build = dict(phases['source_build']['counts'])
    build['row_visits'] += 2*h + 2*n
    build['coefficient_visits'] += l + r + f
    check = dict(phases['source_check']['counts'])
    check['row_visits'] += 3*n
    check['coefficient_visits'] += l + r
    source_upper = {k: build[k] + 2*check[k] for k in build}
    source_floor = 3*w1
    consumer_floor = 2*capacity['support_work_floor']['pair_scoped']['work_lower']
    tensors = 8*v['source_values_with_center']
    assert tensors == v['raw_binary64_tensor_bytes']
    dense_subcase = tensors + 12*w1
    counts = dict(nodes=n, factors=f, memberships=h, locals=l, references=r)
    return dict(
        schema='DIRECT_JOIN_COST_RECONCILIATION_STATIC_V1', date='2026-10-15',
        based_on='8b22bcd9bd249c9e2199e847415239446a31e17f',
        anchors={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in FILES},
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layout=layout, registered_shape_bounds=counts,
        source_equations_and_format_only=dict(
            kind='CONDITIONAL_STRUCTURAL_UPPER_NOT_FULL_REQUEST',
            direct_build=build, one_direct_source_check=check,
            build_and_two_checks=source_upper, total=sum(source_upper.values()),
            omitted=['source-copy IO', 'two terminal source reseals', 'plans', 'consumers',
                     'candidate callback', 'archive/portable inventory', 'reception/cleanup'],
        ),
        conditional_complete_path_floor=dict(
            source_first_weight_passes=3, source_work=source_floor,
            consumer_passes=2, consumer_work=consumer_floor,
            total=source_floor+consumer_floor, current_limit=8000000,
            observed_work=None, quota_recommendation=None,
            condition='all required candidates present; same Work including online replay',
        ),
        source_format=dict(
            exact_formula='36 + h + 32*N + 12*(L+R) + 4*H + 8*F; unique roots=N',
            registered_sections_upper_excluding_header=extent(0,n,l,r,h,f),
            header_upper=8*2**20,
            admitted_by_current_section_caps_bytes_upper=extent(8*2**20,65536,1048576,65536,65536,32768),
            registered_memberships_min=v['mandatory_roots'], pair_output_memberships=28*3072,
            current_membership_limit=65536, actual_counts=None,
            binary64_references_only=True, first_affine_tensor_alias=False,
        ),
        archive=dict(
            raw_tensor_bytes=tensors, current_limit=128*2**20, reserved_terminal=8192,
            shape_upper_source_plus_forest_excluding_header=tensors+extent(0,n,l,r,h,f),
            dense_first_weights_subcase_lower=dense_subcase,
            dense_subcase_exceeds_cap=dense_subcase>128*2**20,
            dense_subcase_condition='all first-affine weight positions nonzero; not a measured checkpoint',
            actual_bytes=None, complete_archive_upper=None,
            native_candidate_stream_upper=capacity['native_stream']['stream_upper'],
            candidate_stream_limit=capacity['unchanged_caps']['candidates'],
            stream_upper_overflow_proves_actual_overflow=False,
        ),
        io=dict(
            successful_path_source_tensor_read_passes_min=10,
            successful_path_forest_and_candidate_file_read_passes_min=5,
            source_tensor_read_bytes_min=10*tensors,
            total_read_formula_lower='10*T + 5*S + 5*C; metadata/code/center re-reads excluded',
            physical_io_measured=False, file_cache_not_assumed_cold=True,
            aggregate_read_and_hash_byte_counters_present=False,
        ),
        lifetime=dict(
            builder_graph_plus_encoder_bytearray_and_bytes=True,
            production_raw_and_checked_forest_live_during_replay=True,
            production_receipts_plan_last_candidate_live_during_replay=True,
            production_owned_arena_released_before_replay=True,
            final_frame_object_release_in_explicit_cleanup=False,
            returned_elapsed_is_full_outer_wall=False,
            replay_owned_arena_upper_any_admitted_frame=capacity['any_admitted_frame_arena']['owned_upper'],
            complete_rss_upper=None,
        ),
        comparison=dict(direct_full_execution_pair_scoped_only=True,
                        dense_same_candidate_control=True, full_matched_native_reference=False),
        decision='NO_FULL_TOPOLOGY_ADMISSION_NO_QUOTA_FROM_PARTIAL_BOUNDS',
        complete_work_upper=None, complete_cost_closed=False, new_quota=None,
        hard_outer_supervisor_integrated=False, native_candidate_integrated=False,
        runtime_changes=False, source_builds=0, model_runs=0, gpu_runs=0, solver_calls=0,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report', type=Path)
    args = parser.parse_args()
    report = analyze()
    if args.check_report:
        if json.loads(args.check_report.read_bytes()) != report:
            raise ValueError('direct join cost report or source binding drift')
        print('PASS: static direct-join cost reconciliation; no runtime admission')
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
