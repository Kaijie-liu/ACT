"""Design-only accounting for a direct, unique-root evidence protocol.

No production imports, source/model construction, candidate calls or cap changes.
Conditional data envelopes are NOT full RSS or completion guarantees.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from scripts.assess_model_scaled_resources import shape_inventory, current_policy
from scripts.analyze_full_resource_ledger import bounds

ROOT = Path(__file__).resolve().parents[1]
MIB = 2**20
ANCHORS = {
    'scripts/analyze_full_resource_ledger.py': 'ae9e11ebf873924e48c7ba89bf42b60aa1968cbf991d839f8b97b67e26ec28fc',
    'scripts/assess_model_scaled_resources.py': '4900b4f3682b27071e8e406c7e5bb955677719183171b13c429f6469999a0789',
    'docs/hz_wire_capacity_20261014.json': '805fd553080207e76aeb17c7efe2b425a8fb35bb519772028453f5de7fe7e3c6',
    'docs/hz_full_resource_ledger_20261014.json': '04c51e4765ada9445dcdd80f63f745459366154037443a68dd565c093acbe9d2',
    'scoped_source/source_dag.py': '38863bafe654242077732bac9d9862d928ceb62eb5a77dcc44991cf2fab934e0',
    'scoped_source/scaled_references.py': '534761cad44cbdf62ed31ee4d999fce0a5bd51aed4461b98d8f986b75d50d75d',
    'scoped_source/check_source_dag.py': '58e5e2158843631c9e6ed324cfeeb2eb38e51df1205d2e14d1ed8ac8ce634458',
    'scoped_source/check_scaled_references.py': '244dc7ee5cf7fe46ab7097b2494dfec64d818e926b4726eb19d59a53c5074f9e',
    'scoped_source/scaled_endpoints.py': '93197c1b329e173b25795c48cf117b7113c2b23cc6958438889a7e912d13e8af',
    'scoped_source/wire_forest.py': '981385019f93f97d41c02eab245cc21e8c430ccdfc8e0f853d5ffacb6981a3b3',
    'scoped_source/wire_archive.py': 'fdf0b0e97522527831a6f03f1c7188403bc8b7b844c3a9df092823e703295abd',
    'scoped_source/capture.py': 'd3ca9b562c9840ace096d790d0fd6eb12d7105560a3c38767c85244c65f651e2',
    'scoped_source/hz_request_store.py': '7cb2be7647039f59d3c896b7856f634f43be9c532d9d2fd43632be09ab3230e0',
    'act/back_end/moe/scaled_forest_endpoints.py': '118e88e3484ff4f4977f3b2535fa3aba8bdf7b1b1b7598da8232c0fe0ea57af1',
}


def inventory():
    return shape_inventory(3072, 8, 10, [128], [256, 128])


def validate(v, width):
    if v != inventory() or type(width) is not int or width not in (1, 8):
        raise ValueError('registered captured topology and width 1 or 8 only')


def rational_record(bits):
    if type(bits) is not int or bits not in (4096, 8192):
        raise ValueError('unchanged stored or checked-result bit profile')
    # Two little-endian magnitudes, canonical sign/length/tag and reserved bytes.
    return 16 + 2*((bits+7)//8)


def source_chunks(v):
    validate(v, 8)
    chunk = lambda elements: (8*elements+MIB-1)//MIB
    graph = lambda dims: sum(chunk(x*y)+chunk(y) for x,y in zip(dims,dims[1:]))
    return graph(v['router_widths'])+v['experts']*graph(v['expert_widths'])+chunk(v['inputs'])


def archive(v, width):
    validate(v, width)
    f, n, h, l, r, vcount = (v[k] for k in (
        'factors_upper', 'nodes_upper', 'roots_upper', 'local_entries_upper',
        'references_upper', 'views'))
    e, d, p, m = (v[k] for k in (
        'endpoints_upper', 'property_duties', 'pairs', 'pair_dual_rows_upper'))
    t = p*((2*(v['classes']-1)+width-1)//width)
    forest = dict(factor_records=256*f, node_records=64*n,
        non_source_local_records=16*(l-v['first_weight_positions']),
        reference_records=16*r, unique_roots=16*n, membership_indices=4*h,
        scales=8*f, view_records=512*vcount, bounded_header=MIB)
    proof = dict(proposed_binary64=8*e*m, both_role_bindings=512*e,
                 tile_framing=256*t, bounded_index=MIB//4)
    qbytes = rational_record(8192)
    diagnostics = dict(endpoint_receipts=e*(512+3*qbytes),
        property_results=d*(128+qbytes), gate_records=p*(128+4*qbytes),
        bounded_index=MIB//4)
    aux = dict(checker_files=32*65536, checker_index=16384,
        execution_metadata=4*65536, store_index=MIB//4,
        events=(504+32)*512, combined_stdio=MIB, failure_context=65536,
        terminal=8192)
    parts = dict(source=v['raw_binary64_tensor_bytes']+4*MIB,
        forest=sum(forest.values()), candidate_proof=sum(proof.values()),
        diagnostics=sum(diagnostics.values()), auxiliary=sum(aux.values()))
    members = dict(source_chunks=source_chunks(v), source_manifest=1, forest=1,
        candidate_tiles=t, candidate_index=1, endpoint_receipts=1,
        diagnostics_index=1, property_results=1, gate_records=1,
        checker_files=32, checker_index=1, execution_metadata=4,
        store_index=1, events=1, stdio=1, failure=1, terminal=1)
    total = sum(parts.values())
    # Same framing; only the proposed role changes scalar encoding.
    general = total + (rational_record(4096)-8)*e*m
    return dict(kind='CONDITIONAL_NEW_FORMAT_SINGLE_ATTEMPT_UPPER',
        forest_parts=forest, candidate_parts=proof, diagnostic_parts=diagnostics,
        auxiliary_parts=aux, component_bytes=parts, member_counts=members,
        members_upper=sum(members.values()), native_single_store_bytes=total,
        native_two_store_bytes=2*total, generic_rational_single_store_bytes=general,
        largest_native_tile_bytes=256+width*(512+8*m),
        incoming_rational_record_bytes=rational_record(4096),
        checked_result_record_bytes=qbytes,
        generic_profile_conditions='rational dual scalars; same fixed bindings and native advisory lower=0 for both roles',
        immutable_single_store=True, no_retry_or_hidden_copy=True,
        conditions=['registered binary64 native candidates; explicit zero role remains checked',
            'direct typed construction; no old logical JSON materialization',
            'all named member envelopes satisfied; any overflow is a refusal',
            'partial/final use the same immutable payload; no additional full copy',
            'no checkpoint/installed environment/data duplication inside this envelope'],
        old_archive_cap=128*MIB, single_native_upper_fits_old_archive=total<=128*MIB,
        two_native_upper_fits_old_archive=2*total<=128*MIB,
        larger_upper_proves_actual_overflow=False, actual_bytes=None)


def exact_core(v, width):
    validate(v, width)
    old = bounds(v, width)
    phases = copy.deepcopy(old['phases'])
    # Only the count of independently parsed root VALUES changes. References
    # remain separately validated/charged; algebraic formulas remain V3.
    phases['source_check']['counts']['row_visits'] -= v['roots_upper']-v['nodes_upper']
    phases['source_check']['derivation'] = '2V+Q+N+F+U, Q<=N; all other V3 source algebra retained'
    keys = ('row_visits', 'coefficient_visits', 'arithmetic')
    prod = {k:sum(x['counts'][k] for x in phases.values()) for k in keys}
    exclude = ('source_build', 'native_preparation', 'native_candidates')
    replay = {k:prod[k]-sum(phases[x]['counts'][k] for x in exclude) for k in keys}
    replay['row_visits'] -= 4
    h,w1,w2,u,p,c,e,m,views,d = (v[k] for k in (
        'roots_upper','first_weight_positions','later_weight_positions','relu_units',
        'pairs','classes','endpoints_upper','pair_dual_rows_upper','views','property_duties'))
    t = old['pair_local_tiles_upper']
    index_ops = 2*h+2*(w1+w2+u)+2*p+2*c*e+2*t*m+4*t*views+e+d+16*views
    return dict(kind='CONDITIONAL_SPECIFIED_CORE_NOT_COMPLETE_RUNTIME_COST',
        phases=phases, production=prod, portable_replay=replay,
        online_with_separate_replay={k:prod[k]+replay[k] for k in keys},
        role_index_access_upper_per_pass=index_ops,
        online_role_index_upper=2*index_ops,
        native_adjoint_calls=129*t, native_forward_calls=128*t,
        proposed_and_zero_scalar_checks=2*e*m,
        serialization_hash_decode_native_ops_in_core=False,
        implementation_correspondence_proved=False,
        complete_request_work_upper=None, wall_seconds_prediction=None)


def memory(v, width, storage):
    validate(v, width)
    n,f,m,d,e = (v[k] for k in ('nodes_upper','factors_upper',
        'pair_dual_rows_upper','inputs','endpoints_upper'))
    slots = 2*width*(n+f)+4*(n+f+m+d)+16*e
    typed = dict(source_and_forest=storage['component_bytes']['source']+
                 storage['component_bytes']['forest'],
                 exact_persistent_slots=slots*rational_record(8192),
                 one_raw_tile=storage['largest_native_tile_bytes'],
                 two_stream_buffers=2*65536)
    return dict(kind='CONDITIONAL_DESIGN_OWNED_DATA_NOT_RSS',
        slot_formula='2w(N+F)+4(N+F+M+n)+16E', rational_slots_upper=slots,
        typed_payload_components=typed, typed_payload_bytes_upper=sum(typed.values()),
        source_construction_is_direct=True, child_lifetimes_sequential=True,
        accepted_working_bits=8192, example_unreduced_addition_bits_upper=16385,
        example_is_not_all_fraction_or_gcd_scratch_bound=True,
        full_rss_formula='parent + max(construction, native_plus_exact, portable); each includes runtime, scratch, allocator and mapped pages',
        full_peak_rss_upper=None, native_workspace_upper=None,
        live_checkpoint_capture_upper=None, arithmetic_runtime_scratch_upper=None,
        missing=['direct implementation and fixed-slot coverage proof',
            'pinned exact-arithmetic/runtime allocation and allocator lifetime bound',
            'pinned native kernel allocation/dispatch or separately controlled replacement',
            'live checkpoint load/capture and complete parent/child/mapped-page lifetime'],
        retained_rss_stop_bytes=8*2**30, retained_address_space_stop_bytes=16*2**30,
        stop_limits_are_not_completion_bounds=True)


def io_contract(storage):
    # A proposed staged full-store verifier scans each immutable member at most
    # once per pass into owned bounded intake. Internal source accesses are RAM
    # traffic, not filesystem re-reads. The current implementation does NOT
    # establish this schedule; it is an implementation obligation.
    b = storage['native_single_store_bytes']
    v = inventory()
    f,h,views,p,d,e,m = (v[k] for k in ('factors_upper','roots_upper','views',
        'pairs','property_duties','endpoints_upper','pair_dual_rows_upper'))
    frames = dict(source_request=4*MIB, contract=8192,
        factor_identity=264*f+64, view_scopes=512*views+4*h+64*views,
        pair_rosters_and_gates=p*(8*m+256+16384),
        properties=4096*d, queries_and_slots=2176*e, root_misc=16384)
    # Member hash plus at most one disjoint payload partition hash. All other
    # overlapping semantic hash inputs are named/bounded separately here.
    contexts = 2*storage['members_upper']+53+views+4*p+d+2*e+16
    per_pass = 2*b+sum(frames.values())+64*contexts
    return dict(kind='CONDITIONAL_SINGLE_STORE_FILE_SCHEDULE',
        named_passes=['input intake','independent online source check',
            'portable full replay','parent inventory/reception'],
        read_bytes_upper=4*b, write_bytes_upper=b,
        hash_semantic_frames_upper_per_pass=frames,
        hash_context_count_upper_per_pass=contexts,
        hash_input_bytes_upper_per_pass=per_pass,
        hash_input_bytes_upper=5*per_pass,
        hash_closure='CONDITIONAL_NAMED_CONTEXT_DISPATCH; not inferred from file traffic',
        one_file_pass_is_not_one_hash_pass=True,
        payload_second_contexts_must_partition=True,
        full_store_root_hashes_index_not_expanded_inventory=True,
        hash_updates_outside_named_schedule_allowed=False,
        optional_relocation=dict(additional_read_bytes=b, additional_write_bytes=b,
            additional_hash_bytes=2*b+128*storage['members_upper'], retained_bytes=2*b,
            authorized=False, independent_recheck_is_extra=True),
        source_reread_cache_required='owned immutable byte bank; native never receives writable alias',
        live_checkpoint_read_bytes_upper=None,
        total_cpu_memory_traffic_upper=None,
        no_disk_throughput_or_seconds_claim=True)


def analyze():
    for name, expected in ANCHORS.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError('changed protocol anchor: '+name)
    v = inventory()
    methods = {}
    for width in (1,8):
        storage = archive(v,width)
        methods[str(width)] = dict(archive=storage, core=exact_core(v,width),
                                   memory=memory(v,width,storage), io=io_contract(storage))
    return dict(schema='UNIQUE_ROOT_DIRECT_PROTOCOL_DESIGN_V1', date='2026-10-14',
        based_on_head='f5baf0e599d6674fa079047b884e594ba36b5475',
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        anchors=ANCHORS, inventory=v, methods=methods,
        identity=dict(new_domain='ACT_MOE_DIRECT_UNIQUE_ROOT_DRAFT_V1',
            replays_old_json=False, old_hash_equals_new_hash=False,
            canonical_root_upper=v['nodes_upper'], membership_upper=v['roots_upper'],
            membership_checks_removed=False, duplicate_owner_merging=False),
        invariant=dict(request_seconds=300,stored_bits=4096,working_bits=8192,
            accept='strictly greater than 1/10000000',
            all_tie_legal_pairs=True,zero_role_checked=True,
            old_results_unchanged=True,reference_and_proposed_share_resource_policy=True),
        current_policy_unchanged=current_policy(),
        closure=dict(archive_format_envelope='CONDITIONAL_CLOSED',
            specified_exact_core='CONDITIONAL_CLOSED',
            full_runtime_and_capture='NOT_CLOSED',
            full_cost_upper_closed=False,execution_admitted=False),
        status='DESIGN_DATA_ENVELOPES_ONLY_FULL_COST_NOT_CLOSED',
        new_quotas=None,absolute_ceiling=None,implementation_authorized=False,
        model_runs=0,gpu_runs=0,solver_calls=0,source_builds=0,
        production_files_changed=False,scientific_claims_upgraded=False,
        next='bound exact/native/capture implementation lifetimes before implementation or quota/execution freeze')


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-report',type=Path)
    args=p.parse_args(); result=analyze()
    if args.check_report:
        if json.loads(args.check_report.read_bytes()) != result:
            raise ValueError('unique-root design report differs')
        print('PASS: static design arithmetic; full runtime closure and execution remain unadmitted')
    else:
        print(json.dumps(result,sort_keys=True,indent=2))
