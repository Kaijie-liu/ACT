"""Static changed-path assessment, not a source/model/native execution.

The archived parent is immutable evidence, not a claim that its old counter
totals apply to the new lazy decoder. No runtime limit is changed here.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.assess_model_scaled_resources import ROOT, current_policy, shape_inventory

ANCHORS = {
    'docs/hz_full_resource_ledger_20261014.json': '04c51e4765ada9445dcdd80f63f745459366154037443a68dd565c093acbe9d2',
    'docs/hz_lossless_evidence_design_20261014.json': 'f09feaed87b25e4ccc49becd2256d6d3b89fa02f6413bf3cae317d1f07f408d1',
    'scripts/assess_model_scaled_resources.py': '4900b4f3682b27071e8e406c7e5bb955677719183171b13c429f6469999a0789',
    'scoped_source/wire_forest.py': '981385019f93f97d41c02eab245cc21e8c430ccdfc8e0f853d5ffacb6981a3b3',
    'scoped_source/wire_proof.py': '1bd8d3bbc88b7d28fb022963c482ee77b909c50afb476bb9ec035cc67ac6d58a',
    'scoped_source/wire_archive.py': 'fdf0b0e97522527831a6f03f1c7188403bc8b7b844c3a9df092823e703295abd',
    'scoped_source/wire_package.py': 'a0ad7a894e91cfb2a6cd1a6518e401a1b8217b7b9b1933d5b97ce958a1275197',
    'scoped_source/check_source_dag.py': '58e5e2158843631c9e6ed324cfeeb2eb38e51df1205d2e14d1ed8ac8ce634458',
    'scoped_source/check_affine_references.py': 'a0d688ee82bf88c42945edfd3abc9591a46998c01cf236cbf752ad5b549e9c62',
    'scoped_source/check_scaled_references.py': '244dc7ee5cf7fe46ab7097b2494dfec64d818e926b4726eb19d59a53c5074f9e',
    'scoped_source/scaled_endpoints.py': '93197c1b329e173b25795c48cf117b7113c2b23cc6958438889a7e912d13e8af',
    'act/back_end/moe/scaled_forest_endpoints.py': '118e88e3484ff4f4977f3b2535fa3aba8bdf7b1b1b7598da8232c0fe0ea57af1',
    'scoped_source/tensor_rows.py': '5c3ad6d642868e257ac9dd2edbc5e066b4cd793d4c7567a4ea800fc8e8f97f82',
    'scoped_source/hz_request_store.py': '7cb2be7647039f59d3c896b7856f634f43be9c532d9d2fd43632be09ab3230e0',
    'scoped_source/factored_io.py': 'd51bcd7eae6734fe6cd676f584a9e9f2220eeea4400d497ccb1dc6950dce120d',
    'scoped_source/capture.py': 'd3ca9b562c9840ace096d790d0fd6eb12d7105560a3c38767c85244c65f651e2',
    'scoped_source/scaled_references.py': '534761cad44cbdf62ed31ee4d999fce0a5bd51aed4461b98d8f986b75d50d75d',
    'scripts/run_wire_archive_controls.py': '272fc7adf1bd2e40b7b3ee9e7d91ffcea3144a257e725a12573b5cff726691ee',
}
MIB = 2**20


def registered(v):
    expected = shape_inventory(3072, 8, 10, [128], [256, 128])
    if v != expected:
        raise ValueError('registered topology/inventory only')


def digit_sum(n):
    if type(n) is not int or n < 1:
        raise ValueError('positive integer count')
    value, start, digits = 1, 1, 1
    while start < n:
        end = min(n, 10*start)
        value += (end-start)*digits
        start, digits = end, digits+1
    return value


def native_header(width, eq_digits=4, ine_digits=4):
    if (type(width) is not int or not 1 <= width <= 8 or
            any(type(v) is not int or not 1 <= v <= 4 for v in (eq_digits, ine_digits))):
        raise ValueError('bounded width and count digits')
    # Six binding hashes, logical hash, 64-character query and slot IDs;
    # both lower fields are the native producer's literal "0".
    return 618+eq_digits+ine_digits+229*width


def storage(v, width):
    registered(v)
    if type(width) is not int or width not in (1, 8):
        raise ValueError('only reference/proposed widths 1 and 8')
    f, n, h, local, refs = (v[k] for k in
        ('factors_upper','nodes_upper','roots_upper','local_entries_upper','references_upper'))
    tail = local-v['first_weight_positions']
    forest_parts = dict(prefix=24, bounded_header=8*MIB, scales=8*f,
        roots=16*h, node_descriptors=64*n, inline_and_tail_entries=16*tail,
        reference_entries=16*refs)
    per_pair = 2*(v['classes']-1)
    full, remainder = divmod(per_pair, width)
    widths = [width]*full+([remainder] if remainder else [])
    rows = v['pair_dual_rows_upper']
    one = lambda w: 12+native_header(w)+8*w*rows
    tiles = v['pairs']*sum(one(w) for w in widths)
    source = v['raw_binary64_tensor_bytes']+4*MIB
    components = dict(source_payload_and_bounded_header=source,
        wire_forest=sum(forest_parts.values()), native_wire_tiles=tiles)
    return dict(kind='CONDITIONAL_COMPONENT_UPPERS_NOT_COMPLETE_ARCHIVE', width=width,
        conditions=['registered V3 topology and implied compensation bounds',
                    'forest header passes unchanged 8MiB cap',
                    'dense finite binary64 proposed duals and native lower fields equal 0',
                    'all logical identities/caps still checked; no structural admission implied'],
        forest_parts=forest_parts, forest_bytes_upper=sum(forest_parts.values()),
        maximum_tile_header_bytes=native_header(width), maximum_tile_bytes=one(width),
        logical_candidate_tile_upper=width*(332+349*rows)+1,
        logical_candidate_upper_scope='native finite-binary64 literals <=342 chars; metadata and both roles included',
        logical_tile_upper_exceeds_4mib=width*(332+349*rows)+1>4*MIB,
        logical_cumulative_cap=16*MIB-8192,
        physical_tiles_do_not_replace_logical_caps=True,
        per_pair_tile_widths=widths, tile_count=v['pairs']*len(widths),
        native_wire_tile_bytes_upper=tiles, components=components,
        single_component_sum_upper=sum(components.values()),
        two_copy_component_sum_upper=2*sum(components.values()),
        two_copy_upper_exceeds_cap=2*sum(components.values())>128*MIB,
        upper_overflow_proves_actual_overflow=False,
        complete_archive_upper=None, complete_archive_actual=None,
        unclosed=['full registered runner not frozen or integrated',
                  'member inventory, source chunk layout, code, receipts, logs and terminal roster',
                  'temporary source/old JSON lifetime and simultaneous retained copy policy'])


def logical_obstruction(v):
    registered(v)
    n = v['inputs']; first_rows = v['first_weight_positions']//n
    literal = '1/9007199254740992'  # exact binary64 2^-53, not a loaded weight
    one = 1+digit_sum(n)+(len(literal)+6)*n
    total = first_rows*one
    return dict(kind='CONSTRUCTIVE_ENCODING_CASE_NOT_CHECKPOINT_OBSERVATION',
        condition='all first-affine weights equal exact binary64 2^-53',
        coefficient_literal=literal, coefficient_characters=len(literal),
        first_rows=first_rows, one_local_array_bytes=one,
        first_local_arrays_bytes=total, retained_logical_forest_cap=128*MIB,
        excess_before_other_fields=total-128*MIB,
        omitted='all wrappers, other nodes, views, scales, source and proof',
        actual_values_observed=False, full_source_constructed=False,
        conclusion='no uniform old-logical-byte capacity guarantee from wire compression')


def changed_work(v):
    registered(v)
    w1 = v['first_weight_positions']; first_rows=w1//v['inputs']
    return dict(scope='complete-path mandatory component counts, not observed prefixes or full uppers',
        two_lazy_first_weight_scans=dict(
            coefficient_visits_lower=2*w1, row_visits_lower=0, arithmetic_lower=0,
            sites=['wire_forest.logical_chunks/document_row','check_source_dag.initialize/document_row'],
            includes_zero_source_positions=True, work_lower=2*w1),
        repeated_input_node_decodes=dict(
            row_visits=2*w1, coefficient_visits=2*w1, added_work=4*w1,
            site='check_scaled_references.Checker.affine accesses nodes[pn][1] and nodes[pn][2]',
            relation_to_old='additional decoding visits; old semantic checks remain'),
        reachability_caveat='unchanged root/other gates may refuse earlier; these are requirements to complete',
        first_source_row_pass_hash_read_bytes_upper=8*w1+2*MIB*first_rows,
        hash_read_derivation='row bytes plus at most two partial chunks of <=1MiB per row',
        hash_read_scope='repeated userspace read/hash volume, not physical disk traffic, peak memory or time',
        old_full_work_totals_reusable=False,
        complete_wire_production_work_upper=None, complete_wire_portable_work_upper=None,
        other_new_costs=['lazy semantic node decoding beyond the two counted components',
            'gate/objective/acceptance pullbacks decode active nodes again',
            'native Operator decodes every node into its whole global COO bank',
            'encode_tile visits both roles; decode_tile visits proposed entries; zero checks retained',
            'source terminal reseal and package inventories, copies and independent replay'],
        native_129_evaluations_do_not_reread_source_rows=True,
        absent_execution_contract='native produce still retains JSON; only finite fixture driver uses wire lifecycle')


def analyze():
    for path, expected in ANCHORS.items():
        if hashlib.sha256((ROOT/path).read_bytes()).hexdigest() != expected:
            raise ValueError('changed assessment anchor: '+path)
    old=json.loads((ROOT/'docs/hz_full_resource_ledger_20261014.json').read_bytes())
    v=old['inventory']; registered(v); policy=current_policy()
    return dict(schema='IMPLEMENTED_WIRE_CAPACITY_STATIC_V1', date='2026-10-14',
        reviewed_head='aa7bc74a95f8a49ede1858bb2fd698a0eab39bb6',
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        anchors=ANCHORS, inventory=v, unchanged_policy=policy,
        methods={str(w):storage(v,w) for w in (1,8)},
        logical_forest=logical_obstruction(v), changed_work=changed_work(v),
        mandatory_roots=dict(lower=v['mandatory_roots'], retained_cap=policy['rows'],
            refusal_proved=v['mandatory_roots']>policy['rows']),
        control_driver_terminal=dict(
            scope='existing finite-fixture driver embeds all result rows, not a full registered runner',
            smallest_result_row_bytes=52,
            result_array_bytes_lower=53*v['property_duties']+1,
            terminal_cap_bytes=8192,
            inline_full_roster_refused=53*v['property_duties']+1>8192,
            next_layout_decision='separate bound result member with small hash-bound terminal; not implemented'),
        memory=dict(complete_peak_rss_upper=None, complete_native_peak_upper=None,
            bounded_consumer_scope='one decoded coefficient row / raw candidate tile; roots and metadata stay resident',
            construction='encode still retains old raw JSON, decoded document and bytearrays before joined wire bytes',
            native='Operator keeps whole rr/cc/vv Python lists while building global COO and transpose',
            native_named_array_components_only={
                'one_coo':24*v['local_entries_upper'],
                'coo_and_transpose':48*v['local_entries_upper'],
                'coalesce_and_sort_branch':88*v['local_entries_upper']},
            native_components_condition='historical pinned upstream dispatch; not full binary/RSS guarantee',
            old_live_capture_replaced=False,
            deadlines_and_memory_caps_are_stop_rules=True),
        next_decision='STOP_CURRENT_CAPACITY_RETRIES; separately scope logical/resource and direct-construction contract',
        execution_admitted=False, caps_changed=False, request_seconds=300,
        new_work_quota=None, absolute_work_ceiling=None, walltime_forecast=None,
        model_loads=0, source_builds=0, solver_calls=0, native_calls=0, gpu_calls=0,
        historical_results_changed=False, G1_G6_closed=False,
        scientific_scope='interface capacity/refusal only; no new certificate, performance or GPU result')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report',type=Path)
    args=parser.parse_args(); result=analyze()
    if args.check_report:
        if json.loads(args.check_report.read_bytes()) != result:
            raise ValueError('saved wire assessment differs')
        print('PASS: static wire assessment; full topology remains unadmitted')
    else:
        print(json.dumps(result,sort_keys=True,indent=2))
