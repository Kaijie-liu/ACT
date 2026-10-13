"""Code-bound topology arithmetic, not source construction or execution admission.

Only standard-library code and the parent static assessment are imported.
Bounds concern the already captured registered seed1/2 topology under the
unchanged V3 algorithm. They are not measured counts, FLOPs or time estimates.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.assess_model_scaled_resources import analyze as parent_analysis, ROOT

PARENT = 'scripts/assess_model_scaled_resources.py'
PARENT_SHA = '4900b4f3682b27071e8e406c7e5bb955677719183171b13c429f6469999a0789'
COUNTERS = ('row_visits', 'coefficient_visits', 'arithmetic')


def digit_sum_below(n):
    """Decimal digits for IDs 0..n-1 without allocating an ID/weight bank."""
    if type(n) is not int or n < 1:
        raise ValueError('positive integer ID count')
    result = 1  # zero
    start, digits = 1, 1
    while start < n:
        end = min(n, start*10)
        result += (end-start)*digits
        start *= 10
        digits += 1
    return result


def closure(inputs, affine_outputs, relus, first_weights, later_weights, guards=0):
    """Existing shape envelope restricted to an ancestor-closed graph set."""
    return dict(factors=inputs+2*affine_outputs+7*relus,
                entries=inputs+first_weights+2*affine_outputs+12*relus+
                        later_weights+relus+2*guards)


def bounds(v, width):
    if type(width) is not int or width not in (1, 8):
        raise ValueError('only the two statically selected widths')
    if (v['router_widths'] != [3072, 128, 8] or
            v['expert_widths'] != [3072, 256, 128, 10]):
        raise ValueError('derivation limited to the bound registered topology')
    n, a, u = v['inputs'], v['affine_outputs'], v['relu_units']
    w1, w2, g = v['first_weight_positions'], v['later_weight_positions'], v['guards']
    f, nodes, local, refs = (v[k] for k in
        ('factors_upper', 'nodes_upper', 'local_entries_upper', 'references_upper'))
    views, roots = v['views'], v['roots_upper']
    pairs, classes, duties = v['pairs'], v['classes'], v['property_duties']
    endpoints, rows = v['endpoints_upper'], v['pair_dual_rows_upper']
    tiles = pairs*((2*(classes-1)+width-1)//width)
    scales = min(f, n+1)  # checked auxiliary factors have unit scale
    router = closure(n, 136, 128, n*128, 128*8)
    pair = closure(n, 136+2*394, 128+2*384, n*(128+2*256),
                   128*8+2*(256*128+128*10), 2*(8-2))
    kr, fr, kp = router['entries'], router['factors'], pair['entries']
    levels = 2  # registered graphs: nonempty reference levels are depth 2 and 3
    phases = {}

    def add(name, r, c, x, formula):
        phases[name] = dict(kind='UPPER', counts=dict(zip(COUNTERS, (r,c,x))),
                            derivation=formula)

    add('source_build', f+nodes+u, w1+w2,
        n+w1+2*w2+2*a+9*u+g+3*local+refs,
        'F+N+U; W1+W2; n+W1+2W2+2A+9U+G+3L+R')
    add('source_check', 2*views+roots+nodes+f+u, f+local+refs+w1+w2,
        n+w1+2*w2+a+9*u+g+3*local+refs,
        '2V+H+N+F+U; F+L+R+W1+W2; n+W1+2W2+A+9U+G+3L+R')
    add('gate_and_obligation_plan', 11+pairs+duties+endpoints+pairs*nodes,
        pairs*(kr+fr), 1+pairs*(2*kr+3*fr+3)+2*classes*endpoints,
        '11+P+D+E+PN; P(Kr+Fr); 1+P(2Kr+3Fr+3)+2CE')
    add('native_preparation', nodes, local+refs+f, 0, 'N; L+R+F; 0')
    add('objective_pullback', endpoints*(nodes+2*classes), endpoints*kp,
        endpoints*(2*kp+12), 'E(N+2C); EK; E(2K+12)')
    add('native_candidates', 3*endpoints+tiles*(rows+129+257*levels),
        4*endpoints+tiles*rows+endpoints*rows, 0,
        '3E+T(M+129+257J); 4E+TM+EM; 0')
    add('dual_check_merge_and_support', tiles*(rows+nodes)+endpoints*(32+2*scales),
        2*endpoints*rows+tiles*kp+3*endpoints*f,
        3*endpoints*rows+2*endpoints*kp+endpoints*(3*f+4*scales+3),
        'T(M+N)+E(32+2S); 2EM+TK+3EF; 3EM+2EK+E(3F+4S+3)')
    add('aggregation_and_proof', endpoints+duties+13, 0, 0, 'E+D+13; 0; 0')
    total = {k:sum(p['counts'][k] for p in phases.values()) for k in COUNTERS}
    excluded = ('source_build', 'native_preparation', 'native_candidates')
    portable = {k:total[k]-sum(phases[p]['counts'][k] for p in excluded) for k in COUNTERS}
    portable['row_visits'] -= 4  # check_proof finishes but does not generate proof()
    return dict(width=width, phases=phases, production_counts_upper=total,
        production_work_upper=sum(total.values()), portable_counts_upper=portable,
        portable_work_upper=sum(portable.values()), pair_local_tiles_upper=tiles,
        closure=dict(router=router, pair=pair, distinct_scales_upper=scales,
                     native_reference_levels_upper=levels),
        native_schedule=dict(adjoint_calls_upper=129*tiles, forward_calls_upper=128*tiles,
            global_sparse_bank_used=True, work_is_not_flops=True),
        scope='complete captured-source V3 call with publication-only event; not live capture or fallback',
        execution_admitted=False)


def storage(v):
    n = v['inputs']; first_outputs = v['first_weight_positions']//n
    # Per row: '[' + comma-separated [ID,"1/2"] + ']'. The +1 already
    # accounts for the omitted final comma and both brackets.
    local_row = 1+digit_sum_below(n)+9*n
    bank = first_outputs*local_row
    source = v['raw_binary64_tensor_bytes']
    witness = dict(kind='CONSTRUCTIVE_ENCODING_LOWER_BOUND_NOT_CHECKPOINT_OBSERVATION',
        condition='all first-affine weights are binary64 1/2; other source terms not counted',
        first_affine_rows=first_outputs, id_digit_sum=digit_sum_below(n),
        one_local_array_lower_bytes=local_row, first_local_arrays_lower_bytes=bank,
        raw_source_payload_bytes=source, single_bundle_lower_bytes=source+bank,
        old_archive_cap_bytes=128*2**20, excess_bytes=source+bank-128*2**20,
        two_bundle_lower_bytes=2*(source+bank),
        two_bundle_condition='if retaining the archived dense runner online plus relocated layout',
        omits=['node wrappers','views','scales','source header','proof','partial','logs','code'],
        observed_checkpoint_values=False, constructed_source=False)
    # A finite binary64 is either an integer (<=309 digits) or a reduced
    # p/2^k with |p|<=2^53-1 and k<=1074: 1+16+1+324=342 suffices.
    native_chars, stored_chars = 342, 2470
    m, e = v['pair_dual_rows_upper'], v['endpoints_upper']
    duals = {}
    for role, chars in (('native_binary64',native_chars), ('general_checker',stored_chars)):
        duals[role] = dict(kind='DUAL_ELEMENT_COMPONENT_UPPER_NOT_FULL_PROOF_UPPER',
            chars_per_proposed_scalar_upper=chars, proposed_and_zero=True,
            tile8_element_bytes_upper=8*m*((chars+3)+4),
            full_element_bytes_upper=e*m*((chars+3)+4),
            complete_metadata_upper=None, observed_bytes=None)
    return dict(source_bytes_upper=source+4*2**20, source_bound='raw payload plus bounded manifest',
        encoding_witness=witness, candidate_components=duals,
        forest_value_components_upper=(v['local_entries_upper']+v['roots_upper'])*(native_chars+12)+
            v['references_upper']*(stored_chars+12)+v['factors_upper']*(native_chars+3),
        forest_component_scope='local/reference/root/scale entries only; not complete metadata',
        full_forest_bytes_upper=None, full_proof_bytes_upper=None, full_archive_bytes_upper=None,
        archive_lifetime_formula='2*source + 2*forest + 2*proof + partial_tiles + headers/logs/code',
        archive_formula_scope='archived dense runner retention pattern only; new full-topology runner not frozen',
        byte_caps_changed=False)


def analyze():
    if hashlib.sha256((ROOT/PARENT).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('parent assessment identity changed')
    parent = parent_analysis(); v = parent['inventory']
    methods = {'reference':bounds(v,1), 'proposed':bounds(v,8)}
    return dict(schema='CAPTURED_FULL_TOPOLOGY_RESOURCE_LEDGER_V1', date='2026-10-14',
        reviewed_head='eb911b09889f1ca74e494d10571189b5bdaf5b11',
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        code_and_metadata={**parent['code_and_metadata'],PARENT:PARENT_SHA}, inventory=v,
        reference_contract=dict(producer='act/back_end/moe/scaled_forest_endpoints.py:produce',
            reference='V3_FULL_ENDPOINT_BATCH_WIDTH_1_CPU', proposed='V3_FULL_ENDPOINT_BATCH_WIDTH_8_CPU',
            changed_factor='joint native candidate and exact dual-consumption batching',
            fixed=['V3 source enclosure','all pairs/properties','128 updates and 129 evaluations',
                   'proposed and explicit zero candidates','CPU float64','one thread','exact acceptance'],
            same_generated_candidates_claimed=False, consumer_only_ablation=False,
            same_candidate_exact_differential_required=True, strong_native_baselines_retained=True,
            real_runner_frozen=False),
        methods=methods, storage=storage(v),
        memory=dict(peak_rss_upper=None, peak_address_space_upper=None,
            current_sampled_rss_cap=8*2**30, current_address_space_cap=16*2**30,
            caps_are_stop_rules_not_completion_guarantees=True,
            single_coo_values_and_two_indices_payload_upper=24*v['local_entries_upper'],
            node_tile_shape_upper=[v['nodes_upper'],8], factor_tile_shape_upper=[v['factors_upper'],8],
            dual_tile_shape_upper=[v['pair_dual_rows_upper'],8],
            unresolved=['Python container/Fraction/allocator high-water lifetime bound',
                        'native coalesce and sparse-mm temporary workspace',
                        'charged live checkpoint capture/base64/chunk conversion',
                        'bounded metadata/logs/terminal handling for a new full-topology runner']),
        work_scope=dict(counter_names=list(COUNTERS), bit_limits_unchanged=True,
            explicitly_counted_captured_path_upper_closed=True,
            deadline_only_operations=['TensorRows parsing/hash/read','compact/decode/string encoding',
                'frozen containers','file publication/copy','native torch scalar/tensor work'],
            new_live_capture_or_fallback_work_upper=None, arbitrary_event_callbacks_not_covered=True,
            no_work_to_walltime_conversion=True),
        actual_counts=None, candidate_work_quota=None, absolute_work_ceiling=None,
        walltime_forecast=None, production_limits_changed=False,
        implementation_admitted=False, real_execution_admitted=False,
        new_source_runs=0, new_bound_queries=0, checkpoint_loads=0, gpu_calls=0,
        decision='WORK_LEDGER_CLOSED_ARCHIVE_COUNTEREXAMPLE_MEMORY_NOT_CLOSED_NO_ADMISSION',
        next_decision='separate evidence-encoding or explicitly approved byte/resource design; no runtime cap patch',
        scientific_limit='static captured-path accounting; no trained result, safety, speedup or G1-G6 closure')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report',type=Path)
    args=parser.parse_args(); result=analyze()
    if args.check_report:
        if result != json.loads(args.check_report.read_bytes()):
            raise ValueError('resource ledger differs')
        print('PASS: static full-path ledger; no execution admission')
    else:
        print(json.dumps(result,sort_keys=True,indent=2))
