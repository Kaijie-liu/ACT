"""One complete authenticated C9 source cost census, no HZ writer or solve."""
import faulthandler
import json
from pathlib import Path
import resource
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c87_masked_tile_cost_v1 import census, COLUMNS
from experiments.neural_hz_20260831.c63_birth_blocks_v1 import route_rows
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout, fingerprint
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import (
    _sha256, _atomic_exclusive_json)

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c87_masked_tile_20260913_v1'
SOURCE = EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA = '616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
REPORT = EXP/'results/c69_prepared_finite_20260913_v1/actual/result.json'
REPORT_SHA = 'c5089a0666646562603208e13cdc735e4cb53ed78c0131996cd59230379723bc'


def complete(pool, branch, held, emit):
    hash_start = time.monotonic()
    if _sha256(REPORT) != REPORT_SHA:
        raise ValueError('sealed complete source report differs')
    old = json.loads(REPORT.read_text())['generation_report']['node_counts']
    separate_hash_s = time.monotonic()-hash_start
    with SOURCE.open('rb') as stream:
        saved, decoder = load(stream, expected_sha256=SOURCE_SHA, pool=pool, enabled=True)
    held['complete_original_checkpoint'] = saved
    if (saved['schema'] != 'c9_integrated_suffix_checkpoint_v1'
            or saved['origin_snapshot_sha256'] != 'd08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'
            or not saved['identity_audit']['all_original_coefficients_exact']
            or not saved['identity_audit']['all_redundant_main_and_radix_boxes_proved']):
        raise ValueError('complete original coefficient/box/source proof required')
    layout = numeric_layout(saved, pool)
    if layout.resident_entries > 64_000_000:
        raise MemoryError('complete original input entry cap')
    pool.charge('c87_complete_original_before_after_fingerprints', 2*int(layout.resident_entries)+2048)
    before, shallow = fingerprint(saved, layout, pool, already_paid=True)
    try:
        _, binding = route_rows(saved, np.empty(0, np.int64), pool=branch, enabled=True)
        nodes = saved['definition_graph']
        if len(nodes) != len(old):
            raise ValueError('complete source graph/report arity differs')
        branch.charge('c87_complete_original_node_mask_scale_headers',
                      64*len(nodes)+8*sum(n['width'] for n in nodes))
        tables, transforms, records = {}, {}, []
        held['all_tile_tables'], held['all_exact_operator_transforms'] = tables, transforms
        emit(dict(event='complete_original_source_bound', decoder=decoder, binding=binding,
                  input_entries=layout.resident_entries, input_bytes=layout.resident_bytes,
                  whole_work=pool.used, branch_work=branch.used))
        for index, (node, prior) in enumerate(zip(nodes, old, strict=True)):
            count = int(node['needed'].sum())
            if (node['kind'] != prior['kind'] or node['width'] != prior['width']
                    or count != prior['auxiliaries']
                    or np.any(node['needed'] & ~node['support'])):
                raise ValueError('complete actual source node mask/report differs')
            rec = dict(node=index, kind=node['kind'], width=node['width'], needed_rows=count,
                       source_report_continuous_edges=prior['continuous_edges'])
            if node['kind'] == 'op':
                op = node['op']
                rec['operator_type'] = type(op).__name__
                if type(op) is ImplicitConv2DOp:
                    rec.update(kernel_shape=list(op._kernel.shape), input_shape=list(op.input_shape),
                        output_shape=list(op.output_shape), stride=list(op._stride),
                        dilation=list(op._dilation), padding=list(op._padding), groups=op._groups)
                    selected = (op._kernel.shape[-2:] == (3, 3) and op._stride == (1, 1)
                                and op._dilation == (1, 1) and op._groups == 1
                                and op.input_shape[0] == op.output_shape[0] == 1)
                    rec['eligible_geometry'] = selected
                    if selected and count:
                        parent = nodes[node['parents'][0]]
                        mask = node['needed']
                        if op._row_mask is not None and np.any(mask & ~op._row_mask):
                            raise ValueError('demand violates actual operator row mask')
                        rec['parent_needed_rows'] = int(parent['needed'].sum())
                        powers = parent['exponents'][parent['needed']]
                        rec['original_parent_exponent_minmax'] = [int(powers.min()), int(powers.max())]
                        result, table, transformed = census(op._kernel,
                            parent['needed'].reshape(op.input_shape)[0],
                            mask.reshape(op.output_shape)[0], op._padding, pool=branch, enabled=True)
                        if result['direct_nnz'] != prior['continuous_edges']+count:
                            raise ValueError('independent complete direct incidence differs from original source')
                        tables[f'node{index}_all_tiles'] = table
                        for name, value in transformed.items():
                            transforms[f'node{index}_{name}'] = value
                        rec['cost'] = result
                        emit(dict(event='complete_source_masked_operator', record=rec,
                                  whole_work=pool.used, branch_work=branch.used))
                    elif selected:
                        rec['zero_demand_exact_noop'] = True
            records.append(rec)
        payload = {**tables, **transforms}
        combined = numeric_layout(dict(original=saved, payload=payload), pool)
        if combined.resident_entries > 64_000_000:
            raise MemoryError('complete original plus new census arrays exceed entry cap')
        pool.charge('c87_complete_numeric_evidence_export', sum(int(v.size) for v in payload.values()))
        with (RUN/'all_actual_source_tiles_and_transforms.npz').open('xb') as out:
            np.savez(out, **payload)
        costs = [r['cost'] for r in records if 'cost' in r]
        return dict(complete_source_graph_nodes=len(nodes), all_node_records=records,
            complete_original_source_binding=binding, independent_support_recomputation=False,
            unchanged_hash_bound_original_support_theorem_reused=True,
            original_decoder=decoder, original_numeric_entries=layout.resident_entries,
            original_numeric_bytes=layout.resident_bytes, original_python_shallow_bytes=shallow,
            complete_original_and_census_entries=combined.resident_entries,
            complete_original_and_census_bytes=combined.resident_bytes,
            eligible_nonzero_demand_operators=len(costs), all_tiles=sum(c['tiles'] for c in costs),
            winning_tiles=sum(c['winning_tiles'] for c in costs),
            necessary_only_nnz_saving=sum(c['necessary_only_nnz_saving'] for c in costs),
            winning_new_factors=sum(c['winning_new_factors'] for c in costs),
            actual_tile_table_columns=COLUMNS, file_hash_seconds_separate=separate_hash_s,
            complete_original_inputs_retained=True, actual_HZ_constructed=False,
            native_normalized_rows_proved=False, whole_HZ_physical_gate_proved=False,
            solver_executed=False, formal_gain=0)
    finally:
        unchanged = fingerprint(saved, layout, pool, already_paid=True)[0] == before
        emit(dict(event='complete_original_input_preservation', unchanged=unchanged))
        if not unchanged or _sha256(SOURCE) != SOURCE_SHA or _sha256(REPORT) != REPORT_SHA:
            raise ValueError('complete original source or reference changed')


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    started, pool, held = time.monotonic(), WorkPool(256_000_000), {}
    branch = BranchPool(pool)
    result = dict(completed=False, necessary_gate_passed=False, formal_gain=0)
    with (RUN/'events.jsonl').open('x') as log, (RUN/'fatal.log').open('x') as fatal:
        def emit(value):
            log.write(json.dumps(dict(value, worker_elapsed_s=time.monotonic()-started), allow_nan=False)+'\n')
            log.flush()
        try:
            faulthandler.enable(file=fatal, all_threads=True)
            data, stats = measured(lambda: complete(pool, branch, held, emit),
                                   observe=lambda s: result.update(measurement=s))
            result.update(completed=True, necessary_gate_passed=True, data=data)
        except Exception as exc:
            result['failure'] = dict(type=type(exc).__name__, reason=str(exc))
            emit(dict(event='complete_source_cost_failed', **result['failure']))
        finally:
            faulthandler.disable()
            result.update(wall_s=time.monotonic()-started, whole_work=pool.used, branch_work=branch.used,
                          work_parts=pool.parts, branch_work_parts=branch.parts)
            _atomic_exclusive_json(RUN/'result.json', result)
            print(json.dumps({k: result[k] for k in ('completed', 'wall_s', 'whole_work', 'branch_work', 'formal_gain')}), flush=True)
    if not result['completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
