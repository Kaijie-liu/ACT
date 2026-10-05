"""Complete original-source reduced-circuit diagnostic, all input owners held."""
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
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import recover_filter
from experiments.neural_hz_20260831.c88_inline_tile_v1 import construct, NativeUnproved, prepare
from experiments.neural_hz_20260831.c63_birth_blocks_v1 import route_rows
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout, fingerprint
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c88_inline_tile_20260913_v1'
SOURCE = EXP/'results/c9_integrated_suffix_20260905_v1/lifted_hz.pickle'
SOURCE_SHA = '616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962'
PREVIOUS = EXP/'results/c87_masked_tile_20260913_v1'
OLD_RESULT_SHA = 'e582041fe220fded5a0a35d5d1331228036ad75a91558e727f3d5f6ff7557106'
OLD_ARRAY_SHA = 'a634b1becc25ae8b33e22246334200c29f6e3a491728a73861ed4a4087c0c3e1'


def complete(pool, branch, held, emit):
    if (_sha256(PREVIOUS/'result.json') != OLD_RESULT_SHA
            or _sha256(PREVIOUS/'all_actual_source_tiles_and_transforms.npz') != OLD_ARRAY_SHA):
        raise ValueError('complete original source census reference changed')
    old = json.loads((PREVIOUS/'result.json').read_text())
    if not old['completed']:
        raise ValueError('complete original source census required')
    with np.load(PREVIOUS/'all_actual_source_tiles_and_transforms.npz', allow_pickle=False) as archive:
        prior_arrays = {name: archive[name] for name in archive.files}
    with SOURCE.open('rb') as stream:
        saved, decoder = load(stream, expected_sha256=SOURCE_SHA, pool=pool, enabled=True)
    inputs = dict(complete_C9=saved, complete_C87_arrays=prior_arrays)
    held['complete_inputs'] = inputs
    if (saved['schema'] != 'c9_integrated_suffix_checkpoint_v1'
            or saved['origin_snapshot_sha256'] != 'd08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed'
            or not saved['identity_audit']['all_original_coefficients_exact']
            or not saved['identity_audit']['all_redundant_main_and_radix_boxes_proved']):
        raise ValueError('complete original source/box proof required')
    layout = numeric_layout(inputs, pool)
    if layout.resident_entries > 64_000_000:
        raise MemoryError('complete original/reference input entry cap')
    pool.charge('c88_complete_input_before_after_fingerprints', 2*int(layout.resident_entries)+2048)
    before, shallow = fingerprint(inputs, layout, pool, already_paid=True)
    try:
        _, binding = route_rows(saved, np.empty(0, np.int64), pool=branch, enabled=True)
        nodes, previous = saved['definition_graph'], old['data']['all_node_records']
        if len(nodes) != len(previous):
            raise ValueError('complete source/census graph arity differs')
        branch.charge('c88_complete_node_mask_scale_headers', 64*len(nodes)+8*sum(n['width'] for n in nodes))
        payload, records = {}, []
        held['all_new_native_packets_and_transforms'] = payload
        native_pass = wins = gains = new_factors = tiles = 0
        emit(dict(event='complete_source_and_reference_bound', input_entries=layout.resident_entries,
                  input_bytes=layout.resident_bytes, whole_work=pool.used, branch_work=branch.used))
        for index, (node, prior) in enumerate(zip(nodes, previous, strict=True)):
            if node['kind'] != prior['kind'] or node['width'] != prior['width'] or int(node['needed'].sum()) != prior['needed_rows']:
                raise ValueError('complete original node/demand differs')
            if 'cost' not in prior:
                continue
            op, parent = node['op'], nodes[node['parents'][0]]
            if (type(op) is not ImplicitConv2DOp or op._stride != (1, 1) or op._dilation != (1, 1)
                    or op._groups != 1 or op._kernel.shape[-2:] != (3, 3)
                    or op.input_shape[0] != 1 or op.output_shape[0] != 1):
                raise ValueError('original selected mathematical geometry differs')
            data = {name: prior_arrays[f'node{index}_{name}'] for name in ('numerator', 'exponent', 'native', 'gauge')}
            branch.charge('c88_cached_transform_independent_original_kernel_binding',
                          64*int(op._kernel.shape[0]*op._kernel.shape[1])+4*int(op._kernel.size))
            recovered = recover_filter(data['numerator'])
            expected = np.ldexp(op._kernel, -data['exponent'][..., None, None])
            if (not np.isfinite(expected).all() or np.any(np.abs(expected) >= 2.**62)
                    or not np.array_equal(expected.astype(np.int64).astype(np.float64), expected)
                    or not np.array_equal(recovered, expected.astype(np.int64))):
                raise ValueError('independent original kernel differs from complete cached transform')
            proof = dict(complete_C87_forward_transform_proof_reused=True,
                         original_kernel_inverse_freshly_rechecked=True, fresh_forward_transform_claimed=False)
            prepared = prepare(data, pool=branch)
            _, channels, height, width = op.input_shape
            _, outputs, oh, ow = op.output_shape
            py, px = op._padding
            pids = parent['slots'].reshape(op.input_shape)[0]
            pexp = parent['exponents'].reshape(op.input_shape)[0]
            pmask = parent['needed'].reshape(op.input_shape)[0]
            oids = node['slots'].reshape(op.output_shape)[0]
            oexp = node['exponents'].reshape(op.output_shape)[0]
            omask = node['needed'].reshape(op.output_shape)[0]
            original_nonzero = op._kernel != 0
            old_table = prior_arrays[f'node{index}_all_tiles']
            local_records = []
            for position, (y, x) in enumerate((y, x) for y in range(0, oh, 2) for x in range(0, ow, 2)):
                branch.charge('c88_complete_original_tile_maps', 1024+128*(channels+outputs))
                ids = np.full((channels, 4, 4), -1, np.int64)
                powers = np.zeros(ids.shape, np.int32)
                out_ids = np.full((outputs, 2, 2), -1, np.int64)
                out_powers = np.zeros(out_ids.shape, np.int32)
                for i in range(4):
                    for j in range(4):
                        sy, sx = y-py+i, x-px+j
                        if 0 <= sy < height and 0 <= sx < width:
                            enabled = pmask[:, sy, sx]
                            ids[enabled, i, j] = pids[enabled, sy, sx]
                            powers[enabled, i, j] = pexp[enabled, sy, sx]
                for i in range(min(2, oh-y)):
                    for j in range(min(2, ow-x)):
                        enabled = omask[:, y+i, x+j]
                        out_ids[enabled, i, j] = oids[enabled, y+i, x+j]
                        out_powers[enabled, i, j] = oexp[enabled, y+i, x+j]
                direct = int((out_ids >= 0).sum())
                for i in range(2):
                    for j in range(2):
                        ks = np.flatnonzero(out_ids[:, i, j] >= 0)
                        for a in range(3):
                            for b in range(3):
                                cs = np.flatnonzero(ids[:, i+a, j+b] >= 0)
                                branch.charge('c87_complete_direct_pairs', 4*int(ks.size*cs.size))
                                direct += int(np.count_nonzero(original_nonzero[ks[:, None], cs[None, :], a, b]))
                if (int(old_table[position, 0]), int(old_table[position, 1]), int(old_table[position, 4])) != (y, x, direct):
                    raise ValueError('independent original tile incidence differs')
                item = dict(y=y, x=x, direct_nnz=direct, prior_raw_tile_win=bool(old_table[position, -1]),
                            native_coefficients_pass=False, strict_nnz_win=False)
                try:
                    report, packet = construct(prepared, ids, powers, out_ids, out_powers,
                        saved['hz'].n_cont, pool=branch, enabled=True)
                    item.update(report, strict_nnz_win=report['nnz'] < direct)
                    for name, value in packet.items():
                        payload[f'node{index}_tile{y}_{x}_{name}'] = value
                    native_pass += 1
                    if item['strict_nnz_win']:
                        wins += 1
                        gains += direct-report['nnz']
                        new_factors += report['new_factors']
                except NativeUnproved as exc:
                    item['native_rejection'] = str(exc)
                tiles += 1
                local_records.append(item)
            rec = dict(node=index, all_tiles=local_records, kernel_proof=proof)
            records.append(rec)
            emit(dict(event='complete_reduced_source_operator', node=index, tiles=len(local_records),
                native_pass=sum(r['native_coefficients_pass'] for r in local_records),
                wins=sum(r['strict_nnz_win'] for r in local_records), whole_work=pool.used, branch_work=branch.used))
        if tiles != old['data']['all_tiles'] or len(records) != old['data']['eligible_nonzero_demand_operators']:
            raise ValueError('complete original source tile population differs')
        combined = numeric_layout(dict(inputs=inputs, packets=payload), pool)
        if combined.resident_entries > 64_000_000:
            raise MemoryError('complete input plus native proof packet entry cap')
        pool.charge('c88_complete_native_packet_export', sum(int(v.size) for v in payload.values()))
        with (RUN/'all_native_tile_packets.npz').open('xb') as out:
            np.savez(out, **payload)
        return dict(all_operator_records=records, tiles=tiles, native_valid_tiles=native_pass,
            winning_tiles=wins, necessary_nnz_saving=gains, winning_new_factors=new_factors,
            summed_winner_auxiliary_count_within_16384=new_factors <= 16384,
            other_global_reserves_proved=False, whole_physical_reduction_proved=False,
            original_source_independent_row_proof_complete=False, complete_source_binding=binding,
            original_decoder=decoder, input_entries=layout.resident_entries, input_bytes=layout.resident_bytes,
            input_shallow_bytes=shallow, complete_original_and_packets_entries=combined.resident_entries,
            complete_original_and_packets_bytes=combined.resident_bytes,
            original_HZ_constructed=False, native_solver_ingested=False, solver_executed=False, formal_gain=0)
    finally:
        unchanged = fingerprint(inputs, layout, pool, already_paid=True)[0] == before
        emit(dict(event='complete_original_and_reference_preservation', unchanged=unchanged))
        if not unchanged or _sha256(SOURCE) != SOURCE_SHA or _sha256(PREVIOUS/'all_actual_source_tiles_and_transforms.npz') != OLD_ARRAY_SHA:
            raise ValueError('complete original inputs changed')


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
            data, stats = measured(lambda: complete(pool, branch, held, emit), observe=lambda s: result.update(measurement=s))
            result.update(completed=True, necessary_gate_passed=True, data=data)
        except Exception as exc:
            result['failure'] = dict(type=type(exc).__name__, reason=str(exc))
            emit(dict(event='complete_inline_source_failed', **result['failure']))
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
