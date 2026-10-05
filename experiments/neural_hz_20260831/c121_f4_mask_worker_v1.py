# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read-only complete fresh affine census; intentionally no native verdict."""
from dataclasses import asdict
import faulthandler
import json
import os
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from act.back_end.solver.solver_hz import HZSolver
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v1 as prefix
from experiments.neural_hz_20260831.c5_runtime_materializer_v2 import installed as prefix_installed
from experiments.neural_hz_20260831.c5_live_transaction_worker_v1 import caller_roots
from experiments.neural_hz_20260831.c5_admitted_phase_audit_v1 import plain_entry
from experiments.neural_hz_20260831.c8_dyadic_balance_v1 import expression_binding
from experiments.neural_hz_20260831 import c9_live_runtime_v1 as runtime
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c24_dense_graph_v1 import graph
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c102_complete_roots_v1 import collect
from experiments.neural_hz_20260831.c121_f4_mask_cost_v1 import census as mask_census
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json

EXP = Path(__file__).resolve().parent
RUN = EXP / 'results/c121_f4_mask_20260922_v1'


class CensusComplete(BaseException):
    """Leave the original pipeline before source lift, without a solver result."""


class StagePool:
    """A prepaid-charge ceiling on the SAME global pool, never a reset."""
    def __init__(self, parent, cap):
        self.parent, self.start, self.cap = parent, parent.used, cap

    @property
    def used(self):
        return self.parent.used

    @property
    def parts(self):
        return self.parent.parts

    def charge(self, name, amount):
        if self.parent.used-self.start+amount > self.cap:
            raise MemoryError('preregistered shared-pool stage ceiling exceeded')
        self.parent.charge(name, amount)


def archive_arrays(evidence):
    """Only fresh packed mask/cost arrays, never bare original CSR payloads."""
    flat = {}
    def walk(value, path):
        if type(value) is np.ndarray:
            if value.dtype.name not in ('bool','uint16','uint64','int32','int64'):
                raise ValueError('unregistered F4 mask/cost evidence dtype')
            if path in flat:
                raise ValueError('ambiguous numeric evidence path')
            flat[path] = value
        elif type(value) is dict:
            for key, child in value.items():
                walk(child, path+'_'+str(key))
        elif type(value) in (tuple, list):
            for i, child in enumerate(value):
                walk(child, path+'_'+str(i))
        else:
            raise ValueError('only complete packed mask/cost evidence may be archived')
    walk(evidence, 'evidence')
    return flat


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    freeze = json.loads((RUN / 'preregistered.json').read_text())
    pool = WorkPool(256_000_000)
    started = time.monotonic()
    record = dict(completed=False, formal_gain=0, selected_entrances=0,
        solver_calls=0, source_lift_executed=False, source_or_LIVE_admitted=False,
        archived_HZ_loaded=False, numerical_verdict=None)
    native_evaluate, original_argv = HZSolver.evaluate_spec, sys.argv[:]
    fatal = (RUN / 'fatal.log').open('x')
    faulthandler.enable(file=fatal, all_threads=True)
    held = {}

    def drift():
        return any(_sha256(EXP / n) != h for n, h in freeze['source_sha256'].items())

    def input_drift():
        return any(_sha256(Path(n)) != h for n, h in freeze['input_sha256'].items())

    try:
        if drift() or input_drift() or _provenance(ROOT) != freeze['provenance']:
            raise ValueError('frozen source/input/production provenance drift')
        with (RUN / 'events.jsonl').open('x') as stream:
            def emit(event):
                row = dict(elapsed_s=time.monotonic() - started, detail=event)
                stream.write(json.dumps(row) + '\n')
                stream.flush()
                if event['event'].startswith('c121_'):
                    print(json.dumps(row), flush=True)

            def before(state):
                record['selected_entrances'] += 1
                if record['selected_entrances'] != 1:
                    raise ValueError('more than one diagnostic entrance')
                tf, expr = state['tf'], state['expression']
                plain_entry(tf)
                binding = expression_binding(expr)
                held.update(original_caller_roots=caller_roots(),
                    original_expression=expr, apply_args=state['apply_args'],
                    apply_kwargs=state['apply_kwargs'], entry_widths=state['entry_widths'],
                    entry_slots=state['entry_slots'], entry_binding=binding)
                record['selected_layer'] = state['layer'].id
                emit(dict(event='c121_fresh_source_entrance', layer=state['layer'].id,
                    output_width=expr.n_out, terms=len(expr.terms)))

                def census():
                    pool.charge('c121_complete_graph_support_reservation', 96_000_000)
                    keep = np.ones(expr.n_out, dtype=bool)
                    nodes, root, counts, owners, uid_bases = graph(expr, keep,
                        96_000_000, uid_start=0)
                    held.update(graph_nodes=nodes, graph_keep=keep, graph_owners=owners,
                        graph_uid_bases=uid_bases, graph_counts=counts)
                    stage_pool = StagePool(pool, 32_000_000)
                    report, evidence = mask_census(nodes, pool=stage_pool, enabled=True)
                    held.update(census_report=report, census_evidence=evidence)
                    held['archive_array_views'] = archive_arrays(evidence)
                    stage_pool.charge('c121_complete_packed_evidence_encoding',
                        1024+16*sum(int(a.size) for a in held['archive_array_views'].values()))
                    with (RUN / 'complete_census_arrays.npz').open('xb') as output:
                        np.savez(output, **held['archive_array_views'])
                        output.flush()
                        os.fsync(output.fileno())
                    result = dict(report=report, graph=counts, root=root,
                        uid_bases=uid_bases, uid_namespace='graph_local_only_not_native_rows',
                        layer=state['layer'].id, expression_frame=expr.frame_id,
                        before_binding=binding, formal_gain=0,
                        candidate_transformation_executed=False,
                        source_or_LIVE_admitted=False,
                        arrays_sha256=_sha256(RUN / 'complete_census_arrays.npz'))
                    _atomic_exclusive_json(RUN / 'complete_census.json', result)
                    return result

                data, stage = measured(census,
                    observe=lambda m: record.update(census_measurement=m))
                record['census_measurement'] = stage
                emit(dict(event='c121_complete_census_saved', work=pool.used,
                    graph_nodes=len(held['graph_nodes'])))

                def ledger():
                    ledger_pool = StagePool(pool, 80_000_000)
                    roots = collect(tf, held, pool=ledger_pool, enabled=True, observe=emit)
                    layout = roots.measure()
                    if layout.resident_entries > 64_000_000:
                        raise MemoryError('complete held diagnostic numeric entries exceed 64M')
                    return dict(numeric=asdict(layout), fingerprint=roots.fingerprint,
                        python_shallow_bytes=roots.python_shallow_bytes,
                        numeric_roots=len(roots.numeric), unique_objects=roots.unique_objects,
                        schema_counts=roots.schema_counts)

                layout, stage = measured(ledger,
                    observe=lambda m: record.update(ledger_measurement=m))
                if pool.used > 208_000_000:
                    raise MemoryError('complete fresh-mask diagnostic bound exceeded')
                if (expression_binding(expr) != binding
                    or tf._sparse_frame_widths != state['entry_widths']
                    or tf._sparse_relu_slots != state['entry_slots']):
                    raise ValueError('read-only census changed expression or shared frame state')
                _atomic_exclusive_json(RUN / 'complete_held_root_ledger.json', layout)
                record.update(completed=True, complete_held_root_ledger=layout,
                    ledger_measurement=stage, original_expression_preserved=True,
                    graph_node_count=len(held['graph_nodes']), census=data['report'])
                emit(dict(event='c121_complete_read_only_diagnostic', work=pool.used,
                    numeric_entries=layout['numeric']['resident_entries']))
                raise CensusComplete()

            def forbidden_solver(*args, **kwargs):
                record['solver_calls'] += 1
                raise runtime.SelectedRejected('C121 census does not authorize a solver call')

            HZSolver.evaluate_spec = forbidden_solver
            command = freeze['fresh_pipeline_command']
            sys.argv = [__file__, str(RUN), *command[3:]]
            with prefix_installed(enabled=True, emit=emit):
                with runtime.installed(enabled=True, before=before, emit=emit):
                    prefix.main()
            raise ValueError('fresh pipeline returned without the structural census entrance')
    except CensusComplete:
        if not record['completed']:
            record['failure'] = dict(type='IncompleteCensus', reason='completion without evidence')
    except (runtime.SelectedRejected, Exception) as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
        record['completed'] = False
    finally:
        HZSolver.evaluate_spec, sys.argv = native_evaluate, original_argv
        faulthandler.disable()
        fatal.close()
        record.update(work=pool.used, work_parts=pool.parts,
            wall_s=time.monotonic() - started, source_drift=drift(), input_drift=input_drift(),
            provenance_drift=_provenance(ROOT) != freeze['provenance'],
            numeric_hash_traffic_in_token_pool=False, all_CPU_work_in_generation_cap=False)
        _atomic_exclusive_json(RUN / 'result.json', record)
        print(json.dumps({k: v for k, v in record.items()
            if k not in ('census', 'complete_held_root_ledger')}), flush=True)
    if (not record['completed'] or record.get('failure') or record['solver_calls']
        or record['source_drift'] or record['input_drift'] or record['provenance_drift']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
