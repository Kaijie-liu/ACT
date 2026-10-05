"""Fresh qualified fused C10 followed by unchanged observed terminal solving."""

from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from act.back_end.solver.solver_hz import HZSolver
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v1 as prefix
from experiments.neural_hz_20260831.c5_runtime_materializer_v2 import installed as c5_installed
from experiments.neural_hz_20260831 import c10_live_runtime_v1 as runtime
from experiments.neural_hz_20260831.c10_portable_binding_v1 import verify
from experiments.neural_hz_20260831.c10_terminal_observer_v1 import observed
from experiments.neural_hz_20260831.c9_terminal_binding_v1 import bind
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_live_transaction_worker_v1 import caller_roots
from experiments.neural_hz_20260831.c7_factored_hz_audit_v1 import reference_subset
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory != EXP / 'results/c10_first_terminal_20260908_v1':
        raise ValueError('not the exclusive terminal directory')
    freeze = json.loads((directory / 'preregistered.json').read_text())
    if any(_sha256(EXP / name) != sha for name, sha in freeze['source_sha256'].items()):
        raise ValueError('terminal source freeze drift')
    proof = json.loads((EXP / 'results/c10_live_relu_20260908_v1/qualification.json').read_text())
    proof_binding = json.loads((EXP / 'results/c10_live_relu_20260908_v1/proof_binding.json').read_text())
    record = {'schema': 'c10_first_terminal_audit_v1', 'formal_gain': 0,
        'terminal_solve_executed': False, 'terminal_gates_passed': False,
        'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance']}
    retained, start = [], time.monotonic()
    native_evaluate = HZSolver.evaluate_spec
    try:
        with (directory / 'events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps({'elapsed_s': time.monotonic() - start, **event}) + '\n')
                stream.flush()

            def ready(state):
                record['pre_relu_binding'] = verify(state['lifted'], proof_binding)
                retained.append(state)

            def consumed(state, fact):
                if len(retained) != 1:
                    raise ValueError('terminal diagnostic requires one qualified structural island')
                record['post_relu_binding'] = bind(state, proof)
                record['construction'] = state['construction']
                record['relu_construction'] = state['consumer_construction']
                emit({'event': 'fresh_post_relu_matches_audited_hz', **record['post_relu_binding']})

            def evaluated(solver, output_hz, out_spec, **kwargs):
                if len(retained) != 1 or not record.get('post_relu_binding') or record['terminal_solve_executed']:
                    raise runtime.SelectedRejected('terminal called without unique qualified live prefix')
                if output_hz is None or kwargs.get('input_hz') is None or output_hz.frame_id != kwargs['input_hz'].frame_id:
                    raise runtime.SelectedRejected('terminal missing original shared input/output HZ')
                if kwargs.get('timelimit') != 45. or solver.neural_hz_projection or solver.neural_hz_phase_fixing:
                    raise runtime.SelectedRejected('terminal budget or forbidden rescue option changed')
                tf = retained[0]['tf']
                extra = {**caller_roots(), 'retained_runtime': [runtime.numeric_roots(item) for item in retained],
                    'actual_terminal_output': output_hz, 'actual_terminal_input': kwargs['input_hz'],
                    'scalar_proof_binding': proof_binding}
                roots = collect(tf, extra)
                candidate = roots.measure()
                witness, reference = reference_subset(roots, tf._net)
                lower = reference['reference_lower_bound']
                physical = candidate.resident_bytes < lower['resident_bytes'] and candidate.resident_entries < lower['resident_entries']
                record.update(whole_live_state=asdict(candidate), whole_live_numeric_roots=len(roots.numeric),
                    reference=reference, physical_decrease=physical)
                if not physical or collect(tf, extra).fingerprint != roots.fingerprint:
                    raise runtime.SelectedRejected('terminal physical/input-integrity gate failed')
                native = inspect(output_hz)
                record['final_native_ingestion'] = native
                if not native['passed']:
                    raise runtime.SelectedRejected('final HZ native fidelity failed')
                record.update(terminal_gates_passed=True, final_hz_sha256=source_digest(output_hz))
                # Retain a completed gate record even if ordinary solving times out.
                _atomic_exclusive_json(directory / 'terminal_gate.json', record)
                emit({'event': 'ordinary_terminal_start', 'final_native_nnz': native['retained_matrix_nnz'],
                    'live_bytes': candidate.resident_bytes, 'solver_seconds': kwargs['timelimit']})
                tick = time.monotonic()
                record['terminal_solve_executed'] = True
                record['ordinary_solver_observations'] = []
                def observed_event(event):
                    record['ordinary_solver_observations'].append(event)
                    emit(event)
                with observed(observed_event):
                    result = native_evaluate(solver, output_hz, out_spec, **kwargs)
                record['ordinary_terminal_wall_s'] = time.monotonic() - tick
                record['ordinary_terminal_returned'] = True
                emit({'event': 'ordinary_terminal_returned', 'wall_s': record['ordinary_terminal_wall_s']})
                return result

            HZSolver.evaluate_spec = evaluated
            with c5_installed(enabled=True, emit=emit):
                with runtime.installed(enabled=True, ready=ready, consumed=consumed, emit=emit):
                    prefix.main()
        if not record.get('ordinary_terminal_returned'):
            raise ValueError('ordinary terminal did not return')
    except (runtime.SelectedRejected, Exception) as exc:
        record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
    finally:
        HZSolver.evaluate_spec = native_evaluate
        record.update(wall_s=time.monotonic() - start,
            max_rss_kib_including_oracles=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(directory / 'terminal_audit.json', record)
        print(json.dumps({'terminal_gates_passed': record['terminal_gates_passed'],
            'terminal_solve_executed': record['terminal_solve_executed'], 'failure': record.get('failure')}), flush=True)
    if record.get('failure'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
