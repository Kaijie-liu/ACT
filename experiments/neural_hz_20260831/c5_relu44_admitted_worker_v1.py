"""One actual admitted-phase execution with independent test-only oracles."""

import copy
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np

from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v2 as prefix
from experiments.neural_hz_20260831.c5_runtime_materializer_v2 import installed
from experiments.neural_hz_20260831.c5_native_budgeted_materialization_v1 import BudgetedMaterializer
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_live_transaction_worker_v1 import caller_roots, matches
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_ordered_row_oracle_v3 import reference_materialize, compare_hz, equal_payload
from experiments.neural_hz_20260831.c5_reference_lower_bound_v1 import lower_bound_roots, reachable_operators
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots
from experiments.neural_hz_20260831.c5_admitted_phase_audit_v1 import (
    plain_entry, stable_fingerprint, verify_phase, verify_precomputed, verify_consumed,
)
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
OUTPUT = EXPERIMENT / 'evidence/c5_relu44_admitted_20260905_v1.json'


class AuditRejected(BaseException):
    pass


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory.parent != EXPERIMENT / 'results' or OUTPUT.exists():
        raise ValueError('unsafe/occupied admitted-phase diagnostic')
    prerequisite = EXPERIMENT / 'evidence/c5_relu36_live_20260905_v2.json'
    if _sha256(prerequisite) != 'b45d017ac76a55fdc535d411f4c1ee9aa93496b0d603d18da1b286dc02d7e2c3':
        raise ValueError('ReLU36 qualification drift')
    started = time.monotonic()
    record = {'schema': 'c5_relu44_admitted_v1', 'formal_gain': 0, 'boundaries': {}, 'terminal_solve_executed': False,
        'measurement_point_only': 44, 'candidate_runtime': 'c5_runtime_materializer_v2',
        'provenance': prefix.v1.worker._provenance(ROOT)}
    state, producer = {}, {}

    def reject(exc):
        record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
        raise AuditRejected() from exc

    def unchanged_sources():
        if [source_digest(term.source) for term in state['expr'].terms] != state['source_sha256']:
            raise ValueError('admitted source/predicate drift')

    def measured_boundary(name, tf, additional=None):
        candidate = state['candidate']
        extra = {**state['extra'], 'retained_probe': state['probe'], 'retained_core': candidate.core,
            'retained_expression': candidate.expression, 'retained_phase_bounds': candidate.output_bounds, **(additional or {})}
        roots = collect(tf, extra)
        if not any(op is state['witness_op'] for op in reachable_operators(roots)):
            raise ValueError('reference subset no longer strongly reachable')
        metric = roots.measure()
        lower = state['reference_lower_bound']
        passed = metric.resident_bytes < lower.resident_bytes and metric.resident_entries < lower.resident_entries
        record['boundaries'][name] = {'complete_candidate': asdict(metric), 'reference_lower_bound': asdict(lower),
            'passed': passed, 'registered_numeric_roots': len(roots.numeric), 'python_shallow_bytes': roots.python_shallow_bytes}
        if not passed:
            raise ValueError('admitted complete-state physical strict bound failed')
        unchanged_sources()
        print(json.dumps({'event': 'admitted_boundary', 'name': name, 'bytes': metric.resident_bytes, 'passed': passed}), flush=True)

    old_apply, old_propagate = HybridzTF.apply, HybridzTF._propagate_sparse_hz
    try:
        with (directory / 'c5_events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps(event) + '\n')
                stream.flush()
            with installed(enabled=True, emit=emit):
                native_phase, native_deferred = cnn._try_phase_selective_exact_relu, cnn._try_deferred_expr_conv_relu

                def phase(expr, bounds, tf, layer):
                    if layer.id != 44:
                        return native_phase(expr, bounds, tf, layer)
                    try:
                        if state or not matches(expr):
                            raise ValueError('unexpected admitted diagnostic structure/visit')
                        entry = plain_entry(tf)
                        extra = {**caller_roots(), **producer, 'current_expr': expr, 'authoritative_bounds': bounds}
                        roots = collect(tf, extra)
                        entry_metric = roots.measure()
                        entry_op_ids = {id(op) for op in reachable_operators(roots)}
                        before = stable_fingerprint(tf, extra)
                        before_all = roots.fingerprint
                        old_count = tf._neural_hz_phase_selective_relus
                        old_profile = copy.deepcopy(tf._neural_hz_phase_selective_profile)
                        state.update(expr=expr, bounds=bounds, extra=extra, entry=entry,
                                     source_sha256=[source_digest(t.source) for t in expr.terms])
                        captured, matrices = {}, {}
                        native_materialize, native_run = cnn._lazy_materialize, BudgetedMaterializer.run

                        def observed_run(budget, current, rows, *, observe=None):
                            def visit(index, bound):
                                matrices[index] = bound.matrix
                                if observe is not None:
                                    observe(index, bound)
                            output, stats = native_run(budget, current, rows, observe=visit)
                            captured['stats'] = stats
                            captured['sequence_products'] = 256_000_000 - budget.remaining_whole
                            return output, stats

                        def materialize(current, mask, limit, **kwargs):
                            if captured.get('probe') is not None or current is not expr:
                                raise ValueError('admitted native phase requested more than one probe')
                            output = native_materialize(current, mask, limit, **kwargs)
                            captured['probe'], captured['rows'] = output, np.flatnonzero(mask)
                            if collect(tf, extra).fingerprint != before_all:
                                raise ValueError('C5 probe mutated incoming registered roots')
                            return output

                        cnn._lazy_materialize, BudgetedMaterializer.run = materialize, observed_run
                        try:
                            candidate, construction = measured_build(lambda: native_phase(expr, bounds, tf, layer))
                        finally:
                            cnn._lazy_materialize, BudgetedMaterializer.run = native_materialize, native_run
                        if candidate is None or len(matrices) != len(expr.terms) or 'probe' not in captured:
                            raise ValueError('registered native admitted phase incomplete')
                        state.update(candidate=candidate, probe=captured['probe'])
                        record.update(candidate_construction=construction, native_requests=1, selected_rows=int(captured['rows'].size),
                            term_stats=captured['stats'], sequence_products=captured['sequence_products'], source_sha256=state['source_sha256'])
                        print(json.dumps({'event': 'admitted_candidate_complete', 'rows': record['selected_rows'], 'construction': construction}), flush=True)
                        checks = {}
                        def observe_oracle(index, matrix):
                            checks[index] = equal_payload(matrices[index], matrix[captured['rows']])
                            print(json.dumps({'event': 'admitted_oracle_branch', 'index': index, 'bitwise': checks[index]}), flush=True)
                        oracle = reference_materialize(expr, captured['rows'], observe=observe_oracle)
                        fields = compare_hz(captured['probe'], oracle)
                        fields['all_operators'] = len(checks) == len(expr.terms) and all(checks.values())
                        record['probe_bitwise_checks'] = fields
                        if not all(fields.values()):
                            raise ValueError('admitted probe/scalar oracle mismatch')
                        record['phase_proof'] = verify_phase(expr, oracle, bounds, layer, candidate, tf, entry)
                        if stable_fingerprint(tf, extra) != before:
                            raise ValueError('unregistered phase side effect')
                        if tf._neural_hz_phase_selective_relus != old_count + 1 or tf._neural_hz_phase_selective_profile[:-1] != old_profile:
                            raise ValueError('native phase profile/count mutation mismatch')
                        record['native_phase_profile'] = copy.deepcopy(tf._neural_hz_phase_selective_profile[-1])
                        state['post_slots'], state['post_widths'] = dict(tf._sparse_relu_slots), dict(tf._sparse_frame_widths)
                        # All actual candidate construction preceded every oracle.
                        bound_roots, certificates = lower_bound_roots(roots, tf._net)
                        state['witness_op'] = max(reachable_operators(roots), key=lambda op: op.logical_expanded_nnz)
                        if id(state['witness_op']) not in entry_op_ids:
                            raise ValueError('reference witness absent at entry')
                        lower = snapshot_partial_csr_owners(WholeStateRoots(active=bound_roots, consumer_gc_enabled=False))
                        state['reference_lower_bound'] = lower
                        record['reference_certificate'] = certificates[0]
                        passed = entry_metric.resident_bytes < lower.resident_bytes and entry_metric.resident_entries < lower.resident_entries
                        record['boundaries']['entry'] = {'complete_candidate': asdict(entry_metric), 'reference_lower_bound': asdict(lower),
                            'passed': passed, 'registered_numeric_roots': len(roots.numeric), 'python_shallow_bytes': roots.python_shallow_bytes}
                        if not passed:
                            raise ValueError('admitted entry physical bound failed')
                        measured_boundary('phase_return', tf)
                        return candidate
                    except Exception as exc:
                        reject(exc)

                def deferred(layer, expr, result, tf):
                    previous = dict(producer)
                    producer.update(incoming_expr=expr, producer_fact=result)
                    try:
                        answer = native_deferred(layer, expr, result, tf)
                        if layer.id == 41 and state.get('candidate') is not None:
                            record['precomputed_proof'] = verify_precomputed(tf, 44, state['candidate'], state['bounds'])
                            measured_boundary('precomputed_publication', tf)
                        return answer
                    except Exception as exc:
                        reject(exc)
                    finally:
                        producer.clear()
                        producer.update(previous)

                def propagated(self, layer, bounds, interval_result):
                    answer = old_propagate(self, layer, bounds, interval_result)
                    if layer.id == 44 and state.get('candidate') is not None:
                        try:
                            record['consumption_proof'] = verify_consumed(self, layer.id, state['candidate'], interval_result, answer)
                        except Exception as exc:
                            reject(exc)
                    return answer

                def applied(self, layer, *args, **kwargs):
                    answer = old_apply(self, layer, *args, **kwargs)
                    if layer.id == 44 and state.get('candidate') is not None:
                        try:
                            if self._sparse_relu_slots != state['post_slots'] or self._sparse_frame_widths != state['post_widths']:
                                raise ValueError('native publication changed reserved slots')
                            measured_boundary('actual_relu_application', self, {'applied_fact': answer})
                            if not record.get('consumption_proof'):
                                raise ValueError('actual native consumption not observed')
                            record['all_registered_gates_passed'] = True
                            record['actual_publication_executed'] = True
                        except Exception as exc:
                            reject(exc)
                    return answer

                cnn._try_phase_selective_exact_relu, cnn._try_deferred_expr_conv_relu = phase, deferred
                HybridzTF.apply, HybridzTF._propagate_sparse_hz = applied, propagated
                try:
                    prefix.main()
                finally:
                    cnn._try_phase_selective_exact_relu, cnn._try_deferred_expr_conv_relu = native_phase, native_deferred
                    HybridzTF.apply, HybridzTF._propagate_sparse_hz = old_apply, old_propagate
    except AuditRejected:
        pass
    except Exception as exc:
        record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
    finally:
        record.update(wall_s=time.monotonic() - started, worker_sha256=_sha256(Path(__file__)),
            max_rss_kib_including_oracles=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(OUTPUT, record)
        print(json.dumps({'output': str(OUTPUT), 'sha256': _sha256(OUTPUT), 'failure': record.get('failure'),
                          'all_registered_gates_passed': record.get('all_registered_gates_passed', False)}), flush=True)


if __name__ == '__main__':
    main()
