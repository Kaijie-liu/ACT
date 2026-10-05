"""Fresh ReLU71 with audited native empty transfers and complete live metric."""

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
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831 import c5_corrected_prefix_worker_v2 as prefix
from experiments.neural_hz_20260831.c5_runtime_materializer_v2 import installed
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c5_live_transaction_worker_v1 import caller_roots
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_reference_lower_bound_v1 import lower_bound_roots
from experiments.neural_hz_20260831.c5_partial_csr_owner_ledger_v3 import snapshot_partial_csr_owners
from experiments.neural_hz_20260831.s0_c2_whole_state_ledger_prototype import WholeStateRoots
from experiments.neural_hz_20260831.c5_zero_suffix_audit_v1 import verify_zero_transfer, verify_negative_relu
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
OUTPUT = EXPERIMENT / 'evidence/c5_relu71_zero_suffix_20260905_v1.json'


class AuditRejected(BaseException):
    pass


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory.parent != EXPERIMENT / 'results' or OUTPUT.exists():
        raise ValueError('unsafe/occupied zero-suffix audit')
    proof_path = EXPERIMENT / 'evidence/c5_relu44_admitted_20260905_v1.json'
    if _sha256(proof_path) != '72a1d01c07f89faed5e5f1545e9db79a274c2226f62ca90c3ca45403ec560931':
        raise ValueError('admitted-phase prerequisite drift')
    proof = json.loads(proof_path.read_text())
    if not proof.get('all_registered_gates_passed') or not proof.get('actual_publication_executed'):
        raise ValueError('admitted-phase prerequisite incomplete')
    started = time.monotonic()
    record = {'schema': 'c5_relu71_zero_suffix_v1', 'formal_gain': 0, 'terminal_solve_executed': False,
        'empty_transfers': [], 'negative_relus': [], 'unqualified_nonempty_suffix': [],
        'candidate_runtime': 'c5_runtime_materializer_v2', 'provenance': prefix.v1.worker._provenance(ROOT)}
    stack, pending = [], {}

    def reject(exc):
        record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
        raise AuditRejected() from exc

    old_apply, old_propagate = HybridzTF.apply, HybridzTF._propagate_sparse_hz
    try:
        with (directory / 'c5_events.jsonl').open('x') as stream:
            def emit(event):
                stream.write(json.dumps(event) + '\n')
                stream.flush()
            with installed(enabled=True, emit=emit):
                old_deferred, old_materialize = cnn._try_deferred_expr_conv_relu, cnn._lazy_materialize

                def deferred(layer, expr, result, tf):
                    island = cnn._deferred_relu_island(layer, tf)
                    if island is None:
                        return old_deferred(layer, expr, result, tf)
                    stack.append((tf, island[1].id))
                    try:
                        return old_deferred(layer, expr, result, tf)
                    finally:
                        stack.pop()

                def materialize(expr, keep, limit, **kwargs):
                    if not stack:
                        return old_materialize(expr, keep, limit, **kwargs)
                    tf, relu_id = stack[-1]
                    mask = np.asarray(keep, dtype=bool).reshape(-1)
                    if np.any(mask):
                        if relu_id > 44:
                            record['unqualified_nonempty_suffix'].append({'relu': relu_id, 'selected_rows': int(mask.sum())})
                        return old_materialize(expr, keep, limit, **kwargs)
                    try:
                        if relu_id in pending:
                            raise ValueError('duplicate native empty transfer for one ReLU')
                        hashes = [source_digest(term.source) for term in expr.terms]
                        slots = dict(tf._sparse_relu_slots)
                        original_row = ImplicitConv2DOp._row
                        def forbidden_row(*args, **kwargs):
                            raise ValueError('native empty transfer requested an implicit Conv row')
                        ImplicitConv2DOp._row = forbidden_row
                        try:
                            actual, construction = measured_build(lambda: old_materialize(expr, keep, limit, **kwargs))
                        finally:
                            ImplicitConv2DOp._row = original_row
                        fields = verify_zero_transfer(expr, actual)
                        if [source_digest(term.source) for term in expr.terms] != hashes or tf._sparse_relu_slots != slots:
                            raise ValueError('native empty transfer mutated sources or slots')
                        pending[relu_id] = (actual, slots)
                        event = {'relu': relu_id, 'source_terms': len(expr.terms), 'unique_sources': len({id(t.source) for t in expr.terms}),
                            'source_sha256': hashes, 'bitwise_checks': fields, 'no_operator_rows': True,
                            'construction': construction, 'n_cont': actual.n_cont, 'n_bin': actual.n_bin,
                            'n_eq': actual.n_eq, 'n_ineq': actual.n_ineq, 'frame_id': actual.frame_id}
                        record['empty_transfers'].append(event)
                        print(json.dumps({'event': 'verified_empty_transfer', 'relu': relu_id, 'source_terms': len(expr.terms),
                                          'n_bin': actual.n_bin, 'construction': construction}), flush=True)
                        return actual
                    except Exception as exc:
                        reject(exc)

                def propagated(self, layer, bounds, interval_result):
                    actual_fact = old_propagate(self, layer, bounds, interval_result)
                    if layer.id in pending:
                        try:
                            preactivation, slots = pending.pop(layer.id)
                            actual = self._sparse_hz_cache.get(layer.id)
                            if actual is None or self._sparse_relu_slots != slots:
                                raise ValueError('native negative ReLU missing HZ or changed slots')
                            checks = verify_negative_relu(preactivation, actual, bounds, self._sparse_frame_widths)
                            record['negative_relus'].append({'relu': layer.id, 'all_fields': checks,
                                'slots_unchanged': True, 'actual_hz_sha256': source_digest(actual),
                                'n_cont': actual.n_cont, 'n_bin': actual.n_bin, 'n_eq': actual.n_eq, 'n_ineq': actual.n_ineq})
                        except Exception as exc:
                            reject(exc)
                    return actual_fact

                def applied(self, layer, *args, **kwargs):
                    result = old_apply(self, layer, *args, **kwargs)
                    if layer.id == 71:
                        try:
                            actual = self._sparse_hz_cache.get(71)
                            record['actual_target_hz'] = actual is not None
                            if actual is None or not actual.exact or not any(e['relu'] == 71 for e in record['negative_relus']):
                                raise ValueError('registered ReLU71 zero-output boundary not established')
                            if record['unqualified_nonempty_suffix'] or pending:
                                raise ValueError('new nonempty or unconsumed suffix requires separate qualification')
                            extra = {**caller_roots(), 'applied_fact': result}
                            roots = collect(self, extra)
                            candidate = roots.measure()
                            reference, certificates = lower_bound_roots(roots, self._net)
                            lower = snapshot_partial_csr_owners(WholeStateRoots(active=reference, consumer_gc_enabled=False))
                            passed = candidate.resident_bytes < lower.resident_bytes and candidate.resident_entries < lower.resident_entries
                            record['actual_target_boundary'] = {'complete_candidate': asdict(candidate), 'reference_lower_bound': asdict(lower),
                                'reference_certificate': certificates[0], 'registered_numeric_roots': len(roots.numeric),
                                'python_shallow_bytes': roots.python_shallow_bytes, 'passed': passed}
                            if not passed or collect(self, extra).fingerprint != roots.fingerprint:
                                raise ValueError('ReLU71 complete-state physical gate or input fingerprint failed')
                            record['all_registered_gates_passed'] = True
                            record['actual_target_hz_sha256'] = source_digest(actual)
                        except Exception as exc:
                            reject(exc)
                    return result

                cnn._try_deferred_expr_conv_relu, cnn._lazy_materialize = deferred, materialize
                HybridzTF.apply, HybridzTF._propagate_sparse_hz = applied, propagated
                try:
                    prefix.main()
                finally:
                    cnn._try_deferred_expr_conv_relu, cnn._lazy_materialize = old_deferred, old_materialize
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
