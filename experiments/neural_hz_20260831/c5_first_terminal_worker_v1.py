"""Unchanged candidate to the terminal, with diagnostic-only composition events."""

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from act.back_end.hybridz_tf import tf_cnn as cnn
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831 import c5_integrated_prefix_worker_v2 as prior
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import live_value_rows


def main():
    directory = Path(sys.argv[1]).resolve()
    evidence = ROOT / 'experiments/neural_hz_20260831/evidence/c5_relu71_zero_suffix_20260905_v1.json'
    if prior._sha256(evidence) != '340508f48f5680eba098572229bf640774685a910d911a4c18dc19fbe34457ec':
        raise ValueError('ReLU71 prerequisite drift')
    proof = json.loads(evidence.read_text())
    if not proof.get('all_registered_gates_passed') or not proof.get('actual_target_hz'):
        raise ValueError('ReLU71 prerequisite incomplete')
    native_apply, native_materialize, native_left = HybridzTF.apply, cnn._lazy_materialize, ImplicitConv2DOp.left_compose
    current = [None]
    started = time.monotonic()
    with (directory / 'composition_events.jsonl').open('x') as stream:
        def event(name, **data):
            stream.write(json.dumps({'event': name, 'layer': current[0], 'elapsed_s': time.monotonic() - started, **data}) + '\n')
            stream.flush()

        def applied(self, layer, *args, **kwargs):
            previous, current[0] = current[0], layer.id
            try:
                return native_apply(self, layer, *args, **kwargs)
            finally:
                current[0] = previous

        def materialize(expr, rows, limit, **kwargs):
            source_records = {}
            terms = []
            for term in expr.terms:
                source = term.source
                if id(source) not in source_records:
                    source_records[id(source)] = {'source_index': len(source_records), 'n_out': source.n_out,
                        'n_cont': source.n_cont, 'n_bin': source.n_bin, 'live_value_rows': int(live_value_rows(source).sum())}
                terms.append({'source_index': source_records[id(source)]['source_index'],
                    'operators': [{'type': type(op).__name__, 'shape': list(op.shape)} for op in term.operators]})
            event('materialization_start', n_out=expr.n_out, sources=list(source_records.values()), terms=terms)
            out = native_materialize(expr, rows, limit, **kwargs)
            event('materialization_end', n_cont=out.n_cont, n_bin=out.n_bin, gc_nnz=out.Gc.nnz)
            return out

        def left(op, matrix, max_nnz):
            event('implicit_left_start', operator_shape=list(op.shape), left_shape=list(matrix.shape), left_nnz=matrix.nnz, max_nnz=max_nnz)
            result = native_left(op, matrix, max_nnz)
            event('implicit_left_end', result_shape=list(result.shape), result_nnz=result.nnz)
            return result

        HybridzTF.apply, cnn._lazy_materialize, ImplicitConv2DOp.left_compose = applied, materialize, left
        try:
            prior.main()
        finally:
            HybridzTF.apply, cnn._lazy_materialize, ImplicitConv2DOp.left_compose = native_apply, native_materialize, native_left


if __name__ == '__main__':
    main()
