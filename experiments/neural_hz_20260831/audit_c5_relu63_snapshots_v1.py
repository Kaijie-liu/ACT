"""Read sealed actual objects, archive a prefix-only audit, never a verdict."""

import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy.sparse as sp

from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest

EXPERIMENT = Path(__file__).resolve().parent
DIRECTORY = EXPERIMENT / 'results/c5_integrated_relu63_20260905_v1'
OUTPUT = EXPERIMENT / 'evidence/c5_relu63_snapshot_audit_20260905_v1.json'


def hz_record(hz):
    if not hz.exact or hz.frame_id != 1:
        raise ValueError('snapshot HZ exactness/frame mismatch')
    for name in ('c', 'Gc', 'Gb', 'Ac', 'Ab', 'b', 'Auc', 'Aub', 'ub'):
        value = getattr(hz, name)
        if sp.issparse(value):
            value = value.data
        if not np.isfinite(value).all():
            raise ValueError('nonfinite snapshot HZ')
    return {name: getattr(hz, name) for name in ('exact', 'frame_id', 'n_out', 'n_cont', 'n_bin', 'n_eq', 'n_ineq')} | {
        'source_sha256': source_digest(hz), 'gc_nnz': hz.Gc.nnz, 'gb_nnz': hz.Gb.nnz}


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    result_sha = '62f33f7aafd320e9804fbc5f22a78fc8b2465552ade32161b7e71de5436a051d'
    if _sha256(DIRECTORY / 'result.json') != result_sha:
        raise ValueError('registered ReLU63 result drift')
    prereg = json.loads((DIRECTORY / 'preregistered.json').read_text())
    if any(_sha256(EXPERIMENT / path) != sha for path, sha in prereg['source_sha256'].items()):
        raise ValueError('candidate source drift before loading local snapshots')
    record = {'schema': 'c5_relu63_actual_snapshot_audit_v1', 'formal_gain': 0, 'prefix_only': True,
        'result_sha256': result_sha, 'auditor_sha256': _sha256(Path(__file__)), 'nodes': [],
        'provenance': prereg['provenance'], 'input_source_manifest_verified': True}
    try:
        for lid in (36, 44, 55, 63):
            path = DIRECTORY / f'layer{lid:02d}.pickle'
            seal = json.loads((DIRECTORY / f'layer{lid:02d}.snapshot.json').read_text())
            if _sha256(path) != seal['pickle_sha256']:
                raise ValueError('snapshot seal mismatch')
            with path.open('rb') as stream:
                state = pickle.load(stream)
            if state['layer'] != lid:
                raise ValueError('snapshot observation-layer mismatch')
            node = {'layer': lid, 'pickle_sha256': seal['pickle_sha256'], 'pickle_bytes': path.stat().st_size}
            hz, expr = state['hz_cache'].get(lid), state['expr_cache'].get(lid)
            if hz is not None:
                node.update(representation='exact_sparse_hz', state=hz_record(hz))
            elif expr is not None:
                if expr.frame_id != 1 or not np.isfinite(expr.bias).all():
                    raise ValueError('expression frame/bias mismatch')
                node.update(representation='exact_affine_expression', n_out=expr.n_out, frame=expr.frame_id, terms=[])
                for term in expr.terms:
                    for op in term.operators:
                        for field in ('data', '_kernel', '_row_mask', '_diagonal'):
                            array = getattr(op, field, None)
                            if array is not None and not np.isfinite(array).all():
                                raise ValueError('nonfinite expression operator')
                    node['terms'].append({'operator_count': len(term.operators), 'source': hz_record(term.source)})
            else:
                raise ValueError('target-named snapshot without an actual exact state')
            record['nodes'].append(node)
        record['actual_relu63_hz_passed'] = record['nodes'][-1]['representation'] == 'exact_sparse_hz'
        record['next_required_gate'] = 'ReLU44 admitted phase-selective live equivalence/publication/physical audit before ReLU71'
    except Exception as exc:
        record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
    finally:
        _atomic_exclusive_json(OUTPUT, record)
        print(json.dumps({'output': str(OUTPUT), 'sha256': _sha256(OUTPUT),
                          'actual_relu63_hz_passed': record.get('actual_relu63_hz_passed', False),
                          'failure': record.get('failure')}), flush=True)


if __name__ == '__main__':
    main()
