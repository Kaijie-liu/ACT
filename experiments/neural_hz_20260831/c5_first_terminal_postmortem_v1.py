"""Inventory the closed actual terminal failure without evaluating a new path."""

import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest, live_value_rows
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
DIRECTORY = EXPERIMENT / 'results/c5_first_terminal_20260905_v1'
OUTPUT = EXPERIMENT / 'evidence/c5_first_terminal_postmortem_20260905_v1.json'


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    exit_record = json.loads((DIRECTORY / 'exit.json').read_text())
    if exit_record.get('timeout_s') != 240 or exit_record.get('source_drift') or exit_record.get('provenance_drift'):
        raise ValueError('closed terminal evidence differs from registered postmortem')
    prereg = json.loads((DIRECTORY / 'preregistered.json').read_text())
    if any(_sha256(EXPERIMENT / name) != sha for name, sha in prereg['source_sha256'].items()):
        raise ValueError('candidate source drift')
    for name in ('layer75.pickle', 'layer75.snapshot.json', 'composition_events.jsonl', 'events.jsonl'):
        if _sha256(DIRECTORY / name) != exit_record['artifacts'][name]:
            raise ValueError('terminal artifact drift')
    with (DIRECTORY / 'layer75.pickle').open('rb') as stream:
        saved = pickle.load(stream)
    expr = saved['expr_cache'][75]
    sources, terms = {}, []
    for index, term in enumerate(expr.terms):
        source = term.source
        key = id(source)
        if key not in sources:
            live = live_value_rows(source)
            sources[key] = {'source_index': len(sources), 'source_sha256': source_digest(source),
                'live_value_rows': int(live.sum()), 'n_out': source.n_out, 'n_cont': source.n_cont,
                'n_bin': source.n_bin, 'n_eq': source.n_eq, 'n_ineq': source.n_ineq, 'exact': source.exact,
                'frame_id': source.frame_id}
        operators = []
        for position, op in enumerate(term.operators):
            item = {'position': position, 'type': type(op).__name__, 'shape': list(op.shape)}
            if type(op) is ImplicitConv2DOp:
                item.update(logical_expanded_nnz=op.logical_expanded_nnz, input_shape=list(op.input_shape),
                            kernel_shape=list(op._kernel.shape), kernel_nonzeros=int(np.count_nonzero(op._kernel)))
            elif sp.isspmatrix_csr(op) and op.shape[0] == op.shape[1]:
                diagonal = op.diagonal()
                remainder = (op - sp.diags(diagonal, format='csr')).tocsr()
                remainder.eliminate_zeros()
                item.update(is_diagonal=remainder.nnz == 0, diagonal_nonzeros=int(np.count_nonzero(diagonal)))
            operators.append(item)
        terms.append({'index': index, 'source_index': sources[key]['source_index'], 'operators': operators})
    events = [json.loads(line) for line in (DIRECTORY / 'composition_events.jsonl').read_text().splitlines()]
    pending = [event for event in events if event['layer'] == 78 and event['event'] == 'implicit_left_start']
    completed = [event for event in events if event['layer'] == 78 and event['event'] == 'implicit_left_end']
    if len(pending) != 1 or completed:
        raise ValueError('first terminal pending composition changed')
    first = expr.terms[0]
    op = first.operators[0]
    if type(op) is not ImplicitConv2DOp or list(op.shape) != pending[0]['operator_shape']:
        raise ValueError('pending operator is not the actual first source path')
    dense_left = pending[0]['left_nnz'] == int(np.prod(pending[0]['left_shape']))
    record = {'schema': 'c5_first_terminal_postmortem_v1', 'formal_gain': 0, 'new_path_executed': False,
        'auditor_sha256': _sha256(Path(__file__)), 'prereg_sha256': _sha256(EXPERIMENT / 'C5_TERMINAL_POSTMORTEM_PREREG_20260905.md'),
        'exit_sha256': _sha256(DIRECTORY / 'exit.json'), 'snapshot_sha256': _sha256(DIRECTORY / 'layer75.pickle'),
        'provenance': prereg['provenance'], 'terms': terms, 'sources': list(sources.values()),
        'zero_source_term_indices': [t['index'] for t in terms if list(sources.values())[t['source_index']]['live_value_rows'] == 0],
        'pending_composition': pending[0], 'pending_source': list(sources.values())[terms[0]['source_index']],
        'recorded_left_has_full_stored_pattern': dense_left,
        'structural_multiply_bound_if_full_left': pending[0]['left_shape'][0] * op.logical_expanded_nnz if dense_left else None,
        'no_terminal_result': not (DIRECTORY / 'result.json').exists(),
        'no_final_hz_checkpoint': not (DIRECTORY / 'final_hz.pickle').exists(),
        'interpretation': 'composition of a zero-value source is avoidable only if its full predicates/frame remain in the joined HZ'}
    _atomic_exclusive_json(OUTPUT, record)
    print(json.dumps({'output': str(OUTPUT), 'sha256': _sha256(OUTPUT), 'zero_source_terms': record['zero_source_term_indices'],
        'unique_sources': len(sources), 'terms': len(terms), 'structural_multiply_bound': record['structural_multiply_bound_if_full_left']}), flush=True)


if __name__ == '__main__':
    main()
