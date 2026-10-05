"""Offline complete suffix qualification; never invokes a verifier or solver."""

import json
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.hybridz_tf import tf_cnn as cnn
from experiments.neural_hz_20260831.c6_support_affine_plan_v2 import plan
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest, live_value_rows
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
PREVIOUS = EXPERIMENT / 'results/c5_first_terminal_20260905_v1'


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory != EXPERIMENT / 'results/c6_support_affine_plan_20260905_v2':
        raise ValueError('not the exclusive preregistered result directory')
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze = json.loads((directory / 'preregistered.json').read_text())
    if any(_sha256(EXPERIMENT / name) != sha for name, sha in freeze['source_sha256'].items()):
        raise ValueError('source freeze drift before unpickling')
    snapshot = PREVIOUS / 'layer75.pickle'
    if _sha256(snapshot) != 'd08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed':
        raise ValueError('actual snapshot drift')
    events = PREVIOUS / 'composition_events.jsonl'
    old_exit = json.loads((PREVIOUS / 'exit.json').read_text())
    if _sha256(events) != old_exit['artifacts'][events.name]:
        raise ValueError('native journal drift')
    with snapshot.open('rb') as stream:
        saved = pickle.load(stream)
    original = saved['expr_cache'][75]
    net = saved['net']
    dense = net.by_id[77]
    if dense.kind != 'DENSE' or net.by_id[76].kind not in {'FLATTEN', 'RESHAPE'}:
        raise ValueError('actual native affine tail changed')
    if net.preds[77] != [76] or net.preds[76] != [75]:
        raise ValueError('actual native tail graph changed')
    weight = sp.csr_matrix(dense.params['weight'].detach().cpu().double().numpy())
    bias = dense.params.get('bias')
    bias = None if bias is None else bias.detach().cpu().double().numpy().reshape(-1)
    expr = cnn._lazy_append_linear(original, weight, bias, 64_000_000)
    journal = [json.loads(line) for line in events.read_text().splitlines()]
    expected = [entry for entry in journal if entry['event'] == 'materialization_start' and entry['layer'] == 78]
    if len(expected) != 1:
        raise ValueError('actual ReLU78 invocation is not unique')
    sources, terms = {}, []
    for term in expr.terms:
        s = term.source
        if id(s) not in sources:
            sources[id(s)] = {'source_index': len(sources), 'n_out': s.n_out, 'n_cont': s.n_cont,
                'n_bin': s.n_bin, 'live_value_rows': int(live_value_rows(s).sum())}
        terms.append({'source_index': sources[id(s)]['source_index'],
            'operators': [{'type': type(op).__name__, 'shape': list(op.shape)} for op in term.operators]})
    if terms != expected[0]['terms'] or list(sources.values()) != expected[0]['sources'] or expr.n_out != expected[0]['n_out']:
        raise ValueError('complete offline expression differs from actual native schema')
    before = collect(SimpleNamespace(), {'saved': saved, 'expr': expr}).fingerprint
    started = time.monotonic()
    with (directory / 'support_events.jsonl').open('x') as stream:
        def observe(record, visits):
            stream.write(json.dumps({'elapsed_s': time.monotonic() - started,
                'support_integer_visits': visits, 'term': record}) + '\n')
            stream.flush()
        bound, construction = measured_build(lambda: plan(expr, np.ones(expr.n_out, dtype=bool), observe=observe))
    bound.validate()
    after = collect(SimpleNamespace(), {'saved': saved, 'expr': expr}).fingerprint
    if after != before:
        raise ValueError('incoming complete snapshot changed during planning')
    record = {**bound.report, 'support_engine': 'sparse_integer_v2', 'construction': construction, 'source_sha256': freeze['source_sha256'],
        'provenance': freeze['provenance'], 'snapshot_sha256': _sha256(snapshot),
        'actual_native_schema_matched': True, 'incoming_snapshot_unchanged': True,
        'source_payloads': [source_digest(pair[0]) for pair in bound.source_bindings],
        'max_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'status': 'WORK_CAPS_CERTIFIED_NOT_LIVE_QUALIFIED' if bound.report['all_branch_caps_certified']
            and bound.report['whole_256m_product_cap_certified'] else 'WORK_CAPS_NOT_CERTIFIED'}
    _atomic_exclusive_json(directory / 'result.json', record)
    print(json.dumps({'status': record['status'], 'total_product_upper_bound': record['uncached_total_product_upper_bound'],
        'support_visits': record['support_integer_visits'], 'source_count': record['unique_source_count'],
        'terms': len(record['terms']), 'formal_gain': 0}), flush=True)


if __name__ == '__main__':
    main()
