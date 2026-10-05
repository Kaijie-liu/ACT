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
from experiments.neural_hz_20260831.c9_radix_predicate_v1 import pack
from experiments.neural_hz_20260831.c9_radix_predicate_audit_v1 import audit
from experiments.neural_hz_20260831.c8_native_ingestion_v1 import inspect
from experiments.neural_hz_20260831.c5_zero_suffix_audit_v1 import zero_transfer_reference
from experiments.neural_hz_20260831.c7_factored_hz_audit_v1 import reference_subset
from dataclasses import asdict
import os
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest, live_value_rows
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXPERIMENT = Path(__file__).resolve().parent
PREVIOUS = EXPERIMENT / 'results/c5_first_terminal_20260905_v1'


def main():
    directory = Path(sys.argv[1]).resolve()
    if directory != EXPERIMENT / 'results/c9_predicate_rows_20260905_v1':
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
    base = zero_transfer_reference(expr)
    before = collect(SimpleNamespace(), {'saved': saved, 'expr': expr, 'base': base}).fingerprint
    started = time.monotonic()
    packed, construction = measured_build(lambda: pack(base, enabled=True))
    print(json.dumps({'event': 'predicate_constructed', 'report': packed.report, 'construction': construction}), flush=True)
    audit_start = time.monotonic()
    identity = audit(packed)
    identity['stage_elapsed_s'] = time.monotonic() - audit_start
    print(json.dumps({'event': 'all_predicates_audited', **identity}), flush=True)
    original_native = inspect(base)
    native = inspect(packed.hz)
    print(json.dumps({'event': 'native_ingestion', 'original_changed_coefficients': original_native['different_coefficients'],
        'encoded_changed_coefficients': native['different_coefficients'], 'passed': native['passed']}), flush=True)
    checkpoint_payload = {'schema': 'c9_predicate_checkpoint_v1', **packed.numeric_roots(),
        'report': packed.report, 'identity': identity, 'provenance': freeze['provenance'],
        'origin_snapshot_sha256': _sha256(snapshot), 'formal_gain': 0}
    checkpoint = directory / 'packed_predicates.pickle'
    with checkpoint.open('xb') as stream:
        pickle.dump(checkpoint_payload, stream, protocol=5)
        stream.flush()
        os.fsync(stream.fileno())
    full = collect(SimpleNamespace(), {'saved': saved, 'expr': expr, 'checkpoint': checkpoint_payload,
        **packed.numeric_roots()})
    complete = full.measure()
    witness, reference = reference_subset(full, net)
    lower = reference['reference_lower_bound']
    physical = complete.resident_bytes < lower['resident_bytes'] and complete.resident_entries < lower['resident_entries']
    if collect(SimpleNamespace(), {'saved': saved, 'expr': expr, 'base': base}).fingerprint != before:
        raise ValueError('source/native state changed')
    packed.validate()
    # Prospective combined allowance, not an executed integrated suffix proof.
    c8 = json.loads((EXPERIMENT / 'results/c8_dyadic_balance_20260905_v1/factor_events.jsonl').read_text().splitlines()[0])
    future_work = c8['support_work'] + (c8['total_work_upper'] - c8['support_work']) // 16 * 12 + packed.report['total_work_upper']
    conservative_branch = c8['largest_branch_work_upper'] + packed.report['total_work_upper']
    passed = native['passed'] and physical
    record = {'schema': 'c9_predicate_rows_actual_v1', 'formal_gain': 0,
        'status': 'PREDICATE_PREREQUISITE_QUALIFIED' if passed else 'PREDICATE_PREREQUISITE_REJECTED',
        'prerequisite_gates_passed': passed, 'full_suffix_executed': False, 'live_publication_executed': False,
        'terminal_solve_executed': False, 'family_retention_claimed': False,
        'report': packed.report, 'construction': construction, 'identity': identity,
        'original_experimental_prefix_native': original_native, 'radix_predicate_native': native,
        'complete_prerequisite_numeric_roots': len(full.numeric), 'complete_prerequisite': asdict(complete),
        'reference': reference, 'strict_complete_prerequisite_physical_decrease': physical,
        'prospective_integrated_work_upper': future_work, 'prospective_branch_upper': conservative_branch,
        'prospective_budget_within_caps': future_work <= 256_000_000 and conservative_branch <= 200_000_000,
        'prospective_work_is_not_an_integrated_run': True, 'source_sha256': freeze['source_sha256'],
        'provenance': freeze['provenance'], 'snapshot_sha256': _sha256(snapshot),
        'checkpoint_sha256': _sha256(checkpoint), 'checkpoint_bytes': checkpoint.stat().st_size,
        'max_rss_kib_including_oracles': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'worker_elapsed_s': time.monotonic() - started}
    _atomic_exclusive_json(directory / 'result.json', record)
    print(json.dumps({'status': record['status'], 'formal_gain': 0, 'extra_aux': packed.def_rows.size,
        'native_passed': native['passed'], 'physical': physical, 'future_work': future_work}), flush=True)
    if not passed:
        raise ValueError('predicate prerequisite failed')

if __name__ == '__main__':
    main()

