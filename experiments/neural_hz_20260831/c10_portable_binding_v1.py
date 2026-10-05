"""Portable ALL-content/sharing proof binding; no representation selection."""

from fractions import Fraction
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np

from experiments.neural_hz_20260831.c10_fused_emission_v1 import FusedIntegrated
from experiments.neural_hz_20260831.c10_fused_rows_v1 import aliases
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import operator_digest, digest_arrays
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
PRIOR = EXP / 'results/c10_fused_emission_20260908_v1'


def identity(candidate):
    candidate.validate()
    h = hashlib.sha256(b'c10_portable_original_program_and_tagged_hz_v1')
    sources, operators = {}, {}
    def token(value):
        h.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        h.update(b'\0')
    def source(value):
        key = sources.setdefault(id(value), len(sources))
        return [key, source_digest(value)]
    def operator(value):
        key = operators.setdefault(id(value), len(operators))
        return [key, operator_digest(value)]
    token([source_digest(candidate.hz), candidate.root, candidate.old_n_cont, candidate.old_n_bin,
           candidate.old_n_eq, candidate.logical_n_cont, candidate.expression.frame_id,
           candidate.expression.n_out, digest_arrays(candidate.expression.bias, candidate.keep)])
    for term in candidate.expression.terms:
        token([source(term.source), [operator(op) for op in term.operators]])
    for node in candidate.nodes:
        token([node['kind'], node['width'], node['parents'], node['support_work'],
               digest_arrays(*(node[k] for k in ('support', 'needed', 'slots', 'exponents')))])
        if node['kind'] == 'source':
            token(source(node['source']))
        elif node['kind'] == 'op':
            token(operator(node['op']))
    token(digest_arrays(candidate.eq_roots, candidate.eq_scales, candidate.ineq_roots,
                        candidate.ineq_scales, candidate.def_rows))
    token(candidate.report)
    return {'schema': 'c10_portable_binding_v1', 'sha256': h.hexdigest(),
        'hz_sha256': source_digest(candidate.hz), 'source_objects': len(sources),
        'operator_objects': len(operators), 'selected_aliases': int(aliases(candidate)[0].size),
        'whole_work': candidate.report['total_work_upper'], 'branch_work': candidate.report['largest_branch_work_upper']}


def verify(candidate, binding):
    actual = identity(candidate)
    if actual != binding['identity'] or binding.get('all_rows_proved') is not True:
        raise ValueError('fresh live content/sharing does not match sealed independent proof')
    return {'all_content_and_sharing_bound': True, 'all_rows_proof_sha256': binding['result_sha256'],
        'portable_identity': actual, 'old_oracle_hz_retained': False}


def reconstruct_fraction(candidate, continuous):
    candidate.validate()
    if len(continuous) != candidate.hz.n_cont:
        raise ValueError('tagged reconstruction requires original global width')
    values = [Fraction(v) for v in continuous]
    if any(abs(v) > 1 for v in values):
        raise ValueError('point outside continuous latent box')
    cols, parents, ratios, unused = aliases(candidate)
    for c, p, r in zip(cols, parents, ratios):
        values[int(c)] = Fraction(float(r)) * values[int(p)]
    return values


def export(path):
    if _sha256(PRIOR / 'result.json') != '315a152e3910b8340f5971be8346434a8d81e5106cd5f67321deea8def29fd5b':
        raise ValueError('fused proof result drift')
    result = json.loads((PRIOR / 'result.json').read_text())
    checkpoint = PRIOR / 'fused_hz.pickle'
    if (_sha256(checkpoint) != '24b22d8a06c0c4d51525dd9d5ee0902941c40f20cc6a711e6231673e3efa9d08'
            or not result['completed'] or not result['native_ingestion']['passed']
            or not result['strict_complete_offline_reference_decrease']
            or result['identity']['status'] != 'EXACT_ORIGINAL_DAG_AND_QUOTIENT'):
        raise ValueError('incomplete or changed offline proof')
    with checkpoint.open('rb') as stream:
        saved = pickle.load(stream)
    if saved['schema'] != 'c10_fused_emission_checkpoint_v1':
        raise ValueError('unexpected fused checkpoint schema')
    fields = saved['fields']
    candidate = FusedIntegrated(**fields, origin_binding=expression_binding(fields['expression']))
    candidate.seal = candidate.fingerprint()
    bound = identity(candidate)
    if bound['hz_sha256'] != result['hz_sha256']:
        raise ValueError('checkpoint HZ differs from proved result')
    _atomic_exclusive_json(path, {'identity': bound, 'all_rows_proved': True,
        'result_sha256': _sha256(PRIOR / 'result.json'), 'checkpoint_sha256': _sha256(checkpoint),
        'formal_gain': 0})
    return bound


if __name__ == '__main__':
    print(json.dumps(export(Path(sys.argv[1]))), flush=True)
