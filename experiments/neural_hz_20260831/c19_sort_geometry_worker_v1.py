"""One exclusive complete geometry diagnostic; no failed-version execution."""

import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from experiments.neural_hz_20260831.c19_sort_geometry_v1 import census
from experiments.neural_hz_20260831.c10_fused_rows_v1 import aliases
from experiments.neural_hz_20260831.c10_fused_emission_v1 import FusedIntegrated
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
from experiments.neural_hz_20260831.c10_portable_binding_v1 import identity
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c19_sort_geometry_20260911_v1'
ORIGINAL = EXP / 'results/c9_live_relu_20260906_v1/relu78.pickle'
FUSED = EXP / 'results/c10_fused_emission_20260908_v1/fused_hz.pickle'
INPUTS = {ORIGINAL: '5bf82fc83205cd9b5f52187e164c70ce3c03abffd9d8bf352a643f38d2966a65',
    FUSED: '24b22d8a06c0c4d51525dd9d5ee0902941c40f20cc6a711e6231673e3efa9d08'}


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze = json.loads((DIRECTORY / 'preregistered.json').read_text())
    started = time.monotonic()
    record = {'completed': False, 'formal_gain': 0, 'generator_executed': False,
        'solver_executed': False, 'native_ingestion_executed': False,
        'source_sha256': freeze['source_sha256'], 'provenance': freeze['provenance']}
    try:
        if any(_sha256(EXP / n) != sha for n, sha in freeze['source_sha256'].items()):
            raise ValueError('frozen diagnostic/source drift')
        if any(_sha256(p) != sha for p, sha in INPUTS.items()):
            raise ValueError('sealed independent input drift')
        with ORIGINAL.open('rb') as f: original_saved = pickle.load(f)
        with FUSED.open('rb') as f: fused_saved = pickle.load(f)
        if (original_saved['schema'] != 'c9_live_relu_checkpoint_v1'
                or fused_saved['schema'] != 'c10_fused_emission_checkpoint_v1'):
            raise ValueError('unexpected archived source schemas')
        hz = original_saved['preactivation_hz']
        fields = fused_saved['fields']
        candidate = FusedIntegrated(**fields, origin_binding=expression_binding(fields['expression']))
        candidate.seal = candidate.fingerprint()
        before = source_digest(hz), identity(candidate)
        cols, parents, ratios, tagged = aliases(candidate)
        erased = original_saved['numeric_roots']['eq_roots'][tagged]
        if len(cols) != 100603:
            raise ValueError('complete sealed alias population differs')
        report, construction = measured_build(lambda: census(hz.Ac, hz.Auc,
            candidate.old_n_cont, cols, parents, erased))
        old = candidate.report['alias_quotient']
        if (report['counts']['rows'] != old['rewritten_rows']
                or report['counts']['occurrences'] != old['rewritten_occurrences']
                or report['counts']['rows'] != 37500 or report['counts']['occurrences'] != 227185):
            raise ValueError('diagnostic did not cover every previously proved rewrite')
        if (source_digest(hz), identity(candidate)) != before:
            raise ValueError('read-only census mutated source HZ/lineage')
        report.update(all_C10_rewritten_rows_covered=True,
            frozen_C10_collision_groups=old['collision_groups'],
            full_archived_dictionaries_retained=True, source_and_lineage_unchanged=True)
        _atomic_exclusive_json(DIRECTORY / 'geometry.json', report)
        record.update(completed=True, report=report, construction=construction)
    except Exception as exc:
        record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
    finally:
        record.update(wall_s=time.monotonic()-started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        _atomic_exclusive_json(DIRECTORY / 'result.json', record)
        print(json.dumps({'completed': record['completed'], 'failure': record.get('failure'), 'formal_gain': 0}), flush=True)
    if not record['completed']: raise SystemExit(1)


if __name__ == '__main__': main()
