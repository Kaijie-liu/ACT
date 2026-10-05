# SPDX-License-Identifier: AGPL-3.0-or-later
"""Complete archived explicit-operator census, never a restored HZ verifier."""
from dataclasses import asdict
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy.sparse as sp
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout, metadata
from experiments.neural_hz_20260831.c118_column_orbits_v1 import census
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json
EXP = Path(__file__).resolve().parent
RUN = EXP / 'results/c118_column_orbits_20260922_v1'
PREV = EXP / 'results/c117_affine_block_census_20260922_v1'


def payload_digest(arrays, authentication):
    digest = hashlib.sha256()
    for value in arrays:
        digest.update(repr((value.dtype.str, value.shape)).encode())
        digest.update(value.tobytes())
        authentication['numeric_payload_bytes_hashed'] += int(value.nbytes)
    authentication['numeric_payload_hash_calls'] += 1
    return digest.hexdigest()


def one_operator(node, arrays, pool, authentication):
    """Numeric arrays outlive temporary sparse wrappers; no root is omitted."""
    geometry = node['operator_geometry']
    data, indices, indptr = arrays
    if (data.size != geometry['stored_nnz'] or indices.size != data.size
        or indptr.size != geometry['shape'][0] + 1):
        raise ValueError('complete CSR population differs from frozen geometry')
    before = payload_digest(arrays, authentication)
    matrix = sp.csr_matrix((data, indices, indptr), shape=tuple(geometry['shape']), copy=False)
    report, evidence = census(matrix, pool=pool, enabled=True)
    if payload_digest(arrays, authentication) != before:
        raise ValueError('column census changed archived input payload')
    if any(type(a) is not np.ndarray or a.dtype.hasobject for a in evidence.values()):
        raise ValueError('complete packed numeric evidence required')
    return report, evidence, before


def build(pool, held, authentication):
    raw = (PREV / 'complete_census.json').read_bytes()
    original = json.loads(raw)
    held.update(complete_original_census_raw=raw, complete_original_census=original,
        inputs={}, reports={}, evidence={}, authentication=authentication)
    nodes = [n for n in original['report']['nodes']
        if n['operator_geometry'] is not None and n['operator_geometry']['type'] == 'csr']
    if [n['node'] for n in nodes] != [2,5,8,11,14,17,20,23,26,27,31,33,35]:
        raise ValueError('complete structure-selected archived CSR inventory changed')
    with np.load(PREV / 'complete_census_arrays.npz', allow_pickle=False) as archive:
        for node in nodes:
            ni, geometry = node['node'], node['operator_geometry']
            pool.charge('c118_complete_operator_archive_loading',
                16 * (2 * geometry['stored_nnz'] + geometry['shape'][0] + 1))
            arrays = tuple(archive[f'operator_{ni}_payload_{j}'] for j in range(3))
            held['inputs'][str(ni)] = arrays
            report, evidence, binding = one_operator(node, arrays, pool, authentication)
            held['reports'][str(ni)] = report
            held['evidence'][str(ni)] = evidence
            with (RUN / f'node_{ni}_complete_orbits.npz').open('xb') as stream:
                np.savez(stream, **evidence)
                stream.flush()
                os.fsync(stream.fileno())
            evidence_path = RUN / f'node_{ni}_complete_orbits.npz'
            evidence_sha = _sha256(evidence_path)
            authentication['emitted_evidence_bytes_hashed'] += evidence_path.stat().st_size
            authentication['emitted_evidence_hash_calls'] += 1
            _atomic_exclusive_json(RUN / f'node_{ni}_census.json', dict(node=ni,
                geometry=geometry, input_payload_sha256=binding, report=report,
                evidence_sha256=evidence_sha,
                source_or_LIVE_admitted=False, formal_gain=0))
            print(json.dumps(dict(event='complete_operator_saved', node=ni,
                work=pool.used)), flush=True)
    layout = numeric_layout(held, pool)
    meta = metadata(held, pool)
    if layout.resident_entries > 64_000_000:
        raise MemoryError('complete held explicit-operator census exceeds64M entries')
    return dict(complete_operator_count=len(nodes), reports=held['reports'],
        complete_numeric_layout=asdict(layout), complete_known_metadata=meta,
        all_original_loaded_operator_payloads_preserved=True,
        all_operator_columns_inspected=True, implicit_convolutions_expanded=False,
        original_HZ_loaded=False, fresh_source_or_LIVE_admitted=False,
        prior_failed_C117_ledger_still_failed=True, solver_calls=0, formal_gain=0)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    freeze = json.loads((RUN / 'preregistered.json').read_text())
    pool, held = WorkPool(256_000_000), {}
    authentication = dict(source_file_bytes_hashed=0, source_file_hash_calls=0,
        numeric_payload_bytes_hashed=0, numeric_payload_hash_calls=0,
        emitted_evidence_bytes_hashed=0, emitted_evidence_hash_calls=0)
    record = dict(completed=False, formal_gain=0)
    started = time.monotonic()
    fatal = (RUN / 'fatal.log').open('x')
    faulthandler.enable(file=fatal, all_threads=True)

    def drift():
        changed = False
        for name, expected in freeze['source_sha256'].items():
            path = EXP / name
            actual = _sha256(path)
            authentication['source_file_bytes_hashed'] += path.stat().st_size
            authentication['source_file_hash_calls'] += 1
            changed |= actual != expected
        return changed

    try:
        if drift() or _provenance(ROOT) != freeze['provenance']:
            raise ValueError('frozen source/production drift')
        data, stats = measured(lambda: build(pool, held, authentication),
            observe=lambda m: record.update(measurement=m))
        record.update(completed=True, data=data, measurement=stats)
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        faulthandler.disable()
        fatal.close()
        source_changed = drift()
        provenance_changed = _provenance(ROOT) != freeze['provenance']
        record.update(work=pool.used, work_parts=pool.parts, wall_s=time.monotonic()-started,
            source_drift=source_changed, provenance_drift=provenance_changed,
            authentication_traffic=authentication,
            numeric_hash_traffic_in_token_pool=False, all_CPU_work_in_generation_cap=False)
        _atomic_exclusive_json(RUN / 'result.json', record)
        print(json.dumps({k:v for k,v in record.items() if k != 'data'}), flush=True)
    if not record['completed'] or record['source_drift'] or record['provenance_drift']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
