"""Bounded complete ordinary F4/F2/direct HZ comparisons; no solver."""
from dataclasses import asdict
import faulthandler
import json
import os
from pathlib import Path
import resource
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout, metadata
from experiments.neural_hz_20260831.c120_complete_source_fixture_v1 import complete_source
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json
EXP = Path(__file__).resolve().parent
RUN = EXP / 'results/c120_denominator_f4_20260922_v1'


def save_packet_stage(mode, stage, packet, report, pool, authentication):
    arrays = {name: value for name, value in packet.items() if type(value) is np.ndarray}
    pool.charge('c120_partial_packet_evidence_encoding',
                1024+16*sum(int(value.size) for value in arrays.values()))
    path = RUN/(mode+'_'+stage+'.npz')
    with path.open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    authentication['emitted_evidence_hash_calls'] += 1
    authentication['emitted_evidence_bytes_hashed'] += path.stat().st_size
    _atomic_exclusive_json(RUN/(mode+'_'+stage+'.json'), dict(
        mode=mode, stage=stage, constructor_report=report, packet_sha256=_sha256(path),
        proof_completed_at_save=False, physical_or_source_admission=False,
        work=pool.used, formal_gain=0))


def build(pool, held, authentication, observe):
    reports = {}
    for mode in ('dense', 'masked'):
        report, case = complete_source(mode, pool=pool, enabled=True, observe=observe,
            retain=lambda stage, packet, report: save_packet_stage(
                mode, stage, packet, report, pool, authentication))
        held[mode] = case
        reports[mode] = report
        arrays = {}
        for prefix, packet in [('f4', case['f4_packet']),
                *((f'f2_{i}', p) for i, p in enumerate(case['f2_packets']))]:
            arrays.update({prefix+'_'+n: a for n, a in packet.items()
                           if type(a) is np.ndarray})
        path = RUN / (mode+'_complete_packets.npz')
        pool.charge('c120_complete_packet_evidence_encoding',
                    1024+16*sum(int(value.size) for value in arrays.values()))
        with path.open('xb') as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        authentication['emitted_evidence_hash_calls'] += 1
        authentication['emitted_evidence_bytes_hashed'] += path.stat().st_size
        _atomic_exclusive_json(RUN / (mode+'_complete_source.json'), dict(
            report=report, packet_sha256=_sha256(path),
            complete_source_record_not_a_network_benchmark=True, formal_gain=0))
        print(json.dumps(dict(event='complete_source_comparison_saved', mode=mode,
                              work=pool.used)), flush=True)
    held['reports'] = reports
    held['authentication'] = authentication
    # This guard pays before serialization, not instead of the complete ledger.
    pre_global_work = pool.used
    paid_c62 = sum(v for k, v in pool.parts.items() if k.startswith('c62_'))
    if pre_global_work > 234_810_164 or paid_c62 > 13_321_560:
        raise MemoryError('complete preregistered pre-global work bound exceeded')
    pool.charge('c120_complete_report_size_guard', 131_072)
    encoded = json.dumps(dict(reports=reports, authentication=authentication),
                         sort_keys=True).encode()
    if len(encoded) > 89_707:
        raise MemoryError('complete report/authentication header bound exceeded')
    held['preflight_guard'] = dict(pre_global_work=pre_global_work,
        paid_case_and_physical_C62_work=paid_c62, report_auth_bytes=len(encoded),
        complete_batch_work_upper=250_419_860)
    held['report_auth_guard_bytes'] = encoded
    layout = numeric_layout(held, pool)
    meta = metadata(held, pool)
    if pool.used > 250_419_860:
        raise MemoryError('complete preregistered whole-batch work bound exceeded')
    if layout.resident_entries > 64_000_000:
        raise MemoryError('complete held ordinary comparison exceeds64M entries')
    return dict(complete_source_count=2, reports=reports,
        complete_numeric_layout=asdict(layout), complete_known_metadata=meta,
        complete_work_preflight=held['preflight_guard'],
        original_network_loaded=False, actual_network_source_or_LIVE_admitted=False,
        solver_calls=0, formal_gain=0)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3, 16*1024**3))
    freeze = json.loads((RUN/'preregistered.json').read_text())
    pool, held = WorkPool(256_000_000), {}
    authentication = dict(source_file_bytes_hashed=0, source_file_hash_calls=0,
        emitted_evidence_bytes_hashed=0, emitted_evidence_hash_calls=0)
    record = dict(completed=False, formal_gain=0)
    started = time.monotonic()
    fatal = (RUN/'fatal.log').open('x')
    phases = (RUN/'phases.jsonl').open('x')
    faulthandler.enable(file=fatal, all_threads=True)

    def observe(event):
        event = dict(event, wall_s=time.monotonic()-started)
        phases.write(json.dumps(event)+'\n')
        phases.flush()
        os.fsync(phases.fileno())
        print(json.dumps(event), flush=True)

    def drift():
        changed = False
        for name, expected in freeze['source_sha256'].items():
            path = EXP/name
            changed |= _sha256(path) != expected
            authentication['source_file_bytes_hashed'] += path.stat().st_size
            authentication['source_file_hash_calls'] += 1
        return changed

    try:
        if drift() or _provenance(ROOT) != freeze['provenance']:
            raise ValueError('frozen source or production drift')
        data, stats = measured(lambda: build(pool, held, authentication, observe),
            observe=lambda stats: record.update(measurement=stats))
        record.update(completed=True, data=data, measurement=stats)
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        faulthandler.disable()
        fatal.close()
        phases.close()
        record.update(work=pool.used, work_parts=pool.parts, wall_s=time.monotonic()-started,
            source_drift=drift(), provenance_drift=_provenance(ROOT) != freeze['provenance'],
            authentication_traffic=authentication,
            numeric_hash_traffic_in_token_pool=False, all_CPU_work_in_generation_cap=False)
        _atomic_exclusive_json(RUN/'result.json', record)
        print(json.dumps({k:v for k,v in record.items() if k != 'data'}), flush=True)
    if not record['completed'] or record['source_drift'] or record['provenance_drift']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
