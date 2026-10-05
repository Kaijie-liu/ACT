"""Bounded complete mixed-channel ordinary sources; all evidence exclusive."""
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
from experiments.neural_hz_20260831.c123_source_binding_worker_v1 import BudgetView
from experiments.neural_hz_20260831.c124_complete_mixed_source_v1 import (
    MODES, CATEGORY_CAPS, COMPLETE_WORK_UPPER, complete_source)
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c124_mixed_source_20260927_v1'


def build(pool, held, authentication, emit):
    budgets = {k:BudgetView(pool,v) for k,v in CATEGORY_CAPS.items()}
    reports, packet_records, qualification_failures = {}, {}, []
    held.update(reports=reports, authentication=authentication, packet_records=packet_records)

    def retain(mode, stage, packet, report):
        arrays = {name:value for name,value in packet.items() if type(value) is np.ndarray}
        budgets['evidence'].charge('c124_complete_native_packet_evidence_encoding',
            1024+16*sum(int(value.size) for value in arrays.values()))
        path = RUN/(mode+'_'+stage+'_arrays.npz')
        with path.open('xb') as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        authentication['evidence_bytes_hashed'] += path.stat().st_size
        authentication['evidence_hash_calls'] += 1
        budgets['evidence'].charge('c124_partial_packet_JSON_reporting_reservation', 65536)
        saved = dict(mode=mode, stage=stage, packet_sha256=_sha256(path),
            new_factors=int(packet['new_factors']), constructor=report,
            proof_completed_at_save=False, physical_or_source_admission=False,
            work=pool.used, formal_gain=0)
        packet_records[mode] = saved
        _atomic_exclusive_json(RUN/(mode+'_'+stage+'.json'), saved)

    expected_selected = dict(dense=16, masked=16, heterogeneous=12, noop=0)
    for mode in MODES:
        report, case = complete_source(mode, budgets, held, emit, retain)
        reports[mode] = report
        budgets['evidence'].charge('c124_complete_case_JSON_reporting_reservation', 131072)
        encoded = json.dumps(report, sort_keys=True).encode()
        if len(encoded)+1024 > 131072:
            raise MemoryError('complete case report exceeds frozen evidence reserve')
        _atomic_exclusive_json(RUN/(mode+'_complete_source.json'), report)
        # Save the complete result BEFORE interpreting its promotion boundary.
        if (not report['original_source_unchanged']
            or report['binding']['all_actual_source_output_rows_bound'] != 512
            or report['route']['selected_channels'] != expected_selected[mode]):
            raise ValueError('registered complete ordinary population differs')
        if mode == 'noop':
            if (case['packet'] is not None or case['state'] is not case['direct']['fields']
                or not report['physical']['exact_original_object_retained']
                or report['constructor']['kernel_transform_prepaid']):
                raise ValueError('complete zero-hit source was not a literal no-op')
        elif (not report['route']['conditional_candidate']
              or not report['physical']['complete_physical_reduction_proved']
              or report['physical']['rejection_reasons']):
            qualification_failures.append(dict(mode=mode,
                reason='complete ordinary mixed HZ physical reduction not established',
                physical=report['physical']))
        emit(dict(event='complete_source_saved', mode=mode, work=pool.used))
    layout = numeric_layout(held, budgets['ledger'])
    meta = metadata(held, budgets['ledger'])
    if layout.resident_entries > 64000000:
        raise MemoryError('complete retained source/candidate/proof state exceeds64M entries')
    budgets['ledger'].charge('c124_complete_ledger_and_result_JSON_reservation', 524288)
    ledger = dict(numeric=asdict(layout), known_metadata=meta)
    _atomic_exclusive_json(RUN/'complete_held_ledger.json', ledger)
    if pool.used > COMPLETE_WORK_UPPER:
        raise MemoryError('complete mixed-source batch exceeds frozen254M bound')
    return dict(reports=reports, complete_source_count=len(MODES), ledger=ledger,
        qualification_pass=not qualification_failures, qualification_failures=qualification_failures,
        category_work={k:v.spent for k,v in budgets.items()}, category_limits=CATEGORY_CAPS,
        full_original_dense_and_historical_masked_sources_retained=True,
        complete_heterogeneous_source_and_literal_noop_guard=True,
        mixed_F4_constructor_executed=True, original_network_loaded=False,
        archived_HZ_loaded=False, source_or_LIVE_admitted=False, solver_calls=0, formal_gain=0)


def main():
    resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,16*1024**3))
    freeze = json.loads((RUN/'preregistered.json').read_text())
    pool,held = WorkPool(256_000_000),dict(freeze=freeze)
    auth = dict(source_file_bytes_hashed=0,source_file_hash_calls=0,
                evidence_bytes_hashed=0,evidence_hash_calls=0)
    record = dict(completed=False,solver_calls=0,formal_gain=0)
    started = time.monotonic()
    fatal = (RUN/'fatal.log').open('x')
    phases = (RUN/'phases.jsonl').open('x')
    faulthandler.enable(file=fatal,all_threads=True)

    def emit(event):
        value = dict(elapsed_s=time.monotonic()-started,detail=event)
        phases.write(json.dumps(value)+'\n')
        phases.flush()
        os.fsync(phases.fileno())
        print(json.dumps(value),flush=True)

    def drift():
        changed = False
        for n,h in freeze['source_sha256'].items():
            path = EXP/n
            changed |= _sha256(path)!=h
            auth['source_file_bytes_hashed'] += path.stat().st_size
            auth['source_file_hash_calls'] += 1
        return changed

    def input_drift():
        return any(_sha256(Path(n))!=h for n,h in freeze['input_sha256'].items())

    try:
        if drift() or input_drift() or _provenance(ROOT)!=freeze['provenance']:
            raise ValueError('frozen source/production drift')
        data,stats = measured(lambda:build(pool,held,auth,emit),
                              observe=lambda m:record.update(measurement=m))
        record.update(completed=True,data=data,measurement=stats)
        if not data['qualification_pass']:
            record['completed'] = False
            raise ValueError('complete four-case physical qualification rejected; all results retained')
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__,reason=str(exc))
    finally:
        faulthandler.disable()
        fatal.close()
        phases.close()
        changed = drift()
        record.update(work=pool.used,work_parts=pool.parts,wall_s=time.monotonic()-started,
            source_drift=changed,input_drift=input_drift(),
            provenance_drift=_provenance(ROOT)!=freeze['provenance'],
            authentication_traffic=auth,numeric_hash_traffic_in_token_pool=False,
            all_CPU_work_in_generation_cap=False)
        _atomic_exclusive_json(RUN/'result.json',record)
        print(json.dumps({k:v for k,v in record.items() if k!='data'}),flush=True)
    if (not record['completed'] or record['source_drift'] or record['input_drift']
        or record['provenance_drift']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
