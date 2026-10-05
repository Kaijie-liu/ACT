"""Complete independent old/new ordinary source generation and exact evidence."""
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
from experiments.neural_hz_20260831.c128_complete_source_v1 import (
    MODES, CATEGORY_CAPS, COMPLETE_WORK_UPPER, complete_source)
from experiments.neural_hz_20260831.c125_exact_json_v1 import JsonAllowance
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c128_channel_support_20260927_v1'


def build(pool, budgets, held, authentication, emit, ledger_allowance):
    reports, receipts = {}, []
    stage_records,stage_arrays,point_receipts = {},{},{}
    held.update(reports=reports, authentication=authentication, serialization_receipts=receipts,
                stage_records=stage_records,stage_arrays=stage_arrays,point_receipts=point_receipts)

    def save_json(name, payload, reserve):
        receipt = JsonAllowance(budgets['evidence'], reserve,
            'c128_complete_source_exact_JSON_reservation').write(RUN/name, payload)
        receipts.append(receipt)
        return receipt

    def retain(mode, stage, arrays, scalar_metadata, *, points=None):
        stages = ('source','old','new','owners')
        if mode not in MODES or stage not in stages:
            raise ValueError('unregistered complete source retention stage')
        key = mode+'_'+stage
        if key in stage_records or key in stage_arrays:
            raise ValueError('complete source stage cannot be retained twice')
        if any(mode+'_'+previous not in stage_records for previous in stages[:stages.index(stage)]):
            raise ValueError('complete source retention stages are out of order')
        if (type(arrays) is not dict or not arrays
            or any(type(name) is not str or type(a) is not np.ndarray
                   or a.dtype.kind not in 'biuf' for name,a in arrays.items())):
            raise ValueError('complete retained stage must contain only named numeric arrays')
        if (stage == 'owners') != (points is not None):
            raise ValueError('full inverse points belong to the completed owner stage')
        # Header/manifest walks were prepaid before the first source inventory.
        # Retain every stage root even if a later encoding or proof rejects.
        stage_arrays[key] = arrays
        entries = sum(int(a.size) for a in arrays.values())
        budgets['evidence'].charge('c128_complete_source_graph_inverse_array_encoding',1024+16*entries)
        path = RUN/(key+'_arrays.npz')
        with path.open('xb') as stream:
            np.savez(stream,**arrays)
            stream.flush()
            os.fsync(stream.fileno())
        authentication['evidence_bytes_hashed'] += path.stat().st_size
        authentication['evidence_hash_calls'] += 1
        array_manifest = {name:dict(dtype=a.dtype.str,shape=list(a.shape),entries=int(a.size))
                          for name,a in arrays.items()}
        artifact = dict(file=path.name,sha256=_sha256(path),stored_bytes=path.stat().st_size,
                        arrays=len(arrays),numeric_entries=entries,encoding_work=1024+16*entries,
                        complete_array_manifest=array_manifest)
        if points is not None:
            point_receipts[mode] = save_json(mode+'_complete_points.json',
                dict(mode=mode,points=points,points_are_not_network_witnesses=True,
                     complete_old_new_original_and_recovered_populations=True,formal_gain=0),262144)
        stage_record = dict(mode=mode,stage=stage,metadata=scalar_metadata,artifact=artifact,
                            proof_completed_at_save=False,formal_gain=0)
        stage_records[key] = stage_record
        save_json(key+'.json',stage_record,65536)
        emit(dict(event='complete_source_stage_retained_before_next_phase',
                  mode=mode,stage=stage,arrays=len(arrays),work=pool.used))
        return artifact

    for mode in MODES:
        report, case = complete_source(mode, budgets, held, emit,retain=retain)
        if set(report['artifacts']) != {'source','old','new','owners'}:
            raise ValueError('complete source stage archive is incomplete')
        report.update(complete_points_receipt=point_receipts[mode])
        reports[mode] = report
        save_json(mode+'_complete_source.json',report,131072)
        emit(dict(event='complete_old_new_source_and_full_evidence_saved',mode=mode,work=pool.used))
    old_support = sum(report['support_work']['old'] for report in reports.values())
    new_support = sum(report['support_work']['new'] for report in reports.values())
    if new_support >= old_support:
        raise ValueError('uniform complete four-source support program has no aggregate logical saving')
    layout = numeric_layout(held,budgets['ledger'])
    meta = metadata(held,budgets['ledger'])
    if layout.resident_entries > 64000000:
        raise MemoryError('complete source/control/proof roots exceed64M numeric entries')
    ledger = dict(numeric=asdict(layout),known_metadata=meta)
    receipt = ledger_allowance.write(RUN/'complete_held_ledger.json',ledger)
    receipts.append(receipt)
    if pool.used > COMPLETE_WORK_UPPER:
        raise MemoryError('complete old/new source qualification exceeds254M fixed bound')
    return dict(qualification_pass=True,qualification_failures=[],reports=reports,
        complete_source_count=4,independent_source_build_count=8,
        unchanged_complete_ordinary_source_geometry=True,
        complete_source_graph_owner_inverse_equivalence=True,
        complete_all_numeric_source_arrays_saved=True,complete_all_point_populations_saved=True,
        full_original_source_sharing_and_nonconvex_predicates_preserved=True,
        complete_support_work=dict(old=old_support,new=new_support,saved=old_support-new_support),
        ledger=ledger,serialization_receipts=receipts,stage_records=stage_records,
        category_work={name:view.spent for name,view in budgets.items()},category_limits=CATEGORY_CAPS,
        source_or_LIVE_admitted=False,original_network_loaded=False,archived_HZ_loaded=False,
        solver_calls=0,formal_gain=0)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3,16*1024**3))
    freeze = json.loads((RUN/'preregistered.json').read_text())
    pool = WorkPool(256000000)
    budgets = {k:BudgetView(pool,v) for k,v in CATEGORY_CAPS.items()}
    final_allowance = JsonAllowance(budgets['ledger'], 524288, 'c128_upfront_terminal_success_JSON_reservation')
    failure_allowance = JsonAllowance(budgets['ledger'], 65536, 'c128_upfront_terminal_failure_JSON_reservation')
    ledger_allowance = JsonAllowance(budgets['ledger'], 131072, 'c128_upfront_complete_ledger_JSON_reservation')
    held = dict(freeze=freeze)
    auth = dict(source_file_bytes_hashed=0, source_file_hash_calls=0,
                evidence_bytes_hashed=0, evidence_hash_calls=0)
    record = dict(completed=False, solver_calls=0, formal_gain=0)
    started = time.monotonic()
    fatal, phases = (RUN/'fatal.log').open('x'), (RUN/'phases.jsonl').open('x')
    faulthandler.enable(file=fatal, all_threads=True)

    def emit(event):
        value = dict(elapsed_s=time.monotonic()-started, detail=event)
        phases.write(json.dumps(value)+'\n')
        phases.flush()
        os.fsync(phases.fileno())
        print(json.dumps(value), flush=True)

    def drift():
        changed = False
        for name,digest in freeze['source_sha256'].items():
            path = EXP/name
            changed |= _sha256(path) != digest
            auth['source_file_bytes_hashed'] += path.stat().st_size
            auth['source_file_hash_calls'] += 1
        return changed

    def input_drift():
        return any(_sha256(Path(n)) != h for n,h in freeze['input_sha256'].items())

    try:
        if drift() or input_drift() or _provenance(ROOT) != freeze['provenance']:
            raise ValueError('frozen source/input/production drift')
        data,stats = measured(lambda:build(pool,budgets,held,auth,emit,ledger_allowance),
                              observe=lambda m:record.update(measurement=m))
        record.update(completed=True, data=data, measurement=stats)
        if not data['qualification_pass']:
            record['completed'] = False
            raise ValueError('complete four-case qualification rejected; all evidence retained')
    except Exception as exc:
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        faulthandler.disable()
        fatal.close()
        phases.close()
        changed = drift()
        record.update(work=pool.used, work_parts=pool.parts, wall_s=time.monotonic()-started,
            source_drift=changed, input_drift=input_drift(),
            provenance_drift=_provenance(ROOT) != freeze['provenance'],
            authentication_traffic=auth, numeric_hash_traffic_in_token_pool=False,
            all_CPU_work_in_generation_cap=False,
            terminal_allowances_prepaid_before_build=True,
            terminal_success_reservation=524288, terminal_failure_reservation=65536,
            complete_ledger_reservation=131072)
        try:
            receipt = final_allowance.write(RUN/'result.json', record)
        except (MemoryError, TypeError, ValueError) as exc:
            # This is a failed version, not a reduced success payload.  All
            # independently saved full case/proof/point artifacts remain.
            record = dict(completed=False, solver_calls=0, formal_gain=0,
                failure=dict(type=type(exc).__name__, reason=str(exc)),
                reporting_failure=True, full_success_payload_not_published=True,
                full_evidence_files_retained=[p.name for p in RUN.iterdir() if p.is_file()],
                work=pool.used, work_parts=pool.parts, source_drift=changed,
                input_drift=input_drift(), provenance_drift=_provenance(ROOT) != freeze['provenance'])
            receipt = failure_allowance.write(RUN/'result.json', record)
        print(json.dumps(dict(event='terminal_exact_bytes_published', receipt=receipt)), flush=True)
        print(json.dumps({k:v for k,v in record.items() if k != 'data'}), flush=True)
    if (not record['completed'] or record['source_drift'] or record['input_drift']
        or record['provenance_drift']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
