"""Complete unchanged ordinary sources and exact-word mixed native comparison."""
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
from experiments.neural_hz_20260831.c127_complete_mixed_source_v1 import (
    MODES, CATEGORY_CAPS, COMPLETE_WORK_UPPER, complete_source)
from experiments.neural_hz_20260831.c125_exact_json_v1 import JsonAllowance
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256

EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c127_systematic_kernel_20260927_v1'


def build(pool, budgets, held, authentication, emit, ledger_allowance):
    reports, packet_records, kernel_records, receipts, failures = {}, {}, {}, [], []
    held.update(reports=reports, authentication=authentication,
                packet_records=packet_records, kernel_records=kernel_records,
                serialization_receipts=receipts)

    def save_json(name, payload, reserve, label):
        receipt = JsonAllowance(budgets['evidence'], reserve, label).write(RUN/name, payload)
        receipts.append(receipt)
        return receipt

    def save_arrays(name, arrays):
        if any(type(a) is not np.ndarray or a.dtype.kind not in 'biuf' for a in arrays.values()):
            raise ValueError('full owned numeric proof arrays required, no opaque objects')
        entries = sum(int(a.size) for a in arrays.values())
        budgets['evidence'].charge('c127_complete_native_and_kernel_evidence_encoding', 1024+16*entries)
        path = RUN/name
        with path.open('xb') as stream:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        authentication['evidence_bytes_hashed'] += path.stat().st_size
        authentication['evidence_hash_calls'] += 1
        return dict(file=name, sha256=_sha256(path), numeric_entries=entries,
                    stored_bytes=path.stat().st_size, encoding_work=1024+16*entries)

    def retain(mode, stage, packet, report):
        arrays = {name:value for name,value in packet.items() if type(value) is np.ndarray}
        artifact = save_arrays(mode+'_'+stage+'_arrays.npz', arrays)
        saved = dict(mode=mode, stage=stage, artifact=artifact,
            new_factors=int(packet['new_factors']), constructor=report,
            proof_completed_at_save=False, physical_or_source_admission=False,
            work=pool.used, formal_gain=0)
        packet_records[mode+'_'+stage] = saved
        save_json(mode+'_'+stage+'.json', saved, 65536, 'c127_partial_packet_exact_JSON_reservation')

    def retain_kernel(mode, preparation, reference, comparison):
        arrays = {prefix+'_'+name:array for prefix,evidence in
                  (('word',preparation),('fraction',reference)) for name,array in evidence.items()}
        artifact = save_arrays(mode+'_complete_kernel_proofs.npz', arrays)
        saved = dict(mode=mode, artifact=artifact, comparison=comparison,
            complete_all36_numeric_evidence_retained=True, formal_gain=0,
            source_or_LIVE_admitted=False, cache_reuse_authorized=False)
        kernel_records[mode] = saved
        save_json(mode+'_complete_kernel_proofs.json', saved, 65536,
                  'c127_complete_kernel_proof_exact_JSON_reservation')

    expected_selected = dict(dense=16, masked=16, heterogeneous=12, noop=0)
    for mode in MODES:
        report, case = complete_source(mode, budgets, held, emit, retain, retain_kernel)
        if case['point_evidence'] is not None:
            all_four_points = set(case['point_evidence']) == {'original','expected','expanded','recovered'}
            points = dict(mode=mode, points=case['point_evidence'],
                complete_original_expected_expanded_recovered=all_four_points,
                exact_pairs_not_network_witnesses=True, formal_gain=0)
            receipt = save_json(mode+'_complete_point_evidence.json', points, 262144,
                               'c127_complete_point_exact_JSON_reservation')
            report['complete_point_evidence'] = dict(receipt=receipt,
                counts={name:len(values) for name,values in case['point_evidence'].items()},
                all_four_full_populations_saved=all_four_points)
        else:
            report['complete_point_evidence'] = None
        reports[mode] = report
        save_json(mode+'_complete_source.json', report, 131072,
                  'c127_complete_case_exact_JSON_reservation')
        if (not report['original_source_unchanged']
            or report['binding']['all_actual_source_output_rows_bound'] != 512
            or report['route']['selected_channels'] != expected_selected[mode]
            or not report['complete_native_comparison']['all_native_arrays_bitwise_equal']):
            raise ValueError('unchanged complete ordinary population differs')
        if mode == 'noop':
            if (case['packet'] is not None or case['state'] is not case['direct']['fields']
                or case['kernel_preparation'] is not None or case['kernel_reference'] is not None
                or case['control_packet'] is not None
                or not report['physical']['exact_original_object_retained']
                or report['constructor']['kernel_transform_prepaid']):
                raise ValueError('complete zero-hit source did not stay literal')
        elif (not report['route']['conditional_candidate']
              or not report['physical']['complete_physical_reduction_proved']
              or report['physical']['rejection_reasons']
              or not report['kernel_comparison']['complete_Fraction_agreement']
              or not report['complete_point_evidence']['all_four_full_populations_saved']):
            failures.append(dict(mode=mode, reason='complete mixed source qualification rejected',
                                 physical=report['physical']))
        emit(dict(event='complete_source_and_full_evidence_saved', mode=mode, work=pool.used))
    layout = numeric_layout(held, budgets['ledger'])
    meta = metadata(held, budgets['ledger'])
    if layout.resident_entries > 64000000:
        raise MemoryError('complete source/native/kernel/point evidence exceeds64M entries')
    ledger = dict(numeric=asdict(layout), known_metadata=meta)
    # This allowance was paid before the experiment, as were both terminal
    # success and failure allowances.  The final result never borrows capacity.
    ledger_receipt = ledger_allowance.write(RUN/'complete_held_ledger.json', ledger)
    receipts.append(ledger_receipt)
    if pool.used > COMPLETE_WORK_UPPER:
        raise MemoryError('complete kernel-proof source batch exceeds frozen254M bound')
    return dict(reports=reports, complete_source_count=len(MODES), ledger=ledger,
        qualification_pass=not failures, qualification_failures=failures,
        category_work={k:v.spent for k,v in budgets.items()}, category_limits=CATEGORY_CAPS,
        serialization_receipts=receipts, full_original_four_source_fixtures_retained=True,
        complete_independent_all36_Fraction_word_agreement=True,
        complete_old_new_native_packet_bitwise_agreement=True,
        complete_control_and_new_native_arrays_saved=True,
        complete_point_populations_saved_without_nested_duplication=True,
        mixed_F4_constructor_executed=True, support_compiled_exact_word_rows_executed=True, original_network_loaded=False,
        systematic_complete_kernel_reconstruction_executed=True,
        archived_HZ_loaded=False, source_or_LIVE_admitted=False,
        no_kernel_cache_or_reuse_claim=True, solver_calls=0, formal_gain=0)


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16*1024**3,16*1024**3))
    freeze = json.loads((RUN/'preregistered.json').read_text())
    pool = WorkPool(256000000)
    budgets = {k:BudgetView(pool,v) for k,v in CATEGORY_CAPS.items()}
    final_allowance = JsonAllowance(budgets['ledger'], 524288, 'c127_upfront_terminal_success_JSON_reservation')
    failure_allowance = JsonAllowance(budgets['ledger'], 65536, 'c127_upfront_terminal_failure_JSON_reservation')
    ledger_allowance = JsonAllowance(budgets['ledger'], 131072, 'c127_upfront_complete_ledger_JSON_reservation')
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
