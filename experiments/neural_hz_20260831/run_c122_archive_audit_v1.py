"""Exclusive C122 saved-JSON/identity audit; no additional numeric search."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json
EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c122_channel_route_20260927_v1'


def main():
    started = time.monotonic()
    output = EXP/'C122_TERMINAL_INTEGRITY_20260927.json'
    seal = EXP/'CHECKPOINT_C122_CHANNEL_ROUTE_20260927_SHA256SUMS'
    if output.exists() or seal.exists():
        raise FileExistsError('exclusive complete archive already exists')
    if (_sha256(RUN/'exit.json') !=
        '8ab6f038d1613eddbc9d922468d23a35883bd065b107cc800c3c263d8f638cc4'
        or _sha256(RUN/'result.json') !=
        '0f6f0dfbc6b4c6a8f17c4023b7eb370fec1d75fae6a8ac2d70f0a800c2bb1d1f'):
        raise ValueError('C122 terminal identity differs')
    freeze = json.loads((RUN/'preregistered.json').read_text())
    exited = json.loads((RUN/'exit.json').read_text())
    result = json.loads((RUN/'result.json').read_text())
    pool = WorkPool(256_000_000)
    pool.charge('carried_complete_C122_diagnostic',result['work'])
    pool.charge('saved_JSON_terminal_reporting_no_new_array_analysis',65536)
    report = result['data']['report']
    layout = result['data']['ledger']['numeric']
    if (not exited['all_stages_passed'] or not result['completed']
        or exited['tests_exit'] or exited['worker_exit'] or exited['tests_count']!=3448
        or exited['test_wall_s']>60 or result['work']!=183589516
        or result['work']>freeze['complete_work_upper']
        or not result['measurement']['measured_transient_gate']
        or report['tiles']!=52 or report['conditional_candidates']!=5 or report['empty_routes']!=17
        or layout['resident_bytes']!=454168 or layout['resident_entries']!=69368
        or layout['numeric_storage_count']!=180
        or result['data']['ledger']['known_metadata']['opaque_inherited_ids']
        or report['original_HZ_loaded'] or report['kernel_transform_executed']
        or report['source_or_LIVE_admitted'] or report['solver_calls'] or result['solver_calls']
        or result['source_drift'] or result['input_drift'] or result['provenance_drift']):
        raise ValueError('complete C122 qualification differs')
    if json.loads((RUN/'complete_channel_census.json').read_text()) != report:
        raise ValueError('saved complete report copies differ')
    prior_report = json.loads((EXP/'results/c121_f4_mask_20260922_v1/complete_census.json').read_text())['report']
    key = lambda row:(row['node'],row['tile'],row['y'],row['x'])
    if (len({key(r) for r in report['records']})!=52
        or {key(r) for r in report['records']}!={key(r) for r in prior_report['records']}):
        raise ValueError('complete original tile coverage differs')
    positives = []
    for row in report['records']:
        r,b = row['report'],row['report']['tight_bill']
        if (r['numeric_admission'] or r['actual_global_admission'] or r['formal_gain']
            or not r['packet_bill_only'] or not r['actual_global_reserves_unbound']):
            raise ValueError('conditional route claimed unproved admission')
        if r['conditional_candidate']:
            if (b['nnz_saving_lower']<=0 or b['byte_saving_after_route_mask_lower']<=0
                or b['entry_delta_with_route_mask_upper']>0):
                raise ValueError('conditional candidate gate differs')
            positives.append(dict(node=row['node'],y=row['y'],x=row['x'],
                selected=r['selected_channels'],saved_bytes_lower=b['byte_saving_after_route_mask_lower'],
                auxiliary_upper=b['new_factors'],emission_upper=b['new_emission_work_upper']))
    expected_positive = [
        (19,0,4,17,175488,5218,5124448),
        (19,4,4,26,655248,5422,5167328),
        (19,4,8,24,569816,5255,4308528),
        (19,8,4,28,233600,5537,5717040),
        (19,8,8,23,522528,5206,4256672)]
    if [tuple(p.values()) for p in positives] != expected_positive:
        raise ValueError('complete conditional positive population differs')
    expected = {str((EXP/n).resolve()):h for n,h in freeze['source_sha256'].items()}
    expected.update({str((RUN/n).resolve()):h for n,h in exited['artifacts'].items()})
    expected[str((RUN/'exit.json').resolve())] = _sha256(RUN/'exit.json')
    old_seal = EXP/'CHECKPOINT_C121_F4_MASK_20260922_SHA256SUMS'
    if _sha256(old_seal)!='d45d53abf49373c6e1cb04420eb3b38cdb97f8f7d14f6b1ee4865c34478c4949':
        raise ValueError('C121 seal drift')
    for line in old_seal.read_text().splitlines():
        digest,name = line.split('  ',1)
        path = str((ROOT/name).resolve())
        if path in expected and expected[path]!=digest:
            raise ValueError('historical binding conflict')
        expected[path] = digest
    mismatches = [n for n,h in expected.items() if _sha256(Path(n))!=h]
    input_mismatches = [n for n,h in freeze['input_sha256'].items() if _sha256(Path(n))!=h]
    provenance = _provenance(ROOT)
    record = dict(completed=not mismatches and not input_mismatches and provenance==freeze['provenance'],
        checked_unique_files=len(expected),checked_file_bytes=sum(Path(n).stat().st_size for n in expected),
        mismatches=mismatches,input_mismatches=input_mismatches,provenance=provenance,
        provenance_unchanged=provenance==freeze['provenance'],
        all_52_original_tiles_rechecked=True,conditional_candidates=positives,
        all_five_declared_auxiliary_upper=sum(p['auxiliary_upper'] for p in positives),
        all_five_declared_emission_upper=sum(p['emission_upper'] for p in positives),
        joint_declared_upper_plan_not_admitted=True,actual_old_reserves_still_unproved=True,
        current_archive_scope_not_original_runtime=True,
        prior_C121_all_channel_zero_screen_unchanged=True,
        carried_run_work=result['work'],aggregate_reporting_work=pool.used,reporting_work_parts=pool.parts,
        native_HZ_or_source_admitted=False,solver_calls=0,formal_gain=0,
        archived_numeric_arrays_reanalysed_by_terminal_audit=False,numerical_proofs_rerun=False,
        hash_traffic_separate_from_generation_tokens=True,
        wall_s=time.monotonic()-started,exit_sha256=_sha256(RUN/'exit.json'))
    _atomic_exclusive_json(output,record)
    if not record['completed']:
        raise ValueError('terminal archive integrity failed')
    names = [EXP/n for n in ('C122_CHANNEL_ROUTE_PREREG_20260927.md',
        'C122_CHANNEL_ROUTE_AUDIT_20260927.md','C122_MIXED_NATIVE_HANDOFF_20260927.md',
        'CHECKPOINT_C122_CHANNEL_ROUTE_20260927.md','c122_channel_route_v1.py',
        'test_c122_channel_route_v1.py','c122_channel_route_worker_v1.py',
        'run_c122_channel_route_supervisor_v1.py','run_c122_archive_audit_v1.py')]
    names += [output,*sorted(p for p in RUN.iterdir() if p.is_file())]
    with seal.open('x') as stream:
        for path in sorted(names):
            stream.write(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n')
    print(json.dumps(dict(**record,sealed_files=len(names),seal_sha256=_sha256(seal))),flush=True)


if __name__ == '__main__':
    main()
