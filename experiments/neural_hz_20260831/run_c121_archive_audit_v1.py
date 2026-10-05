"""Exclusive complete C121 saved-JSON/identity audit; no new numeric search."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json
EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c121_f4_mask_20260922_v1'


def main():
    started = time.monotonic()
    output = EXP/'C121_TERMINAL_INTEGRITY_20260922.json'
    seal = EXP/'CHECKPOINT_C121_F4_MASK_20260922_SHA256SUMS'
    if output.exists() or seal.exists():
        raise FileExistsError('exclusive complete archive already exists')
    if (_sha256(RUN/'exit.json') !=
            '48bfe7301010b991ffff9488e9b944ab906f2af91a41a64801b098608578386f'
        or _sha256(RUN/'result.json') !=
            'a622c11882d253ff2231f0f18cb3611a1aac437f4cf6df153117d1d15cb708c4'):
        raise ValueError('complete C121 terminal identity differs')
    freeze = json.loads((RUN/'preregistered.json').read_text())
    exited = json.loads((RUN/'exit.json').read_text())
    result = json.loads((RUN/'result.json').read_text())
    pool = WorkPool(256_000_000)
    pool.charge('carried_complete_C121_diagnostic', result['work'])
    pool.charge('saved_JSON_terminal_reporting_only_no_array_analysis', 65536)
    report = result['census']
    numeric = result['complete_held_root_ledger']['numeric']
    if (not exited['all_stages_passed'] or not result['completed']
        or exited['tests_exit'] or exited['worker_exit'] or exited['tests_count'] != 3426
        or exited['test_wall_s'] > 60 or result['work'] != 163210652
        or result['work'] > freeze['complete_work_upper']
        or not result['census_measurement']['measured_transient_gate']
        or not result['ledger_measurement']['measured_transient_gate']
        or not result['original_expression_preserved']
        or report['nodes_scanned'] != 36 or report['eligible_operators'] != 8
        or report['active_operators'] != 4 or report['total_tiles'] != 52
        or report['topology_candidates'] or report['selected_positions']
        or numeric['resident_bytes'] != 198227972
        or numeric['resident_entries'] != 21638431
        or numeric['numeric_storage_count'] != 968
        or result['solver_calls'] or result['source_lift_executed']
        or result['source_or_LIVE_admitted'] or result['archived_HZ_loaded']
        or result['source_drift'] or result['input_drift'] or result['provenance_drift']):
        raise ValueError('complete saved C121 diagnostic differs')
    census = json.loads((RUN/'complete_census.json').read_text())
    if census['report'] != report:
        raise ValueError('complete independent saved report copy differs')
    nnz_only = []
    for row in report['records']:
        cost, bill = row['cost'], row['cost']['bill']
        if (cost['topology_qualified'] or cost['numeric_admission']
            or cost['actual_global_admission'] or not cost['packet_bill_only']
            or bill['byte_saving_lower'] >= 0 or bill['entry_delta_upper'] <= 0):
            raise ValueError('all-channel conditional upper screen differs')
        if bill['nnz_saving_lower'] > 0:
            nnz_only.append(dict(node=row['node'],y=row['y'],x=row['x'],
                nnz_saving_lower=bill['nnz_saving_lower'],
                byte_increase_upper=-bill['byte_saving_lower'],
                packet_entry_delta_upper=bill['entry_delta_upper']))
    if nnz_only != [dict(node=19,y=4,x=4,nnz_saving_lower=14772,
                        byte_increase_upper=472272,packet_entry_delta_upper=67712),
                    dict(node=19,y=4,x=8,nnz_saving_lower=4265,
                        byte_increase_upper=565004,packet_entry_delta_upper=83799)]:
        raise ValueError('complete raw-nnz-only evidence differs')
    expected = {str((EXP/n).resolve()):h for n,h in freeze['source_sha256'].items()}
    expected.update({str((RUN/n).resolve()):h for n,h in exited['artifacts'].items()})
    expected[str((RUN/'exit.json').resolve())] = _sha256(RUN/'exit.json')
    old_seal = EXP/'CHECKPOINT_C120_INTEGER_F4_20260922_SHA256SUMS'
    if _sha256(old_seal) != '1b8339e775d503c7a29aa4f05f2fe0ca75bf5a8b73033e11e4f0bdeb89187dc3':
        raise ValueError('C120 seal drift')
    for line in old_seal.read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = str((ROOT/name).resolve())
        if path in expected and expected[path] != digest:
            raise ValueError('historical binding conflict')
        expected[path] = digest
    mismatches = [name for name,h in expected.items() if _sha256(Path(name)) != h]
    input_mismatches = [n for n,h in freeze['input_sha256'].items() if _sha256(Path(n)) != h]
    provenance = _provenance(ROOT)
    record = dict(completed=not mismatches and not input_mismatches and provenance == freeze['provenance'],
        checked_unique_files=len(expected),
        checked_file_bytes=sum(Path(n).stat().st_size for n in expected),
        mismatches=mismatches, input_mismatches=input_mismatches, provenance=provenance,
        provenance_unchanged=provenance == freeze['provenance'],
        complete_fresh_diagnostic_rechecked=True, all_52_conditional_candidates=0,
        raw_nnz_only_positives=nnz_only,
        upper_bound_failure_is_not_impossibility=True,
        prior_failed_C117_C119_runs_still_failed=True,
        qualified_C120_ordinary_sources_unchanged=True,
        carried_run_work=result['work'], aggregate_reporting_work=pool.used,
        reporting_work_parts=pool.parts,
        actual_network_source_or_LIVE_admitted=False, solver_calls=0, formal_gain=0,
        archived_numeric_arrays_reanalysed=False, numerical_proofs_rerun=False,
        hash_traffic_separate_from_generation_tokens=True,
        wall_s=time.monotonic()-started, exit_sha256=_sha256(RUN/'exit.json'))
    _atomic_exclusive_json(output, record)
    if not record['completed']:
        raise ValueError('terminal archive integrity failed')
    names = [EXP/n for n in ('C121_F4_MASK_PREREG_20260922.md',
        'C121_F4_MASK_AUDIT_20260922.md', 'C121_CHANNEL_ROUTED_F4_HANDOFF_20260922.md',
        'CHECKPOINT_C121_F4_MASK_20260922.md', 'c121_f4_mask_cost_v1.py',
        'test_c121_f4_mask_cost_v1.py', 'c121_f4_mask_worker_v1.py',
        'run_c121_f4_mask_supervisor_v1.py', 'run_c121_archive_audit_v1.py')]
    names += [output, *sorted(p for p in RUN.iterdir() if p.is_file())]
    with seal.open('x') as stream:
        for path in sorted(names):
            stream.write(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n')
    print(json.dumps(dict(**record,sealed_files=len(names),seal_sha256=_sha256(seal))),flush=True)


if __name__ == '__main__':
    main()
