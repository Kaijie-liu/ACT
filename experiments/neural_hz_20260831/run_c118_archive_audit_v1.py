# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exclusive terminal reporting and hash seal; no new numerical search."""
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
RUN = EXP / 'results/c118_column_orbits_20260922_v1'


def main():
    start = time.monotonic()
    output = EXP / 'C118_TERMINAL_INTEGRITY_20260922.json'
    seal = EXP / 'CHECKPOINT_C118_COLUMN_ORBITS_20260922_SHA256SUMS'
    if output.exists() or seal.exists():
        raise FileExistsError('exclusive terminal archive already exists')
    if _sha256(RUN/'exit.json') != '915cea4a9fcb4c8692db4d278e8d36cf0e2ccd0f84bfe48508a3c459e1c8aa3d':
        raise ValueError('complete C118 terminal identity differs')
    freeze = json.loads((RUN/'preregistered.json').read_text())
    exited = json.loads((RUN/'exit.json').read_text())
    result = json.loads((RUN/'result.json').read_text())
    reports = result['data']['reports']
    pool = WorkPool(256_000_000)
    pool.charge('carried_complete_C118_diagnostic',result['work'])
    pool.charge('terminal_reporting_only',8192+4096*len(reports))
    duplicates = [(node,group) for node,r in reports.items() for group in r['duplicate_column_groups']]
    if (len(reports)!=13 or not exited['all_stages_passed'] or not result['completed']
        or exited['tests_exit'] or exited['tests_count']!=3363
        or sum(r['total_columns'] for r in reports.values())!=175616
        or sum(r['old_coefficient_nnz'] for r in reports.values())!=1417596
        or any(r['strict_nnz_positive_group_count'] or r['candidate_conditional_nnz_saving']
            for r in reports.values())
        or len(duplicates)!=1 or duplicates[0][0]!='27'
        or (duplicates[0][1]['size'],duplicates[0][1]['degree'])!=(6148,0)
        or reports['35']['class_count']!=6272 or result['data']['solver_calls']):
        raise ValueError('complete negative diagnostic report mismatch')
    expected = {str((EXP/n).resolve()):h for n,h in freeze['source_sha256'].items()}
    expected.update({str((RUN/n).resolve()):h for n,h in exited['artifacts'].items()})
    old_seal = EXP/'CHECKPOINT_C117_AFFINE_BLOCK_20260922_SHA256SUMS'
    if _sha256(old_seal)!='4761e8da4c43244378d789b58058f298fe01fe76e3b2e873caa32202016cd3b7':
        raise ValueError('C117 seal drift')
    for line in old_seal.read_text().splitlines():
        digest,name=line.split('  ',1)
        path=str((ROOT/name).resolve())
        if path in expected and expected[path]!=digest:
            raise ValueError('historical binding conflict')
        expected[path]=digest
    mismatches=[name for name,h in expected.items() if _sha256(Path(name))!=h]
    provenance=_provenance(ROOT)
    record=dict(completed=not mismatches and provenance==freeze['provenance'],
        checked_unique_files=len(expected),checked_file_bytes=sum(Path(n).stat().st_size for n in expected),
        mismatches=mismatches,provenance=provenance,provenance_unchanged=provenance==freeze['provenance'],
        complete_negative_column_report_rechecked=True,old_C117_seal_unchanged=True,
        prior_failed_C117_ledger_still_failed=True,carried_run_work=result['work'],
        aggregate_reporting_work=pool.used,reporting_work_parts=pool.parts,
        archived_numeric_arrays_reanalysed=False,numeric_hash_traffic_in_token_pool=False,
        wall_s=time.monotonic()-start,formal_gain=0,exit_sha256=_sha256(RUN/'exit.json'))
    _atomic_exclusive_json(output,record)
    if not record['completed']:
        raise ValueError('terminal integrity failed')
    names=[EXP/n for n in ('C118_COLUMN_ORBITS_PREREG_20260922.md',
        'C118_COLUMN_ORBITS_AUDIT_20260922.md','C118_EXACT_COLUMN_THEOREM_20260922.md',
        'C118_LARGER_CIRCUIT_HANDOFF_20260922.md','CHECKPOINT_C118_COLUMN_ORBITS_20260922.md',
        'c118_column_orbits_v1.py','test_c118_column_orbits_v1.py',
        'c118_column_orbit_worker_v1.py','run_c118_column_orbit_supervisor_v1.py',
        'run_c118_archive_audit_v1.py')]
    names += [output,*sorted(p for p in RUN.iterdir() if p.is_file())]
    with seal.open('x') as stream:
        for path in sorted(names):
            stream.write(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n')
    print(json.dumps(dict(**record,sealed_files=len(names),seal_sha256=_sha256(seal))),flush=True)


if __name__=='__main__':
    main()
