"""Hash-only immutable archive of a FAILED numerical run; no requalification.

Source/evidence authentication was separately counted by preregistration.
The exhausted 256M numerical pool stays exhausted. This script never loads
numeric packet arrays, reruns a proof, finishes the missing ledger or admits HZ.
"""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _sha256, _atomic_exclusive_json
EXP = Path(__file__).resolve().parent
RUN = EXP/'results/c119_denominator_f4_20260922_v1'


def main():
    started = time.monotonic()
    output = EXP/'C119_TERMINAL_INTEGRITY_20260922.json'
    seal = EXP/'CHECKPOINT_C119_DENOMINATOR_F4_20260922_SHA256SUMS'
    if output.exists() or seal.exists():
        raise FileExistsError('exclusive terminal archive already exists')
    if (_sha256(RUN/'exit.json') !=
            'adf8b07777e65b1022a1939870e109de66187d6e168d272b680ed9e1fbdbfdcd'
        or _sha256(RUN/'result.json') !=
            '185235a4634fbee10ee59e981b6511fcf8286da7d3e07aa91e4700873d1e5147'):
        raise ValueError('complete failed C119 terminal identity differs')
    freeze = json.loads((RUN/'preregistered.json').read_text())
    exited = json.loads((RUN/'exit.json').read_text())
    result = json.loads((RUN/'result.json').read_text())
    if (exited['all_stages_passed'] or result['completed']
        or exited['tests_exit'] or exited['tests_count'] != 3384
        or result['work'] != 255999988 or result['failure']['type'] != 'MemoryError'
        or not result['measurement']['measured_transient_gate']
        or result['source_drift'] or result['provenance_drift']):
        raise ValueError('saved failed-run classification differs')
    expected = {str((EXP/n).resolve()): h for n,h in freeze['source_sha256'].items()}
    expected.update({str((RUN/n).resolve()): h for n,h in exited['artifacts'].items()})
    expected[str((RUN/'exit.json').resolve())] = _sha256(RUN/'exit.json')
    old_seal = EXP/'CHECKPOINT_C118_COLUMN_ORBITS_20260922_SHA256SUMS'
    if _sha256(old_seal) != '59a2e37ff939f27b8a0e75c30227e6843709c34deb1bc5fe41559fbf44b45cf5':
        raise ValueError('C118 seal drift')
    for line in old_seal.read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = str((ROOT/name).resolve())
        if path in expected and expected[path] != digest:
            raise ValueError('historical binding conflict')
        expected[path] = digest
    mismatches = [name for name,h in expected.items() if _sha256(Path(name)) != h]
    provenance = _provenance(ROOT)
    record = dict(archive_integrity_pass=not mismatches and provenance == freeze['provenance'],
        checked_unique_files=len(expected),
        checked_file_bytes=sum(Path(n).stat().st_size for n in expected),
        mismatches=mismatches, provenance=provenance,
        provenance_unchanged=provenance == freeze['provenance'],
        scope='hash_only_failed_run_archive_not_numerical_requalification',
        recorded_numerical_work=result['work'], numerical_budget_exhausted_preserved=True,
        combined_ledger_completed=False, failed_run_requalified=False,
        archived_numeric_arrays_reanalysed=False, numerical_proofs_rerun=False,
        source_or_LIVE_admitted=False, solver_calls=0, formal_gain=0,
        hash_traffic_separate_from_generation_tokens=True,
        wall_s=time.monotonic()-started, exit_sha256=_sha256(RUN/'exit.json'))
    _atomic_exclusive_json(output, record)
    if not record['archive_integrity_pass']:
        raise ValueError('terminal archive integrity failed')
    names = [EXP/n for n in ('C119_DENOMINATOR_F4_PREREG_20260922.md',
        'C119_DENOMINATOR_F4_AUDIT_20260922.md', 'C119_EXACT_F4_THEOREM_20260922.md',
        'C119_EXACT_INTEGER_F4_HANDOFF_20260922.md', 'CHECKPOINT_C119_DENOMINATOR_F4_20260922.md',
        'c119_denominator_f4_v1.py', 'c119_f4_oracle_v1.py', 'test_c119_denominator_f4_v1.py',
        'c119_complete_source_fixture_v1.py', 'c119_denominator_f4_worker_v1.py',
        'run_c119_denominator_f4_supervisor_v1.py', 'run_c119_archive_integrity_v1.py')]
    names += [output, *sorted(p for p in RUN.iterdir() if p.is_file())]
    with seal.open('x') as stream:
        for path in sorted(names):
            stream.write(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n')
    print(json.dumps(dict(**record, sealed_files=len(names), seal_sha256=_sha256(seal))), flush=True)


if __name__ == '__main__':
    main()
