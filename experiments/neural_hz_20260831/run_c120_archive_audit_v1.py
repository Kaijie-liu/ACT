"""Exclusive complete qualification archive; saved-JSON checks, no new search."""
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
RUN = EXP/'results/c120_denominator_f4_20260922_v1'


def main():
    started = time.monotonic()
    output = EXP/'C120_TERMINAL_INTEGRITY_20260922.json'
    seal = EXP/'CHECKPOINT_C120_INTEGER_F4_20260922_SHA256SUMS'
    if output.exists() or seal.exists():
        raise FileExistsError('exclusive complete archive already exists')
    if (_sha256(RUN/'exit.json') !=
            'a7bb985a22ad88012b1947e29564b33154aa390852ca7664b26d646c64a9f798'
        or _sha256(RUN/'result.json') !=
            'd7222792018b031063c37ee8112395b465409488ad618eb77ece09471ed8f30d'):
        raise ValueError('complete C120 terminal identity differs')
    freeze = json.loads((RUN/'preregistered.json').read_text())
    exited = json.loads((RUN/'exit.json').read_text())
    result = json.loads((RUN/'result.json').read_text())
    pool = WorkPool(256_000_000)
    pool.charge('carried_complete_C120_diagnostic', result['work'])
    pool.charge('saved_JSON_terminal_reporting_and_geometry_only', 65536)
    data = result['data']
    if (not exited['all_stages_passed'] or not result['completed']
        or exited['tests_exit'] or exited['tests_count'] != 3405
        or result['work'] != 245413428 or result['work'] > freeze['complete_work_upper']
        or not result['measurement']['measured_transient_gate']
        or data['complete_source_count'] != 2 or data['solver_calls']
        or data['complete_numeric_layout']['resident_bytes'] != 7851956
        or data['complete_numeric_layout']['resident_entries'] != 1003745
        or result['source_drift'] or result['provenance_drift']):
        raise ValueError('complete saved C120 qualification differs')
    for mode, before, after in (('dense',1095680,820800),('masked',863560,786632)):
        r = data['reports'][mode]
        p = r['f4_physical']
        if (not p['complete_physical_reduction_proved'] or p['rejection_reasons']
            or p['before_numeric_bytes'] != before or p['after_numeric_bytes'] != after
            or not p['actual_complete_inverse_equal'] or p['new_factors'] != 1728
            or r['complete_source_binding']['all_actual_source_output_rows_bound'] != 512
            or p['complete_original_audit']['original_binary_factors'] != 1):
            raise ValueError('complete ordinary F4 source report differs')
    # Full packet hashes agree, but the current actual oracle was ALSO rerun.
    previous = EXP/'results/c119_denominator_f4_20260922_v1'
    old_exit = json.loads((previous/'exit.json').read_text())
    packet_names = [n for n in exited['artifacts'] if n.endswith('.npz')]
    if len(packet_names) != 12 or any(exited['artifacts'][n] != old_exit['artifacts'][n]
                                      for n in packet_names):
        raise ValueError('same complete exact packet evidence differs from C119')
    expected = {str((EXP/n).resolve()):h for n,h in freeze['source_sha256'].items()}
    expected.update({str((RUN/n).resolve()):h for n,h in exited['artifacts'].items()})
    expected[str((RUN/'exit.json').resolve())] = _sha256(RUN/'exit.json')
    old_seal = EXP/'CHECKPOINT_C119_DENOMINATOR_F4_20260922_SHA256SUMS'
    if _sha256(old_seal) != 'a86304296d4c7363fe1c61ad1c8c7bac7ac04e44366dc6afc5771edb4a29f728':
        raise ValueError('C119 seal drift')
    for line in old_seal.read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = str((ROOT/name).resolve())
        if path in expected and expected[path] != digest:
            raise ValueError('historical binding conflict')
        expected[path] = digest
    mismatches = [name for name,h in expected.items() if _sha256(Path(name)) != h]
    provenance = _provenance(ROOT)
    record = dict(completed=not mismatches and provenance == freeze['provenance'],
        checked_unique_files=len(expected),
        checked_file_bytes=sum(Path(n).stat().st_size for n in expected),
        mismatches=mismatches, provenance=provenance,
        provenance_unchanged=provenance == freeze['provenance'],
        complete_ordinary_qualification_rechecked=True,
        all_12_packet_archive_hashes_identical_to_C119=True,
        prior_failed_C119_run_still_failed=True,
        carried_run_work=result['work'], aggregate_reporting_work=pool.used,
        reporting_work_parts=pool.parts,
        actual_network_source_or_LIVE_admitted=False, solver_calls=0, formal_gain=0,
        archived_numeric_arrays_reanalysed=False, numerical_proofs_rerun=False,
        hash_traffic_separate_from_generation_tokens=True,
        wall_s=time.monotonic()-started, exit_sha256=_sha256(RUN/'exit.json'))
    _atomic_exclusive_json(output, record)
    if not record['completed']:
        raise ValueError('terminal archive integrity failed')
    names = [EXP/n for n in ('C120_DENOMINATOR_F4_PREREG_20260922.md',
        'C120_INTEGER_F4_AUDIT_20260922.md', 'C120_INTEGER_F4_THEOREM_20260922.md',
        'C120_REAL_F4_SOURCE_HANDOFF_20260922.md', 'CHECKPOINT_C120_INTEGER_F4_20260922.md',
        'c120_word_f4_v1.py', 'test_c120_word_f4_v1.py',
        'c120_complete_source_fixture_v1.py', 'c120_denominator_f4_worker_v1.py',
        'run_c120_denominator_f4_supervisor_v1.py', 'run_c120_archive_audit_v1.py')]
    names += [output, *sorted(p for p in RUN.iterdir() if p.is_file())]
    with seal.open('x') as stream:
        for path in sorted(names):
            stream.write(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n')
    print(json.dumps(dict(**record,sealed_files=len(names),seal_sha256=_sha256(seal))),flush=True)


if __name__ == '__main__':
    main()
