# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reporting-only terminal integrity; carries failed C117 work, no new search."""
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
RUN = EXP / 'results/c117_affine_block_census_20260922_v1'


def main():
    start = time.monotonic()
    output = EXP / 'C117_TERMINAL_INTEGRITY_20260922.json'
    seal = EXP / 'CHECKPOINT_C117_AFFINE_BLOCK_20260922_SHA256SUMS'
    if output.exists() or seal.exists():
        raise FileExistsError('terminal archive is exclusive')
    if _sha256(RUN / 'exit.json') != '381e5448b7b092e47a558ddaee824323953130c13342511712caad0a4c225da7':
        raise ValueError('failed terminal identity changed')
    freeze = json.loads((RUN / 'preregistered.json').read_text())
    exited = json.loads((RUN / 'exit.json').read_text())
    result = json.loads((RUN / 'result.json').read_text())
    data = json.loads((RUN / 'complete_census.json').read_text())
    pool = WorkPool(256_000_000)
    pool.charge('carried_complete_C117_diagnostic', result['work'])
    nodes = data['report']['nodes']
    pool.charge('terminal_reporting_only', 8192 + 4096 * len(nodes))
    if (len(nodes) != 36 or data['report']['repeated_complete_programs']
        or data['report']['repeated_operator_content']
        or sum(n['consumer_occurrences'] for n in nodes) !=
            sum(n['continuous_edges'] for n in nodes if n['kind'] != 'source')
        or exited['tests_count'] != 3341 or exited['tests_exit']
        or exited['all_stages_passed'] or result['completed'] or result['solver_calls']):
        raise ValueError('complete negative/failure/test report mismatch')
    expected = {str((EXP / n).resolve()): h for n, h in freeze['source_sha256'].items()}
    expected.update(freeze['input_sha256'])
    expected.update({str((RUN / n).resolve()): h for n, h in exited['artifacts'].items()})
    old_seal = EXP / 'CHECKPOINT_C116_ROW_COMPOSITION_20260920_SHA256SUMS'
    if _sha256(old_seal) != 'fe9ff752fbb0de3cfd6668525d5ea8a0cacbd9bad515eb9011b987eb488b7c05':
        raise ValueError('frozen C116 seal changed')
    for line in old_seal.read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = str((ROOT / name).resolve())
        if path in expected and expected[path] != digest:
            raise ValueError('historical binding conflict')
        expected[path] = digest
    mismatches = [name for name, digest in expected.items() if _sha256(Path(name)) != digest]
    provenance = _provenance(ROOT)
    report = dict(completed=not mismatches and provenance == freeze['provenance'],
        checked_unique_files=len(expected), checked_file_bytes=sum(Path(n).stat().st_size for n in expected),
        mismatches=mismatches, provenance=provenance,
        provenance_unchanged=provenance == freeze['provenance'],
        complete_negative_report_rechecked=True, old_C116_seal_unchanged=True,
        carried_failed_run_work=result['work'], aggregate_reporting_work=pool.used,
        reporting_work_parts=pool.parts, wall_s=time.monotonic() - start,
        archived_numeric_arrays_reanalysed=False, numerical_hash_traffic_in_token_pool=False,
        failed_ledger_remains_failed=True, post_expression_preservation_unproved=True,
        formal_gain=0, exit_sha256=_sha256(RUN / 'exit.json'))
    _atomic_exclusive_json(output, report)
    if not report['completed']:
        raise ValueError('archive integrity failed')
    names = [EXP / n for n in ('C117_AFFINE_BLOCK_CENSUS_PREREG_20260922.md',
        'C117_AFFINE_BLOCK_CENSUS_AUDIT_20260922.md', 'C117_STRUCTURAL_HANDOFF_20260922.md',
        'CHECKPOINT_C117_AFFINE_BLOCK_20260922.md', 'c117_affine_block_census_v1.py',
        'test_c117_affine_block_census_v1.py', 'c117_affine_block_worker_v1.py',
        'run_c117_affine_block_supervisor_v1.py', 'run_c117_archive_audit_v1.py')]
    names += [output, *sorted(p for p in RUN.iterdir() if p.is_file())]
    with seal.open('x') as stream:
        for path in sorted(names):
            stream.write(_sha256(path) + '  ' + str(path.relative_to(ROOT)) + '\n')
    print(json.dumps(dict(**report, sealed_files=len(names), seal_sha256=_sha256(seal))), flush=True)


if __name__ == '__main__':
    main()
