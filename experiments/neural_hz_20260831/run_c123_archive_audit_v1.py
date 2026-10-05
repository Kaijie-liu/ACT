"""Exclusive C123 saved-JSON/identity audit; no additional numeric search."""
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
RUN = EXP / 'results/c123_source_binding_20260927_v1'


def main():
    started = time.monotonic()
    output = EXP / 'C123_TERMINAL_INTEGRITY_20260927.json'
    seal = EXP / 'CHECKPOINT_C123_SOURCE_BINDING_20260927_SHA256SUMS'
    if output.exists() or seal.exists():
        raise FileExistsError('exclusive complete archive already exists')
    if (_sha256(RUN / 'exit.json') !=
        'a63e1df28d43b338223c5e7f76fa1070fc14cee132bec236547736e22ab7550d'
        or _sha256(RUN / 'result.json') !=
        '5555cd2cf582f5a711f8eaa44baf5328ad71bc12bc09267a820f666298cd7ca3'):
        raise ValueError('C123 terminal identity differs')
    freeze = json.loads((RUN / 'preregistered.json').read_text())
    exited = json.loads((RUN / 'exit.json').read_text())
    result = json.loads((RUN / 'result.json').read_text())
    inventory = json.loads((RUN / 'inventory.json').read_text())
    pool = WorkPool(256_000_000)
    pool.charge('carried_complete_C123_diagnostic', result['work'])
    pool.charge('saved_JSON_terminal_reporting_no_new_array_analysis', 65536)
    data, measurement = result['data'], result['measurement']
    layout = data['ledger']['numeric']
    metadata = data['ledger']['known_metadata']
    if (not exited['all_stages_passed'] or not result['completed']
        or exited['tests_exit'] or exited['worker_exit']
        or exited['tests_count'] != 3470 or freeze['required_test_count'] != 3470
        or inventory['count'] != 3470 or len(set(inventory['nodeids'])) != 3470
        or len(inventory['nodeids']) != 3470 or len(freeze['tests']) != 150
        or exited['test_wall_s'] > 60 or result['work'] != 141178782
        or exited['work'] != result['work'] or freeze['complete_work_upper'] != 164000000
        or result['work'] > freeze['complete_work_upper']
        or sum(result['work_parts'].values()) != result['work']
        or not measurement['measured_transient_gate']
        or measurement['resident_growth_upper_bound_bytes'] > 1073741824
        or measurement['traced_peak_bytes'] + measurement['tracer_metadata_bytes'] > 1073741824
        or layout['resident_bytes'] != 6297096 or layout['resident_entries'] != 722699
        or layout['numeric_storage_count'] != 188
        or metadata['nonoverlapping_known_metadata_bytes'] != 31767055
        or len(metadata['opaque_inherited_ids']) != 4
        or not metadata['python_allocator_occupancy_not_measured']
        or data['complete_source_count'] != 2 or exited['complete_source_count'] != 2
        or not data['all_original_source_rows_compared']
        or any(data[k] for k in ('archived_HZ_loaded', 'mixed_F4_constructor_executed',
            'original_network_loaded', 'source_or_LIVE_admitted', 'solver_calls', 'formal_gain'))
        or result['solver_calls'] or result['formal_gain'] or exited['formal_gain']
        or any(result[k] or exited[k] for k in ('source_drift', 'input_drift', 'provenance_drift'))):
        raise ValueError('complete C123 qualification differs')
    expected_categories = dict(source=64000000, setup=265568, old=28632960,
        new=13361024, evidence=15093824, ledger=19825406)
    if (data['category_work'] != expected_categories
        or sum(expected_categories.values()) != result['work']
        or data['category_limits'] != freeze['category_caps']
        or sum(data['category_limits'].values()) != freeze['complete_work_upper']
        or any(v > data['category_limits'][k] for k, v in expected_categories.items())
        or json.loads((RUN / 'complete_held_ledger.json').read_text()) != data['ledger']):
        raise ValueError('complete category/ledger reconciliation differs')
    expected_cases = dict(dense=(512, 74240, 14318208, 7566976),
                          masked=(512, 55808, 14314752, 5794048))
    if set(data['reports']) != set(expected_cases):
        raise ValueError('complete original source population differs')
    summaries = []
    for mode, (rows, nnz, old_work, new_work) in expected_cases.items():
        report = data['reports'][mode]
        binding = report['source_binding']
        if (json.loads((RUN / (mode + '_complete_binding.json')).read_text()) != report
            or report['mode'] != mode or report['all_original_rows'] != rows
            or report['all_original_nnz'] != nnz or report['fraction_work'] != old_work
            or report['word_work'] != new_work or report['work_saved'] != old_work - new_work
            or not report['complete_literals_gauges_pivot_lookup_and_survival_equal']
            or not report['original_source_unchanged']
            or report['original_source_reserved_work'] != 32000000
            or report['original_nonconvex_binary_factors'] != 1
            or report['original_inequality_count'] != 1
            or report['word_wall_s'] <= report['fraction_wall_s']
            or binding['all_actual_source_output_rows_bound'] != rows
            or binding['all_actual_native_coefficients_compared'] != nnz
            or binding['complete_observed_work'] != new_work
            or not binding['no_fraction_arithmetic']
            or not binding['no_approximate_coefficient_comparison']
            or not binding['exact_coefficients_and_original_row_gauges_equal']
            or binding['numeric_admission'] or binding['actual_global_admission']
            or binding['formal_gain'] or report['formal_gain'] or report['source_or_LIVE_admitted']
            or report['arrays_sha256'] != exited['artifacts'][mode + '_complete_binding_arrays.npz']):
            raise ValueError('complete case qualification differs: ' + mode)
        summaries.append(dict(mode=mode, rows=rows, native_coefficients=nnz,
            old_work=old_work, new_work=new_work, saved_work=old_work-new_work,
            old_wall_s=report['fraction_wall_s'], new_wall_s=report['word_wall_s'],
            measured_new_binder_slower=True))

    expected = {}

    def bind(path, digest):
        name = str(path.resolve())
        if name in expected and expected[name] != digest:
            raise ValueError('historical binding conflict: ' + name)
        expected[name] = digest

    for name, digest in freeze['source_sha256'].items():
        bind(EXP / name, digest)
    for name, digest in exited['artifacts'].items():
        bind(RUN / name, digest)
    bind(RUN / 'exit.json', _sha256(RUN / 'exit.json'))
    old_seal = EXP / 'CHECKPOINT_C122_CHANNEL_ROUTE_20260927_SHA256SUMS'
    if _sha256(old_seal) != 'c52b49877b7974c1aafbaa421086d9efd9cc5c602141b52f9568a7febe9f19a5':
        raise ValueError('C122 seal drift')
    for line in old_seal.read_text().splitlines():
        digest, name = line.split('  ', 1)
        bind(ROOT / name, digest)
    mismatches = [n for n, h in expected.items() if _sha256(Path(n)) != h]
    input_mismatches = [n for n, h in freeze['input_sha256'].items() if _sha256(Path(n)) != h]
    provenance = _provenance(ROOT)
    record = dict(completed=not mismatches and not input_mismatches and provenance == freeze['provenance'],
        checked_unique_files=len(expected), checked_file_bytes=sum(Path(n).stat().st_size for n in expected),
        mismatches=mismatches, input_mismatches=input_mismatches, provenance=provenance,
        provenance_unchanged=provenance == freeze['provenance'], complete_source_cases=summaries,
        all_original_output_rows=1024, all_actual_native_coefficients=130048,
        complete_old_new_binding_agreement=True, original_nonconvex_sources_unchanged=True,
        both_measured_transient_gates_pass=True, speedup_claimed=False,
        metadata_is_known_nonoverlap_not_complete_allocator_occupancy=True,
        carried_run_work=result['work'], aggregate_reporting_work=pool.used, reporting_work_parts=pool.parts,
        native_mixed_HZ_or_source_admitted=False, solver_calls=0, formal_gain=0,
        archived_numeric_arrays_reanalysed_by_terminal_audit=False, numerical_proofs_rerun=False,
        hash_traffic_separate_from_generation_tokens=True,
        wall_s=time.monotonic()-started, exit_sha256=_sha256(RUN / 'exit.json'))
    _atomic_exclusive_json(output, record)
    if not record['completed']:
        raise ValueError('terminal archive integrity failed')
    names = [EXP / n for n in ('C123_SOURCE_BINDING_PREREG_20260927.md',
        'C123_SOURCE_BINDING_AUDIT_20260927.md', 'C123_NATIVE_MIXED_HANDOFF_20260927.md',
        'CHECKPOINT_C123_SOURCE_BINDING_20260927.md', 'c123_support_word_binding_v1.py',
        'test_c123_support_word_binding_v1.py', 'c123_source_binding_worker_v1.py',
        'run_c123_source_binding_supervisor_v1.py', 'run_c123_archive_audit_v1.py')]
    names += [output, *sorted(p for p in RUN.iterdir() if p.is_file())]
    with seal.open('x') as stream:
        for path in sorted(names):
            stream.write(_sha256(path) + '  ' + str(path.relative_to(ROOT)) + '\n')
    print(json.dumps(dict(**record, sealed_files=len(names), seal_sha256=_sha256(seal))), flush=True)


if __name__ == '__main__':
    main()
