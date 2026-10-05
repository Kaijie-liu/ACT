"""Saved-JSON/hash audit: retain C124 evidence, reject unpaid reporting."""
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
RUN = EXP / 'results/c124_mixed_source_20260927_v1'


def main():
    started = time.monotonic()
    output = EXP / 'C124_TERMINAL_INTEGRITY_20260927.json'
    seal = EXP / 'CHECKPOINT_C124_MIXED_SOURCE_20260927_SHA256SUMS'
    if output.exists() or seal.exists():
        raise FileExistsError('exclusive terminal record already exists')
    if (_sha256(RUN / 'exit.json') != '01c44e0f5aaa08025a241df83692f8cbfd0c98c52735ac6f35b69e5ce9c42aab'
        or _sha256(RUN / 'result.json') != 'a4fa77b2d99a0fc166886b0d51b2c1142c80c2bd813763ad4872a3ab03d401ac'):
        raise ValueError('terminal C124 identity differs')
    freeze = json.loads((RUN / 'preregistered.json').read_text())
    exited = json.loads((RUN / 'exit.json').read_text())
    result = json.loads((RUN / 'result.json').read_text())
    data, measurement = result['data'], result['measurement']
    pool = WorkPool(256000000)
    pool.charge('carried_RECORDED_C124_counter_not_a_complete_payment_certificate', result['work'])
    pool.charge('saved_JSON_terminal_reporting_no_new_array_analysis', 65536)
    if (not exited['all_stages_passed'] or not result['completed']
        or not data['qualification_pass'] or data['qualification_failures']
        or exited['tests_exit'] or exited['worker_exit'] or exited['tests_count'] != 3495
        or len(freeze['tests']) != 151 or exited['test_wall_s'] > 60
        or data['complete_source_count'] != 4 or result['work'] != 222618270
        or sum(result['work_parts'].values()) != result['work']
        or sum(data['category_work'].values()) != result['work']
        or data['category_limits'] != freeze['category_caps']
        or sum(data['category_limits'].values()) != 254000000
        or any(v > data['category_limits'][k] for k, v in data['category_work'].items())
        or not measurement['measured_transient_gate']
        or measurement['resident_growth_upper_bound_bytes'] > 1073741824
        or measurement['traced_peak_bytes']+measurement['tracer_metadata_bytes'] > 1073741824
        or any(result[k] or exited[k] for k in ('source_drift', 'input_drift', 'provenance_drift'))
        or result['solver_calls'] or result['formal_gain']
        or any(data[k] for k in ('original_network_loaded', 'archived_HZ_loaded',
                                 'source_or_LIVE_admitted', 'solver_calls', 'formal_gain'))):
        raise ValueError('raw numerical evidence differs from recorded observation')
    layout, metadata = data['ledger']['numeric'], data['ledger']['known_metadata']
    if (layout['resident_bytes'] != 6845956 or layout['resident_entries'] != 770827
        or layout['numeric_storage_count'] != 408
        or metadata['nonoverlapping_known_metadata_bytes'] != 26453156
        or len(metadata['opaque_inherited_ids']) != 8
        or json.loads((RUN / 'complete_held_ledger.json').read_text()) != data['ledger']):
        raise ValueError('complete held ledger observation differs')
    inventory = json.loads((RUN / 'inventory.json').read_text())
    if inventory['count'] != 3495 or len(inventory['nodeids']) != 3495 or len(set(inventory['nodeids'])) != 3495:
        raise ValueError('complete original/new test inventory differs')
    cases, encoding = [], []
    expected_cases = dict(dense=(16,74240,-274864,-18344,1728,38784),
        masked=(16,55808,-76912,-1848,1728,36848),
        heterogeneous=(12,55936,-145456,-7896,1584,32224))
    if set(data['reports']) != {*expected_cases, 'noop'}:
        raise ValueError('complete fixed source population differs')

    def encoding_check(names, reserve):
        sizes = {name:(RUN / name).stat().st_size for name in names}
        required = sum(sizes.values())+1024*len(names)
        entry = dict(files=sizes, header_bytes=1024*len(names), required=required,
                     fixed_reservation=reserve, excess=max(0, required-reserve), passed=required <= reserve)
        encoding.append(entry)
        return entry

    for mode in ('dense', 'masked', 'heterogeneous', 'noop'):
        report = data['reports'][mode]
        if (json.loads((RUN / (mode+'_complete_source.json')).read_text()) != report
            or not report['original_source_unchanged']
            or report['binding']['all_actual_source_output_rows_bound'] != 512
            or report['original_nonconvex_binary_factors'] != 1 or report['original_inequalities'] != 1
            or report['source_or_LIVE_admitted'] or report['formal_gain']):
            raise ValueError('complete original-source case differs: '+mode)
        item = encoding_check([mode+'_complete_source.json'], 131072)
        physical, constructor = report['physical'], report['constructor']
        if mode == 'noop':
            if (report['proof'] is not None or not physical['exact_original_object_retained']
                or not constructor['literal_noop'] or constructor['kernel_transform_prepaid']
                or report['route']['selected_channels'] or report['binding']['actual_direct_output_nnz'] != 5120
                or not item['passed']):
                raise ValueError('complete no-hit source differs')
            continue
        selected, direct, byte_delta, entry_delta, aux, nnz = expected_cases[mode]
        if (report['route']['selected_channels'] != selected
            or report['binding']['actual_direct_output_nnz'] != direct
            or physical['numeric_byte_delta'] != byte_delta or physical['numeric_entry_delta'] != entry_delta
            or not physical['complete_physical_reduction_proved'] or physical['rejection_reasons']
            or not physical['actual_complete_inverse_equal'] or physical['route_mask_bytes_and_entries'] != 16
            or constructor['new_factors'] != aux or constructor['nnz'] != nnz
            or report['proof']['all_native_nnz_proved'] != nnz
            or report['proof']['all_transformed_kernel_coefficients_rebuilt'] != 18432
            or report['proof']['all_original_output_equations_proved'] != 512
            or item['passed']):
            raise ValueError('numerical reduction or expected reporting failure differs: '+mode)
        if not encoding_check([mode+'_constructed.json'], 65536)['passed']:
            raise ValueError('unexpected partial constructor reporting overflow')
        cases.append(dict(mode=mode, selected_channels=selected, original_nnz=direct,
            native_packet_nnz=nnz, new_factors=aux, complete_numeric_byte_delta=byte_delta,
            complete_numeric_entry_delta=entry_delta, exact_original_inverse=True))
    combined = encoding_check(['complete_held_ledger.json', 'result.json'], 524288)
    if combined['passed'] or len([r for r in encoding if not r['passed']]) != 4:
        raise ValueError('complete reporting failure population differs')

    expected = {}

    def bind(path, digest):
        name = str(path.resolve())
        if name in expected and expected[name] != digest:
            raise ValueError('inherited identity conflict: '+name)
        expected[name] = digest

    for name, digest in freeze['source_sha256'].items():
        bind(EXP / name, digest)
    for name, digest in exited['artifacts'].items():
        bind(RUN / name, digest)
    bind(RUN / 'exit.json', _sha256(RUN / 'exit.json'))
    previous = EXP / 'CHECKPOINT_C123_SOURCE_BINDING_20260927_SHA256SUMS'
    if _sha256(previous) != '06da4cfed3244aa3df3a12c99a77a1b5ddf105ce08847a8be6670ab17b5746ed':
        raise ValueError('C123 archive seal drift')
    for line in previous.read_text().splitlines():
        digest, name = line.split('  ', 1)
        bind(ROOT / name, digest)
    mismatches = [n for n, h in expected.items() if _sha256(Path(n)) != h]
    input_mismatches = [n for n, h in freeze['input_sha256'].items() if _sha256(Path(n)) != h]
    provenance = _provenance(ROOT)
    record = dict(completed=not mismatches and not input_mismatches and provenance == freeze['provenance'],
        completion_means_archive_integrity_only=True,
        checked_unique_files=len(expected), checked_file_bytes=sum(Path(n).stat().st_size for n in expected),
        mismatches=mismatches, input_mismatches=input_mismatches, provenance=provenance,
        provenance_unchanged=provenance == freeze['provenance'],
        complete_qualification_passed=False, candidate_version_closed=True,
        reason='actual_indented_case_and_aggregate_JSON_exceed_fixed_prepaid_reservations',
        raw_wrapper_green_flags_preserved_not_rewritten=True,
        this_terminal_audit_overrides_raw_complete_qualification_claim=True,
        mathematical_and_full_ordinary_physical_evidence_positive=True,
        complete_source_cases=cases, complete_nohit_literal_source_preserved=True,
        all_original_output_rows=2048, all_actual_original_coefficients=191104,
        complete_native_rows_proved=6576, complete_native_coefficients_proved=107856,
        serialization_checks=encoding,
        global_or_category_headroom_not_retroactive_reporting_payment=True,
        recorded_run_counter=result['work'], recorded_counter_plus_archive_reporting=pool.used,
        recorded_counter_is_NOT_a_complete_prepaid_work_certificate=True,
        reporting_counter_parts=pool.parts, numerical_proofs_rerun=False,
        archived_numeric_arrays_reanalysed=False, target_source_or_LIVE_admitted=False,
        solver_calls=0, formal_gain=0, speedup_claimed=False,
        hash_traffic_separate_from_generation_tokens=True,
        wall_s=time.monotonic()-started, exit_sha256=_sha256(RUN / 'exit.json'))
    if len((json.dumps(record, sort_keys=True, indent=2, allow_nan=False)+'\n').encode())+1024 > 65536:
        raise MemoryError('this terminal report exceeds its own exact writer-format reserve')
    _atomic_exclusive_json(output, record)
    if not record['completed']:
        raise ValueError('terminal archive integrity failed')
    names = [EXP / n for n in ('C124_MIXED_SOURCE_PREREG_20260927.md',
        'C124_MIXED_SOURCE_AUDIT_20260927.md', 'C124_KERNEL_PROOF_HANDOFF_20260927.md',
        'CHECKPOINT_C124_MIXED_SOURCE_20260927.md', 'c124_mixed_f4_v1.py',
        'c124_mixed_f4_oracle_v1.py', 'c124_complete_mixed_source_v1.py',
        'test_c124_mixed_f4_v1.py', 'c124_mixed_source_worker_v1.py',
        'run_c124_mixed_source_supervisor_v1.py', 'run_c124_archive_audit_v1.py')]
    names += [output, *sorted(p for p in RUN.iterdir() if p.is_file())]
    with seal.open('x') as stream:
        for path in sorted(names):
            stream.write(_sha256(path)+'  '+str(path.relative_to(ROOT))+'\n')
    print(json.dumps(dict(**record, sealed_files=len(names), seal_sha256=_sha256(seal))), flush=True)


if __name__ == '__main__':
    main()
