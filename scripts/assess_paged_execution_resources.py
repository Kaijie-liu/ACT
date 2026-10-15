"""Post-implementation resource reconciliation, not an execution/admission test.

Read frozen reports and bound source files; evaluate small integer expressions and
schema lengths only. No source/forest/candidate, worker, solver or GPU is built.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from scripts.assess_paged_result_implementation import assess
from scoped_source.direct_lifecycle_package import code_files
from scoped_source.direct_method import method

ROOT = Path(__file__).resolve().parents[1]
BASE = 'f87e4c7ef958035a89cd87aecb50fc8885e82206'
CONTROLS = 'docs/hz_paged_result_controls_20261015.json'
ARCHIVE = 'docs/hz_shared_archive_contract_20261015.json'
WORK = 'docs/hz_shared_execution_envelope_20261015.json'
FLOOR = 'docs/hz_shared_source_decision_20261015.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(name):
    return json.loads((ROOT / name).read_bytes())


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def check_bindings(bindings):
    require(type(bindings) is dict and len(bindings) == 50, 'frozen implementation roster')
    for name, expected in bindings.items():
        require(sha(name) == expected, 'frozen implementation drift: ' + name)


def check_parent_links(archive):
    # capture.py defines the conditional source manifest/trust layout, but is
    # deliberately not part of the small-source worker's 50 implementation files.
    names = (WORK, FLOOR, 'scoped_source/capture.py')
    for name in names:
        require(archive['anchors'][name] == sha(name), 'inherited layout/account drift: ' + name)
    return {name:sha(name) for name in names}


def counted_delta_controls():
    # These syntactic controls support the reviewed delta; they are NOT a proof
    # of arbitrary Python cost or of the inherited mathematical loop formulas.
    for name in ('result_stream', 'check_result_stream'):
        tree = ast.parse((ROOT / ('scoped_source/' + name + '.py')).read_text())
        charged = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Attribute) and n.func.attr in ('visit', 'exact')]
        require(not charged, 'new result path changes counted Work')
    tree = ast.parse((ROOT / 'scoped_source/scaled_endpoints.py').read_text())
    session = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Session')
    finish = next(n for n in session.body if isinstance(n, ast.FunctionDef) and n.name == 'finish')
    calls = [n for n in ast.walk(finish) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'visit']
    require([ast.unparse(n) for n in calls] == ["work.visit('row_visits', 9)",
            "work.visit('row_visits')", "work.visit('row_visits')"], 'retained finish Work charges')


def analyze():
    controls, archive, work, floor = map(load, (CONTROLS, ARCHIVE, WORK, FLOOR))
    check_bindings(controls['bindings'])
    inherited_bindings = check_parent_links(archive)
    require(controls['status'] == 'PASS' and controls['tests'] == 171
            and controls['errors'] == controls['failures'] == 0, 'bound completed controls')
    layout = assess()
    require(layout == controls['success_layout'], 'implemented successful schema drift')
    counted_delta_controls()
    files = code_files(method('pair_scoped', 1, shared=True, paged_results=True))
    sizes = {name:(ROOT / name).stat().st_size for name in files}
    require(len(sizes) == 28 and all(0 < size <= 65536 for size in sizes.values()), 'code member cap')
    code_upper = len(sizes) * 65536
    totals = {}
    for profile, row in archive['conditional_success_archive'].items():
        total = row['source_and_forest'] + row['candidate_stream'] + code_upper + layout['metadata_upper']
        totals[profile] = dict(source_and_forest_upper=row['source_and_forest'],
            candidate_stream_upper=row['candidate_stream'], checker_code_upper=code_upper,
            success_metadata_upper=layout['metadata_upper'], request_upper=total,
            archive_cap=2**27, whole_upper_fits_archive_cap=total <= 2**27,
            candidate_stream_cap=floor['candidate_layout']['stream_cap'],
            candidate_upper_fits_stream_cap=row['candidate_stream'] <= floor['candidate_layout']['stream_cap'],
            upper_excess_proves_actual_overflow=False, actual_bytes=None)
    per_pass = layout['members_upper']['result_pages'] + layout['index_upper']
    mandatory = floor['membership']['mandatory']
    source_floor = floor['semantic_floor_min_endpoint_roster']['source_coefficient_floor']
    require(source_floor == 2 * (floor['inventory']['first_weight_positions']
            + floor['inventory']['later_weight_positions']), 'two independent affine checks')
    require(archive['unchanged'] == work['unchanged'], 'unchanged resource contract')
    return dict(schema='PAGED_IMPLEMENTED_RESOURCE_RECONCILIATION_V1', based_on=BASE,
        anchors={p:sha(p) for p in (CONTROLS, ARCHIVE, WORK, FLOOR,
            'scripts/assess_paged_execution_resources.py', 'scripts/test_paged_execution_resources.py')},
        frozen_implementation_bindings_verified=len(controls['bindings']),
        inherited_layout_and_account_bindings=inherited_bindings,
        inherited_conditions=archive['source_layout']['conditions'] + layout['hypotheses'],
        success_layout=layout, conditional_success_archive=totals,
        code=dict(files=sizes, actual_current_bytes=sum(sizes.values()), payload_upper=code_upper),
        result_file_reads=dict(child_passes=5, parent_passes=5, online_passes=10,
            pages_and_index_bytes_per_pass_upper=per_pass, online_logical_bytes_upper=10*per_pass,
            offline_audit_passes_separate=4, physical_device_io=False,
            includes_code_other_metadata_imports_or_native_capture=False),
        counted_work=dict(new_result_modules_increment=0, total_machine_cost_increment_is_zero=False,
            source_subpath_upper=work['source_counted_work']['build_and_two_checks_upper'],
            postcapture_upper={arm:v['postcapture_counted_work_upper'] for arm,v in work['consumers'].items()},
            added_row_parse_and_sign_checks_upper=3*252,
            prior_formulas_reviewed_not_machine_proved=True, actual_full_work=None),
        refusals=dict(source_two_check_lower=source_floor, source_work_cap=work['unchanged']['work'],
            source_lower_exceeds_cap=source_floor > work['unchanged']['work'],
            mandatory_expanded_memberships=mandatory, memberships_cap=work['unchanged']['memberships'],
            membership_lower_exceeds_cap=mandatory > work['unchanged']['memberships']),
        lifetime=dict(writer_persistent_row_limit=8, page_reference_limit=32,
            private_roster_duties_limit=252, private_roster_slots_limit=504,
            endpoint_values_and_receipts_still_retained=True,
            reader_old_and_new_decoded_pages_can_overlap=True,
            reader_json_and_canonical_encoding_temporaries=True,
            replay_last_candidate_frame_survives_into_comparison=True,
            production_packet_released_before_consumer=False,
            parsed_and_encoded_storage_is_not_total_heap=True,
            full_heap_upper=None, full_rss_upper=None,
            parent_reception_in_worker_rss_sample=False),
        bounded_failures=dict(error_str_or_repr_called=False,
            incomplete_terminal_cost_complete=False, arbitrary_failure_heap_bound=None),
        supervision=dict(timely_acceptance_seconds=300,
            parent_finalization_hard_preemption=False, whole_process_return_upper_seconds=None,
            late_return_cannot_accept=True, offline_audit_can_rescue_timeout=False),
        unchanged=work['unchanged'], runtime_changed=False, mathematical_checks_removed=0,
        source_builds=0, candidate_builds=0, worker_runs=0, model_runs=0, gpu_runs=0,
        solver_calls=0, new_quota=None, full_capacity_admitted=False,
        decision='PAGED_RESULT_COMPONENT_CLOSED_FULL_ADMISSION_STILL_REFUSED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report', type=Path)
    args = parser.parse_args()
    result = analyze()
    if args.check_report:
        require(result == json.loads(args.check_report.read_bytes()), 'resource report drift')
        print('PASS: implemented static resource reconciliation; no request executed')
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
