"""Static approval proposal and archived-reference checks, never a runtime driver.

No setrlimit, mmap, subprocess, source construction or mathematical solver calls.
"""
import hashlib
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = 'docs/hz_staged_memory_approval_20261015.json'
PARENT = 'docs/hz_staged_memory_protocol_20261015.json'
OLD = 'docs/hz_paged_result_controls_20261015.json'
AS_BYTES = 256 * 2**20
U64 = 2**64 - 1
ANCHORS = (PARENT, 'docs/hz_staged_memory_protocol_20261015.md', OLD,
           'scoped_source/direct_method.py', 'scoped_source/direct_lifecycle.py',
           'scoped_source/result_stream.py',
           'scripts/test_paged_results.py', 'scripts/assess_staged_memory_approval.py')
# The eight contiguous same-candidate differential controls in the frozen r3 log.
REFERENCE_IDS = (
    '10907779523618713', '10907780150337977', '10907780827554813', '10907781431397528',
    '10907782114224938', '10907782810573507', '10907783531084611', '10907784306159668')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def pairs(items):
    result = {}
    for k, v in items:
        require(k not in result, 'duplicate JSON field')
        result[k] = v
    return result


def load(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs)


def reference_rows(observation):
    root = Path(observation['root']) / 'artifacts/package'
    require(root.is_relative_to('/data1/Kane/MOE/runs/hz_paged_result_controls_20261015_r3'),
            'frozen r3 reference only')
    require(all(not p.is_symlink() for p in (root, *root.parents)), 'reference path alias')

    def member(name, expected, size=None, cap=2**20):
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'reference member path')
        p = root / rel
        require(all(not q.is_symlink() for q in (p, *p.parents)), 'reference member alias')
        require(p.is_file() and p.stat().st_size <= cap, 'bounded reference member')
        raw = p.read_bytes()
        require(sha(raw) == expected and (size is None or len(raw) == size), 'reference hash/length')
        return json.loads(raw, object_pairs_hook=pairs)

    accepted = observation['accepted']
    bundle = member('bundle.json', accepted['package_sha256'])
    entries = {x['file']: x for x in bundle['members']}
    require(len(entries) == len(bundle['members']), 'duplicate reference member')
    manifest_ref = entries['manifest.json']
    manifest = member('manifest.json', manifest_ref['sha256'], manifest_ref['bytes'])
    ref = manifest['result']
    require(ref == {k: entries[ref['file']][k] for k in ('file', 'bytes', 'sha256')}, 'index inventory')
    index = member(ref['file'], ref['sha256'], ref['bytes'])
    require(index['summary'] == accepted['result'], 'reference aggregate')
    require(index['summary']['native_float_proof'] is False and
            index['summary']['checkpoint_correspondence_checked'] is False,
            'no upgrade of reference guarantee')
    rows = []
    for ordinal, page in enumerate(index['pages']):
        require({k: page[k] for k in ('file', 'bytes', 'sha256')} == entries[page['file']], 'page inventory')
        data = member(page['file'], page['sha256'], page['bytes'], cap=65536)
        require(data['ordinal'] == ordinal and data['start'] == len(rows)
                and page['start'] == len(rows) and len(data['rows']) == page['count'], 'ordered reference pages')
        rows.extend(data['rows'])
    require(len(rows) == index['summary']['required'], 'reference duty count')
    logical = dict(summary=index['summary'], rows=rows)
    return dict(request_root=observation['root'], bundle_sha256=accepted['package_sha256'],
                result_index_sha256=ref['sha256'], source_sha256=accepted['source'],
                logical_result_sha256=sha(compact(logical)), required=len(rows),
                result_status=index['summary']['status'])


def layout():
    # Proposed flat ASCII JSON control record. All values have fixed widths.
    # This sizes the schema, not a production encoder/decoder implementation.
    widths = dict(schema=32, role=16, state=32, result_status=64, error_code=32,
                  case=8, arm=16, profile=1)
    hashes = ('invocation', 'approval', 'implementation', 'method', 'source', 'forest',
              'contract', 'candidates', 'result_index', 'bundle', 'predecessor',
              'prefix_names', 'stage_chain')
    widths.update({k + '_sha256': 64 for k in hashes})
    numbers = ('start_ns', 'deadline_ns', 'run_deadline_ns', 'cleanup_deadline_ns',
               'observed_exit_ns', 'as_bytes', 'pid', 'pgid', 'required', 'positive',
               'required_endpoints', 'checked_endpoints', 'missing_endpoints',
               'row_visits', 'coefficient_visits', 'arithmetic', 'prefix_bytes',
               'prefix_files', 'retained_bytes', 'retained_files', 'bytes_read',
               'bytes_written', 'bytes_hashed', 'sampled_peak_rss', 'observed_vmsize',
               'elapsed_ns', 'cpu_user_ns', 'cpu_system_ns')
    widths.update({k: 20 for k in numbers})
    for stage in ('production', 'portable', 'receive'):
        widths.update({stage + '_' + k: 20 for k in
                       ('start_ns', 'end_ns', 'cleanup_ns', 'row_visits', 'coefficient_visits', 'arithmetic')})
    for i in range(6):
        widths['suffix_' + str(i) + '_bytes'] = 20
        widths['suffix_' + str(i) + '_sha256'] = 64
    widths.update({k: 1 for k in ('proof_complete', 'work_complete', 'storage_complete',
                                 'cleanup_complete', 'limit_active', 'numerical_imports_absent')})
    physical = len(compact({k: '0' * n for k, n in widths.items()}))
    require(physical == 2 + sum(len(k) + n + 5 for k, n in widths.items()) + len(widths) - 1,
            'independent JSON length arithmetic')
    require(physical < 65536, 'handoff cap')
    return dict(schema='STAGED_MEMORY_CONTROL_V1', field_widths=widths,
                record_bytes=physical, field_count=len(widths), numeric_max=U64,
                log_line_bytes=256, max_log_lines=256, log_bytes=65536,
                six_suffix_reserve=6 * 65536,
                successful_suffix_upper=3 * physical + 3 * 65536,
                online_caller_file=None, runtime_codec_implemented=False)


def assess():
    parent, old = load(ROOT / PARENT), load(ROOT / OLD)
    require(parent['as_bytes'] is None and not parent['runtime_controls_authorized'], 'parent remains unapproved')
    require(old['status'] == 'PASS' and old['tests'] == 171, 'registered reference suite')
    complete = [x for x in old['observations'] if x.get('status') == 'COMPLETE']
    baseline = []
    for i, ((case, arm, profile), rid) in enumerate(zip(
            itertools.product(('half', 'guarded'), ('dense', 'pair_scoped'), (0, 1)), REFERENCE_IDS), 1):
        records = [x for x in complete if Path(x['root']).name == 'request-' + rid]
        require(len(records) == 1, 'one explicitly bound reference')
        r = records[0]
        require((r['case'], r['arm'], r['profile'], r['fault'], r['budget']) ==
                (case, arm, profile, '', 300), 'reference request identity')
        baseline.append(dict(id=f'B{i:02}', case=case, arm=arm, profile=profile, seconds=300,
                             expected='COMPLETE_EXACT_ARCHIVED_DIFFERENTIAL', reference=reference_rows(r)))
    faults = []
    descriptions = (
        ('rehashed_false_lower_dense', 'guarded', 'dense', 300, 'ERROR_AT_RECOMPUTED_ROW'),
        ('rehashed_false_lower_scoped', 'guarded', 'pair_scoped', 300, 'ERROR_AT_RECOMPUTED_ROW'),
        ('complete_missing_candidates', 'guarded', 'pair_scoped', 300, 'COMPLETE_UNKNOWN_MISSING_EVIDENCE'),
        ('exception_after_first_page', 'guarded', 'pair_scoped', 300, 'ERROR_PREFIX_RETAINED'),
        ('block_construction', 'guarded', 'pair_scoped', 10, 'TIMEOUT_AFTER_TARGET_ENTERED'),
        ('block_production_endpoint', 'guarded', 'pair_scoped', 10, 'TIMEOUT_AFTER_TARGET_ENTERED'),
        ('block_portable', 'guarded', 'pair_scoped', 10, 'TIMEOUT_AFTER_TARGET_ENTERED'),
        ('block_receive', 'guarded', 'pair_scoped', 10, 'TIMEOUT_AFTER_TARGET_ENTERED'),
        ('block_final_publication', 'half', 'pair_scoped', 10, 'TIMEOUT_AFTER_TARGET_ENTERED'),
        ('late_caller_handoff', 'half', 'pair_scoped', 10, 'TIMEOUT_NO_FILE_ONLY_ACCEPT'),
        ('watchdog_abnormal_after_terminal', 'half', 'pair_scoped', 300, 'INCOMPLETE_TERMINAL'),
        ('receive_injected_memory_error', 'guarded', 'pair_scoped', 300, 'ERROR_NOT_OS_ENFORCEMENT_PROOF'))
    for i, (fault, case, arm, seconds, expected) in enumerate(descriptions, 1):
        faults.append(dict(id=f'F{i:02}', fault=fault, case=case, arm=arm, profile=0,
                           seconds=seconds, expected=expected))
    components = (
        ('default_off_and_missing_approval', ['disabled', 'missing', 'wrong_hash']),
        ('limit_and_role_binding', ['different_as', 'wrong_role', 'wrong_predecessor']),
        ('source_method_property_binding', ['source', 'method', 'property']),
        ('preimport_external_code_binding', ['changed_code_rehashed_bundle', 'old_path_import']),
        ('roster_completeness', ['missing', 'duplicate', 'reordered', 'extra']),
        ('cumulative_cost_binding', ['work_reset', 'deadline_reset', 'unknown_as_zero']),
        ('prefix_suffix_closure', ['extra_path', 'changed_length', 'symlink', 'open_log']),
        ('fixed_record_and_log_limits', ['oversize', 'unknown_field', 'wrong_width', 'line_257']),
        ('relocation_errors', ['destination_exists', 'exdev']),
        ('supervision_observation', ['unexpected_member', 'observe_error', 'reap_failure']),
        ('bounded_errors', ['str_raises', 'repr_raises']))
    return dict(schema='STAGED_MEMORY_APPROVAL_PROPOSAL_V1', status='AWAITING_EXPLICIT_APPROVAL',
                base_commit='648378f8482709fba5a329d659d98999952cf09e',
                approval_granted=False, implementation_authorized=False, runtime_controls_authorized=False,
                proposed_as_bytes=AS_BYTES, active_as_bytes=None, automatic_fallback_as_bytes=None,
                conditional_post_arm_virtual_bytes=3 * AS_BYTES, existing_limits_changed=False,
                full_capacity_admitted=False, success_seconds=300,
                arm_label_mapping={'dense_reference': 'dense', 'pair_scoped': 'pair_scoped'},
                anchors={n: sha((ROOT / n).read_bytes()) for n in ANCHORS},
                archived_observation=dict(complete=len(complete),
                    sampled_rss_min=min(x['child']['sampled_peak_rss'] for x in complete),
                    sampled_rss_max=max(x['child']['sampled_peak_rss'] for x in complete),
                    owned_arena_peak=max(x['accepted']['snapshots']['owner']['peak_bytes'] for x in complete),
                    retained_bytes_max=max(x['storage']['bytes'] for x in complete),
                    work_max=max(sum(x['work'].values()) for x in complete),
                    vmsize_recorded=False, proves_proposed_as_sufficient=False),
                layout=layout(), baseline=baseline, full_chain_faults=faults,
                components=[dict(id=f'C{i:02}', family=n, variants=v) for i, (n, v) in enumerate(components, 1)],
                component_driver=dict(seconds=300, actor_runs=1, real_children_for_fake_faults=False),
                os_probe=dict(id='M01', seconds=300, actor_runs=1,
                    lengths=['page_size', AS_BYTES], flags='MAP_PRIVATE|MAP_ANONYMOUS', prot=0,
                    touch_pages=False, sweep=False, expected=['MAPPED_AND_CLOSED', 'OSERROR_ENOMEM']),
                counts=dict(full_chain_requests=20, component_actor_runs=1, os_probe_actor_runs=1,
                    component_variants=sum(len(v) for _, v in components),
                    new_reference_solver_calls=0, new_requests_executed=0, gpu_runs=0, model_runs=0))


if __name__ == '__main__':
    report = assess()
    if sys.argv[1:] == ['--check']:
        require(load(ROOT / OUTPUT) == report, 'approval record drift')
        print(json.dumps(dict(status='PASS_STATIC_APPROVAL_PROPOSAL', references=8,
                             record_bytes=report['layout']['record_bytes'],
                             control_requests_executed=0, limits_changed=False)))
    else:
        require(not sys.argv[1:], 'only --check supported')
        print(json.dumps(report, indent=2, sort_keys=True))
