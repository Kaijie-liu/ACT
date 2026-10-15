"""Static encoded-size contract for a proposed result stream, not a writer/checker.

Only symbolic schema lengths and existing source hashes are evaluated. No source,
candidate, forest, worker, solver or GPU is constructed or imported.
"""
import argparse
import json
from pathlib import Path

from scripts import assess_shared_archive_contract as old

BASE = '3aad7f87f61497cc61905483d9b2c953de6f279b'
PARENT = 'docs/hz_shared_archive_contract_20261015.json'
H = old.H
Size, length, array = old.Size, old.length, old.array


def schemas(*, duties=252, tiles=84, added_code_files=2, added_implementation_files=3):
    """Closed proposal; field lengths are symbolic, never expanded row arrays."""
    old.prior.require(type(duties) is int and 1 <= duties <= 252, 'registered duties')
    old.prior.require(type(tiles) is int and 1 <= tiles <= 84, 'registered tiles')
    old.prior.require(type(added_code_files) is int and 0 <= added_code_files <= 2,
                      'at most two proposed packaged modules')
    old.prior.require(type(added_implementation_files) is int
                      and added_code_files <= added_implementation_files <= 3,
                      'packaged modules plus at most one new protocol binding')
    S = lambda n: Size(n + 2)
    I, F, B = Size(20), Size(24), Size(5)
    work = dict.fromkeys(old.previous.COUNTERS, Size(12))
    consumer = dict(kind='pair_scoped', check_width=2)
    method = dict(schema='RESULT_STREAM_LIFECYCLE_METHOD_V1', protocol=H,
        producer=dict(kind='FIXED_SCALAR_STUB_V1', profile=1), candidate_width=8,
        production=consumer, replay=consumer, result_format='SHARED_RESULT_STREAM_V1')
    ref = dict(file=S(200), bytes=I, sha256=H)
    result_ref = dict(file='result/index.json', bytes=I, sha256=H)
    summary = dict(status='CHECKED_POSITIVE_CAPTURED_REAL_SOURCE', required=252, positive=252,
        required_endpoints=504, checked_endpoints=504, missing_endpoints=504,
        native_float_proof=False, checkpoint_correspondence_checked=False)
    # All result values retain their original 8192-bit rational bound.
    fraction_chars = old.previous.metadata(8192, duties, tiles, 12)['fraction_text_upper']
    row = dict(pair=[7, 7], other=9, lower=S(fraction_chars), positive=False)
    context = dict(schema='SHARED_RESULT_CONTEXT_V1', source=H, forest=H,
        forest_index=H, contract=H, method=H, invocation=H, roster=H)
    def page(count):
        return dict(schema='SHARED_RESULT_PAGE_V1', context=H, ordinal=31, start=251,
                    rows=array(count, row))
    count = (duties + 7) // 8
    full, tail = divmod(duties, 8)
    page_bytes = full*length(page(8)) + (length(page(tail)) if tail else 0)
    page_ref = dict(file='result/0031.json', bytes=65536, sha256=H, start=251, count=8)
    index = dict(schema='SHARED_RESULT_STREAM_V1', context=context, candidates=H,
                 pages=array(count, page_ref), summary=summary)
    # Compared with the old layout: result.json outside the package disappears;
    # count pages + one index are added INSIDE the package. Code is bounded anew.
    ns, forest_pages, code = 96, 107, 26 + added_code_files
    artifacts = ns + forest_pages + tiles + code + 6 + count
    members = artifacts - 2  # only bundle and outer inner-terminal excluded
    phases = array(18 + 2*tiles, dict(name=S(22), complete=B, seconds=F, work=work))
    progress = (32 + 4*tiles)*(length(dict(name=S(22), event=S(5), work=work, at=F)) + 1)
    source_stats = dict.fromkeys(('nodes', 'local_nnz', 'references', 'factors', 'views',
        'forest_bytes', 'unique_roots', 'memberships', 'stored_memberships', 'sequence_count',
        'physical_bytes', 'packed_source_bytes', 'alias_rows', 'explicit_locals', 'aliased_locals'), I)
    transfer = dict(bytes=I, members=I, seconds=F, copies=1, source=H)
    owner = dict.fromkeys(('live_bytes', 'peak_bytes', 'allocated_bytes', 'copied_bytes',
                          'released_bytes', 'allocations'), I)
    owner['cleanup_seconds'] = F
    cost = dict.fromkeys(('init_seconds', 'prepare_seconds', 'encode_seconds', 'receive_seconds',
        'publish_seconds', 'check_seconds', 'terminal_seconds', 'release_seconds'), F)
    cost.update(publication_attempted_bytes=I, acknowledged_bytes=I)
    snapshots = dict(source_transfer=transfer, source_stats=source_stats, consumer=consumer,
        lifecycle=dict(setup_calls=1, propose_calls=Size(2), close_calls=1, root_terms=I),
        owner=owner, driver_cost=cost)
    event = dict(file=S(200), bytes=I, seconds=F, bytes_fully_written=B, timely_published=B, complete=B)
    events = Size(length(array(artifacts, event)) + ns*(length(dict(ownership_transfer=True)) - 2 + 1))
    ledger = dict(logical_bytes=I, limit=I, reserved_terminal=I, written=I, attempted=I,
                  events=events, elapsed_seconds=F, elapsed_after_deadline=F)
    contract = dict(schema='SCALED_FOREST_ENDPOINT_CONTRACT_V3', forest=H, source=H, request=H,
        factors=H, classes=10, label=9, margin=S(4096), gate='SELECTED_SOFTMAX',
        top_k=2, tie_policy='ANY_LEGAL_TOPK')
    manifest = dict(schema='RESULT_STREAM_LIFECYCLE_MANIFEST_V1',
        source=dict(directory='source', sha256=H), forest=ref, contract=ref,
        tiles=array(tiles, ref), method=method, invocation=H, code=array(code, ref), result=result_ref)
    bundle = dict(schema='RESULT_STREAM_LIFECYCLE_BUNDLE_V1', manifest='manifest.json',
                  members=array(members, ref))
    replay = dict(summary=summary, result_ref=result_ref, owner=owner, driver_cost=cost,
        candidate_bytes=I, consumer=consumer, package_sha256=H, package_bytes=I,
        method=method, invocation=H, work=work, seconds=F)
    inner = dict(status='PENDING_RECEPTION', method=method, invocation=H, source=H,
                 result=result_ref, package=ref, work=work)
    construction = dict(native_seconds=F, native_coordinates=I, packed_source_bytes=I, forest_bytes=I, pages=I)
    candidate = dict(schema='RESULT_STREAM_CANDIDATE_V1', status='COMPLETE', summary=summary,
        result_ref=result_ref, terminal_ref=ref, package=ref, replay=replay,
        construction=construction, method=method, invocation=H, source=H,
        phases=phases, snapshots=snapshots, work=work, initial_work=work, deadline=F,
        seconds=F, archive=ledger, isolated=True, site_disabled=True,
        fixture_seconds=F, numerical_imports=[])
    # Implementation path/hash map is not an inventory of result pages.
    impls = 44 + added_implementation_files
    impl = Size(2 + impls*(200 + 3 + length(H)) + impls - 1)
    spec = dict(schema='RESULT_STREAM_SUPERVISED_STUB_V1', case='guarded', method=method, fault='')
    invocation = dict(schema='RESULT_STREAM_INVOCATION_V1', id=H, spec=spec, start=F,
        deadline=F, run_deadline=F, cleanup_deadline=F, initial_work=work, implementations=impl)
    child = dict(schema='OWNED_BOUNDED_CPU_V1', status='COMPLETED', error=None, pid=I,
        returncode=0, seconds=F, execution_seconds=F, cleanup_seconds=F,
        cleanup_status='LEADER_REAPED_NO_LIVE_GROUP', remaining_group=dict(live=[], zombies=[], rss=0),
        descendant_on_leader_exit=B, exit_observed_at=F, sampled_peak_rss=I,
        run_deadline=F, cleanup_deadline=F, escaped_descendants_or_driver_cleanup=B)
    accepted = dict(summary=summary, result_ref=result_ref, package_sha256=H, source=H,
        method=method, invocation=H, package_bytes=I, worker_work=work,
        snapshots=snapshots, phases=phases, worker_seconds=F, fixture_seconds=F)
    receipt = dict(schema='RESULT_STREAM_RECEIPT_V1', status='COMPLETE', invocation=H,
        invocation_sha256=H, method=method, candidate_sha256=H, accepted=accepted,
        error=None, child=child, work=work, work_complete=True,
        seconds_before_publication=F, budget=F)
    return dict(row=row, context=context, page=page(8), index=index, summary=summary,
        contract=contract, manifest=manifest, bundle=bundle, inner_terminal=inner,
        candidate_report=candidate, invocation=invocation, parent_receipt=receipt,
        pages=count, page_bytes=page_bytes, progress=progress,
        artifact_files=artifacts, whole_request_files=artifacts+5, bundle_members=members,
        code_files=code, implementation_files=impls, fraction_chars=fraction_chars)


def analyze():
    previous = old.prior.load(PARENT)
    old.prior.require(previous == old.analyze(), 'bound archive contract drift')
    s = schemas()
    names = ('contract', 'manifest', 'bundle', 'inner_terminal', 'candidate_report',
             'invocation', 'parent_receipt')
    sizes = {name:length(s[name]) for name in names}
    sizes.update(result_pages=s['page_bytes'], result_index=length(s['index']),
                 progress=s['progress'], worker_log=0)
    demand = sum(sizes.values())
    code_upper = s['code_files'] * 65536
    archives = {name:dict(metadata_upper=demand, checker_code_upper=code_upper,
        request_upper=row['source_and_forest'] + row['candidate_stream'] + code_upper + demand)
        for name, row in previous['conditional_success_archive'].items()}
    anchors = {p:old.prior.sha(p) for p in (PARENT,
        'docs/hz_next_runtime_decision_20261015.md', 'scripts/assess_paged_result_contract.py',
        'scripts/test_paged_result_contract.py')}
    return dict(schema='PAGED_RESULT_STATIC_CONTRACT_V1', based_on=BASE, anchors=anchors,
        schema_sizes=dict(row_upper=length(s['row']), context_upper=length(s['context']),
            full_page_upper=length(s['page']), summary_upper=length(s['summary']),
            fraction_text_chars=s['fraction_chars']),
        members_upper=sizes, metadata_upper=demand,
        files={k:s[k] for k in ('pages', 'artifact_files', 'whole_request_files', 'bundle_members',
                               'code_files', 'implementation_files')},
        success_archive=archives,
        capacity=dict(page_bytes=65536, report_bytes=2**20,
            all_success_reports_fit=all(sizes[k]<=2**20 for k in names),
            page_fits=length(s['page'])<=65536, index_fits=length(s['index'])<=2**20),
        result_reads=dict(complete_online_page_passes=10,
                         logical_page_bytes=10*s['page_bytes'], mathematical_replays=1),
        conditions=['proposed schema, not emitted runtime; all rows and both candidate roles retained',
            'old captured-source profile and no-fault fixed-stub scope',
            'at most 28 packaged checker files, each obeying existing 64 KiB cap; actual new code size unknown',
            'at most 47 implementation bindings: old 44, two packaged modules, one new protocol; no other new execution files',
            'old upper bounds on phase count, names, progress and nonnegative counters preserved',
            'fixed JSON-safe relative paths using A-Z/a-z/0-9/_/./- and length <=200',
            'twelve Work-counter digits are representation, not a raised quota'],
        open_contract=dict(new_implementation_work_upper=None, full_heap_upper=None, full_rss_upper=None,
            failure_generation_upper=None, native_capture_upper=None, completion_seconds_upper=None),
        unchanged=previous['unchanged'], mathematical_checks_removed=0,
        full_capacity_admitted=False, runtime_changed=False, new_quota=None,
        source_builds=0, candidate_builds=0, model_runs=0, gpu_runs=0, solver_calls=0,
        decision='PROPOSED_SUCCESS_METADATA_FITS_PER_FILE_CAPS_NOT_FULL_EXECUTION_ADMISSION')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-report', type=Path)
    args = p.parse_args()
    result = analyze()
    if args.check_report:
        old.prior.require(result == json.loads(args.check_report.read_bytes()), 'paged-result report drift')
        print('PASS: static proposed metadata; no runtime execution or resource change')
    else:
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
