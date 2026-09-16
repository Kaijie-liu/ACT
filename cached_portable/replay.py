"""One frozen offline input114 integration replay; not a verification rerun."""
import json
from pathlib import Path
import time

from portable_proof.runtime import digest
from cached_portable.execution import supervise, audit_outer, read, save_new
from cached_portable.controls import hashes

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'docs/general_evidence_execution_v1_results.json'
PARENT_SHA = '67cf5602723221df1b6ac1d10a04363de8b71bd385722c8a87f272353b9a21fe'
OUT = ROOT/'data/moe/results/cached_portable_saved114_20260916_v2'


def run():
    from scripts.optional_evidence_dev_contract import git
    from evidence_cohort.run import resource, resource_ok
    controls_path = sorted((ROOT/'docs').glob('cached_portable_v2_controls_attempt*.json'))[-1]
    control = read(controls_path)
    if control['status'] != 'PASS' or control['sources'] != hashes(): raise ValueError('passing source-bound controls required')
    if git('branch', '--show-current') != 'feat/moe-route-verification' or git('status', '--porcelain'):
        raise ValueError('clean feature branch required')
    head = git('rev-parse', 'HEAD')
    if git('ls-remote', 'origin', 'refs/heads/feat/moe-route-verification').split()[0] != head:
        raise ValueError('publish protocol and implementation first')
    resources = resource()
    if not resource_ok(resources): raise RuntimeError('resource gate unavailable; no launch')
    if digest(PARENT.read_bytes()) != PARENT_SHA: raise ValueError('parent changed')
    parent = read(PARENT); row = parent['evidence_rows'][0]
    if row['dataset_index'] != 114 or row['saved_precheck'] is None: raise ValueError('fixed subject changed')
    source = ROOT/parent['raw_root']/'rank0_evidence'
    binding = next(b for b in parent['terminal_bindings'] if b['job_id'] == 'rank0_evidence')
    for name, sha in [('request.json', binding['request_sha256']), ('terminal.json', binding['terminal_sha256']),
                      *row['saved_source_sha256'].items()]:
        if digest((source/name).read_bytes()) != sha: raise ValueError('saved artifact changed: '+name)
    started = time.monotonic()  # offline tail clock, explicitly no upstream computation
    request = read(source/'request.json')['evidence_request']
    terminal = supervise(source, request, OUT, started=started, enabled=True)
    ended = time.monotonic()
    reviewed = audit_outer(OUT)  # archival review, outside offline execution budget
    if reviewed != terminal: raise ValueError('outer replay audit mismatch')
    checked = read(OUT/'tail/check.log') if terminal['complete_independent_check'] else None
    expected = read(source/'independent.json')
    identical = checked is not None and checked['result'] == expected
    if digest((source/'terminal.json').read_bytes()) != binding['terminal_sha256']:
        raise ValueError('historical terminal changed')
    record = {'schema': 'CACHED_PORTABLE_SAVED114_V2', 'head': head,
              'controls_sha256': digest(controls_path.read_bytes()), 'parent_sha256': PARENT_SHA,
              'subject': 'archived rank0/input114', 'raw_root': str(OUT.relative_to(ROOT)),
              'status': 'PASS' if identical else 'INCOMPLETE_OR_DIFFERENT',
              'resources_before': resources, 'outer': terminal,
              'observed_tail_including_outer_publication_seconds': ended-started,
              'separate_archival_review_seconds': time.monotonic()-ended,
              'exact_saved_result_equal': identical,
              'expected_status': expected['status'], 'new_solver_queries': 0, 'new_real_requests': 0,
              'original_terminal': 'TIMEOUT', 'original_terminal_unchanged': True,
              'upstream_cost': 'not executed; offline saved-evidence tail, not production acceleration',
              'promotion': False,
              'files': {str(p.relative_to(OUT)): digest(p.read_bytes()) for p in OUT.rglob('*') if p.is_file()}}
    for name in ('candidate.json', 'packing.json', 'precheck.json'):
        path = OUT/'tail'/name
        if path.exists():
            value = read(path)
            record[name] = value if name != 'precheck.json' else {k: value[k] for k in ('cache', 'scope')}
    if checked is not None:
        record['isolated_check'] = {k: checked[k] for k in ('cache', 'check_seconds', 'isolated', 'site_disabled', 'solver_imported')}
        record['result_sha256'] = digest(json.dumps(checked['result'], sort_keys=True, separators=(',', ':')).encode())
    save_new(OUT/'summary.json', record)
    save_new(ROOT/'docs/cached_portable_v2_saved114.json', record)
    print(json.dumps({k: record[k] for k in ('status', 'exact_saved_result_equal', 'outer', 'observed_tail_including_outer_publication_seconds')}, indent=2))


if __name__ == '__main__': run()
