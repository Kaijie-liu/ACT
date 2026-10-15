"""Fixed stdlib worker for direct lifecycle controls; no generic dispatch."""
import os
from pathlib import Path
import resource
import sys
import time


def main():
    from scoped_source.direct_supervised import (ROOT, POLICY, LOG_CAP, REPORT_CAP, META_RESERVE,
        bindings, validate_spec, fixed_source, read_json, write_json, census)
    from scoped_source.hz_request_store import Work, compact, require
    from scoped_source.direct_lifecycle_execution import execute, EXEC_FAULTS
    root, expected = Path(sys.argv[1]), sys.argv[2]
    inv, _ = read_json(root / 'invocation.json', LOG_CAP, expected)
    require(bindings() == inv['implementations'], 'frozen worker code')
    s = validate_spec(inv['spec'])
    resource.setrlimit(resource.RLIMIT_FSIZE, (REPORT_CAP, REPORT_CAP))
    # Frozen tiny fixtures and code members fit this stricter file bound.
    require(sys.flags.isolated == 1 and sys.flags.no_site == 1, 'isolated no-site worker')
    w = Work(inv['run_deadline'])
    w.counts.update(inv['initial_work'])
    if s['fault'] == 'startup_error':
        raise RuntimeError('fixed startup error')
    progress_bytes = 0
    def progress(name, event, work):
        nonlocal progress_bytes
        raw = compact(dict(name=name, event=event, work=dict(work.counts), at=time.monotonic())) + b'\n'
        progress_bytes += len(raw)
        require(progress_bytes <= LOG_CAP, 'bounded progress evidence')
        with (root / 'progress.jsonl').open('ab') as stream:
            stream.write(raw); stream.flush()
    fixture_start = time.monotonic()
    source = fixed_source(root / 'source', s['case'])
    w.tick()
    fixture_seconds = time.monotonic() - fixture_start
    before = census(root, w.deadline)
    result = execute(root / 'artifacts', root / 'source', source, s['method'], inv['id'], ROOT,
                     work=w, enabled=True, fault=s['fault'] if s['fault'] in EXEC_FAULTS else '',
                     archive_limit=POLICY['total_bytes'] - META_RESERVE - before['bytes'],
                     member_limit=POLICY['members'] - len(before['files']) - 3, progress=progress)
    result.update(isolated=True, site_disabled=True, fixture_seconds=fixture_seconds,
                  numerical_imports=sorted(n for n in sys.modules if n.split('.')[0] in ('torch', 'numpy', 'scipy')))
    require(not result['numerical_imports'], 'no numerical/native imports')
    require(bindings() == inv['implementations'], 'implementation unchanged at worker close')
    if s['fault'] == 'wrong_invocation':
        result['invocation'] = '0' * 64
    if s['fault'] == 'wrong_method':
        result['method']['replay']['kind'] = 'wrong'
    if s['fault'] == 'output_block':
        while True:
            time.sleep(.05)
    if s['fault'] == 'output_partial':
        with (root / 'candidate.json').open('xb') as stream:
            stream.write(b'{')
        return
    write_json(root / 'candidate.json', result, REPORT_CAP)
    if s['fault'] == 'late_exit':
        while True:
            time.sleep(.05)
    # If this exits late, the parent must reject the already-written candidate.
    w.tick()


if __name__ == '__main__':
    main()
