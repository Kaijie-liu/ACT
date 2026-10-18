"""Read saved finite-control records only; never launches or repairs a request.

This post-execution archive/audit is separately timed. It cannot promote a late,
partial or failed execution into online acceptance. Raw failure files remain.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from scoped_source.hz_request_store import compact, digest, require
from scripts.audit_native_controls import audit

REPO = Path(__file__).resolve().parents[1]
FREEZE = REPO/'docs/hz_native_controls_freeze_20261018.json'


def inventory(directory):
    rows = []
    for path in sorted(directory.rglob('*')):
        require(not path.is_symlink(), 'archive does not follow evidence symlinks')
        if not path.is_file():
            continue
        h = hashlib.sha256(); size = 0
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(65536), b''):
                h.update(block); size += len(block)
        require(size == path.stat().st_size, 'quiescent retained evidence')
        rows.append(dict(file=path.relative_to(directory).as_posix(), bytes=size, sha256=h.hexdigest()))
    return dict(files=rows, bytes=sum(r['bytes'] for r in rows), members=len(rows),
                inventory_sha256=digest(compact(rows)))


def summarize(directory):
    started = time.monotonic(); root = Path(directory)
    frozen = json.loads(FREEZE.read_bytes()); batch = json.loads((root/'batch.json').read_bytes())
    checked = audit(root, FREEZE)
    specs = {r['id']: r for r in frozen['records']}
    records = []
    for outcome in batch['outcomes']:
        name = outcome['id']; spec = specs[name]; run = root/'runs'/name
        receipt = json.loads((root/'receipts'/(name+'.json')).read_bytes())
        terminal = receipt['result']; actors = {}; last_progress = {}
        for role in ('production', 'replay', 'reception'):
            path = run/(role+'.json')
            if path.exists():
                a = json.loads(path.read_bytes())
                actors[role] = {k: a.get(k) for k in ('status', 'error', 'initial_work', 'work',
                    'delta_work', 'actor_seconds', 'limits', 'native_setup')}
                if role == 'production' and a.get('output') is not None:
                    actors[role]['phases'] = a['output']['phases']
                    actors[role]['snapshots'] = a['output']['snapshots']
            path = run/(role+'.progress')
            if path.exists():
                # A killed writer may leave a partial last line. Keep the last
                # complete JSON prefix; never interpret an incomplete line as 0.
                for line in path.read_bytes().splitlines():
                    try:
                        last_progress[role] = json.loads(line)
                    except (ValueError, UnicodeError):
                        break
        stored = inventory(run)
        record = dict(id=name, kind=spec['kind'], fault=spec['profile'].get('control'),
            arm=spec['profile']['method']['producer']['representation'],
            source_case=spec['profile']['source']['case'], source_sha256=spec['profile']['source']['source'],
            profile_sha256=spec['sha256'], control_passed=outcome['ok'],
            full_caller_seconds=outcome['full_caller_seconds'], error=receipt['error'],
            terminal_status=None if terminal is None else terminal['status'],
            online_acceptance=False if terminal is None else terminal['online_acceptance'],
            result=None if terminal is None else terminal['result'],
            online_work=None if terminal is None else terminal['cumulative_work'],
            online_work_complete=False if terminal is None else terminal['work_complete'],
            online_storage_complete=False if terminal is None else terminal['storage_complete'],
            owned_execution=None if terminal is None else terminal['owned_execution'],
            actors=actors, last_complete_progress=last_progress,
            retained_inventory=stored)
        records.append(record)
    component = None
    if batch['component'] is not None:
        component = dict(receipt=batch['component'], retained_inventory=inventory(root/'components'),
            contaminated_relocation_directory_is_fault_evidence_not_clean_delivery=True)
    references = {}
    for name in ('batch.json', 'admission.json', 'component-receipt.json'):
        p = root/name
        if p.exists():
            b = p.read_bytes(); references[name] = dict(bytes=len(b), sha256=digest(b))
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    normal = {(r['source_case'], r['arm']): r for r in records if r['kind']=='normal'}
    paired = []
    for source in ('mixed', 'clipped', 'guarded'):
        keys = [(source, arm) for arm in ('native_coo_reference', 'native_source_row_blocks')]
        if not all(k in normal for k in keys):
            continue
        a, b = (normal[k] for k in keys)
        def positive(r):
            value = r['result']
            return bool(r['online_acceptance'] and value is not None
                and value['missing_endpoints']==0 and value['positive']==value['required'])
        paired.append(dict(source=source, reference_positive=positive(a), blocks_positive=positive(b),
            complete_positive_gained=int(positive(b) and not positive(a)),
            complete_positive_lost=int(positive(a) and not positive(b)),
            result_equal=a['result']==b['result']))
    return dict(schema='NATIVE_FINITE_CONTROLS_ARCHIVE_V1', date='2026-10-18',
        execution_freeze_commit='f72acda2414c38e9d21de08ef3ac20bf1d18416b', archival_read_head=head,
        freeze_sha256=digest(FREEZE.read_bytes()), execution_directory=str(root), references=references,
        independent_terminal_audit=checked, records=records, component=component, paired_results=paired,
        attempted=len(records), planned=26, unexecuted=batch['unexecuted'], stop=batch['stop'],
        online_batch_seconds=batch['seconds'], offline_inspection_seconds=time.monotonic()-started,
        scientific_scope=dict(fixed_synthetic_declarations=True, real_models=0, gpu_runs=0,
            full_topology_admitted=False, checkpoint_correspondence_proved=False,
            native_float_execution_proved=False, external_comparison=False,
            complete_positive_gained=sum(p['complete_positive_gained'] for p in paired),
            complete_positive_lost=sum(p['complete_positive_lost'] for p in paired)),
        failure_cost_semantics='whole caller time is measured; missing online Work/storage stays null/partial; offline inventory does not rescue it',
        inspection_scope='saved record/identity/cost audit, not a human review or second independent mathematical solver run')


if __name__ == '__main__':
    begin = time.monotonic()
    p = argparse.ArgumentParser(); p.add_argument('directory'); p.add_argument('output'); a = p.parse_args()
    output = Path(a.output)
    require(output.is_absolute() and output.resolve() == output and output.is_relative_to(REPO/'docs')
            and not output.exists(), 'new local derived archive only')
    result = summarize(a.directory); raw = compact(result)
    require(len(raw) <= 4*2**20, 'bounded derived record, not a raw checkpoint or dataset')
    with output.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    print(compact(dict(status=result['independent_terminal_audit']['status'], archive=str(output),
        bytes=len(raw), sha256=digest(raw), attempted=result['attempted'], planned=26,
        offline_inspection_seconds=result['offline_inspection_seconds'],
        offline_seconds_through_publish=time.monotonic()-begin)).decode())
