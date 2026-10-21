"""Frozen 41-request B1 online queue; no new solver or diagnostic strategy.

The controller runs from a git-archive copy. Offline diagnostics have separate
call IDs and never replace these online terminals. All source paths are bound
before launch; donor packages supply tensors only, never another arm's answers.
"""
import time
ENTRY = time.monotonic()
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import traceback

CODE = Path(__file__).resolve().parents[1]
SOURCE = Path('/data1/Kane/MOE/ACT')
sys.path[:0] = [str(CODE), str(CODE/'scripts')]
from scoped_proof.io import load, save, sha, tick, PYTHON
from scoped_proof.owned_session import execute
from scoped_source.development_work import DevelopmentWork, POLICY

ROWS = SOURCE/'data/moe/results/schedule_confirmation_100_full_20260912_r1/rows.jsonl'
SELECTION = SOURCE/'act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json'
CONFIG = SOURCE/'act/pipeline/moe/configs/route_complexity_reuse_v1.json'
STAGE_BYTES = 2*2**30
REQUEST_SECONDS = 300
SEALED = {98, 4088, 4096, 4098, 4099}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sizes(root):
    return sum(p.stat().st_size for p in root.rglob('*') if p.is_file())


def select(rows):
    return sorted((r for r in rows if r['method'] == 'adaptive'
                   and r['status'] in ('UNKNOWN', 'TIMEOUT') and r['dataset_index'] not in SEALED),
                  key=lambda r: (r['dataset_index'], int(r['model'][4:])))


def freeze(path):
    """Read-only source inventory, no model forward, propagation or solve."""
    rows = [json.loads(line) for line in ROWS.read_text().splitlines()]
    selected = select(rows)
    require(len(selected) == 41 and sum(r['status'] == 'UNKNOWN' for r in selected) == 25,
            'approved 25 UNKNOWN and 16 TIMEOUT')
    selection = load(SELECTION)
    requests = []
    for i, row in enumerate(selected):
        own = row.get('package')
        donors = sorted((r for r in rows if r['method'] == 'adaptive'
                         and r['dataset_index'] == row['dataset_index'] and r.get('package')),
                        key=lambda r: int(r['model'][4:]))
        donor = Path(own if own else donors[0]['package'])/'request.pt'
        sample = selection['samples'][row['rank']]
        require(sample['dataset_index'] == row['dataset_index'], 'sample identity')
        subject = selection['models'][row['model']]
        require(sha(subject['checkpoint']) == subject['checkpoint_sha256'], 'checkpoint binding')
        requests.append(dict(ordinal=i, job_id=f"{i:02d}_{row['model']}_{row['dataset_index']}",
            model=row['model'], dataset_index=row['dataset_index'], previous_status=row['status'],
            historical_job=row['job_id'], subject=subject, sample=sample,
            tensors=dict(path=str(donor), sha256=sha(donor)),
            epsilon=selection['request']['epsilon']))
    value = dict(schema='B1_OBSERVED_STAGE_V1', policy=POLICY,
        sources={str(p): sha(p) for p in (ROWS, SELECTION, CONFIG)},
        config=load(CONFIG), requests=requests, full_calls=41,
        order='dataset_index then numeric model seed', full_capture_ordinals=10,
        request_seconds=300, stage_seconds=21600, artifact_bytes=STAGE_BYTES,
        worker_alarm_seconds=294, run_seconds=296, cleanup_seconds=298,
        reception_deadline_seconds=300, historical_results_unchanged=True,
        grade='NUMERICAL_POLICY_DIAGNOSIS_NOT_COMPLETE_SOURCE_PROOF',
        diagnostics=dict(local_max=150, local_seconds_max=300,
            recipes=[dict(name='candidate replay and fixed PGD', slots=41, seconds=60),
                     dict(name='same native matrix continuous LP', slots=10, seconds=120),
                     dict(name='integer difference minimum and maximum', slots=20, seconds=60),
                     dict(name='replace only difference range', slots=10, seconds=60),
                     dict(name='first two retained sources sixteen closed margin slices two endpoints',
                          slots=64, seconds=30)],
            registered_local_slots=145, maximum_local_seconds=7380,
            unavailable='retain ordinal and UNEXECUTED/MISSING; no replacement',
            extended=dict(max_calls=8, seconds_per_call=1800, seconds_total=14400,
                          outside_stage_seconds=True, outside_full_slots=True,
                          eligibility='lower < 0 and finite candidate >= 0',
                          order='request ordinal then native query ordinal then pair/property'),
            offline_never_repairs_online=True),
        success_gate=None, next='multilabel attribution then B2/B3; no positive requirement')
    save(path, value)
    print(json.dumps(dict(freeze=str(path), sha256=sha(path), requests=len(requests))))


def publish_raw(path, raw, digest):
    require(hashlib.sha256(raw).hexdigest() == digest, 'payload binding')
    partial = path.with_name(path.name+'.partial')
    with partial.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    os.link(partial, path); partial.unlink()


def worker(folder):
    request = load(folder/'launch.json')
    start, deadline = request['started_at'], request['deadline']
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (128*2**20, 128*2**20))
    def alarm(*_):
        # The parent distinguishes this authorized cutoff from arbitrary exit124.
        os.write(2, b'B1_AUTHORIZED_WORKER_CUTOFF\n')
        os._exit(124)
    signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, max(.001, start+294-time.monotonic()))
    work = DevelopmentWork(deadline, policy=POLICY)
    with (folder/'native.log').open('xb', buffering=0) as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        try:
            from act.util.typing_compat import install_typing_override
            install_typing_override()
            import torch
            from act.util.device_manager import initialize_device
            from act.back_end.moe.factory import load_output_moe_checkpoint
            from act.pipeline.moe.staged_verifier import (verify_staged_linf, write_evidence_package,
                _tensor_identity, _model_state_identity)
            from act.pipeline.moe.request_diagnostics import RequestDiagnostics
            torch.set_num_threads(1); initialize_device('cpu', 'float64')
            subject, sample = request['item']['subject'], request['item']['sample']
            require(sha(subject['checkpoint']) == subject['checkpoint_sha256'], 'checkpoint identity')
            model, _ = load_output_moe_checkpoint(subject['checkpoint'], map_location='cpu')
            model.cpu().double().eval()
            require(_model_state_identity(model) == subject['model_state'], 'model state identity')
            source = request['item']['tensors']
            require(sha(source['path']) == source['sha256'], 'materialized source file')
            saved = torch.load(source['path'], map_location='cpu', weights_only=True)
            tensors = {k: saved[k] for k in ('center', 'lower', 'upper')}
            for k, tensor in tensors.items():
                require(_tensor_identity(tensor) == sample[k], 'materialized '+k)
            epsilon = request['item']['epsilon']
            for k, sign in (('lower', -1), ('upper', 1)):
                require(torch.equal(tensors[k], (tensors['center']+sign*epsilon).clamp(0, 1)),
                        'same production input box')
            events = folder/'diagnostics'; events.mkdir()
            count = 0
            def sink(kind, raw, digest):
                nonlocal count
                work.tick()
                require(len(raw)+sizes(folder) <= request['remaining_artifact_bytes'], 'stage artifact budget')
                publish_raw(events/f'{count:05d}_{kind}_{digest}.json', raw, digest)
                count += 1
                work.tick()
            observer = RequestDiagnostics(work=work, byte_limit=min(128*2**20,
                request['remaining_artifact_bytes']), publish=sink, numerical_all=True,
                full_capture_limit=request['full_capture_remaining'])
            progress_count = 0
            def progress(value):
                nonlocal progress_count
                work.tick(); save(folder/f'progress_{progress_count:03d}.json', value)
                progress_count += 1; work.tick()
            report = verify_staged_linf(model, tensors['center'], epsilon, request['config'],
                expected_clean_prediction=sample['label'], budget_started_at=start,
                checkpoint_identity={'path': subject['checkpoint'], 'sha256': subject['checkpoint_sha256']},
                progress_callback=progress, diagnostics=observer)
            work.tick()
            report.evidence['execution'] = dict(frozen_commit=request['commit'],
                freeze_sha256=request['freeze_sha256'], dataset_index=sample['dataset_index'],
                job_id=request['item']['job_id'], diagnostic_grade='NUMERICAL_POLICY')
            manifest = write_evidence_package(report, folder/'package')
            work.tick()
            save(folder/'worker.json', dict(status=report.status, reason=report.reason,
                manifest=manifest, diagnostics=report.evidence.get('diagnostics'),
                counts=work.counts, terminal_clock=time.monotonic(), deadline=deadline,
                freeze_sha256=request['freeze_sha256'], commit=request['commit']))
            work.tick()
        except BaseException as exc:
            traceback.print_exc()
            if time.monotonic() < start+294:
                save(folder/'worker_error.json', dict(error=repr(exc), kind=type(exc).__name__,
                    clock=time.monotonic(), counts=work.counts, freeze_sha256=request['freeze_sha256']))
            return 1
        finally:
            sys.stdout.flush(); sys.stderr.flush()
    return 0


def session_ok(r):
    return (r['status'] == 'COMPLETED' and r['returncode'] == 0 and r['execution_error'] is None
            and r['cleanup_error'] is None and r['cleanup_status'] == 'LEADER_REAPED_NO_LIVE_SESSION')


def run(root, freeze_path, commit):
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    spec = load(freeze_path); anchor = sha(freeze_path)
    require(not root.exists() and root.parent == Path('/data1/Kane/MOE/results'), 'new result root')
    require(CODE != SOURCE and (CODE/'.frozen_commit').read_text().strip() == commit,
            'independent frozen commit copy')
    require(spec['policy'] == POLICY and len(spec['requests']) == 41, 'frozen B1 queue')
    for path, digest in spec['sources'].items():
        require(sha(path) == digest, 'unchanged historical input/config binding')
    root.mkdir(); stage_deadline = ENTRY+21600
    save(root/'start.json', dict(commit=commit, freeze_sha256=anchor, entry=ENTRY,
         stage_deadline=stage_deadline, code=str(CODE), policy=POLICY,
         stage_call_slots=41, historical_real_calls_before=0))
    reserved = 0; rows = []
    for item in spec['requests']:
        if time.monotonic()+300 >= stage_deadline or sizes(root)+2**20 >= STAGE_BYTES:
            break
        folder = root/item['job_id']; folder.mkdir(); (folder/'tmp').mkdir()
        started = time.monotonic(); deadline = started+300
        launch = dict(item=item, started_at=started, deadline=deadline, commit=commit,
                      freeze_sha256=anchor, config=spec['config'],
                      full_capture_remaining=max(0, 10-reserved),
                      remaining_artifact_bytes=STAGE_BYTES-sizes(root)-2**20)
        save(folder/'launch.json', launch)
        env = dict(PATH='/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='', TMPDIR=str(folder/'tmp'))
        outcome = None; reception_error = None; summary = None
        try:
            receipt = execute([PYTHON, '-I', '-B', __file__, '--worker', str(folder)],
                run_deadline=started+296, cleanup_deadline=started+298,
                env=env, rss_limit=None, development_policy=POLICY)
            tick(deadline)
            inventories = sorted((folder/'diagnostics').glob('*_inventory_*.json'))
            if inventories:
                summary = load(inventories[-1], limit=2**20)
                reserved += len(summary.get('full_capture_events', []))
            if (folder/'worker.json').exists():
                outcome = load(folder/'worker.json', limit=2**20)
                require(outcome['commit'] == commit and outcome['freeze_sha256'] == anchor
                        and outcome['deadline'] == deadline, 'online worker identity')
                work = DevelopmentWork(deadline, policy=POLICY); work.continue_from(outcome['counts'])
                manifest = outcome['manifest']
                require(sha(manifest['evidence_path']) == manifest['evidence_sha256'], 'evidence identity')
                work.visit('coefficient_visits', (Path(manifest['evidence_path']).stat().st_size+4095)//4096)
                outcome['received_counts'] = work.counts
            tick(deadline)
            require(sizes(root) <= STAGE_BYTES, 'complete stage artifacts before reception')
        except BaseException as exc:
            reception_error = repr(exc)
            if 'receipt' not in locals():
                receipt = dict(status='ERROR', returncode=None, execution_error=repr(exc),
                    cleanup_error=None, cleanup_status='NOT_LAUNCHED')
        accepted = reception_error is None and outcome is not None and session_ok(receipt)
        status = outcome['status'] if accepted else ('TIMEOUT' if receipt['status'] == 'TIMEOUT' else 'ERROR')
        if not accepted and receipt.get('returncode') == 124 and (folder/'native.log').exists():
            with (folder/'native.log').open('rb') as log:
                log.seek(max(0, (folder/'native.log').stat().st_size-256))
                if b'B1_AUTHORIZED_WORKER_CUTOFF' in log.read(): status = 'TIMEOUT'
        row = dict(ordinal=item['ordinal'], model=item['model'], dataset_index=item['dataset_index'],
            status=status, online_accepted=accepted, session=receipt, reception_error=reception_error,
            worker=outcome, reserved_full_ordinals=reserved,
            obligation_count=None if summary is None else len(summary['obligations']),
            native_calls=None if summary is None else summary.get('native_calls'),
            elapsed_seconds=time.monotonic()-started, deadline=deadline)
        save(folder/'terminal.json', row); rows.append(row)
        save(root/f'ledger_{len(rows):02d}.json', dict(completed_calls=len(rows), full_slots=len(rows),
            online_seconds=sum(r['elapsed_seconds'] for r in rows), rows=rows,
            reserved_full_ordinals=reserved, stage_elapsed=time.monotonic()-ENTRY,
            local_calls=0, extended_calls=0))
        # Include terminal and ledger publication, not only solver/worker time.
        # This final compact clock receipt records its own explicit measurement
        # boundary; its subsequent write tail remains in stage elapsed time.
        received_at = time.monotonic(); artifact_bytes = sizes(root)
        final_ok = received_at < deadline and artifact_bytes+2**16 <= STAGE_BYTES
        row['elapsed_seconds'] = time.monotonic()-started
        save(folder/'reception.json', dict(accepted=accepted and final_ok,
            status=status if final_ok else 'RECEPTION_LIMIT',
            through_terminal_ledger_seconds=row['elapsed_seconds'],
            clock=received_at, deadline=deadline, artifact_bytes=artifact_bytes,
            measurement_tail='final compact receipt write charged to stage clock'))
        if not final_ok:
            row.update(status='RECEPTION_LIMIT', online_accepted=False)
        print(json.dumps(dict(completed=len(rows), total=41, index=item['dataset_index'],
                              model=item['model'], status=status)), flush=True)
        del receipt
        if (not final_ok or status == 'ERROR' or reception_error
                or row['session']['cleanup_status'] != 'LEADER_REAPED_NO_LIVE_SESSION'):
            break
    save(root/'online_complete.json', dict(completed_calls=len(rows), expected=41,
        status='COMPLETE' if len(rows)==41 else 'PREFIX_STOP', stage_elapsed=time.monotonic()-ENTRY,
        online_seconds=sum(r['elapsed_seconds'] for r in rows), bytes=sizes(root),
        reserved_full_ordinals=reserved, next='offline diagnostics; no online verdict replacement'))
    return 0 if len(rows)==41 else 1


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--freeze', type=Path)
    p.add_argument('--worker', type=Path); p.add_argument('--run', type=Path)
    p.add_argument('--manifest', type=Path); p.add_argument('--commit')
    args = p.parse_args()
    if args.freeze: freeze(args.freeze)
    elif args.worker: raise SystemExit(worker(args.worker))
    elif args.run: raise SystemExit(run(args.run, args.manifest, args.commit))
    else: p.error('explicit operation required')
