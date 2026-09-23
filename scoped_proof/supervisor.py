"""Hard owned-process supervision; a single budget includes receipt/publication.

No worker can extend its deadline. Partial evidence survives, but never fills a
missing obligation. Only this invocation's process groups are killed.
"""
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
from source_enclosure.format import identity
from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof.evidence import POSITIVE, roster

PHASES = ('intake', 'construct', 'source_check', 'propose', 'aggregate')


def group_rss(pgid):
    rss = 0; members = []
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal(): continue
        try:
            fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) != pgid: continue
            if fields[0] != 'Z': members.append(int(p.name))
            for line in (p/'status').read_text().splitlines():
                if line.startswith('VmRSS:'): rss += int(line.split()[1])*1024
        except (OSError, ValueError, IndexError): continue  # process exited during sample
    return rss, members


def own_rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'): return int(line.split()[1])*1024
    raise ValueError('parent RSS unavailable')


def execute(command, log, deadline, env, rss_limit):
    start = time.monotonic(); peak = 0; p = None; state = 'TIMEOUT'; error = None
    if start >= deadline: return {'status': state, 'seconds': 0., 'pid': None, 'returncode': None, 'sampled_peak_rss': 0}
    try:
        with Path(log).open('xb') as f:
            p = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                now = time.monotonic(); rss, members = group_rss(p.pid); rss += own_rss(); peak = max(peak, rss)
                if now >= deadline: state = 'TIMEOUT'; break
                if rss > rss_limit: state = 'RESOURCE_LIMIT'; break
                code = p.poll()
                if code is not None:
                    state = 'COMPLETED' if code == 0 else 'ERROR'
                    if any(pid != p.pid for pid in members): state = 'ERROR'; error = 'live descendant after leader exit'
                    break
                time.sleep(min(.02, max(0, deadline-time.monotonic())))
    except Exception as exc: state = 'ERROR'; error = repr(exc)
    finally:
        if p is not None:
            try: os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            p.wait()
    end = time.monotonic()
    if end >= deadline and state == 'COMPLETED': state = 'TIMEOUT'
    return {'status': state, 'seconds': end-start, 'pid': None if p is None else p.pid,
        'returncode': None if p is None else p.returncode, 'sampled_peak_rss': peak, 'error': error,
        'deadline_monotonic': deadline, 'cleanup_included': True}


def accept(root, scope, token):
    candidate = load(root/'result_candidate.json', limit=1024**2)
    record = candidate['evidence_check']
    if (root/'evidence_check.json').stat().st_size != record['bytes']: raise ValueError('check artifact byte count')
    checked = load(root/'evidence_check.json', record['sha256'], limit=8*1024**2)
    expected = roster(scope)
    for key in ('status','request_sha256','invocation','required','checked_bounds','positive_bounds',
                'proposal_complete','complete_output_positive_proof','native_float_proof','route_changing_established'):
        if candidate[key] != checked[key]: raise ValueError('checker summary/receipt disagreement')
    if (candidate['status'] not in (POSITIVE, 'NOT_CLOSED') or candidate['request_sha256'] != identity(scope) or
        candidate['invocation'] != token or candidate['required'] != len(expected) or
        candidate['native_float_proof'] is not False or candidate['route_changing_established'] is not False):
        raise ValueError('unbound or overclaimed checker result')
    if len(checked['rows']) != len(expected): raise ValueError('missing checked obligation inventory')
    for i, (row, obligation) in enumerate(zip(checked['rows'], expected)):
        if row['index'] != i or any(row[k] != v for k,v in obligation.items()): raise ValueError('checker property identity')
        if row['status'] not in ('POSITIVE_BOUND','NONPOSITIVE_BOUND','MISSING','NO_CANDIDATE'): raise ValueError('row status')
    missing = [r['index'] for r in checked['rows'] if r['status'] in ('MISSING','NO_CANDIDATE')]
    nonpositive = [r['index'] for r in checked['rows'] if r['status'] == 'NONPOSITIVE_BOUND']
    closed = candidate['proposal_complete'] is True and not missing and not nonpositive
    if (checked['missing'] != missing or checked['nonpositive'] != nonpositive or
        candidate['missing'] != len(missing) or candidate['nonpositive'] != len(nonpositive) or
        candidate['checked_bounds'] != len(expected)-len(missing) or
        candidate['positive_bounds'] != len(expected)-len(missing)-len(nonpositive) or
        candidate['complete_output_positive_proof'] != closed or (candidate['status'] == POSITIVE) != closed):
        raise ValueError('inconsistent complete-coverage aggregation')
    return candidate


def supervise(root, spec, *, budget=300., rss_limit=8*2**30, command_factory=None):
    if (type(budget) not in (int,float) or not math.isfinite(budget) or not 0 < budget <= 300 or
            type(rss_limit) is not int or rss_limit <= 0): raise ValueError('bounded request policy')
    root = Path(root)
    if root.exists(): raise FileExistsError(root)
    started = time.monotonic(); deadline = started+budget; work_deadline = deadline-min(2., budget/10)
    root.mkdir(parents=True, exist_ok=False); token = uuid.uuid4().hex
    spec_record = save(root/'spec.json', spec)
    save(root/'invocation.json', {'invocation': token, 'started_monotonic': started,
        'deadline_monotonic': deadline, 'work_deadline_monotonic': work_deadline,
        'spec_file_sha256': spec_record['sha256'], 'request_sha256': identity(spec['scope']),
        'budget_seconds': budget, 'rss_limit': rss_limit})
    env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
        NUMEXPR_NUM_THREADS='2', CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
    stages = []; accepted = None; status = 'ERROR'; error = None; proposal_state = None
    try:
        for phase in PHASES:
            begin = time.monotonic()
            # Proposal gets half the ACTUAL remaining work time; final exact check is charged too.
            phase_deadline = begin+max(0., work_deadline-begin)/2 if phase == 'propose' else work_deadline
            if command_factory is None:
                command = [PYTHON]+(['-S'] if phase in ('source_check','aggregate') else [])+[
                    '-m','scoped_proof.worker', phase, str(root), '--deadline', str(phase_deadline)]
            else: command = command_factory(phase, root, phase_deadline)
            result = execute(command, root/(phase+'.log'), phase_deadline, env, rss_limit)
            result.update(phase=phase, start_seconds=begin-started, end_seconds=time.monotonic()-started)
            stages.append(result); save(root/(phase+'_stage.json'), result)
            if phase == 'propose':
                proposal_state = result['status']
                if proposal_state in ('TIMEOUT','ERROR') and time.monotonic() < work_deadline:
                    continue  # Check partial prefix; cannot promote it to a full proof.
            if result['status'] != 'COMPLETED': status = result['status']; break
        else:
            accepted = accept(root, spec['scope'], token)
            status = accepted['status']
            if proposal_state != 'COMPLETED':
                status = 'ERROR' if proposal_state == 'ERROR' else 'NOT_CLOSED'
                accepted = None
    except BaseException as exc: error = repr(exc); status = 'ERROR'; accepted = None
    if time.monotonic() >= work_deadline: status = 'TIMEOUT'; accepted = None
    terminal = {'schema': 'SCOPED_PROOF_TERMINAL_V1', 'status_before_publication': status,
        'invocation': token, 'request_sha256': identity(spec['scope']), 'stages': stages, 'candidate': accepted,
        'error': error, 'required': len(roster(spec['scope'])), 'budget_seconds': budget,
        'seconds_before_publication': time.monotonic()-started,
        'acceptance_requires_cost_receipt': True, 'production_verdict_changed': False}
    record = save(root/'terminal.json', terminal)
    receipt = {'schema': 'SCOPED_PROOF_RECEIPT_V1', 'invocation': token,
        'terminal_sha256': record['sha256'], 'status': status, 'seconds_before_receipt': time.monotonic()-started}
    save(root/'receipt.json', receipt)
    end = time.monotonic(); elapsed = end-started
    if elapsed >= budget: status = 'TIMEOUT'
    cost = {'schema': 'SCOPED_PROOF_COST_V1', 'status': status, 'invocation': token,
        'request_sha256': identity(spec['scope']), 'budget_seconds': budget,
        'end_to_end_seconds': elapsed, 'stage_seconds': sum(r['seconds'] for r in stages),
        'overhead_seconds': elapsed-sum(r['seconds'] for r in stages),
        'sampled_peak_rss': max((r['sampled_peak_rss'] for r in stages), default=0),
        'complete_output_positive_proof': status == POSITIVE, 'terminal_sha256': record['sha256'],
        'receipt_sha256': sha(root/'receipt.json'), 'candidate_prefix_may_be_partial': True,
        'includes': 'imports, hashes, checkpoint/input, capture, propagation, source check, LPs, exact aggregation, serialization, receipt, owned cleanup',
        'excludes': 'final cost-ledger write and later administrative audit; ledger overrun invalidates acceptance'}
    save(root/'cost.json', cost)
    # Receipt/ledger latency must never resurrect an expired positive candidate.
    if time.monotonic() >= deadline:
        save(root/'publication_timeout.json', {'status':'TIMEOUT','seconds':time.monotonic()-started})
        status = 'TIMEOUT'
    return {'status': status, 'complete_output_positive_proof': status == POSITIVE,
        'seconds': time.monotonic()-started, 'root': str(root)}
