"""Versioned phases/coverage receipt; original owned-process watchdog and budgets."""
from fractions import Fraction as F
import math
import os
from pathlib import Path
import time
import uuid

from frontier_proof.contract import phases, policy
from scoped_proof.evidence import POSITIVE, roster
from scoped_proof.io import PYTHON, load, save, sha
from scoped_proof.supervisor import execute, accept as exhaustive_accept
from source_enclosure.format import identity


def command(phase, root, deadline, spec):
    mode = policy(spec)['mode']
    module = 'scoped_proof.worker' if mode == 'exhaustive' or phase == 'intake' else 'frontier_proof.worker'
    no_site = phase in ('route_propose','route_check','construct','source_check','aggregate')
    return [PYTHON]+(['-S'] if no_site else [])+['-m',module,phase,str(root),'--deadline',str(deadline)]


def accept(root, scope, token, mode):
    if mode == 'exhaustive':
        return exhaustive_accept(root, scope, token)
    small = load(root/'result_candidate.json', limit=1024**2)
    ref = small['evidence_check']
    if (root/'evidence_check.json').stat().st_size != ref['bytes']:
        raise ValueError('check file size')
    checked = load(root/'evidence_check.json', ref['sha256'], limit=8*1024**2)
    for key in ('status','request_sha256','invocation','required','checked_bounds','positive_bounds',
                'proposal_complete','complete_output_positive_proof','native_float_proof','route_changing_established',
                'discharged_by_exclusion','required_output_bounds'):
        if small[key] != checked[key]:
            raise ValueError('checker receipt differs')
    expected = roster(scope)
    if (small['status'] not in (POSITIVE, 'NOT_CLOSED') or small['request_sha256'] != identity(scope) or
            small['invocation'] != token or small['required'] != len(expected) or
            small['native_float_proof'] is not False or small['route_changing_established'] is not False or
            len(checked['rows']) != len(expected) or type(small['proposal_complete']) is not bool):
        raise ValueError('request/coverage/guarantee binding')
    decisions = checked['source_check']['frontier']['pairs']
    expected_pairs = [list(p) for p in dict.fromkeys(tuple(r['pair']) for r in expected)]
    if [p['pair'] for p in decisions] != expected_pairs:
        raise ValueError('route ledger missing pair')
    by_pair = {tuple(d['pair']): d for d in decisions}
    missing, nonpositive, excluded = [], [], []
    for index, (row, want) in enumerate(zip(checked['rows'], expected)):
        if row['index'] != index or any(row[k] != v for k, v in want.items()):
            raise ValueError('original property index/identity')
        decision = by_pair[tuple(want['pair'])]
        if row['status'] == 'DISCHARGED_BY_CHECKED_ROUTE_EXCLUSION':
            if (decision['status'] != 'EXCLUDED_BY_CHECKED_STRICT_MARGIN' or row['exclusion'] != decision or
                    decision['lower'] not in want['pair'] or decision['higher'] in want['pair'] or
                    not 0 <= decision['higher'] < scope['experts'] or F(decision['checked_lower_bound']) <= 0):
                raise ValueError('invalid exclusion receipt')
            excluded.append(index)
        else:
            if decision['status'] != 'RETAINED':
                raise ValueError('output result on excluded pair')
            if row['status'] in ('MISSING','NO_CANDIDATE'):
                missing.append(index)
            elif row['status'] == 'NONPOSITIVE_BOUND':
                if F(row['checked_lower_bound']) > 0: raise ValueError('nonpositive flag')
                nonpositive.append(index)
            elif row['status'] != 'POSITIVE_BOUND' or F(row['checked_lower_bound']) <= 0:
                raise ValueError('invalid positive flag')
    required = len(expected)-len(excluded)
    closed = small['proposal_complete'] and not missing and not nonpositive
    if (required <= 0 or checked['missing'] != missing or checked['nonpositive'] != nonpositive or
            small['missing'] != len(missing) or small['nonpositive'] != len(nonpositive) or
            small['discharged_by_exclusion'] != len(excluded) or small['required_output_bounds'] != required or
            small['checked_bounds'] != required-len(missing) or small['positive_bounds'] != required-len(missing)-len(nonpositive) or
            small['complete_output_positive_proof'] != closed or (small['status'] == POSITIVE) != closed):
        raise ValueError('all-original-obligation aggregation')
    return small


def supervise(root, spec, *, budget=300., rss_limit=8*2**30, command_factory=None):
    if (type(budget) not in (int,float) or not math.isfinite(budget) or not 0 < budget <= 300 or
            type(rss_limit) is not int or rss_limit <= 0):
        raise ValueError('bounded request policy')
    root = Path(root)
    if root.exists(): raise FileExistsError(root)
    started = time.monotonic(); deadline = started+budget; work_deadline = deadline-min(2., budget/10)
    root.mkdir(parents=True, exist_ok=False); token = uuid.uuid4().hex
    record = save(root/'spec.json', spec)
    save(root/'invocation.json', {'invocation': token, 'started_monotonic': started,
        'deadline_monotonic': deadline, 'work_deadline_monotonic': work_deadline,
        'spec_file_sha256': record['sha256'], 'request_sha256': identity(spec['scope']),
        'budget_seconds': budget, 'rss_limit': rss_limit})
    env = dict(os.environ, OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
               NUMEXPR_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    stages, accepted, status, error, proposal_state = [], None, 'ERROR', None, None
    try:
        mode = policy(spec)['mode']
        for phase in phases(spec):
            begin = time.monotonic()
            phase_deadline = begin+max(0.,work_deadline-begin)/2 if phase == 'propose' else work_deadline
            cmd = command_factory(phase,root,phase_deadline) if command_factory else command(phase,root,phase_deadline,spec)
            row = execute(cmd, root/(phase+'.log'), phase_deadline, env, rss_limit)
            # An expired, never-started call has no process group to clean up.
            row.setdefault('deadline_monotonic', phase_deadline)
            row.setdefault('cleanup_included', True)
            row.update(phase=phase,start_seconds=begin-started,end_seconds=time.monotonic()-started)
            stages.append(row); save(root/(phase+'_stage.json'),row)
            if phase == 'propose':
                proposal_state = row['status']
                if proposal_state in ('TIMEOUT','ERROR') and time.monotonic() < work_deadline:
                    continue  # inspect partial output only; never promote failed phase
            if row['status'] != 'COMPLETED':
                status = row['status']; break
        else:
            accepted = accept(root,spec['scope'],token,mode); status = accepted['status']
            if proposal_state != 'COMPLETED':
                status = 'ERROR' if proposal_state == 'ERROR' else 'NOT_CLOSED'; accepted = None
    except BaseException as exc:
        error, status, accepted = repr(exc), 'ERROR', None
    if time.monotonic() >= work_deadline:
        status, accepted = 'TIMEOUT', None
    terminal = {'schema':'FRONTIER_PROOF_TERMINAL_V1','status_before_publication':status,
        'invocation':token,'request_sha256':identity(spec['scope']),'stages':stages,'candidate':accepted,
        'error':error,'required':len(roster(spec['scope'])),'budget_seconds':budget,
        'seconds_before_publication':time.monotonic()-started,'acceptance_requires_cost_receipt':True,
        'production_verdict_changed':False}
    record = save(root/'terminal.json',terminal)
    save(root/'receipt.json',{'schema':'FRONTIER_PROOF_RECEIPT_V1','invocation':token,
        'terminal_sha256':record['sha256'],'status':status,'seconds_before_receipt':time.monotonic()-started})
    elapsed = time.monotonic()-started
    if elapsed >= budget: status = 'TIMEOUT'
    stage_seconds = sum(row['seconds'] for row in stages)
    save(root/'cost.json',{'schema':'FRONTIER_PROOF_COST_V1','status':status,'invocation':token,
        'request_sha256':identity(spec['scope']),'budget_seconds':budget,'end_to_end_seconds':elapsed,
        'stage_seconds':stage_seconds,'overhead_seconds':elapsed-stage_seconds,
        'sampled_peak_rss':max((row['sampled_peak_rss'] for row in stages),default=0),
        'complete_output_positive_proof':status==POSITIVE,'terminal_sha256':record['sha256'],
        'receipt_sha256':sha(root/'receipt.json'),'candidate_prefix_may_be_partial':True,
        'includes':'imports, intake, router proposal/check, propagation, source check, native LPs, exact aggregation, serialization, receipt, cleanup',
        'excludes':'final ledger write and later audit; ledger overrun invalidates acceptance'})
    if time.monotonic() >= deadline:
        save(root/'publication_timeout.json',{'status':'TIMEOUT','seconds':time.monotonic()-started}); status='TIMEOUT'
    return {'status':status,'complete_output_positive_proof':status==POSITIVE,
            'seconds':time.monotonic()-started,'root':str(root)}
