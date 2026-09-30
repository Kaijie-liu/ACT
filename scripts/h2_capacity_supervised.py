"""Separate full-size synthetic H2 supervision; no real intake.

The outer execution contract is versioned separately from the mathematical
checker. A positive execution also requires an observed completed API return.
"""
from fractions import Fraction as F
from itertools import combinations
import math
import os
from pathlib import Path
import time
import uuid
from scoped_proof.io import ROOT, PYTHON, load, save, sha, tick
from scoped_proof.supervisor import execute
from source_enclosure.format import identity
from scoped_source.rowwise_verify import CODE, envelope, sha_value
from scoped_source.factored_io import referenced, HEADER_LIMIT

POSITIVE='CHECKED_DECLARED_SOURCE_POSITIVE'
STATES=(POSITIVE,'UNKNOWN_NONPOSITIVE','UNKNOWN_MISSING_EVIDENCE')
PHASES=('produce','check','receive')
CONFIG='configs/h2_capacity_execution_20261001.json'
PROTOCOL_SHA256='cbdc74ef27cccdb940c979b9cd417ecef2b82f82a1073a1d44d151e40158e2fb'
EXTRA_FILES=(CONFIG,'scripts/h2_capacity_supervised.py','scripts/h2_capacity_worker.py',
    'scripts/h2_capacity_build.py','scripts/h2_capacity_native.py')
from scoped_source.rowwise_supervised import FAULTS as OLD_FAULTS
FAULTS=OLD_FAULTS+('pair_delay','endpoint_delay','native_delay','missing_stdout')


def protocol():
    cfg=load(ROOT/CONFIG)
    if identity(cfg)!=PROTOCOL_SHA256: raise ValueError('capacity execution protocol identity')
    from scoped_source.capacity_intake import protocol as recipe
    recipe()
    prepared=load(ROOT/cfg['preparation_archive'],required_sha(cfg['preparation_archive_sha256']))
    full=next(v for v in cfg['cases'] if v['case']=='full_size')
    candidate=prepared['audit']['source_identity']['candidate']
    if any(candidate[k]!=full[k] for k in ('declared_source_sha256','source_manifest_sha256','request','request_sha256')):
        raise ValueError('frozen prepared identity differs')
    return cfg


def specification(case='tiny_control',mode='endpoints',control=''):
    cfg=protocol(); item=next(v for v in cfg['cases'] if v['case']==case)
    value={'schema':'H2_CAPACITY_SPEC_V1',**item,'mode':mode,'control':control,'protocol_sha256':PROTOCOL_SHA256}
    validate_spec(value); return value


def validate_spec(spec):
    cfg=protocol()
    keys={'case','intake','declared_source_sha256','source_manifest_sha256','request','request_sha256','reuse'}
    if (set(spec)!=keys|{'schema','mode','control','protocol_sha256'} or
        spec['schema']!='H2_CAPACITY_SPEC_V1' or spec['mode'] not in cfg['arms'] or
        spec['control'] not in FAULTS or spec['protocol_sha256']!=PROTOCOL_SHA256 or
        {k:spec[k] for k in keys} not in cfg['cases'] or
        spec['case']=='full_size' and spec['control']):
        raise ValueError('fixed synthetic capacity specification')
    if spec['control'] in ('missing_endpoint','missing_both','endpoint_delay') and spec['mode']!='endpoints':
        raise ValueError('endpoint fault only')
    if spec['control']=='rewrite_check_stdout' and spec['mode']!='mccormick':
        raise ValueError('MC output fault only')


def capture_receipt(doc,spec,inv):
    if identity(doc)!=spec['declared_source_sha256'] or doc['request']!=spec['request']:
        raise ValueError('captured source/request differs from frozen preparation')
    return {'schema':'H2_CAPACITY_CAPTURE_V1','invocation':inv['invocation'],
        'declared_source_sha256':spec['declared_source_sha256'],
        'source_manifest_sha256':spec['source_manifest_sha256'],'request':spec['request'],
        'request_sha256':spec['request_sha256'],'producer_sources_sha256':identity(inv['producer_sources']),
        'recipe_sha256':protocol()['recipe_sha256'],'native_float_proof':False,
        'checkpoint_loaded':False,'dataset_loaded':False,'forward_executed':False}


def required_sha(value):
    return sha_value(value)


def trusted_checker_sources():
    return {**{'code/'+n:sha(ROOT/n) for n in CODE},'verify.py':sha(ROOT/'scoped_source/rowwise_verify.py')}


def producer_sources(spec):
    from scoped_source.capacity_intake import sources
    names=sources()
    names.update({n:sha(ROOT/n) for n in EXTRA_FILES})
    return dict(sorted(names.items()))


def bind_producer(spec,inv):
    if inv.get('producer_sources')!=producer_sources(spec): raise ValueError('source producer implementation changed')


def bind_checker(root,manifest,expected):
    required={'code/'+n for n in CODE}|{'verify.py'}
    if set(expected)!=required: raise ValueError('trusted checker inventory')
    for name in required:
        path=root/'bundle'/name
        if (manifest['files'].get(name,{}).get('sha256')!=expected[name] or
                path.is_symlink() or sha(path)!=expected[name]): raise ValueError('trusted mathematical checker identity')


def receive(root,spec,invocation,invocation_sha256,checker_stdout_sha256,deadline=None):
    # Offline audits have a separate check budget; online passes its SAME deadline.
    deadline=time.monotonic()+300 if deadline is None else deadline
    required_sha(checker_stdout_sha256); required_sha(invocation_sha256); tick(deadline); validate_spec(spec)
    built=load(root/'built.json',limit=2**20)
    if (built['invocation']!=invocation or built['source_manifest_sha256']!=spec['source_manifest_sha256'] or
            built['mode']!=spec['mode']): raise ValueError('build invocation/source/mode')
    manifest,_=envelope(root/'bundle',built['sha256'],spec['source_manifest_sha256'],
                        built['proof_manifest_sha256'],spec['mode'],deadline)
    if manifest['invocation']!=invocation: raise ValueError('portable invocation')
    inv=load(root/'invocation.json',invocation_sha256)
    if inv['invocation']!=invocation or inv['spec_sha256']!=identity(spec): raise ValueError('receiver invocation')
    bind_producer(spec,inv); bind_checker(root,manifest,inv['checker_sources'])
    tree=root/'bundle/proof'
    proof=load(tree/'manifest.json',limit=HEADER_LIMIT)
    source=load(tree/'source/manifest.json',limit=HEADER_LIMIT)
    if (identity(proof)!=built['proof_manifest_sha256'] or identity(source)!=spec['source_manifest_sha256'] or
            proof['mode']!=spec['mode'] or proof['reuse_requested']!=spec['reuse'] or
            proof['source_manifest_sha256']!=spec['source_manifest_sha256']): raise ValueError('inner source/proof identity')
    request=source['declaration']['request']
    checked=load(root/'check.stdout',checker_stdout_sha256,limit=8*2**20)
    if (checked['schema']!='HR_PORTABLE_CHECK_V1' or checked['invocation']!=invocation or
            checked['source_manifest_sha256']!=spec['source_manifest_sha256'] or
            checked['proof_manifest_sha256']!=built['proof_manifest_sha256'] or
            checked['manifest_sha256']!=built['sha256'] or checked['mode']!=spec['mode'] or
            checked['isolated'] is not True or checked['site_disabled'] is not True or
            checked['solver_imported'] is not False or checked['producer_executed'] is not False):
        raise ValueError('independent checker binding')
    result=checked['result']; pairs=list(combinations(range(request['experts']),2))
    competitors=[j for j in range(request['classes']) if j!=request['label']]
    expected=[(list(p),j) for p in pairs for j in competitors]; rows=result['duties']
    if ([(r['pair'],r['competitor']) for r in rows]!=expected or
            any(type(r['competitor']) is not int or any(type(p) is not int for p in r['pair']) for r in rows) or
            result['required']!=len(expected) or result['mode']!=spec['mode'] or
            result['source_manifest_sha256']!=spec['source_manifest_sha256'] or
            result['proof_manifest_sha256']!=built['proof_manifest_sha256'] or
            result['deployed_float_SAFE'] is not False or result['hard_budget_supervision'] is not False or
            result['routing_coverage']!='ALL_UNORDERED_TOP2_PAIRS_NO_DUTIES_DROPPED' or
            len(proof['pairs'])!=len(pairs)): raise ValueError('complete request coverage/guarantee')
    missing=checked_lps=0; origins=[]; cursor=0
    for pair,ref in zip(pairs,proof['pairs']):
        tick(deadline); part=referenced(tree,ref)
        if (ref['pair']!=list(pair) or part['context']['pair']!=list(pair) or
                [r['competitor'] for r in part['duties']]!=competitors): raise ValueError('pair/property inventory')
        for record in part['duties']:
            row=rows[cursor]; cursor+=1; origin=record['origin']; origins.append(origin)
            if spec['mode']=='endpoints':
                weights=sorted(set(map(F,part['context']['gate'])))
                if ([F(e['weight']) for e in record['endpoints']]!=weights or
                        len(row['endpoint_bounds'])!=len(weights)): raise ValueError('endpoint coverage')
                values=[None if v is None else F(v) for v in row['endpoint_bounds']]
                if any((v is None)!=(e['certificate'] is None) for v,e in zip(values,record['endpoints'])):
                    raise ValueError('missing endpoint evidence')
                lower=None if None in values else min(values)
                if row['lower_bound']!=(None if lower is None else str(lower)): raise ValueError('endpoint minimum')
                missing+=sum(v is None for v in values); checked_lps+=sum(v is not None for v in values)
            else:
                lower=None if row['lower_bound'] is None else F(row['lower_bound'])
                if (lower is None)!=(record['certificate'] is None): raise ValueError('missing MC evidence')
                missing+=lower is None; checked_lps+=lower is not None and origin!='SOURCE_BOX_REUSE'
            if row['positive'] is not (lower is not None and lower>F(1,10_000_000)):
                raise ValueError('exact positive aggregation')
    positive=sum(r['positive'] for r in rows)
    status=POSITIVE if positive==len(expected) else 'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE'
    if (result['positive']!=positive or result['missing']!=missing or result['status']!=status or
            result['lp_bounds_checked']!=checked_lps or result['origins']!=origins):
        raise ValueError('whole-request aggregation disagreement')
    accepted={'schema':'H2_CAPACITY_ACCEPTED_CHECK_V1','status':status,'invocation':invocation,
        'source_manifest_sha256':spec['source_manifest_sha256'],'declared_source_sha256':spec['declared_source_sha256'],
        'proof_manifest_sha256':built['proof_manifest_sha256'],'manifest_sha256':built['sha256'],
        'checker_stdout_sha256':checker_stdout_sha256,'required':len(expected),'positive':positive,
        'missing':missing,'mode':spec['mode'],'result':result}
    if request!=spec['request'] or identity(request)!=spec['request_sha256']:
        raise ValueError('frozen request identity')
    captured=load(root/'model_intake.json',limit=2**20)
    expected_capture={'schema':'H2_CAPACITY_CAPTURE_V1','invocation':invocation,
        'declared_source_sha256':spec['declared_source_sha256'],
        'source_manifest_sha256':spec['source_manifest_sha256'],'request':request,
        'request_sha256':spec['request_sha256'],'producer_sources_sha256':identity(inv['producer_sources']),
        'recipe_sha256':protocol()['recipe_sha256'],'native_float_proof':False,
        'checkpoint_loaded':False,'dataset_loaded':False,'forward_executed':False}
    if captured!=expected_capture: raise ValueError('capacity capture receipt')
    accepted['model_capture_receipt_content_sha256']=identity(captured)
    accepted.update(pipeline_complete=True,all_obligations_checked=missing==0,
                    complete_declared_source_positive=status==POSITIVE,real_model_admitted=False)
    tick(deadline); return accepted


def supervise(root, spec, *, budget=300., rss_limit=2*2**30):
    start = time.monotonic()
    if (type(budget) not in (int,float) or not math.isfinite(budget) or not 0 < budget <= 300 or
            type(rss_limit) is not int or not 0 < rss_limit <= 2*2**30): raise ValueError('request budget')
    validate_spec(spec)
    if spec['case']=='full_size' and (budget!=300 or rss_limit!=2*2**30): raise ValueError('full capacity fixed limits')
    root = Path(root)
    if not root.is_absolute() or not root.resolve().is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('output outside project')
    root.mkdir(parents=True, exist_ok=False)
    deadline = start+budget; work_deadline = deadline-min(1.,budget/5)
    invocation = uuid.uuid4().hex
    save(root/'spec.json', spec)
    invocation_record = save(root/'invocation.json', {'schema':'H2_CAPACITY_INVOCATION_V1', 'invocation':invocation,
        'spec_sha256':identity(spec), 'start':start, 'deadline':deadline,
        'work_deadline':work_deadline, 'budget':budget, 'rss_limit':rss_limit,
        'checker_sources':trusted_checker_sources(),'producer_sources':producer_sources(spec)})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    stages = []; accepted = None; status = 'ERROR'; error = None; checker_stdout_sha256 = None
    try:
        for phase in PHASES:
            anchored_inv = load(root/'invocation.json',invocation_record['sha256'])
            bind_producer(spec,anchored_inv)
            before = time.monotonic()
            phase_deadline = work_deadline-min(5.,budget/4) if phase == 'produce' else work_deadline
            command = [PYTHON,'-B']+(['-S'] if phase != 'produce' else [])+[
                '-m','scripts.h2_capacity_worker', phase, str(root), '--deadline',str(phase_deadline),
                '--invocation-sha',invocation_record['sha256']]
            if phase == 'receive':
                if checker_stdout_sha256 is None: raise ValueError('checker output not anchored')
                command += ['--checker-stdout-sha',checker_stdout_sha256]
            record = execute(command, root/(phase+'.log'), phase_deadline, env, rss_limit)
            # Anchor actual output after the owned checker has exited, before a
            # receiving worker can modify it. Never take this hash from a worker.
            record.update(phase=phase, start_seconds=before-start, end_seconds=time.monotonic()-start)
            stages.append(record)
            try:
                if phase == 'check' and record['status'] == 'COMPLETED':
                    path = root/'check.stdout'
                    if path.is_symlink() or path.stat().st_size > 8*2**20:
                        raise ValueError('invalid checker output file')
                    checker_stdout_sha256 = sha(path)
                    record['checker_stdout_sha256'] = checker_stdout_sha256
            except Exception as exc:
                record['output_reception_error']=repr(exc)
                raise
            finally:
                record['end_seconds']=time.monotonic()-start
                save(root/(phase+'_stage.json'),record)
            load(root/'invocation.json',invocation_record['sha256'])
            if record['status'] != 'COMPLETED': status = record['status']; break
        else:
            accepted = load(root/'accepted.json', limit=8*2**20)
            if (accepted['invocation'] != invocation or accepted['source_manifest_sha256'] != spec['source_manifest_sha256'] or
                    accepted['status'] not in STATES or
                    accepted['checker_stdout_sha256'] != checker_stdout_sha256 or
                    sha(root/'check.stdout') != checker_stdout_sha256):
                raise ValueError('accepted receipt identity/status')
            status = accepted['status']
    except Exception as exc:
        status = 'ERROR'; error = repr(exc); accepted = None
    if time.monotonic() >= work_deadline: status = 'TIMEOUT'; accepted = None
    charged = time.monotonic()-start
    terminal = {'schema':'H2_CAPACITY_SUPERVISED_TERMINAL_V1','invocation':invocation,
        'invocation_sha256':invocation_record['sha256'],
        'checker_stdout_sha256':checker_stdout_sha256,
        'spec_sha256':identity(spec), 'status_before_publication':status, 'stages':stages,
        'candidate':accepted,'error':error,'seconds_before_publication':charged,
        'stage_seconds':sum(s['seconds'] for s in stages),
        'overhead_before_publication_seconds':charged-sum(s['seconds'] for s in stages),
        'positive_requires_final_receipt_and_no_timeout_marker':True}
    terminal_record = save(root/'terminal.json',terminal)
    # Tiny final receipt is still charged, not a free post-budget proof stage.
    receipt = {'schema':'H2_CAPACITY_FINAL_RECEIPT_V1','invocation':invocation,
        'terminal_sha256':terminal_record['sha256'],'status':status,
        'seconds_before_receipt':time.monotonic()-start,'budget':budget,
        'complete_declared_source_proof':status == POSITIVE,
        'real_requests_started':0,'native_float_SAFE':False}
    save(root/'receipt.json',receipt)
    finish = time.monotonic()
    final = {'schema':'H2_CAPACITY_PUBLICATION_FINISH_V1','receipt_sha256':sha(root/'receipt.json'),
             'invocation':invocation,'seconds_before_finish_marker':finish-start,
             'status':status if finish < deadline else 'TIMEOUT'}
    save(root/'finish.json',final)
    finish_hash = sha(root/'finish.json')
    elapsed = time.monotonic()-start
    if elapsed >= budget:
        status = 'TIMEOUT'
        save(root/'publication_timeout.json', {'status':status,'seconds':time.monotonic()-start})
        elapsed = time.monotonic()-start
    return {'status':status,'seconds':elapsed,'root':str(root),
            'invocation':invocation,'finish_sha256':finish_hash,
            'complete_declared_source_proof':status == POSITIVE}


def audit(root, completed_call=None):
    """Accounting, not a mathematical reproof or proof that a live job exited.

    A positive EXECUTION requires an external observer's actual completed API
    return. Files alone cannot close the crash window during final publication.
    """
    root = Path(root); spec = load(root/'spec.json'); validate_spec(spec)
    # Each link hashes and parses the SAME bytes. A completed-call observation
    # anchors the chain; without it this remains only stored-record accounting.
    finish = load(root/'finish.json',None if completed_call is None else required_sha(completed_call.get('finish_sha256')))
    receipt = load(root/'receipt.json',required_sha(finish.get('receipt_sha256')))
    terminal = load(root/'terminal.json',required_sha(receipt.get('terminal_sha256')))
    try: inv = load(root/'invocation.json',required_sha(terminal.get('invocation_sha256')))
    except ValueError as exc: raise ValueError('invocation identity') from exc
    if [v.get('schema') for v in (inv,terminal,receipt,finish)] != [
            'H2_CAPACITY_INVOCATION_V1','H2_CAPACITY_SUPERVISED_TERMINAL_V1','H2_CAPACITY_FINAL_RECEIPT_V1','H2_CAPACITY_PUBLICATION_FINISH_V1']:
        raise ValueError('supervision record schema')
    if (type(inv['budget']) not in (int,float) or not math.isfinite(inv['budget']) or not 0 < inv['budget'] <= 300 or
            not all(type(inv[k]) in (int,float) and math.isfinite(inv[k]) for k in ('start','deadline','work_deadline')) or
            abs(inv['deadline']-inv['start']-inv['budget']) > 1e-7 or
            abs(inv['work_deadline']-(inv['deadline']-min(1.,inv['budget']/5))) > 1e-7):
        raise ValueError('frozen budget/deadline contract')
    if (inv['spec_sha256'] != identity(spec) or terminal['spec_sha256'] != identity(spec) or
            any(v['invocation'] != inv['invocation'] for v in (terminal,receipt,finish)) or
            receipt['budget'] != inv['budget']):
        raise ValueError('terminal identity')
    if (type(inv['rss_limit']) is not int or not 0<inv['rss_limit']<=2*2**30 or
        spec['case']=='full_size' and (inv['budget']!=300 or inv['rss_limit']!=2*2**30)):
        raise ValueError('frozen capacity resource contract')
    stages = terminal['stages']
    if [s['phase'] for s in stages] != list(PHASES[:len(stages)]): raise ValueError('stage prefix')
    end = 0.
    for s in stages:
        if load(root/(s['phase']+'_stage.json')) != s: raise ValueError('stage record identity')
        if any(type(s[k]) not in (int,float) or not math.isfinite(s[k]) for k in ('seconds','start_seconds','end_seconds')):
            raise ValueError('nonfinite stage cost')
        expected_deadline=inv['work_deadline']-min(5.,inv['budget']/4) if s['phase']=='produce' else inv['work_deadline']
        if s.get('deadline_monotonic',expected_deadline) != expected_deadline:
            raise ValueError('worker extended deadline')
        if s['pid'] is not None and 'deadline_monotonic' not in s:
            raise ValueError('missing launched worker deadline')
        if s['status'] not in ('COMPLETED','ERROR','TIMEOUT','RESOURCE_LIMIT'):
            raise ValueError('stage status')
        if s['status']!='COMPLETED' and s is not stages[-1]:
            raise ValueError('continued after failed stage')
        if s['pid'] is None:
            before_launch=(s['status']=='TIMEOUT' and s['seconds']==0)
            launch_failed=(s['status']=='ERROR' and s.get('cleanup_included') is True and
                           isinstance(s.get('error'),str) and bool(s['error']) and 'deadline_monotonic' in s)
            if not (before_launch or launch_failed) or s['returncode'] is not None or s['sampled_peak_rss']!=0:
                raise ValueError('unlaunched stage contract')
        elif (type(s['pid']) is not int or s['pid']<=0 or s.get('cleanup_included') is not True or
              type(s['returncode']) is not int or s['status']=='COMPLETED' and s['returncode']!=0):
            raise ValueError('worker exit and cleanup contract')
        if (type(s['sampled_peak_rss']) is not int or s['sampled_peak_rss']<0 or
            s['status']=='RESOURCE_LIMIT' and s['sampled_peak_rss']<=inv['rss_limit'] or
            s['status']=='COMPLETED' and s['sampled_peak_rss']>inv['rss_limit']):
            raise ValueError('sampled resource contract')
        if not 0 <= end <= s['start_seconds'] <= s['end_seconds'] or s['seconds'] < 0:
            raise ValueError('stage chronology')
        if s['seconds']>s['end_seconds']-s['start_seconds']+1e-8:
            raise ValueError('stage duration exceeds enclosing interval')
        end = s['end_seconds']
    costs = [terminal['seconds_before_publication'],terminal['stage_seconds'],
             terminal['overhead_before_publication_seconds'],receipt['seconds_before_receipt'],
             finish['seconds_before_finish_marker']]
    if any(type(v) not in (int,float) or not math.isfinite(v) or v < 0 for v in costs):
        raise ValueError('finite complete cost')
    if (abs(costs[0]-costs[1]-costs[2]) > 1e-8 or
            abs(costs[1]-sum(s['seconds'] for s in stages)) > 1e-8 or not end <= costs[0] <= costs[3] <= costs[4]):
        raise ValueError('cost accounting')
    status = receipt['status']
    if status not in STATES+('TIMEOUT','ERROR','RESOURCE_LIMIT'): raise ValueError('terminal status')
    if status != terminal['status_before_publication']: raise ValueError('terminal/receipt status')
    if (root/'publication_timeout.json').exists() or costs[4] >= inv['budget']:
        status = 'TIMEOUT'
    elif finish['status'] != status: raise ValueError('finish status')
    if status in STATES:
        if len(stages) != 3 or any(s['status'] != 'COMPLETED' for s in stages): raise ValueError('incomplete stages')
        if stages[1].get('checker_stdout_sha256') != terminal['checker_stdout_sha256']:
            raise ValueError('parent checker-output anchor')
        accepted = receive(root,spec,inv['invocation'],terminal['invocation_sha256'],terminal['checker_stdout_sha256'])
        if accepted != terminal['candidate'] or accepted != load(root/'accepted.json') or accepted['status'] != status:
            raise ValueError('accepted proof receipt')
    if terminal['status_before_publication'] not in STATES and terminal['candidate'] is not None:
        raise ValueError('failed execution cannot accept partial evidence')
    if receipt['real_requests_started']!=0 or receipt['native_float_SAFE'] is not False:
        raise ValueError('capacity guarantee boundary')
    if receipt['complete_declared_source_proof'] != (receipt['status'] == POSITIVE): raise ValueError('claim status')
    confirmed = completed_call is not None
    if confirmed:
        if completed_call.get('complete_declared_source_proof') is not (completed_call.get('status') == POSITIVE):
            raise ValueError('caller proof claim/status')
        if (completed_call['invocation'] != inv['invocation'] or
                completed_call['root'] != str(root) or type(completed_call['seconds']) not in (int,float) or
                not math.isfinite(completed_call['seconds']) or completed_call['seconds'] < costs[4]):
            raise ValueError('external completed-call observation')
        if completed_call['seconds'] >= inv['budget']:
            status = 'TIMEOUT'
        if completed_call['status'] != status: raise ValueError('caller/terminal status')
    return {'status':'PASS','execution_status':status,
            'pipeline_complete':confirmed and status in STATES,
            'all_obligations_checked':confirmed and status in (POSITIVE,'UNKNOWN_NONPOSITIVE'),
            'real_model_admitted':False,'complete_declared_source_proof':confirmed and status == POSITIVE,
            'stored_complete_source_claim':status == POSITIVE,
            'positive_execution_accepted':confirmed and status == POSITIVE,
            'external_completion_observed':confirmed,
            'seconds_before_finish_marker':costs[4], 'new_solves':0, 'mathematical_recheck':False}
