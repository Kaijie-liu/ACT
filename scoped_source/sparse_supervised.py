"""Opt-in SYNTHETIC H1 request supervision. No real-request intake/new holdout.

Single API-entry clock; worker startup, all imports, fixture loading/creation,
construction/proposals, packaging, checking, reception and cleanup are charged.
Final bookkeeping is charged and can invalidate success. The OS may delay
cleanup/publication beyond the deadline; no strict wall-time return guarantee.
"""
from fractions import Fraction as F
from itertools import combinations
import math
import os
from pathlib import Path
import time
import uuid
from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof.supervisor import execute
from source_enclosure.format import identity
from scoped_source.sparse_portable import CODE

POSITIVE = 'CHECKED_DECLARED_SOURCE_POSITIVE'
STATES = (POSITIVE, 'UNKNOWN_NONPOSITIVE', 'UNKNOWN_MISSING_EVIDENCE')
PHASES = ('produce', 'check', 'receive')


def trusted_checker_sources():
    return {**{'code/'+n:sha(ROOT/n) for n in CODE},
            'verify.py':sha(ROOT/'scoped_source/sparse_verify.py')}


def bind_checker(root, manifest, expected):
    required = {'code/'+n for n in CODE} | {'verify.py'}
    if set(expected) != required: raise ValueError('trusted checker inventory')
    for name in required:
        if manifest['files'].get(name) != expected[name] or sha(root/'bundle'/name) != expected[name]:
            raise ValueError('trusted mathematical checker identity')


def validate_spec(spec):
    if (set(spec) != {'schema','fixture','mode','reuse','source_sha256','control'} or
            spec['schema'] not in ('H1_SYNTHETIC_SUPERVISION_V1','H1_CAPTURE_CONTROL_V1') or
            spec['mode'] not in ('dependency','full') or
            not isinstance(spec['fixture'], dict) or not isinstance(spec['control'], str) or
            len(spec['source_sha256']) != 64):
        raise ValueError('synthetic-only execution specification')
    if spec['schema'] == 'H1_CAPTURE_CONTROL_V1':
        from scoped_source.sparse_intake import validate_fixture
        validate_fixture(spec['fixture'])
    else:
        allowed = {'experts','classes','width','tied','unsafe','dense','relu_bias','constant','relational','router_coordinate'}
        if not set(spec['fixture']) <= allowed: raise ValueError('not a synthetic fixture option')
        if any(type(spec['fixture'].get(k, default)) is not int or not lo <= spec['fixture'].get(k, default) <= hi
               for k, default, lo, hi in [('experts',3,2,5), ('classes',3,2,5), ('width',3,1,16)]):
            raise ValueError('bounded synthetic fixture only')
    if spec['control'] not in ('','produce_delay','check_delay','receive_delay','partial_output',
            'exception_after_bundle','missing_certificate','omit_property','wrong_invocation',
            'late_publish','descendant','memory','rebind_checker_context',
            'capture_delay','capture_exception','mutate_model'):
        raise ValueError('unknown synthetic fault control')
    if spec['control'] in ('capture_delay','capture_exception','mutate_model') and spec['schema'] != 'H1_CAPTURE_CONTROL_V1':
        raise ValueError('capture-specific fault control')


def producer_sources(spec):
    if spec['schema'] != 'H1_CAPTURE_CONTROL_V1': return {}
    from scoped_source.sparse_intake import PRODUCER_FILES
    return {name:sha(ROOT/name) for name in PRODUCER_FILES}


def bind_producer(spec, inv):
    if spec['schema'] == 'H1_CAPTURE_CONTROL_V1' and inv.get('producer_sources') != producer_sources(spec):
        raise ValueError('captured-object producer implementation changed')


def receive(root, spec, invocation, invocation_sha256):
    build = load(root/'built.json', limit=1024**2)
    if build['invocation'] != invocation or build['source_sha256'] != spec['source_sha256']:
        raise ValueError('build invocation/source')
    manifest = load(root/'bundle/manifest.json', build['sha256'], limit=1024**2)
    if (manifest['schema'] != 'H1_PORTABLE_DECLARED_SOURCE_V1' or manifest['invocation'] != invocation or
            manifest['source_sha256'] != spec['source_sha256']): raise ValueError('manifest context')
    inv = load(root/'invocation.json',invocation_sha256)
    if inv['invocation'] != invocation: raise ValueError('receiver invocation')
    bind_producer(spec,inv)
    bind_checker(root,manifest,inv['checker_sources'])
    required_files = {'source.json','proof.json','verify.py'} | {'code/'+n for n in CODE}
    if set(manifest['files']) != required_files: raise ValueError('bundle file coverage')
    for name in required_files:
        path = root/'bundle'/name
        if path.is_symlink() or sha(path) != manifest['files'][name]: raise ValueError('changed checked artifact')
    proof = load(root/'bundle/proof.json', manifest['files']['proof.json'], limit=64*2**20)
    if proof['mode'] != spec['mode'] or proof['omitted_blocks'] != []: raise ValueError('frozen construction arm')
    doc = load(root/'bundle/source.json', manifest['files']['source.json'], limit=64*2**20)
    if identity(doc) != spec['source_sha256']: raise ValueError('receiver source binding')
    if spec['schema'] == 'H1_CAPTURE_CONTROL_V1':
        capture = load(root/'captured_source_identity.json',limit=1024**2)
        if capture != {'source_sha256':spec['source_sha256'],'model_state':doc['request']['model_state'],
                'request':doc['request'],'invocation':invocation,'producer_sources':inv['producer_sources'],
                'native_float_proof':False,'real_requests_started':0}:
            raise ValueError('captured object receipt binding')
    checked = load(root/'check.stdout', limit=8*2**20)
    if (checked['schema'] != 'H1_PORTABLE_CHECK_RESULT_V1' or
            checked['invocation'] != invocation or checked['source_sha256'] != spec['source_sha256'] or
            checked['manifest_sha256'] != build['sha256'] or
            checked['isolated'] is not True or checked['site_disabled'] is not True or
            checked['solver_imported'] is not False or checked['producer_imported'] is not False):
        raise ValueError('independent check result binding')
    result = checked['result']; request = doc['request']
    expected = [(list(p), j) for p in combinations(range(request['experts']),2)
                for j in range(request['classes']) if j != request['label']]
    rows = result['obligations']
    if (result['source_sha256'] != spec['source_sha256'] or result['required'] != len(expected) or
            [(r['pair'],r['competitor']) for r in rows] != expected or
            result['deployed_float_SAFE'] is not False or result['route_change_witness_checked'] is not False or
            result['routing_coverage'] != 'ALL_UNORDERED_TOP2_PAIRS_NO_EXCLUSIONS'):
        raise ValueError('complete request coverage/guarantee')
    for row in rows:
        if row['positive'] is not (row['lower_bound'] is not None and F(row['lower_bound']) > F(1,10_000_000)):
            raise ValueError('exact sign/count disagreement')
    positive = sum(r['positive'] for r in rows)
    missing = sum(r['lower_bound'] is None for r in rows)
    status = POSITIVE if positive == len(expected) else 'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE'
    if result['positive'] != positive or result['status'] != status:
        raise ValueError('whole-request aggregation disagreement')
    return {'schema':'H1_ACCEPTED_CHECK_V1', 'status':status, 'invocation':invocation,
            'source_sha256':spec['source_sha256'], 'manifest_sha256':build['sha256'],
            'checker_stdout_sha256':sha(root/'check.stdout'), 'required':len(expected),
            'positive':positive, 'missing':missing, 'result':result}


def supervise(root, spec, *, budget=300., rss_limit=2*2**30):
    start = time.monotonic()
    if (type(budget) not in (int,float) or not math.isfinite(budget) or not 0 < budget <= 300 or
            type(rss_limit) is not int or rss_limit <= 0): raise ValueError('request budget')
    validate_spec(spec); root = Path(root)
    if not root.is_absolute() or not root.resolve().is_relative_to(Path('/data1/Kane/MOE')):
        raise ValueError('output outside project')
    root.mkdir(parents=True, exist_ok=False)
    deadline = start+budget; work_deadline = deadline-min(1.,budget/5)
    invocation = uuid.uuid4().hex
    save(root/'spec.json', spec)
    invocation_record = save(root/'invocation.json', {'schema':'H1_INVOCATION_V1', 'invocation':invocation,
        'spec_sha256':identity(spec), 'start':start, 'deadline':deadline,
        'work_deadline':work_deadline, 'budget':budget, 'rss_limit':rss_limit,
        'checker_sources':trusted_checker_sources(),'producer_sources':producer_sources(spec)})
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    stages = []; accepted = None; status = 'ERROR'; error = None
    try:
        for phase in PHASES:
            anchored_inv = load(root/'invocation.json',invocation_record['sha256'])
            bind_producer(spec,anchored_inv)
            before = time.monotonic()
            phase_deadline = work_deadline-min(5.,budget/4) if phase == 'produce' else work_deadline
            command = [PYTHON,'-B']+(['-S'] if phase != 'produce' else [])+[
                '-m','scoped_source.sparse_worker', phase, str(root), '--deadline',str(phase_deadline),
                '--invocation-sha',invocation_record['sha256']]
            record = execute(command, root/(phase+'.log'), phase_deadline, env, rss_limit)
            record.update(phase=phase, start_seconds=before-start, end_seconds=time.monotonic()-start)
            stages.append(record); save(root/(phase+'_stage.json'),record)
            load(root/'invocation.json',invocation_record['sha256'])
            if record['status'] != 'COMPLETED': status = record['status']; break
        else:
            accepted = load(root/'accepted.json', limit=8*2**20)
            if (accepted['invocation'] != invocation or accepted['source_sha256'] != spec['source_sha256'] or
                    accepted['status'] not in STATES): raise ValueError('accepted receipt identity/status')
            status = accepted['status']
    except Exception as exc:
        status = 'ERROR'; error = repr(exc); accepted = None
    if time.monotonic() >= work_deadline: status = 'TIMEOUT'; accepted = None
    charged = time.monotonic()-start
    terminal = {'schema':'H1_SUPERVISED_TERMINAL_V1','invocation':invocation,
        'invocation_sha256':invocation_record['sha256'],
        'spec_sha256':identity(spec), 'status_before_publication':status, 'stages':stages,
        'candidate':accepted,'error':error,'seconds_before_publication':charged,
        'stage_seconds':sum(s['seconds'] for s in stages),
        'overhead_before_publication_seconds':charged-sum(s['seconds'] for s in stages),
        'positive_requires_final_receipt_and_no_timeout_marker':True}
    terminal_record = save(root/'terminal.json',terminal)
    # Tiny final receipt is still charged, not a free post-budget proof stage.
    receipt = {'schema':'H1_FINAL_RECEIPT_V1','invocation':invocation,
        'terminal_sha256':terminal_record['sha256'],'status':status,
        'seconds_before_receipt':time.monotonic()-start,'budget':budget,
        'complete_declared_source_proof':status == POSITIVE,
        'real_requests_started':0,'native_float_SAFE':False}
    save(root/'receipt.json',receipt)
    finish = time.monotonic()
    final = {'schema':'H1_PUBLICATION_FINISH_V1','receipt_sha256':sha(root/'receipt.json'),
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
    inv = load(root/'invocation.json'); terminal = load(root/'terminal.json')
    if sha(root/'invocation.json') != terminal['invocation_sha256']: raise ValueError('invocation identity')
    receipt = load(root/'receipt.json'); finish = load(root/'finish.json')
    if (type(inv['budget']) not in (int,float) or not math.isfinite(inv['budget']) or not 0 < inv['budget'] <= 300 or
            not all(type(inv[k]) in (int,float) and math.isfinite(inv[k]) for k in ('start','deadline','work_deadline')) or
            abs(inv['deadline']-inv['start']-inv['budget']) > 1e-7 or
            abs(inv['work_deadline']-(inv['deadline']-min(1.,inv['budget']/5))) > 1e-7):
        raise ValueError('frozen budget/deadline contract')
    if (inv['spec_sha256'] != identity(spec) or terminal['spec_sha256'] != identity(spec) or
            any(v['invocation'] != inv['invocation'] for v in (terminal,receipt,finish)) or
            receipt['terminal_sha256'] != sha(root/'terminal.json') or
            finish['receipt_sha256'] != sha(root/'receipt.json') or receipt['budget'] != inv['budget']):
        raise ValueError('terminal identity')
    stages = terminal['stages']
    if [s['phase'] for s in stages] != list(PHASES[:len(stages)]): raise ValueError('stage prefix')
    end = 0.
    for s in stages:
        if load(root/(s['phase']+'_stage.json')) != s: raise ValueError('stage record identity')
        if any(type(s[k]) not in (int,float) or not math.isfinite(s[k]) for k in ('seconds','start_seconds','end_seconds')):
            raise ValueError('nonfinite stage cost')
        if s.get('deadline_monotonic',inv['work_deadline']) > inv['work_deadline']:
            raise ValueError('worker extended deadline')
        if not 0 <= end <= s['start_seconds'] <= s['end_seconds'] or s['seconds'] < 0:
            raise ValueError('stage chronology')
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
        accepted = receive(root,spec,inv['invocation'],terminal['invocation_sha256'])
        if accepted != terminal['candidate'] or accepted != load(root/'accepted.json') or accepted['status'] != status:
            raise ValueError('accepted proof receipt')
    if receipt['complete_declared_source_proof'] != (receipt['status'] == POSITIVE): raise ValueError('claim status')
    confirmed = completed_call is not None
    if confirmed:
        if (completed_call['invocation'] != inv['invocation'] or completed_call['finish_sha256'] != sha(root/'finish.json') or
                completed_call['root'] != str(root) or type(completed_call['seconds']) not in (int,float) or
                not math.isfinite(completed_call['seconds']) or completed_call['seconds'] < costs[4]):
            raise ValueError('external completed-call observation')
        if completed_call['seconds'] >= inv['budget']:
            status = 'TIMEOUT'
        if status != 'TIMEOUT' and completed_call['status'] != status: raise ValueError('caller/terminal status')
    return {'status':'PASS','execution_status':status,'complete_declared_source_proof':confirmed and status == POSITIVE,
            'stored_complete_source_claim':status == POSITIVE,
            'positive_execution_accepted':confirmed and status == POSITIVE,
            'external_completion_observed':confirmed,
            'seconds_before_finish_marker':costs[4], 'new_solves':0, 'mathematical_recheck':False}
