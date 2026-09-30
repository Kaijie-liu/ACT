"""Opt-in SYNTHETIC H2 request supervision. No real-request intake/new holdout.

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
from scoped_source.endpoint_portable import CODE

POSITIVE = 'CHECKED_DECLARED_SOURCE_POSITIVE'
STATES = (POSITIVE, 'UNKNOWN_NONPOSITIVE', 'UNKNOWN_MISSING_EVIDENCE')
PHASES = ('produce', 'check', 'receive')
PROTOCOL_SHA256 = '154e6171c8c2ffad14b6394ae19419827f22184214522a64a27cd3a727a7180a'
CASE_REUSE = {'weighted_sign': [], 'tied_partial_reuse': [[[0,1],1],[[0,2],2],[[2,3],3]],
              'unsafe_tied': [], 'unresolved_sign': []}
PRODUCER_FILES = ('configs/h2_source_controls_20260930.json',
    'scoped_source/endpoint_source_controls.py','scoped_source/endpoint_source_build.py',
    'scoped_source/endpoint_build.py','scoped_source/endpoint_worker.py',
    'scoped_source/endpoint_portable.py','scoped_source/endpoint_supervised.py',
    'scoped_source/sparse_controls.py','scoped_source/endpoint_controls.py',
    'scoped_proof/io.py','scoped_proof/supervisor.py',
    'act/back_end/solver/lp_certificate.py','act/back_end/solver/sparse_lp_certificate.py')


def trusted_checker_sources():
    return {**{'code/'+n:sha(ROOT/n) for n in CODE},
            'verify.py':sha(ROOT/'scoped_source/endpoint_verify.py')}


def bind_checker(root, manifest, expected):
    required = {'code/'+n for n in CODE} | {'verify.py'}
    if set(expected) != required: raise ValueError('trusted checker inventory')
    for name in required:
        if manifest['files'].get(name) != expected[name] or sha(root/'bundle'/name) != expected[name]:
            raise ValueError('trusted mathematical checker identity')


def validate_spec(spec):
    if spec.get('schema') == 'H2_CAPTURE_CONTROL_V1':
        from scoped_source.endpoint_intake import validate_spec as validate_capture
        validate_capture(spec); return
    if (set(spec) != {'schema','case','mode','reuse','source_sha256','control','protocol_sha256'} or
            spec['schema'] != 'H2_SYNTHETIC_SUPERVISION_V1' or
            spec['mode'] not in ('endpoints','mccormick') or spec['case'] not in CASE_REUSE or
            spec['reuse'] != CASE_REUSE[spec['case']] or
            not isinstance(spec['control'], str) or type(spec['source_sha256']) is not str or
            len(spec['source_sha256']) != 64 or any(c not in '0123456789abcdef' for c in spec['source_sha256']) or
            spec['protocol_sha256'] != PROTOCOL_SHA256 or
            identity(load(ROOT/'configs/h2_source_controls_20260930.json')) != PROTOCOL_SHA256):
        raise ValueError('synthetic-only execution specification')
    if spec['control'] not in ('','produce_delay','check_delay','receive_delay','partial_output',
            'exception_after_bundle','missing_certificate','omit_property','wrong_invocation',
            'late_publish','descendant','memory','rebind_checker_context',
            'proposal_delay','proposal_exception','serialization_delay','wrong_mode','missing_endpoint','missing_both',
            'rewrite_check_stdout'):
        raise ValueError('unknown synthetic fault control')
    if spec['control'] in ('missing_endpoint','missing_both') and spec['mode'] != 'endpoints':
        raise ValueError('endpoint-only fault control')
    if spec['control'] == 'rewrite_check_stdout' and spec['mode'] != 'mccormick':
        raise ValueError('MC checker-output corruption control')


def producer_sources(spec):
    names = PRODUCER_FILES
    if spec['schema'] == 'H2_CAPTURE_CONTROL_V1':
        from scoped_source.endpoint_intake import PRODUCER_FILES as CAPTURE_FILES
        names += CAPTURE_FILES
    return {name:sha(ROOT/name) for name in names}


def bind_producer(spec, inv):
    if inv.get('producer_sources') != producer_sources(spec):
        raise ValueError('source producer implementation changed')


def required_sha(value):
    # io.load(None) intentionally means unanchored read. Never let a mandatory
    # chain reference or caller observation silently select that optional mode.
    if (type(value) is not str or len(value) != 64 or
            any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('mandatory SHA-256 identity')
    return value


def receive(root, spec, invocation, invocation_sha256, checker_stdout_sha256):
    required_sha(checker_stdout_sha256); required_sha(invocation_sha256)
    build = load(root/'built.json', limit=1024**2)
    if build['invocation'] != invocation or build['source_sha256'] != spec['source_sha256']:
        raise ValueError('build invocation/source')
    manifest = load(root/'bundle/manifest.json', required_sha(build.get('sha256')), limit=1024**2)
    if (manifest['schema'] != 'H2_PORTABLE_DECLARED_SOURCE_V1' or manifest['invocation'] != invocation or
            manifest['source_sha256'] != spec['source_sha256'] or manifest['mode'] != spec['mode']):
        raise ValueError('manifest context')
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
    if proof['mode'] != spec['mode'] or proof['reuse_requested'] != spec['reuse']:
        raise ValueError('frozen construction arm/facts')
    doc = load(root/'bundle/source.json', manifest['files']['source.json'], limit=64*2**20)
    if identity(doc) != spec['source_sha256']: raise ValueError('receiver source binding')
    checked = load(root/'check.stdout', checker_stdout_sha256, limit=8*2**20)
    if (checked['schema'] != 'H2_PORTABLE_CHECK_RESULT_V1' or
            checked['invocation'] != invocation or checked['source_sha256'] != spec['source_sha256'] or
            checked['manifest_sha256'] != build['sha256'] or checked['mode'] != spec['mode'] or
            checked['isolated'] is not True or checked['site_disabled'] is not True or
            checked['solver_imported'] is not False or checked['producer_imported'] is not False):
        raise ValueError('independent check result binding')
    result = checked['result']; request = doc['request']
    expected = [(list(p), j) for p in combinations(range(request['experts']),2)
                for j in range(request['classes']) if j != request['label']]
    rows = result['duties']
    if (result['source_sha256'] != spec['source_sha256'] or result['required'] != len(expected) or
            result['mode'] != spec['mode'] or result['request_sha256'] != identity(proof['request']) or
            [(r['pair'],r['competitor']) for r in rows] != expected or
            len(proof['request']['duties']) != len(expected) or
            any(type(r['competitor']) is not int or any(type(p) is not int for p in r['pair']) for r in rows) or
            result['deployed_float_SAFE'] is not False or result['declared_real_graph_source_checked'] is not True or
            result['hard_budget_supervision'] is not False or
            result['routing_coverage'] != 'ALL_UNORDERED_TOP2_PAIRS_NO_DUTIES_DROPPED'):
        raise ValueError('complete request coverage/guarantee')
    checked_lps = 0
    for duty, row, origin in zip(proof['request']['duties'], rows, proof['origins']):
        if spec['mode'] == 'endpoints':
            weights = set(map(F, duty['gate']))
            if len(row['endpoint_bounds']) != len(weights): raise ValueError('endpoint coverage')
            values = [None if v is None else F(v) for v in row['endpoint_bounds']]
            lower = None if None in values else min(values)
            if row['lower_bound'] != (None if lower is None else str(lower)):
                raise ValueError('endpoint minimum')
            checked_lps += sum(v is not None for v in values)
        else:
            checked_lps += row['lower_bound'] is not None and origin != 'SOURCE_BOX_REUSE'
        if row['positive'] is not (row['lower_bound'] is not None and F(row['lower_bound']) > F(1,10_000_000)):
            raise ValueError('exact sign/count disagreement')
    positive = sum(r['positive'] for r in rows)
    missing = (sum(v is None for r in rows for v in r['endpoint_bounds']) if spec['mode'] == 'endpoints'
               else sum(r['lower_bound'] is None for r in rows))
    status = POSITIVE if positive == len(expected) else 'UNKNOWN_MISSING_EVIDENCE' if missing else 'UNKNOWN_NONPOSITIVE'
    if (len(proof['origins']) != len(rows) or result['positive'] != positive or result['status'] != status or
            result['missing'] != missing or result['lp_bounds_checked'] != checked_lps or result['origins'] != proof['origins']):
        raise ValueError('whole-request aggregation disagreement')
    accepted = {'schema':'H2_ACCEPTED_CHECK_V1', 'status':status, 'invocation':invocation,
            'source_sha256':spec['source_sha256'], 'manifest_sha256':build['sha256'],
            'checker_stdout_sha256':checker_stdout_sha256, 'required':len(expected),
            'positive':positive, 'missing':missing, 'mode':spec['mode'], 'result':result}
    if spec['schema'] == 'H2_CAPTURE_CONTROL_V1':
        from scoped_source.endpoint_intake import receipt
        captured = load(root/'model_intake.json',limit=1024**2)
        if captured != receipt(doc,invocation,inv['producer_sources']):
            raise ValueError('captured model/source receipt binding')
        accepted['model_capture_receipt_content_sha256'] = identity(captured)
    return accepted


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
    invocation_record = save(root/'invocation.json', {'schema':'H2_INVOCATION_V1', 'invocation':invocation,
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
                '-m','scoped_source.endpoint_worker', phase, str(root), '--deadline',str(phase_deadline),
                '--invocation-sha',invocation_record['sha256']]
            if phase == 'receive':
                if checker_stdout_sha256 is None: raise ValueError('checker output not anchored')
                command += ['--checker-stdout-sha',checker_stdout_sha256]
            record = execute(command, root/(phase+'.log'), phase_deadline, env, rss_limit)
            # Anchor actual output after the owned checker has exited, before a
            # receiving worker can modify it. Never take this hash from a worker.
            if phase == 'check' and record['status'] == 'COMPLETED':
                path = root/'check.stdout'
                if path.is_symlink() or path.stat().st_size > 8*2**20:
                    raise ValueError('invalid checker output file')
                checker_stdout_sha256 = sha(path)
                record['checker_stdout_sha256'] = checker_stdout_sha256
            record.update(phase=phase, start_seconds=before-start, end_seconds=time.monotonic()-start)
            stages.append(record); save(root/(phase+'_stage.json'),record)
            load(root/'invocation.json',invocation_record['sha256'])
            if record['status'] != 'COMPLETED': status = record['status']; break
        else:
            accepted = load(root/'accepted.json', limit=8*2**20)
            if (accepted['invocation'] != invocation or accepted['source_sha256'] != spec['source_sha256'] or
                    accepted['status'] not in STATES or
                    accepted['checker_stdout_sha256'] != checker_stdout_sha256 or
                    sha(root/'check.stdout') != checker_stdout_sha256):
                raise ValueError('accepted receipt identity/status')
            status = accepted['status']
    except Exception as exc:
        status = 'ERROR'; error = repr(exc); accepted = None
    if time.monotonic() >= work_deadline: status = 'TIMEOUT'; accepted = None
    charged = time.monotonic()-start
    terminal = {'schema':'H2_SUPERVISED_TERMINAL_V1','invocation':invocation,
        'invocation_sha256':invocation_record['sha256'],
        'checker_stdout_sha256':checker_stdout_sha256,
        'spec_sha256':identity(spec), 'status_before_publication':status, 'stages':stages,
        'candidate':accepted,'error':error,'seconds_before_publication':charged,
        'stage_seconds':sum(s['seconds'] for s in stages),
        'overhead_before_publication_seconds':charged-sum(s['seconds'] for s in stages),
        'positive_requires_final_receipt_and_no_timeout_marker':True}
    terminal_record = save(root/'terminal.json',terminal)
    # Tiny final receipt is still charged, not a free post-budget proof stage.
    receipt = {'schema':'H2_FINAL_RECEIPT_V1','invocation':invocation,
        'terminal_sha256':terminal_record['sha256'],'status':status,
        'seconds_before_receipt':time.monotonic()-start,'budget':budget,
        'complete_declared_source_proof':status == POSITIVE,
        'real_requests_started':0,'native_float_SAFE':False}
    save(root/'receipt.json',receipt)
    finish = time.monotonic()
    final = {'schema':'H2_PUBLICATION_FINISH_V1','receipt_sha256':sha(root/'receipt.json'),
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
    if (type(inv['budget']) not in (int,float) or not math.isfinite(inv['budget']) or not 0 < inv['budget'] <= 300 or
            not all(type(inv[k]) in (int,float) and math.isfinite(inv[k]) for k in ('start','deadline','work_deadline')) or
            abs(inv['deadline']-inv['start']-inv['budget']) > 1e-7 or
            abs(inv['work_deadline']-(inv['deadline']-min(1.,inv['budget']/5))) > 1e-7):
        raise ValueError('frozen budget/deadline contract')
    if (inv['spec_sha256'] != identity(spec) or terminal['spec_sha256'] != identity(spec) or
            any(v['invocation'] != inv['invocation'] for v in (terminal,receipt,finish)) or
            receipt['budget'] != inv['budget']):
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
        if stages[1].get('checker_stdout_sha256') != terminal['checker_stdout_sha256']:
            raise ValueError('parent checker-output anchor')
        accepted = receive(root,spec,inv['invocation'],terminal['invocation_sha256'],terminal['checker_stdout_sha256'])
        if accepted != terminal['candidate'] or accepted != load(root/'accepted.json') or accepted['status'] != status:
            raise ValueError('accepted proof receipt')
    if receipt['complete_declared_source_proof'] != (receipt['status'] == POSITIVE): raise ValueError('claim status')
    confirmed = completed_call is not None
    if confirmed:
        if (completed_call['invocation'] != inv['invocation'] or
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
