"""Finite immutable-source endpoint policy for the existing lifecycle engine.

Only CPU synthetic controls are admitted. Execution completion and a checked
declared-real-source result are distinct from a trained/native-float proof.
"""
from fractions import Fraction as F
from pathlib import Path
import time

from scoped_proof.io import ROOT, PYTHON, load, sha, tick
from scoped_source.hz_request_store import identity, require
from scoped_source.check_store_endpoints import EndpointWork, MAX_VISITS, ALGORITHM, SCHEMA
from scripts import hz_endpoint_supervised as base

DONE='CHECKED_STORE_CPU_EXECUTION'
WORKER='scripts.hz_store_worker'
DESIGN='docs/hz_store_supervision_design_20261013.md'
LIMIT=4*2**20
FAULTS=('import_delay','source_delay','candidate_delay','serialization_delay',
        'check_delay','receive_delay','producer_exception','checker_exception',
        'invalid_candidate','missing_check')
COUNTERS=('row_visits','coefficient_visits','arithmetic')
cleanup_confirmed=base.cleanup_confirmed
admitted=base.admitted
release_gate=base.release_gate
visible=base.visible


def fixture(case):
    if case=='negative':
        from scripts.test_store_endpoints import fixture as prior
        return prior('negative3_tile8')
    return dict(schema='REQUEST_STORE_SOURCE_V1',lower=['-1'],upper=['1'],
        router=[dict(kind='affine',rows=[['0',[]],['1',[]],['0',[]]])],
        experts=[[dict(kind='affine',rows=[['0',[[0,'1']]]]),dict(kind='relu'),
                  dict(kind='affine',rows=[['2',[[0,'1']]],['0',[[0,'1/32']]]])]
                 for _ in range(3)])


def sources():
    from scripts.test_store_endpoints import FILES
    names=tuple(dict.fromkeys((*base.FILES,*FILES,DESIGN,
        'scripts/hz_store_supervised.py','scripts/hz_store_worker.py',
        'scripts/test_hz_store_supervised.py','scripts/run_hz_store_supervision.py',
        'scripts/run_hz_source_supervision.py','scripts/run_hz_endpoint_supervision.py',
        'scoped_source/hz_store_source_controls.py','scoped_proof/io.py')))
    return {name:sha(ROOT/name) for name in names}


def spec(case='half',control=''):
    require(case in ('half','negative','partial') and control in ('',*FAULTS)
            and (not control or case=='half'), 'frozen store control roster')
    return dict(schema='STORE_CPU_EXECUTION_V1',case=case,control=control,
        design_sha256=sha(ROOT/DESIGN),source=identity(fixture(case)),tile_width=8,
        algorithm=ALGORITHM,mode='checked_source',classes=3 if case=='negative' else 2,
        label=0,margin='0',semantics='selected-softmax-top2',ties='ANY_LEGAL_TOPK')


def validate_spec(s,permit):
    require(permit is None and s==spec(s['case'],s['control']), 'frozen source/property/execution spec')


def budget(s): return 300


def cutoff(inv,phase,begin):
    end=(min(inv['work'],begin+5) if phase in ('admit','release') else
         inv['work']-30 if phase=='produce' else inv['work']-3 if phase=='check' else inv['work'])
    fault=inv['spec']['control']
    target=('check' if fault=='check_delay' else 'receive' if fault=='receive_delay' else 'produce')
    if fault.endswith('_delay') and phase==target: end=min(end,begin+15)
    return end


def worker_command(phase,root,inv_sha,plan_sha):
    args=[str(root),'--invocation-sha',inv_sha,'--plan-sha',plan_sha]
    if phase=='produce': return [PYTHON,'-B','-m',WORKER,*args]
    # Fixed hashed repository modules only; no site/user-site or PYTHONPATH.
    code=f'import sys,runpy;sys.path.insert(0,{str(ROOT)!r});runpy.run_module({WORKER!r},run_name="__main__")'
    return [PYTHON,'-B','-I','-S','-c',code,*args]


def counts(values):
    require(type(values) is dict and set(values)==set(COUNTERS)
            and all(type(v) is int and v>=0 for v in values.values())
            and sum(values.values())<=MAX_VISITS, 'cumulative work counters')
    return dict(values)


def resume_work(deadline, prior):
    w=EndpointWork(deadline); w.counts.update(counts(prior)); return w


def evidence_inventory(root):
    """Bounded hashed inventory, including diagnostic .partial files on failure."""
    basepath=Path(root)/'evidence'; out={}; total=0
    if not basepath.exists(): return out
    require(not basepath.is_symlink(), 'nested evidence root symlink')
    pending=[basepath]; visited=0
    while pending:
        for path in pending.pop().iterdir():
            visited+=1; require(visited<=8192, 'nested inventory entry capacity')
            require(not path.is_symlink(), 'nested evidence symlink')
            if path.is_dir(): pending.append(path); continue
            size=path.stat().st_size; total+=size
            require(path.is_file() and size<=8*2**20 and total<=64*2**20,
                    'bounded nested evidence inventory')
            out[str(path.relative_to(basepath))]=dict(bytes=size,sha256=sha(path))
    return out


def source_payload(root,inv,digest,extra_fields=()):
    """Common source/cost envelope; device policies must add their own checks."""
    p=load(root/'produce.json',base.required_hash(digest),LIMIT); s=inv['spec']
    require(set(p)=={'invocation','spec_sha256','contract','proof','counts','costs','candidate'}|set(extra_fields)
            and p['invocation']==inv['invocation'] and p['spec_sha256']==identity(s), 'producer identity')
    stage=load(root/'produce_stage.json',limit=LIMIT)
    require(stage.get('output_sha256')==digest, 'parent producer receipt')
    contract=load(root/'evidence/contract.json',base.required_hash(p['contract']),8192)
    require(set(contract)=={'schema','store','request','mode','classes','label','margin','semantics','ties'}
            and contract['schema']==SCHEMA and contract['request']==s['source']
            and all(contract[k]==s[k] for k in ('mode','classes','label','margin','semantics','ties')),
            'external declared source and complete property binding')
    base.required_hash(contract['store']); base.required_hash(p['proof']); counts(p['counts'])
    proof=load(root/'evidence/proof/manifest.json',p['proof'],2**20)
    require(proof['tile_width']==s['tile_width'], 'frozen endpoint tile width')
    require(set(p['costs'])=={'imports','source','plan','candidates'}, 'producer cost inventory')
    for value in p['costs'].values(): base.finite(value)
    require(sum(p['costs'].values())<=stage['seconds'], 'producer cost envelope')
    return p


def payload(root,inv,digest):
    p=source_payload(root,inv,digest); s=inv['spec']
    c=p['candidate']
    require(c['algorithm']==s['algorithm'] and c['device']=='cpu' and c['dtype']=='float64'
            and c['iterations_per_tile']==128 and c['proposal_tiles']==3
            and c['proposal_columns']==(6 if s['case']=='negative' else 5), 'candidate algorithm/device')
    return p


def produced(root,inv,stage): payload(root,inv,stage['output_sha256']); return True


def check(root,inv,digest,deadline):
    return check_payload(root,inv,digest,deadline,payload(root,inv,digest))


def check_payload(root,inv,digest,deadline,p):
    from scoped_source.check_store_endpoints import check_directory
    w=resume_work(deadline,p['counts'])
    agg=check_directory(root/'evidence',expected_contract=p['contract'],expected_proof=p['proof'],work=w)
    w.tick()
    return dict(invocation=inv['invocation'],payload_sha256=digest,counts=w.counts,
                result=dict(results=agg['results'],aggregation=agg))


def expected_rows(contract,anchor,s):
    """Independent inventory oracle for this *finite* frozen source roster."""
    rows=[]
    for pair in ((0,1),(0,2),(1,2)):
        weights=([F(1,2)] if s['case']=='negative' or pair==(0,2) else
                 [F(0),F(1,2)] if pair==(0,1) else [F(1,2),F(1)])
        for k in range(1,s['classes']):
            slots=[]
            for t in weights:
                q=[F(int(j==s['label'])-int(j==k)) for j in range(s['classes'])]
                query=dict(store=contract['store'],request=s['source'],pair=list(pair),
                    left=[str(t*v) for v in q],right=[str((1-t)*v) for v in q],offset=str(-F(s['margin'])))
                slots.append(identity(dict(contract=anchor,query=identity(query),other=k,weight=str(t))))
            rows.append((list(pair),k,slots))
    return rows


def receive(root,inv,inputs):
    return receive_payload(root,inv,inputs,payload(root,inv,inputs['produce']))


def receive_payload(root,inv,inputs,p):
    """Aggregate only after the caller validates its complete producer envelope."""
    c=load(root/'check.json',base.required_hash(inputs['check']),LIMIT)
    require(set(c)=={'invocation','payload_sha256','counts','result'} and c['invocation']==inv['invocation']
            and c['payload_sha256']==inputs['produce'], 'independent checker receipt binding')
    after=counts(c['counts'])
    require(all(after[k]>=p['counts'][k] for k in COUNTERS), 'no phase work reset')
    agg=c['result']['aggregation']; rows=c['result']['results']; s=inv['spec']
    contract=load(root/'evidence/contract.json',p['contract'],8192)
    expected=expected_rows(contract,p['contract'],s)
    require(rows==agg['results'] and [(r['pair'],r['other'],r['endpoints']) for r in rows]==expected,
            'complete source pair/property inventory')
    endpoint_counts=[len(slots) for _,_,slots in expected]
    require([len(r['endpoints']) for r in rows]==endpoint_counts
            and len({v for r in rows for v in r['endpoints']})==sum(endpoint_counts), 'distinct complete endpoints')
    positive=sum(r['lower'] is not None and F(r['lower'])>F(1,10000000) for r in rows)
    require(all(r['positive']==(r['lower'] is not None and F(r['lower'])>F(1,10000000)) for r in rows),
            'positive threshold aggregation')
    missing=1 if s['case']=='partial' else 0
    require([i for i,r in enumerate(rows) if r['lower'] is None]==([len(rows)-1] if missing else []),
            'registered missing endpoint duty pattern')
    status=('UNKNOWN_MISSING_EVIDENCE' if missing else 'CHECKED_POSITIVE_DECLARED_SOURCE'
            if positive==len(rows) else 'UNKNOWN_NONPOSITIVE')
    require((agg['required'],agg['required_endpoints'],agg['checked_endpoints'],agg['missing_endpoints'],
             agg['positive'],agg['status'])==(len(rows),sum(endpoint_counts),sum(endpoint_counts)-missing,
                                            missing,positive,status)
            and agg['native_float_proof'] is False and agg['checkpoint_correspondence_checked'] is False,
            'coverage/guarantee aggregation')
    return dict(status=DONE,invocation=inv['invocation'],inputs=inputs,checked=c['result'],counts=after,
                obligations_complete=not missing,declared_source_positive=not missing and status=='CHECKED_POSITIVE_DECLARED_SOURCE',
                complete_moe_proof=False,real_model_proof=False,cuda_execution=False)


def decision(root,inv,stages):
    records={s['phase']:s for s in stages}
    if any(not cleanup_confirmed(s) for s in stages): return 'CLEANUP_INCOMPLETE',None,None
    release=release_gate(root,inv,records) if 'release' in records else None
    for s in stages:
        if s['status']!='COMPLETED': return s['status'],release,None
    if 'admit' in records: admitted(root,inv,records['admit'])
    if 'produce' in records and not (release or {}).get('release_confirmed'): return 'CPU_RELEASE_UNCONFIRMED',release,None
    if list(records)!=['admit','produce','release','check','receive']: return 'ERROR',release,None
    result=receive(root,inv,{k:records[k]['output_sha256'] for k in ('produce','release','check')})
    require(load(root/'receive.json',records['receive']['output_sha256'],LIMIT)==result, 'receiver receipt')
    return DONE,release,result
