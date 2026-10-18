"""Closed finite midpoint control contract. No execution approvals are granted here."""
import ast
from pathlib import Path
import time

from scoped_source.hz_request_store import POLICY, identity, require, sha, digest, decode, compact
from scoped_source.native_protocol import (approval_registry, validate_profile, read_json,
    closure, REPORT_CAP, LOG_CAP, META_CAPS, META_RESERVE, META_MEMBERS)
from scoped_source.native_method import method
from scoped_proof.io import ROOT

REGISTRY = ROOT/'docs/hz_endpoint_execution_approvals.json'
FREEZE_CAP = 128*1024
RECEIPT_CAP = 64*1024
BATCH_META_CAP = 4*REPORT_CAP
FINAL_RESERVE = 2
ORDER = ('components','guarded_baseline','guarded_midpoint','mixed_midpoint',
         'mixed_baseline','midpoint_block','partial_candidate')
SEEDS = ('scoped_source/endpoint_batch.py','scoped_source/endpoint_component_supervised.py',
         'scoped_source/endpoint_batch_observer.py','scoped_proof/owned_session.py',
         'scripts/endpoint_midpoint_caller.py',
         'scripts/run_endpoint_midpoint_controls.py','scripts/audit_endpoint_midpoint_controls.py',
         'scripts/endpoint_midpoint_components.py','scoped_source/scaled_endpoint_controls.py')
LIMITS = dict(seconds=300,cumulative_seconds=2100,final_reserve_seconds=FINAL_RESERVE,
    as_bytes=8*2**30,rss_bytes=2*2**30,file_bytes=128*2**20,
    run_seconds=298,cleanup_seconds=298.5,native_receive_seconds=299,
    receipt_bytes=RECEIPT_CAP,freeze_bytes=FREEZE_CAP,batch_metadata_bytes=BATCH_META_CAP,
    batch_metadata_members=4,batch_run_seconds=2098,batch_cleanup_seconds=2098.5,
    batch_log_bytes=16384,batch_session_members=512,
    observation='TRUSTED_CALLER_ENTRY_AND_POST_PUBLICATION_RETURN_V1')
ADMISSION = dict(attempts=1,wait_seconds=60,available_bytes=4*2**30,disk_free_bytes=8*2**30)
COMPONENT_META = {'component-invocation.json':FREEZE_CAP,'controller.log':REPORT_CAP,
    'component.log':LOG_CAP,'component.json':REPORT_CAP,'controller.json':REPORT_CAP,
    'slot.json':RECEIPT_CAP,'components/progress.jsonl':LOG_CAP,
    'components/evidence/external.json':REPORT_CAP,
    'components/portable.stdout':REPORT_CAP,'components/portable.stderr':REPORT_CAP}


def batch_closure():
    pending=list(closure(ROOT))+list(SEEDS); found=set()
    while pending:
        name=pending.pop()
        if name in found: continue
        require(len(found)<256,'bounded endpoint dependency closure')
        p=ROOT/name
        require(not p.is_symlink() and p.is_file() and p.stat().st_nlink==1
                and p.stat().st_size<=128*1024,'bounded endpoint code')
        tree=ast.parse(p.read_bytes(),filename=name); found.add(name)
        for node in ast.walk(tree):
            modules=([a.name for a in node.names] if isinstance(node,ast.Import) else
                     [node.module] if isinstance(node,ast.ImportFrom) else [])
            for module in modules:
                if module and module.split('.')[0] in ('scripts','scoped_source','scoped_proof',
                        'source_enclosure','upstream_source','router_source'):
                    target=module.replace('.','/')+'.py'
                    if (ROOT/target).is_file(): pending.append(target)
    return tuple(sorted(found))


def approved_batch(expected):
    require(not REGISTRY.is_symlink() and REGISTRY.is_file() and REGISTRY.stat().st_nlink==1
            and REGISTRY.stat().st_size<=65536,'bounded external endpoint registry')
    raw=REGISTRY.read_bytes(); require(len(raw)<=65536,'endpoint registry capacity')
    r=decode(raw)
    require(compact(r)==raw.rstrip(b'\n'),'canonical endpoint authority')
    require(set(r)=={'schema','batches'} and r['schema']=='ENDPOINT_BATCH_APPROVALS_V1'
            and type(r['batches']) is list and len(r['batches'])<=16
            and r['batches']==sorted(set(r['batches'])),'closed external batch authority')
    for h in r['batches']: sha(h)
    require(sha(expected) in r['batches'],'no separately approved endpoint batch')


def validate(freeze, expected, *, authorize=True, verify=True, deadline=None):
    """No source generation/import of native code; old approvals cannot run a new batch."""
    require(identity(freeze)==sha(expected),'endpoint freeze identity')
    if authorize: approved_batch(expected)  # Before any setup, source or process side effect.
    require(type(freeze) is dict and set(freeze)=={'schema','limits','admission','policy',
        'records','component','code','native_registry'}
        and freeze['schema']=='ENDPOINT_FINITE_FREEZE_V2','closed finite freeze')
    require(freeze['limits']==LIMITS and freeze['admission']==ADMISSION
            and freeze['policy']==dict(POLICY),'unchanged finite resource contract')
    require(sum(META_CAPS.values())+RECEIPT_CAP<=META_RESERVE and len(META_CAPS)+1<=META_MEMBERS
            and sum(COMPONENT_META.values())<=META_RESERVE and len(COMPONENT_META)<=META_MEMBERS,
            'metadata carved out of original request allowance')
    records=freeze['records']
    require(type(records) is list and len(records)==7
            and [r['id'] for r in records]==list(ORDER),'exact ordered seven positions')
    profiles=[]
    cases=('guarded','guarded','guarded','mixed','mixed','guarded','guarded')
    midpoint=(True,False,True,True,False,True,True)
    common=None; sources={}
    for i,r in enumerate(records):
        require(type(r) is dict and set(r)=={'id','profile','sha256'}
                and identity(r['profile'])==r['sha256'],'bound position profile')
        p=r['profile']
        if authorize: validate_profile(p)
        require(p['method']==method('native_source_row_blocks',endpoint_midpoint=midpoint[i])
                and p['source']==dict(kind='fixed_control',case=cases[i],source=p['source']['source'])
                and p.get('control')==(ORDER[i] if i>=5 else None),'fixed source/method/fault roster')
        require(p['seconds']==300 and p['policy']==dict(POLICY)
                and all(p[k]==LIMITS[k] for k in ('as_bytes','rss_bytes','file_bytes')),'same request caps')
        current=(p['environment'],p['implementation'])
        if common is None: common=current
        require(current==common,'same environment and implementation for both arms')
        h=sha(p['source']['source']); require(sources.get(cases[i],h)==h,'same source per pair')
        sources[cases[i]]=h; profiles.append(p)
    require(records[0]['profile']==records[2]['profile'],'component uses registered midpoint profile')
    b=freeze['component']
    from scripts.endpoint_midpoint_components import IDS
    require(set(b)=={'schema','profile','sources','code','cases'}
            and b['schema']=='ENDPOINT_COMPONENT_BINDING_V1'
            and b['profile']==records[0]['sha256'] and b['cases']==list(IDS)
            and set(b['sources'])=={'guarded','mixed','half'}
            and all(b['sources'][k]==v for k,v in sources.items()),'complete component authority')
    for h in b['sources'].values(): sha(h)
    extra={'scripts/endpoint_midpoint_components.py','scoped_source/scaled_endpoint_controls.py'}
    require(set(b['code'])==set(common[1])|extra,'component dependency authority')
    code=freeze['code']
    require(type(code) is dict and 1<=len(code)<=256
            and all(code.get(n)==h for n,h in common[1].items())
            and all(code.get(n)==h for n,h in b['code'].items()),'one frozen code union')
    for n,h in code.items():
        require(type(n) is str and not Path(n).is_absolute() and '..' not in Path(n).parts
                and str(Path(n))==n,'local code path'); sha(h)
    if authorize: require(freeze['native_registry']==approval_registry()[1],'bound native execution authority')
    if verify:
        require(set(code)==set(batch_closure()),'complete endpoint dependency roster')
        for n,h in code.items():
            if deadline is not None: require(time.monotonic()<deadline,'batch verification deadline')
            require(digest((ROOT/n).read_bytes())==h,'frozen endpoint implementation')
        for name in ('scripts','scoped_source','scoped_proof','source_enclosure','upstream_source','router_source'):
            require(not (ROOT/name/'__init__.py').exists(),'no unbound initializer')
    return freeze
