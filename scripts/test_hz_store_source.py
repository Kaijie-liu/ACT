"""Execute frozen source/view component controls once, retaining all receipts."""
import argparse
import copy
import json
import os
from pathlib import Path
import resource
import signal
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scoped_source.hz_request_store import Store,Work,compact,digest,identity,require
from scoped_source.hz_store_source import build
from scoped_source.check_hz_store_source import check_source
from scoped_source.hz_store_sparse import instantiate,match
from scoped_source.hz_store_source_controls import fixture,analytic,signature,negative,NEGATIVES,EXPECTED

ROOT=Path(__file__).resolve().parents[1]
CODE=['scoped_source/hz_request_store.py','scoped_source/hz_store_write.py',
      'scoped_source/hz_store_source.py','scoped_source/hz_store_sparse.py',
      'scoped_source/check_hz_store_source.py','scoped_source/hz_store_source_controls.py',
      'scripts/check_hz_store_source.py','scripts/test_hz_store_source.py',
      'act/back_end/solver/solver_hz.py']


def save(path,value):
    with Path(path).open('xb') as f: f.write(compact(value))


def stage_bytes(parent):
    return sum(p.stat().st_size for root in parent.glob('hz_store_source_20261012_*')
               for p in root.rglob('*') if p.is_file())


def live_negatives(store,work):
    view=store.views['router/1']; reports=[]
    for key in ('c','Gc','Gb','Ac','Ab','b','Auc','Aub','ub','map','frame','exact',
                'cached_duplicate','trailing_data','bad_pointer'):
        start=time.monotonic(); live,cs,bs=instantiate(store.factors,view,work)
        import numpy as np
        if key in ('c','b','ub'):
            getattr(live,key)[0]+=1
        elif key in ('Gc','Gb','Ac','Ab','Auc','Aub'):
            m=getattr(live,key)
            if len(m.data): m.data[0]+=1
            else:
                # Gb has zero output nnz but retains all binary columns.
                import scipy.sparse as sp
                setattr(live,key,sp.csr_matrix(([1.],([0],[0])),shape=m.shape))
        elif key=='map': cs=tuple(reversed(cs))
        elif key=='frame': live.frame_id+=1
        elif key=='exact': live.exact=True
        elif key=='cached_duplicate':
            m=live.Ac; require(m.indptr[1]>=2,'duplicate fixture width')
            _=m.has_canonical_format
            m.indices[1]=m.indices[0]
        elif key=='trailing_data':
            live.Gc.data=np.append(live.Gc.data,1.); live.Gc.indices=np.append(live.Gc.indices,0)
        else: live.Gc.indptr[0]=1
        try: match(store.factors,view,live,cs,bs,work)
        except ValueError as ex:
            expected=('live RHS mismatch' if key in ('c','b','ub') else
                'live coefficient mismatch' if key in ('Gc','Gb','Ac','Ab','Auc','Aub') else
                'live factor bijection/order' if key=='map' else
                'live frame/enclosure flag' if key in ('frame','exact') else
                'live coefficient/column' if key=='cached_duplicate' else 'live CSR shape/dtype/canonical')
            require(str(ex)==expected,f'wrong live refusal: {key}: {ex}')
            reports.append(dict(name=key,reason=str(ex),seconds=time.monotonic()-start))
        else: raise AssertionError(f'live mutation accepted: {key}')
    return reports


def run(output):
    start=time.monotonic(); output.mkdir(parents=True,exist_ok=False)
    report=dict(status='INCOMPLETE',normal=[],negative=[],live_negative=[],other=[],code={},error=None)
    def capacity():
        amount=stage_bytes(output.parent)
        require(amount<=63*2**20,'source component archive cap/terminal reserve')
        return amount
    try:
        require(capacity()+24*2**20<=63*2**20,'projected complete source suite admission')
        snap=output/'executed_code'
        for name in CODE:
            data=(ROOT/name).read_bytes(); report['code'][name]=digest(data)
            target=snap/name; target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write(data)
        stores={}; docs={}
        for recipe,page in (('wide',64),('mixed',64),('mixed',1),('input',64)):
            begin=time.monotonic(); work=Work(min(begin+290,start+290))
            name=f'{recipe}_page{page}'; doc=fixture(recipe); anchor=identity(doc)
            created=time.monotonic()
            save(output/f'{name}_source.json',doc)
            declared=time.monotonic()
            manifest,cost=build(output/name,doc,work=work,page_rows=page)
            loaded=time.monotonic()
            store=Store(output/name,expected_manifest=manifest,expected_request=anchor,work=work)
            parsed=time.monotonic(); result=check_source(store,doc,expected_source=anchor,work=work)
            checked=time.monotonic(); oracle=analytic(store,recipe,work); audited=time.monotonic()
            matches=[]
            for view in store.views.values():
                live,cs,bs=instantiate(store.factors,view,work)
                matches.append(dict(view=view.name,**match(store.factors,view,live,cs,bs,work)))
                del live
            matched=time.monotonic(); sig=signature(store)
            row=dict(name=name,recipe=recipe,page=page,source=anchor,manifest=manifest,result=result,
                     oracle=oracle,signature=sig,matches=matches,stats=dict(store.stats),
                     work_counts=dict(work.counts),cost=dict(**cost,load_seconds=parsed-loaded,
                         declaration_creation_seconds=created-begin,declaration_serialization_seconds=declared-created,
                         source_check_seconds=checked-parsed,oracle_seconds=audited-checked,
                         live_instantiation_and_matching_seconds=matched-audited,signature_seconds=time.monotonic()-matched))
            save(output/f'{name}_receipt.json',row)
            report['other'].append(dict(name=f'publication_{name}',seconds_through_receipt=time.monotonic()-begin,
                                        receipt_sha256=digest((output/f'{name}_receipt.json').read_bytes())))
            report['normal'].append(row); stores[name]=store; docs[name]=doc; capacity()
        require(report['normal'][1]['signature']==report['normal'][2]['signature'],'page size mathematical differential')
        for name in NEGATIVES:
            begin=time.monotonic(); work=Work(start+290)
            base='wide_page64' if name.startswith(('input_','affine_')) else 'mixed_page64'
            try: negative(stores[base],docs[base],name,work)
            except (ValueError,TimeoutError) as ex:
                require(isinstance(ex,TimeoutError)==(name=='source_deadline_tail'), 'wrong refusal category')
                require(str(ex)==EXPECTED[name],f'wrong source refusal {name}: {ex}')
                report['negative'].append(dict(name=name,base=base,reason=str(ex),
                    seconds=time.monotonic()-begin,work_counts=dict(work.counts)))
            else: raise AssertionError(f'accepted source mutation {name}')
        report['live_negative']=live_negatives(stores['mixed_page64'],Work(start+290))
        # Mutating a caller declaration after capture cannot alter the proof read.
        original=fixture('input'); expected=identity(original)
        class AliasWork(Work):
            def tick(self):
                super().tick()
                if hasattr(self,'armed') and self.armed:
                    original['lower'][0]='999'; self.armed=False
        work=AliasWork(start+290); work.armed=True
        alias=check_source(stores['input_page64'],original,expected_source=expected,work=work)
        require(original['lower'][0]=='999' and alias['source']==expected,'private declaration capture')
        report['other'].append(dict(name='caller_alias',result=alias))
        # Standard-library relocation; copy only a checker, its store and source.
        for name in ('wide_page64','mixed_page64'):
            begin=time.monotonic(); target=output/f'portable_{name}'; target.mkdir()
            for src in ('scripts/check_hz_store_source.py','scoped_source/hz_request_store.py',
                        'scoped_source/check_hz_store_source.py'):
                dest=target/src; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(ROOT/src,dest)
            shutil.copytree(output/name,target/'object'); shutil.copyfile(output/f'{name}_source.json',target/'source.json')
            old=next(x for x in report['normal'] if x['name']==name)
            command=[sys.executable,'-B','-I','-S',str(target/'scripts/check_hz_store_source.py'),
                     str(target/'object'),str(target/'source.json'),'--store',old['manifest'],'--source',old['source']]
            completed=subprocess.run(command,capture_output=True,text=True,timeout=max(0.1,start+290-time.monotonic()),cwd=target)
            save(target/'process.json',dict(argv=command,returncode=completed.returncode,stdout=completed.stdout,stderr=completed.stderr))
            require(completed.returncode==0,'relocated source checker process')
            result=json.loads(completed.stdout)
            require(not result.pop('forbidden_modules') and result.pop('work_counts')
                    and result==old['result'],'relocated complete source equality')
            report['other'].append(dict(name=f'portable_{name}',seconds=time.monotonic()-begin,result=result))
            capacity()
        report['status']='PASS_SOURCE_VIEW_CONTROLS_ONLY'
    except BaseException as ex:
        report['status']='FAILED'; report['error']=dict(type=type(ex).__name__,message=str(ex))
        raise
    finally:
        report['seconds_before_report']=time.monotonic()-start
        report['peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report['stage_bytes_before_report']=stage_bytes(output.parent)
        save(output/'report.json',report)
        save(output/'terminal.json',dict(status=report['status'],report_sha256=digest((output/'report.json').read_bytes()),
            seconds_through_report=time.monotonic()-start,stage_bytes_through_report=stage_bytes(output.parent)))


def observed(output):
    """Control-harness wall receipt, not the production request supervisor."""
    begin=time.monotonic(); deadline=begin+300
    require(not output.exists(),'new attempt required')
    require(stage_bytes(output.parent)+24*2**20<=63*2**20,'parent projected archive admission')
    command=[sys.executable,'-B',str(Path(__file__).resolve()),str(output),'--child']
    proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
    timed_out=False; released=False; out=err=''
    try:
        out,err=proc.communicate(timeout=max(.01,deadline-time.monotonic())); released=True
    except subprocess.TimeoutExpired:
        timed_out=True
        try: os.killpg(proc.pid,signal.SIGKILL)
        except ProcessLookupError: pass
        try: out,err=proc.communicate(timeout=5); released=True
        except subprocess.TimeoutExpired: pass
    returned=time.monotonic()
    output.mkdir(parents=True,exist_ok=True)
    status='ERROR'
    if timed_out or returned>deadline: status='TIMEOUT'
    elif released and proc.returncode==0: status='COMPLETE'
    receipt=dict(status=status,returncode=proc.returncode,leader_reaped=released,
                 timed_out=timed_out,seconds_spawn_through_return=returned-begin,
                 seconds_budget=300,argv=command,stdout=out,stderr=err,
                 report_sha256=digest((output/'report.json').read_bytes()) if (output/'report.json').exists() else None)
    save(output/'outer.json',receipt)
    save(output/'outer_terminal.json',dict(status=status,outer_sha256=digest((output/'outer.json').read_bytes()),
         seconds_through_receipt=time.monotonic()-begin,stage_bytes=stage_bytes(output.parent)))
    require(status=='COMPLETE','source control process incomplete; preserve prefix and stop')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--child',action='store_true');a=p.parse_args()
    run(a.output) if a.child else observed(a.output)
