"""Frozen controls and retained receipts, not a numerical model experiment."""
import argparse
import copy
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scoped_source.hz_request_store import (Store, Work, POLICY, compact, digest,
    identity, require, evaluate, check_candidate)
from scoped_source.hz_store_control_build import build
from scoped_source.check_hz_store_control import inspect, records, oracle


def save(path, value):
    with Path(path).open('xb') as f: f.write(compact(value))


def replace(path, value):
    # Control mutation, never called on a frozen parent or normal archive.
    with Path(path).open('wb') as f: f.write(compact(value))


def add_page(root,m,kind,rows):
    raw = compact(dict(kind=kind,rows=rows)); h = digest(raw); name = f'rows/{h}.json'
    with (root/name).open('xb') as f: f.write(raw)
    m['members'].append(dict(file=name,bytes=len(raw),sha256=h))
    return h


def edit_page(root,m,kind,fn,prefix=None):
    ref = next(r for r in m['members'] if json.loads((root/r['file']).read_bytes())['kind'] == kind
        and (prefix is None or json.loads((root/r['file']).read_bytes())['rows'][0][0].startswith(prefix)))
    page = json.loads((root/ref['file']).read_bytes()); fn(page)
    new = add_page(root,m,kind,page['rows']); old = ref['sha256']
    m['members'].remove(ref); (root/ref['file']).unlink()
    for v in m['views']:
        for k in ('output','eq','ine'): v[k] = [new if h == old else h for h in v[k]]


def small_expanded(root, pair, nonzero):
    """Small-only flat oracle reads archived bytes, not Store or its iterator."""
    m = json.loads((root/'manifest.json').read_bytes()); views = {v['name']:v for v in m['views']}
    def rows(view,kind):
        return [r for h in view[kind] for r in json.loads((root/f'rows/{h}.json').read_bytes())['rows']]
    n = len(m['factors']); r = [F(0)]*n; constant = F(1,8)
    for expert,weight in zip(pair,(1,-1)):
        for _,rhs,entries in rows(views[f'expert/{expert}'],'output'):
            constant += weight*F(rhs)
            for j,v in entries: r[j] += weight*F(v)
    for kind,dual in (('eq',F(1,64)),('ine',F(-1,32))):
        dual = dual if nonzero else F(0)
        full = rows(views['common'],kind)+rows(views[f'pair/{pair[0]}/{pair[1]}'],kind)
        for i in pair: full += rows(views[f'expert/{i}'],kind)
        for _,rhs,entries in full:
            constant += F(rhs)*dual
            for j,v in entries: r[j] -= F(v)*dual
    active = set(range(4)) | {4+2*i+j for i in pair for j in range(2)}
    return constant-sum(map(abs,r)),tuple((j,r[j]) for j in sorted(active))


class Suite:
    def __init__(self, root):
        self.root = root; root.mkdir(parents=True,exist_ok=False)
        self.started = time.monotonic(); self.normal = []; self.negative = []; self.other = []
        self.status = 'INCOMPLETE'; self.error = None
        self.code = {str(p):digest(p.read_bytes()) for p in [Path(__file__).resolve(),
            *[Path(__file__).resolve().parents[1]/'scoped_source'/n for n in
              ('hz_request_store.py','hz_store_write.py','hz_store_control_build.py','check_hz_store_control.py')]]}

    def envelope(self):
        require(time.monotonic()-self.started <= 1800,'suite deadline')
        size = self.stage_bytes()
        require(size <= 63*2**20,'stage archive byte cap (1 MiB reserved for terminals)')
        return size

    def stage_bytes(self):
        return sum(p.stat().st_size for root in self.root.parent.glob('hz_request_store_20261012_*')
                   for p in root.rglob('*') if p.is_file())

    def run_normal(self,layout,page=64):
        started = time.monotonic(); work = Work(started+300); name = f'{layout}_page{page}'
        target = self.root/name; ids = build(target,large=layout=='large',work=work,page_rows=page)
        built = time.monotonic(); fixed = records(layout,**ids); save(self.root/f'{name}_queries.json',fixed)
        out = inspect(target,**ids,layout=layout,work=work,saved_records=fixed)
        for pair in combinations(range(3),2) if layout == 'small' else ():
            for nz in (False,True): require(small_expanded(target,pair,nz)==oracle(layout,pair,nz),'small expanded differential')
        work.tick(); receipt = dict(name=name,ids=ids,report=out,
            generation_seconds=built-started,total_seconds=time.monotonic()-started,
            archive_bytes=self.envelope())
        self.normal.append(receipt); save(self.root/f'{name}_receipt.json',receipt)
        return receipt

    def mutate(self,name,fn,expected,*,stale=False,raw=False):
        started=time.monotonic(); work=Work(started+300); target=self.root/name
        shutil.copytree(self.root/'small_page64',target)
        m=json.loads((target/'manifest.json').read_bytes()); fn(target,m)
        if not raw:
            m['members'].sort(key=lambda r:r['sha256']); replace(target/'manifest.json',m)
        actual=digest((target/'manifest.json').read_bytes())
        anchor=self.normal[0]['ids']['manifest'] if stale else actual
        request=self.normal[0]['ids']['request']
        try: Store(target,expected_manifest=anchor,expected_request=request,work=work)
        except ValueError as ex:
            require(expected in str(ex),f'{name}: unexpected refusal {ex}'); reason=str(ex)
        else: raise AssertionError(f'{name}: wrongly accepted')
        record=dict(name=name,path=name,kind='store',expected_manifest=anchor,
            actual_manifest=actual,request=request,reason=reason,expected_reason=expected,
            seconds=time.monotonic()-started,work_counts=dict(work.counts))
        self.negative.append(record); self.envelope()

    def candidate(self,name,fn,expected,store,base):
        started=time.monotonic(); item=copy.deepcopy(base); fn(item)
        save(self.root/f'{name}.json',item)
        try: check_candidate(store,item['query'],item['candidate'],expected_query=item['expected_query'],work=Work(started+300))
        except ValueError as ex:
            require(expected in str(ex),f'{name}: unexpected refusal {ex}'); reason=str(ex)
        else: raise AssertionError(f'{name}: wrongly accepted')
        self.negative.append(dict(name=name,kind='candidate',path=f'{name}.json',
            input_sha256=identity(item),expected_reason=expected,reason=reason,seconds=time.monotonic()-started))

    def run(self):
        # A second complete generated suite would exceed the retained-stage cap.
        # Repairs should recheck existing bound inputs, not copy them all again.
        require(self.envelope()+40*2**20 <= 63*2**20,'full suite archive admission')
        small=self.run_normal('small'); small1=self.run_normal('small',1)
        same=lambda receipt:[(r['pair'],r['nonzero'],r['lower'],r['residual_sha256']) for r in receipt['report']['results']]
        require(same(small)==same(small1),'small page partition differential')
        large=self.run_normal('large')
        ids=small['ids']; store=Store(self.root/'small_page64',expected_manifest=ids['manifest'],expected_request=ids['request'],work=Work(time.monotonic()+300))
        self.mutate('wrong_external_anchor',lambda p,m:m.update(experts=2),'bytes/identity',stale=True)
        self.mutate('cross_request',lambda p,m:m.update(request='0'*64),'manifest/request/policy')
        self.mutate('wrong_policy',lambda p,m:m['policy'].update(active=32768),'manifest/request/policy')
        self.mutate('factor_alias',lambda p,m:m['factors'][1].__setitem__(0,m['factors'][0][0]),'factor identity')
        self.mutate('factor_kind',lambda p,m:m['factors'][0].__setitem__(1,'real'),'factor identity')
        self.mutate('changed_valid_kind',lambda p,m:m['factors'][0].__setitem__(1,'binary'),'bytes/identity',stale=True)
        self.mutate('forward_parent',lambda p,m:m['views'][1].update(parent='expert/2'),'parent order')
        self.mutate('wrong_private_parent',lambda p,m:m['views'][2].update(parent='expert/0'),'private inheritance')
        self.mutate('private_alias',lambda p,m:m['views'][2].update(add_factors=m['views'][1]['add_factors']),'addition identity')
        self.mutate('common_private_row',lambda p,m:edit_page(p,m,'output',lambda page:page['rows'][0][2].append([9,'1']),prefix='common/'),'private factor contamination')
        def promote(p,m):
            v=m['views'][-1]; m['views'].append(dict(name='promoted',scope='common',guard=None,parent=v['name'],frame=1,add_factors=[],output=v['output'],eq=[],ine=[]))
        self.mutate('guard_promotion',promote,'guard fact promoted')
        self.mutate('missing_pair',lambda p,m:m['roles']['entries'].pop(),'all unordered pair')
        self.mutate('duplicate_pair',lambda p,m:m['roles']['entries'].append(m['roles']['entries'][0]),'all unordered pair')
        self.mutate('duplicate_row',lambda p,m:m['views'][1]['eq'].extend(m['views'][0]['eq']),'duplicate inherited row')
        self.mutate('missing_row',lambda p,m:m['views'][1].update(eq=[]),'unused row members')
        self.mutate('missing_member',lambda p,m:(p/m['members'][0]['file']).unlink(),'member absent')
        self.mutate('extra_member',lambda p,m:save(p/'extra.json',{}),'complete store member inventory')
        self.mutate('changed_member_bytes',lambda p,m:(p/m['members'][0]['file']).write_bytes(b'{}'),'bytes/identity')
        self.mutate('coefficient_noncanonical',lambda p,m:edit_page(p,m,'eq',lambda page:page['rows'][0][2][0].__setitem__(1,'2/2')),'rational capacity/canonical')
        self.mutate('coefficient_zero',lambda p,m:edit_page(p,m,'eq',lambda page:page['rows'][0][2][0].__setitem__(1,'0')),'explicit zero')
        self.mutate('coefficient_wrong_column',lambda p,m:edit_page(p,m,'eq',lambda page:page['rows'][0][2][0].__setitem__(0,99)),'coefficient column')
        self.mutate('stored_bits',lambda p,m:edit_page(p,m,'eq',lambda page:page['rows'][0][2][0].__setitem__(1,str(1<<4096))),'rational capacity')
        self.mutate('bad_json_key',lambda p,m:(p/'manifest.json').write_bytes(b'{"x":1,"x":2}'),'duplicate JSON key',raw=True)
        self.mutate('factor_capacity',lambda p,m:m.update(factors=[['x','continuous','common','c']]*(POLICY['factors']+1)),'factor capacity')
        def active(p,m):
            m['factors']=[[f'c{i}','continuous','common','c'] for i in range(POLICY['active']+1)]
            m['views'][0]['add_factors']=list(range(POLICY['active']+1))
        self.mutate('active_capacity',active,'active factor capacity')
        self.mutate('member_count',lambda p,m:m.update(members=[dict(file='x',bytes=1,sha256=f'{i:064x}') for i in range(POLICY['members']+1)]),'member count')
        self.mutate('member_bytes',lambda p,m:m['members'][0].update(bytes=POLICY['member_bytes']+1),'row member byte capacity')
        self.mutate('total_bytes',lambda p,m:m.update(members=[dict(file=f'rows/{i:064x}.json',bytes=POLICY['member_bytes'],sha256=f'{i:064x}') for i in range(129)]),'total byte capacity')
        self.mutate('header_bytes',lambda p,m:(p/'manifest.json').write_bytes(b' '*(POLICY['header_bytes']+1)),'member absent/capacity',raw=True)
        def rows(p,m,count,bind=False):
            refs=[]
            for first in range(0,count,64): refs.append(add_page(p,m,'eq',[[f'cap/{j}','0',[]] for j in range(first,min(count,first+64))]))
            if bind: m['views'][0]['eq']=refs
        self.mutate('unique_rows',lambda p,m:rows(p,m,POLICY['rows']+1),'unique row capacity')
        self.mutate('view_rows',lambda p,m:rows(p,m,POLICY['view_rows']+1,True),'view row capacity')
        def nnz(p,m):
            count=POLICY['factors']; m['factors']=[[f'c{i}','continuous','common','c'] for i in range(count)]
            for j in range(33): add_page(p,m,'eq',[[f'nnz/{j}','0',[[i,'1'] for i in range(count)]]])
        self.mutate('unique_nnz',nnz,'unique coefficient capacity')
        base=records('small',**ids)[0]
        self.candidate('wrong_query',lambda i:i['query'].update(offset='1'), 'query external binding',store,base)
        self.candidate('wrong_pair_query',lambda i:i['query'].update(pair=[0,2]), 'query external binding',store,base)
        self.candidate('wrong_candidate_binding',lambda i:i['candidate'].update(query='0'*64), 'candidate query binding',store,base)
        self.candidate('raised_claim',lambda i:i['candidate'].update(lower=str(F(i['candidate']['lower'])+1)), 'candidate claimed bound',store,base)
        self.candidate('missing_dual',lambda i:i['candidate']['inequality'].pop(), 'dual row coverage',store,base)
        self.candidate('wrong_dual_sign',lambda i:i['candidate']['inequality'].__setitem__(0,'1/32'), 'dual sign',store,base)
        # Ordinary aliases cannot change the captured representation.
        before=check_candidate(store,base['query'],base['candidate'],expected_query=base['expected_query'],work=Work(time.monotonic()+300))
        rejected=[]
        for label,fn in [('store',lambda:setattr(store,'templates',())),('view',lambda:setattr(store.common,'frame',999)),
                         ('row',lambda:setattr(next(iter(store.common.output)),'rhs',F(99))),
                         ('mapping',lambda:store.entries.__setitem__((0,1),None))]:
            try: fn()
            except (AttributeError,TypeError,FrozenInstanceError): rejected.append(label)
            else: raise AssertionError('alias write accepted')
        caller=json.loads((self.root/'small_page64'/'manifest.json').read_bytes()); caller['factors'].clear()
        after=check_candidate(store,base['query'],base['candidate'],expected_query=base['expected_query'],work=Work(time.monotonic()+300))
        require(before==after,'caller alias changed capture'); self.other.append(dict(control='immutable_capture',rejected=rejected))
        # Actual work-limit counter, not a patched quota.
        started=time.monotonic(); work=Work(started+300)
        try:
            for _ in range(POLICY['visits']+1): work.visit('row_visits')
        except ValueError as ex: require(str(ex)=='store work capacity','work limit reason')
        else: raise AssertionError('work limit accepted')
        self.other.append(dict(control='work_capacity',counts=work.counts,seconds=time.monotonic()-started))
        for name,fn in [('expired',lambda:Work(time.monotonic()-1)),
                        ('exact_bits',lambda:Work(time.monotonic()+300).exact(F(1<<8192)))]:
            try: fn()
            except (TimeoutError,ValueError) as ex: self.other.append(dict(control=name,reason=str(ex)))
            else: raise AssertionError(f'{name}: accepted')
        class Partial(Work):
            def visit(self,kind,amount=1):
                super().visit(kind,amount)
                if sum(self.counts.values()) == 7: self.deadline=time.monotonic()-1
        work=Partial(time.monotonic()+300)
        try: check_candidate(store,base['query'],base['candidate'],expected_query=base['expected_query'],work=work)
        except TimeoutError as ex: self.other.append(dict(control='partial_exact_deadline',reason=str(ex),counts=work.counts,accepted=False))
        else: raise AssertionError('partial result accepted')
        # Independent processes, copied code/data, no repo import or model.
        for receipt in (small,large):
            started=time.monotonic(); portable_work=Work(started+300)
            bundle=self.root/f"relocated_{receipt['name']}"; bundle.mkdir()
            for filename in ('hz_request_store.py','check_hz_store_control.py'):
                shutil.copy2(Path(__file__).resolve().parents[1]/'scoped_source'/filename,bundle/filename)
            shutil.copytree(self.root/receipt['name'],bundle/'store')
            call=[sys.executable,'-B','-I','-S',str((bundle/'check_hz_store_control.py').resolve()),'--store',str((bundle/'store').resolve()),
                '--manifest',receipt['ids']['manifest'],'--request',receipt['ids']['request'],'--layout',receipt['report']['layout']]
            portable_work.tick()
            proc=subprocess.run(call,cwd=bundle,capture_output=True,text=True,timeout=portable_work.deadline-time.monotonic())
            (bundle/'stdout.json').write_text(proc.stdout); (bundle/'stderr.txt').write_text(proc.stderr)
            require(proc.returncode==0,f'relocated checker failed: {proc.stderr}')
            out=json.loads(proc.stdout); require(out['results']==receipt['report']['results'] and out['stats']==receipt['report']['stats'] and not out['forbidden_modules'],'portable exact differential')
            portable_work.tick()
            self.other.append(dict(control=f"portable_{receipt['name']}",seconds=time.monotonic()-started,
                checker_sha256=digest((bundle/'hz_request_store.py').read_bytes()),audit_sha256=digest((bundle/'check_hz_store_control.py').read_bytes()),
                result_sha256=digest(proc.stdout.encode()),status=out['status']))
        self.status='PASS_GIVEN_HZ_STORE_CONTROLS_ONLY'; self.envelope()

    def finish(self):
        report=dict(status=self.status,error=self.error,code=self.code,normal=self.normal,negative=self.negative,
            other=self.other,seconds=time.monotonic()-self.started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            archive_bytes_before_report=self.stage_bytes(),
            cost_scope='legacy per-control intervals precede some publication; whole-suite pre-report interval includes those operations',
            not_established=['network/source lowering','252 output duties','GPU','speedup','real model'])
        save(self.root/'report.json',report); return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    suite=Suite(Path(args.output).resolve())
    try: suite.run()
    except BaseException as ex:
        suite.error=f'{type(ex).__name__}: {ex}'; suite.finish(); raise
    report=suite.finish(); print(json.dumps(dict(status=report['status'],normal=len(report['normal']),negative=len(report['negative']),other=len(report['other']),seconds=report['seconds'],archive_bytes=report['archive_bytes_before_report'])))


if __name__=='__main__': main()
