"""Separate-process saved-evidence audit and bounded hardening controls.

Reads R1 mathematical objects without regenerating or copying mutation fixtures.
This is still a given-HZ audit, not network provenance or an optimization run.
"""
import argparse
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scoped_source import hz_request_store as c
from scoped_source.check_hz_store_control import inspect, records, oracle

STORE_NEGATIVES = ('wrong_external_anchor cross_request wrong_policy factor_alias factor_kind changed_valid_kind '
    'forward_parent wrong_private_parent private_alias common_private_row guard_promotion missing_pair duplicate_pair '
    'duplicate_row missing_row missing_member extra_member changed_member_bytes coefficient_noncanonical coefficient_zero '
    'coefficient_wrong_column stored_bits bad_json_key factor_capacity active_capacity member_count member_bytes '
    'total_bytes header_bytes unique_rows view_rows unique_nnz').split()
CANDIDATE_NEGATIVES = ('wrong_query wrong_pair_query wrong_candidate_binding raised_claim missing_dual wrong_dual_sign').split()


def read(path): return json.loads(Path(path).read_bytes())
def save(path,obj):
    with Path(path).open('xb') as stream: stream.write(c.compact(obj))


def stage_bytes(parent):
    return sum(p.stat().st_size for root in parent.glob('hz_request_store_20261012_*')
               for p in root.rglob('*') if p.is_file())


def audit(source,output):
    begin=time.monotonic(); report=read(source/'report.json'); events=[]; result={}
    c.require(report['status']=='PASS_GIVEN_HZ_STORE_CONTROLS_ONLY' and report['error'] is None,'parent execution state')
    c.require([v['name'] for v in report['normal']]==['small_page64','small_page1','large_page64'],'normal inventory')
    c.require([v['name'] for v in report['negative']]==STORE_NEGATIVES+CANDIDATE_NEGATIVES,'negative inventory')
    c.require([v['control'] for v in report['other']]==['immutable_capture','work_capacity','expired','exact_bits',
        'partial_exact_deadline','portable_small_page64','portable_large_page64'],'auxiliary inventory')
    c.require(stage_bytes(source.parent)+8*2**20 <= 63*2**20,'retained-stage admission plus terminal reserve')
    # Preserve and verify the exact R1 implementation, not only current code.
    snapshot=source.parent/'hz_request_store_20261012_r1_code_snapshot'
    for path,h in report['code'].items(): c.require(c.digest((snapshot/Path(path).name).read_bytes())==h,'executed source archive binding')
    code={}
    for path in [Path(__file__).resolve(),*[Path(__file__).resolve().parents[1]/'scoped_source'/n
               for n in ('hz_request_store.py','check_hz_store_control.py')]]:
        raw=path.read_bytes(); code[path.name]=c.digest(raw)
        with (output/path.name).open('xb') as stream: stream.write(raw)

    def measured(name,fn):
        c.require(time.monotonic()-begin <= 1800,'audit suite deadline')
        start=time.monotonic(); work=c.Work(start+300); obj=fn(work)
        target=output/f'{name}.json'; save(target,obj)
        size=stage_bytes(source.parent); work.tick()
        c.require(size<=63*2**20,'retained-stage capacity (terminal reserve)')
        end=time.monotonic()
        events.append(dict(name=name,seconds_through_publication=end-start,
            record_sha256=c.digest(target.read_bytes()),work_counts=dict(work.counts),stage_bytes=size))
        return obj

    normals={}
    for old in report['normal']:
        def normal(work,old=old):
            layout=old['report']['layout']; fixed=read(source/f"{old['name']}_queries.json")
            out=inspect(source/old['name'],**old['ids'],layout=layout,work=work,saved_records=fixed)
            c.require(out['results']==old['report']['results'] and out['stats']==old['report']['stats'],'unchanged parent mathematics')
            c.require(not out['forbidden_modules'],'no numerical library')
            return out
        normals[old['name']]=measured(old['name'],normal)
    ids=report['normal'][0]['ids']
    captured={}
    def load_small(work):
        captured['store']=c.Store(source/'small_page64',expected_manifest=ids['manifest'],expected_request=ids['request'],work=work)
        return dict(identity=ids,stats=dict(captured['store'].stats))
    measured('auxiliary_store_capture',load_small); store=captured['store']
    for old in report['negative']:
        def negative(work,old=old):
            inventory=None
            if old['kind']=='store':
                c.require(c.digest((source/old['path']/'manifest.json').read_bytes())==old['actual_manifest'],'mutated input identity')
                inventory=[]
                for path in sorted((source/old['path']).rglob('*')):
                    work.tick()
                    if path.is_file(): inventory.append(dict(path=path.relative_to(source/old['path']).as_posix(),sha256=c.digest(path.read_bytes())))
                call=lambda:c.Store(source/old['path'],expected_manifest=old['expected_manifest'],expected_request=old['request'],work=work)
            else:
                item=read(source/old['path']); c.require(c.identity(item)==old['input_sha256'],'candidate mutation identity')
                call=lambda:c.check_candidate(store,item['query'],item['candidate'],expected_query=item['expected_query'],work=work)
            try: call()
            except ValueError as ex:
                c.require(str(ex)==old['reason'] and old['expected_reason'] in str(ex),'registered refusal reason')
                return dict(name=old['name'],input_identity=old.get('actual_manifest',old.get('input_sha256')),
                            member_inventory=inventory,inventory_sha256=c.identity(inventory),refused=True,reason=str(ex))
            raise AssertionError('mutant accepted')
        measured(old['name'],negative)
    base=records('small',**ids)[0]
    def aliases(work):
        before=c.check_candidate(store,base['query'],base['candidate'],expected_query=base['expected_query'],work=work)
        rejected=[]
        actions={'assign_store':lambda:setattr(store,'templates',()),'delete_seal':lambda:delattr(store,'_sealed'),
            'delete_view':lambda:delattr(store,'common'),'assign_view':lambda:setattr(store.common,'frame',99),
            'assign_row':lambda:setattr(next(iter(store.common.output)),'rhs',F(9)),
            'mutate_map':lambda:store.entries.__setitem__((0,1),None)}
        for name,fn in actions.items():
            try: fn()
            except (AttributeError,TypeError): rejected.append(name)
            else: raise AssertionError('ordinary alias pollution')
        item=copy.deepcopy(base)
        class Mutating(c.Work):
            def visit(self,kind,amount=1):
                item['query']['left'][0]='123'; item['candidate']['inequality'][0]='999'
                super().visit(kind,amount)
        mutated=Mutating(work.deadline)
        after=c.check_candidate(store,item['query'],item['candidate'],expected_query=item['expected_query'],work=mutated)
        c.require(before==after and c.identity(item['query'])!=base['expected_query'],'captured candidate alias control')
        for k,v in mutated.counts.items(): work.counts[k]+=v
        return dict(rejected=rejected,captured_result=str(after[0]),caller_changed=True,
                    before_identity=c.identity(base),after_identity=c.identity(item))
    measured('alias_capture_and_delete',aliases)
    def work_limit(work):
        try:
            for _ in range(c.POLICY['visits']+1): work.visit('row_visits')
        except ValueError as ex:
            c.require(str(ex)=='store work capacity' and sum(work.counts.values())==8000001,'actual work exhaustion')
            return dict(reason=str(ex),counts=dict(work.counts),accepted=False)
        raise AssertionError('work capacity missing')
    measured('work_limit',work_limit)
    def other_faults(work):
        out=[]
        for name,fn,kind,reason in [('expired',lambda:c.Work(time.monotonic()-1),TimeoutError,'store deadline'),
                ('bits',lambda:work.exact(F(1<<8192)),ValueError,'store intermediate rational capacity')]:
            try: fn()
            except kind as ex:
                c.require(str(ex)==reason,'fault reason'); out.append(dict(name=name,reason=str(ex),accepted=False))
            else: raise AssertionError('fault accepted')
        class Partial(c.Work):
            def visit(self,kind,amount=1):
                super().visit(kind,amount)
                if sum(self.counts.values())==7: self.deadline=time.monotonic()-1
        partial=Partial(work.deadline)
        try: c.check_candidate(store,base['query'],base['candidate'],expected_query=base['expected_query'],work=partial)
        except TimeoutError as ex: out.append(dict(name='partial',reason=str(ex),counts=partial.counts,accepted=False))
        else: raise AssertionError('partial accepted')
        return out
    measured('faults',other_faults)
    for name in ('small_page64','large_page64'):
        def portable(work,name=name):
            bundle=output/f'portable_{name}'; bundle.mkdir()
            for file in ('hz_request_store.py','check_hz_store_control.py'): shutil.copy2(output/file,bundle/file)
            shutil.copytree(source/name,bundle/'store')
            old=next(v for v in report['normal'] if v['name']==name)
            command=[sys.executable,'-B','-I','-S',str((bundle/'check_hz_store_control.py').resolve()),
                '--store',str((bundle/'store').resolve()),'--manifest',old['ids']['manifest'],
                '--request',old['ids']['request'],'--layout',old['report']['layout']]
            work.tick(); proc=subprocess.run(command,cwd=bundle,capture_output=True,text=True,timeout=work.deadline-time.monotonic())
            (bundle/'stdout.json').write_text(proc.stdout); (bundle/'stderr.txt').write_text(proc.stderr)
            c.require(proc.returncode==0,f'portable failure {proc.stderr}'); out=json.loads(proc.stdout)
            c.require(out['results']==normals[name]['results'] and out['pairs']==normals[name]['pairs']
                and out['stats']==normals[name]['stats'] and not out['forbidden_modules'],'portable complete result equality')
            work.tick()
            return dict(code=code,stdout_sha256=c.digest(proc.stdout.encode()),status=out['status'])
        measured(f'relocated_{name}',portable)
    result=dict(status='PASS_GIVEN_HZ_STORE_AND_DIRECT_VIEW_AUDIT',parent_report_sha256=c.digest((source/'report.json').read_bytes()),
        code=code,events=events,normal_count=3,refusal_count=38,
        same_normal_objects=True,all_56_large_residuals_checked=True,
        exact_recipe_checked=True,stage_bytes=stage_bytes(source.parent),seconds_before_final_publication=time.monotonic()-begin,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        parent_cost_note='R1 per-control intervals are incomplete; preserved as pre-hardening, not end-to-end benchmarks.',
        cost_scope='Each audit event includes input reading, checking, record publication and archive-size scan; final audit publication is separate.',
        not_established=['source/network provenance','complete 252 duties','GPU','speedup','real model'])
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--source',required=True); parser.add_argument('--output',required=True)
    args=parser.parse_args(); output=Path(args.output).resolve(); source=Path(args.source).resolve()
    output.mkdir(parents=True,exist_ok=False); begin=time.monotonic()
    try:
        result=audit(source,output); save(output/'audit.json',result)
        elapsed=time.monotonic()-begin
        c.require(elapsed<=1800 and stage_bytes(source.parent)<=64*2**20,'final publication envelope')
        terminal=dict(status=result['status'],seconds_through_final_audit_publication=elapsed,
            audit_sha256=c.digest((output/'audit.json').read_bytes()),stage_bytes_before_terminal=stage_bytes(source.parent))
        save(output/'terminal.json',terminal); print(json.dumps(terminal))
    except BaseException as ex:
        # Never rerun a failed deadline/capacity admission just to emit failure.
        save(output/'failed_terminal.json',dict(status='INCOMPLETE',error=f'{type(ex).__name__}: {ex}',seconds=time.monotonic()-begin))
        raise


if __name__=='__main__': main()
