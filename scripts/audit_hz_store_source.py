"""Independent saved-source recheck; never invokes the source producer."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scoped_source.hz_request_store import Store,Work,compact,digest,identity,require
from scoped_source.check_hz_store_source import check_source
from scoped_source.hz_store_source_controls import fixture,analytic,signature,negative,NEGATIVES,EXPECTED

NORMALS=['wide_page64','mixed_page64','mixed_page1','input_page64']
LIVE=['c','Gc','Gb','Ac','Ab','b','Auc','Aub','ub','map','frame','exact',
      'cached_duplicate','trailing_data','bad_pointer']


def read(p): return json.loads(p.read_bytes())


def audit(root, previous):
    begin=time.monotonic(); result=[]; report=read(root/'report.json'); first=read(previous/'report.json')
    require(report['status']=='PASS_SOURCE_VIEW_CONTROLS_ONLY' and report['error'] is None,'suite completion')
    require([x['name'] for x in report['normal']]==NORMALS,'complete normal inventory')
    require([x['name'] for x in report['negative']]==list(NEGATIVES),'complete source refusal inventory')
    require([x['name'] for x in report['live_negative']]==LIVE,'complete live refusal inventory')
    expected_other=[f'publication_{x}' for x in NORMALS]+['caller_alias','portable_wide_page64','portable_mixed_page64']
    require([x['name'] for x in report['other']]==expected_other,'complete auxiliary inventory')
    terminal=read(root/'terminal.json'); outer=read(root/'outer.json'); end=read(root/'outer_terminal.json')
    report_hash=digest((root/'report.json').read_bytes())
    require(terminal['status']==report['status'] and terminal['report_sha256']==report_hash
            and outer['status']=='COMPLETE' and outer['report_sha256']==report_hash
            and outer['returncode']==0 and outer['leader_reaped'] and not outer['timed_out']
            and end['status']=='COMPLETE' and end['outer_sha256']==digest((root/'outer.json').read_bytes()),
            'outer completion and report binding')
    require(0 < report['seconds_before_report'] <= terminal['seconds_through_report']
            <= outer['seconds_spawn_through_return'] <= end['seconds_through_receipt'] <= 300,
            'complete cost clock ordering')
    for name,h in report['code'].items():
        require(digest((root/'executed_code'/name).read_bytes())==h,'executed code snapshot identity')
        if name in ('scoped_source/hz_request_store.py','scoped_source/check_hz_store_source.py',
                    'scoped_source/hz_store_source_controls.py'):
            require(digest((Path(__file__).resolve().parents[1]/name).read_bytes())==h,
                    'active independent checker version differs from recorded source contract')
        if name!='scripts/test_hz_store_source.py':
            require(first['code'][name]==h,'receipt repair changed algorithm/fixtures')
    stores={}; docs={}
    for i,old in enumerate(report['normal']):
        work=Work(begin+290); start=time.monotonic(); name=old['name']
        stored=read(root/f'{name}_receipt.json'); require(stored==old,'report and receipt equality')
        publication=report['other'][i]
        require(publication['receipt_sha256']==digest((root/f'{name}_receipt.json').read_bytes())
                and 0 < publication['seconds_through_receipt'] <= report['seconds_before_report'], 'normal publication cost')
        for value in old['cost'].values(): require(type(value) in (int,float) and math.isfinite(value) and value>=0,'cost field')
        require(old['cost']['generation_seconds'] >= old['cost']['native_workspace_and_affine_seconds']+
                old['cost']['source_member_serialization_seconds'], 'generation cost subintervals')
        doc=read(root/f'{name}_source.json')
        require(doc==fixture(old['recipe']) and identity(doc)==old['source'],'fixed source coefficient recipe')
        require(old['manifest']==first['normal'][i]['manifest'] and old['source']==first['normal'][i]['source'],
                'unchanged R1/R2 mathematical objects')
        store=Store(root/name,expected_manifest=old['manifest'],expected_request=old['source'],work=work)
        checked=check_source(store,doc,expected_source=old['source'],work=work)
        oracle=analytic(store,old['recipe'],work)
        require(checked==old['result'] and oracle==old['oracle'] and signature(store)==old['signature']
                and dict(store.stats)==old['stats'],'independent complete source and oracle')
        expected_matches=[]
        for v in store.views.values():
            cs=sum(store.factors[j][1]=='continuous' for j in v.factors)
            expected_matches.append(dict(view=v.name,outputs=v.output.count,equalities=v.eq.count,
                inequalities=v.ine.count,continuous=cs,binary=len(v.factors)-cs))
        require(old['matches']==expected_matches,'complete recorded live field inventory')
        stores[name]=store; docs[name]=doc
        result.append(dict(name=name,result=checked,oracle=oracle,stats=dict(store.stats),
                           signature=old['signature'],seconds=time.monotonic()-start,work_counts=work.counts))
    require(result[1]['signature']==result[2]['signature'],'page-size exact coefficient differential')
    refusals=[]
    for old in report['negative']:
        start=time.monotonic(); work=Work(begin+290)
        expected_base='wide_page64' if old['name'].startswith(('input_','affine_')) else 'mixed_page64'
        require(old['base']==expected_base,'mutation fixed parent binding')
        try: negative(stores[old['base']],docs[old['base']],old['name'],work)
        except (ValueError,TimeoutError) as ex:
            require(str(ex)==old['reason']==EXPECTED[old['name']]
                    and isinstance(ex,TimeoutError)==(old['name']=='source_deadline_tail'), 'specific reconstructed refusal')
            refusals.append(dict(name=old['name'],reason=str(ex),seconds=time.monotonic()-start))
        else: raise AssertionError('saved mutation accepted')
    for old in report['other']:
        if old['name'].startswith('portable_'):
            target=root/old['name']; process=read(target/'process.json')
            require(process['returncode']==0 and not process['stderr'],'portable process status')
            out=json.loads(process['stdout']); require(not out.pop('forbidden_modules'),'portable no numerical imports')
            out.pop('work_counts'); require(out==old['result'],'portable complete result')
            name=old['name'].removeprefix('portable_'); original=next(v for v in report['normal'] if v['name']==name)
            require(digest((target/'object/manifest.json').read_bytes())==original['manifest']
                    and read(target/'source.json')==docs[name], 'portable object/source identities')
            for path in ('scripts/check_hz_store_source.py','scoped_source/hz_request_store.py','scoped_source/check_hz_store_source.py'):
                require(digest((target/path).read_bytes())==report['code'][path],'portable checker code')
    stage=sum(p.stat().st_size for d in root.parent.glob('hz_store_source_20261012_*') for p in d.rglob('*') if p.is_file())
    require(stage <= 63*2**20,'complete stage archive cap with terminal reserve')
    require(not any(x in sys.modules for x in ('numpy','scipy','torch','gurobipy','highspy')), 'source audit imports')
    return dict(status='PASS_SAVED_SOURCE_VIEW_AUDIT',report_sha256=report_hash,
        parent_report_sha256=digest((previous/'report.json').read_bytes()),normal=result,
        reconstructed_source_refusals=refusals,recorded_live_refusals=len(report['live_negative']),
        unchanged_normal_manifests=True,stage_bytes=stage,seconds_before_audit_publication=time.monotonic()-begin,
        scope='declared finite source inclusion; no output bounds, native-float proof, real source or CUDA',
        live_scope='recorded executed field/mutation controls; no archived native CSR snapshot is independently re-proved')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('previous',type=Path)
    p.add_argument('output',type=Path);a=p.parse_args();begin=time.monotonic();out=audit(a.root,a.previous)
    out['audit_code_sha256']=digest(Path(__file__).read_bytes())
    with a.output.open('xb') as f:f.write(compact(out))
    print(json.dumps(dict(status=out['status'],sha256=digest(a.output.read_bytes()),seconds_through_publication=time.monotonic()-begin)))
