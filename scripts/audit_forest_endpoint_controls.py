"""Offline stdlib audit of source-complete forest support controls."""
import argparse
import json
import math
from pathlib import Path
import sys
import time

PROTOCOL='ed44073ef5b251aec1088bf5c7fb87e7ef7ad49f5533f3db1cad697ba6a09c4a'
CASES=('small40','mixed','guarded_w8','half','guarded_w1')
ANCHORS={'small40':'28c41d0da9493f2b80de1dab910a769868f12240ac00dc55727b9a716dc8fe65',
 'mixed':'24155a8113346d6964b6a4497be0abc0e5dcebbb169c949931468d6fdbb92a22',
 'guarded_w8':'0be644fc20d28ab6d811f4c933f58456d9039e7a1d513e38d41f929b92cfc8da',
 'half':'cf4dbc3dff90da9d71fc9a056e29cdefa69a2aa8dae768f08555a7b84fb4c8a0',
 'guarded_w1':'0be644fc20d28ab6d811f4c933f58456d9039e7a1d513e38d41f929b92cfc8da'}
LIMIT=64*2**20
PHASES={'imports_seconds','capture_seconds','source_build_encode_seconds','forest_publish_seconds',
        'source_check_seconds','planning_seconds','candidate_and_exact_selection_seconds',
        'proof_encode_publish_seconds','complete_endpoint_check_seconds',
        'test_only_eager_differential_seconds','binding_publish_seconds'}


def seconds(x,limit=300): return type(x) in (int,float) and math.isfinite(x) and 0<=x<=limit


def costs(r,call):
    from scoped_source.hz_request_store import require
    require(r['status']=='COMPLETE' and r['real_models']==r['gpu_calls']==0, 'execution/claim boundary')
    require(set(r['work'])=={'row_visits','coefficient_visits','arithmetic'} and
            all(type(v) is int and v>=0 for v in r['work'].values()) and sum(r['work'].values())<=8000000, 'cumulative work')
    p=r['phases']; require(set(p)==PHASES and all(seconds(x) for x in p.values()), 'finite complete phases')
    require(seconds(r['worker_before_receipt_seconds']) and sum(p.values())<=r['worker_before_receipt_seconds']
            <=call['parent_observed_seconds'], 'full parent/worker versus additive cost')
    n=r['native']
    nested=sum(n[k] for k in ('preparation_seconds','proposal_seconds','exact_selection_seconds','capacity_encoding_seconds'))
    require(all(seconds(n[k]) for k in ('preparation_seconds','proposal_seconds','exact_selection_seconds','capacity_encoding_seconds'))
            and nested<=p['candidate_and_exact_selection_seconds'], 'native costs nested, not additive again')
    require(n['device']=='cpu' and n['dtype']=='float64' and n['expanded_constraint_entries']==0
            and n['algorithm']=='CPU_REFERENCE_PROJECTED_SUBGRADIENT_128_V1'
            and n['local_nnz']==n['coefficient_conversions']==r['stats']['local_nnz']
            and n['references']==r['stats']['references'], 'reference native identity/count')
    require(seconds(r['generation']['native_seconds']) and r['generation']['native_seconds']<=p['source_build_encode_seconds'], 'native source cost nested')


def audit(root):
    from scoped_source.hz_request_store import digest,require,decode
    from scoped_source.verify_forest_endpoints import verify
    from scripts.run_forest_endpoint_controls import FILES,DESIGN
    root=Path(root); start=time.monotonic()
    require(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*')), 'archive symlink')
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()); require(total<=LIMIT, 'archive size')
    e=json.loads((root/'execution.json').read_bytes()); t=json.loads((root/'terminal.json').read_bytes())
    require(e['protocol']==e['code'][DESIGN]==PROTOCOL and e['cases']==list(CASES) and e['sources']==ANCHORS, 'protocol and case roster')
    require(e['archive_cap']==t['archive_cap']==LIMIT and e['work_cap']==8000000 and e['device']=='cpu', 'caps/device')
    require(t['status']=='PASS_COMPONENT_CONTROLS' and t['real_models']==t['gpu_calls']==0, 'archive terminal')
    require(set(e['code'])==set(FILES) and t['code']==e['code'], 'complete code identity')
    for name,h in e['code'].items():
        require(digest((root/'code'/name).read_bytes())==h and digest((root/'relocated'/'code'/name).read_bytes())==h,
                'original and moved code identity')
    require([c['name'] for c in t['calls']]==['unit',*CASES,'relocated'] and
            all(c['status']=='COMPLETE' and c['returncode']==0 and c['hard_cap_seconds']==300 and
                seconds(c['parent_observed_seconds']) for c in t['calls']), 'call inventory/cost/status')
    require(seconds(t['outer_before_terminal_seconds'],2100) and
            sum(c['parent_observed_seconds'] for c in t['calls'])<=t['outer_before_terminal_seconds'], 'outer full cost')
    calls={c['name']:c for c in t['calls']}; results=[]
    moved=json.loads((root/'relocated.stdout').read_bytes()); require(len(moved)==4,'moved request inventory')
    for i,name in enumerate(CASES):
        r=json.loads((root/name/'receipt.json').read_bytes()); costs(r,calls[name])
        require(r['name']==name and r['source']==ANCHORS[name], 'receipt name/frozen source')
        result=verify(root/name/'bundle',r['binding'])
        require(result['result']==r['result'] and result['stats']==r['stats'] and result['source']==r['source']
                and result['forest']==r['forest'], 'independent source and complete bound differential')
        require(r['differential']['exact_endpoints']==r['result']['required_endpoints']==r['native']['columns'], 'full endpoint differential count')
        proof=decode((root/name/'bundle'/'proof.json').read_bytes())
        require(len(proof['tiles'])==r['native']['tiles'] and proof['width']==(1 if name=='guarded_w1' else 8), 'frozen tiling')
        require((root/name/'bundle'/'proof.json').stat().st_size==r['native']['proof_bytes'],'proof bytes')
        partial=root/name/'partial'
        require({p.name for p in partial.iterdir()}=={f'{j:05d}.json' for j in range(len(proof['tiles']))}, 'partial terminal inventory')
        for j,records in enumerate(proof['tiles']):
            require(decode((partial/f'{j:05d}.json').read_bytes())==records,'published prefixes retained')
        if i<4:
            a=root/name/'bundle'; b=root/'relocated'/name
            am={str(p.relative_to(a)):digest(p.read_bytes()) for p in a.rglob('*') if p.is_file()}
            bm={str(p.relative_to(b)):digest(p.read_bytes()) for p in b.rglob('*') if p.is_file()}
            require(am==bm and moved[i]['result']==r['result'] and moved[i]['binding']==r['binding'], 'moved identity/check result')
        results.append(dict(name=name,source=r['source'],forest=r['forest'],binding=r['binding'],
            result=result['result'],independent_work=result['work'],independent_seconds=result['seconds'],
            execution_work=r['work'],parent_seconds=calls[name]['parent_observed_seconds'],phases=r['phases'],stats=r['stats']))
    require(results[2]['source']==results[4]['source'] and results[2]['forest']==results[4]['forest'], 'width differential same source')
    width_equal=results[2]['result']==results[4]['result']
    require(not {'torch','numpy','scipy','highspy'}&set(sys.modules), 'offline native import')
    return dict(status='PASS_CHECKED_FOREST_ENDPOINT_RECHECK',cases=results,guarded_width_results_equal=width_equal,
        archive_bytes=total,audit_seconds=time.monotonic()-start,real_models=0,gpu_calls=0,
        boundary='captured synthetic real sources; no trained checkpoint or native-float guarantee')


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1])); p=argparse.ArgumentParser(); p.add_argument('root',type=Path)
    print(json.dumps(audit(p.parse_args().root.resolve()),indent=2))
