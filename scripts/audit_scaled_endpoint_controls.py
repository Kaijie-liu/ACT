"""Independent stdlib all-source/all-raw-candidate and terminal-cost audit."""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from scoped_source.hz_request_store import require,compact,digest
from scoped_source.verify_scaled_endpoints import verify

PROTOCOL='c298d3f0b41dbdb28e73e78b2638b4c76e723f49e939f439a969703f2847e8b8'
CODE_INVENTORY='be1f800a9af0ca9ef236799996daa61e8cc49825aa6449b5acc1370257fac622'
PHASES={'imports_seconds','capture_and_place_seconds','source_build_encode_seconds','forest_publish_seconds',
        'source_check_seconds','complete_candidate_and_check_seconds','proof_and_binding_publish_seconds',
        'test_only_expanded_differential_seconds'}
NATIVE_PHASES={'planning_seconds','preparation_seconds','objective_seconds','proposal_seconds',
               'exact_check_seconds','partial_publish_seconds','terminal_encode_seconds'}
ANCHORS={'mixed':'24155a8113346d6964b6a4497be0abc0e5dcebbb169c949931468d6fdbb92a22',
 'deep':'91a9cd3551e5f63e500e6db435a3039d6e158d7d671e88b7780a735267f4ba62',
 'clipped':'fec50a03b10e4d0f0c139b7702e3cdc0705e0c167eb7e5a08d35c31f14609806',
 'zero':'de658c8fa7bf0c7e090e0b31d7830d48c84822235b0fb84c70b9ff2fe0452a84',
 'half':'cf4dbc3dff90da9d71fc9a056e29cdefa69a2aa8dae768f08555a7b84fb4c8a0',
 'guarded':'0be644fc20d28ab6d811f4c933f58456d9039e7a1d513e38d41f929b92cfc8da'}


def accounting(record,call):
    require(record['status']=='COMPLETE_FINITE_SCALED_CONTROL' and record['real_models']==record['gpu_calls']==0,'worker boundary')
    require(0<record['worker_before_receipt_seconds']<=call['parent_observed_seconds']<300,'worker/parent cost')
    require(set(record['phases'])==PHASES,'complete worker phase inventory')
    require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in record['phases'].values())
            and sum(record['phases'].values())<=record['worker_before_receipt_seconds'],'phase wall cost')
    before=None
    for field in ('source_work','production_work','work'):
        counts=record[field]
        require(set(counts)=={'row_visits','coefficient_visits','arithmetic'} and
                all(type(v) is int and v>=0 for v in counts.values()) and sum(counts.values())<=8000000,'work count contract')
        if before is not None: require(all(before[k]<=counts[k] for k in counts),'cumulative source/production/oracle work')
        before=counts
    native=record['native']
    require({k for k in native if k.endswith('_seconds')}==NATIVE_PHASES,'complete native phase inventory')
    require(native['device']=='cpu' and native['dtype']=='float64' and native['expanded_constraint_entries']==0
            and native['algorithm']=='CPU_SCALED_REFERENCE_PROJECTED_SUBGRADIENT_128_V3','fixed native contract')
    seconds=[v for k,v in native.items() if k.endswith('_seconds')]
    require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in seconds)
            and sum(seconds)<=record['phases']['complete_candidate_and_check_seconds'],'nested native costs')


def code_inventory(root,code):
    require(type(code) is dict and digest(compact(sorted(code)))==CODE_INVENTORY,'fixed code key inventory')
    for dirname in ('code','relocated/code'):
        directory=Path(root)/dirname
        require({str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}==set(code),
                'complete archived code directory')
        for name,h in code.items():
            require(hashlib.sha256((directory/name).read_bytes()).hexdigest()==h,'archived code bytes')


def audit(root):
    start=time.monotonic(); root=Path(root); load=lambda name:json.loads((root/name).read_bytes())
    require(not root.is_symlink() and not any(p.is_symlink() for p in root.rglob('*')),'self-contained nonsymlink archive')
    e=load('execution.json'); terminal=load('terminal.json'); names=list(ANCHORS)
    require(terminal['status']=='PASS_FINITE_SCALED_ENDPOINT_CONTROLS' and terminal['code']==e['code'],'complete terminal/code')
    require(e['protocol']==PROTOCOL==e['code']['docs/hz_scaled_batch_design_20261013.md'],'frozen protocol')
    require(e['source_anchors']==ANCHORS and e['cases']==names and e['width']==8,'fixed source/order/width')
    require(e['archive_cap']==terminal['archive_cap']==16*2**20 and e['work_cap']==8000000,'unchanged capacities')
    require(e['dense']==terminal['dense']=='NOT_ADMITTED' and e['gpu']=='NOT_POLLED_OR_RUN'
            and terminal['real_models']==terminal['gpu_calls']==0,'no expanded admission')
    require([c['name'] for c in terminal['calls']]==['unit',*names,'relocated'],'complete call inventory')
    for c in terminal['calls']:
        require(c['status']=='COMPLETE' and c['returncode']==0 and c['hard_cap_seconds']==300
                and 0<c['parent_observed_seconds']<300,'hard terminal status/cost')
    require(math.isfinite(terminal['outer_before_terminal_seconds']) and
            sum(c['parent_observed_seconds'] for c in terminal['calls'])<=terminal['outer_before_terminal_seconds'],'outer cost ledger')
    code_inventory(root,e['code'])
    relocated=load('relocated.stdout'); require(type(relocated) is list and len(relocated)==len(names),'relocated results')
    output=[]
    for i,name in enumerate(names):
        record=load(name+'/receipt.json'); call=terminal['calls'][i+1]; accounting(record,call)
        require(record['source']==ANCHORS[name],'source anchor')
        result=verify(root/name/'bundle',record['binding'])
        require(result['source']==record['source'] and result['forest']==record['forest'] and
                result['result']==record['result'] and result['stats']==record['stats'],'independent source/output recheck')
        require(result['result']==record['differential']['result'] and record['differential']['candidates']==
                2*result['result']['required_endpoints'],'both-candidate eager differential inventory')
        require(record['native']['candidates']==record['differential']['candidates'],'native both-candidate inventory')
        r=relocated[i]
        require(all(r[k]==result[k] for k in ('binding','source','forest','stats','result','work')),'relocated independent result')
        original=root/name/'bundle'; moved=root/'relocated'/name
        for p in original.rglob('*'):
            if p.is_file(): require((moved/p.relative_to(original)).read_bytes()==p.read_bytes(),'portable unchanged bytes')
        p=load(name+'/bundle/proof.json'); partial=root/name/'partial'
        require(sorted(x.name for x in partial.iterdir())==[f'{j:05d}.json' for j in range(len(p['tiles']))],'partial inventory')
        for j,records in enumerate(p['tiles']):
            require((partial/f'{j:05d}.json').read_bytes()==compact(records),'raw partial candidates preserved')
        output.append(dict(name=name,source=record['source'],forest=record['forest'],binding=record['binding'],
            result=result['result'],source_work=sum(record['source_work'].values()),
            production_work=sum(record['production_work'].values()),control_work=sum(record['work'].values()),
            checker_work=sum(result['work'].values()),differential_candidates=record['differential']['candidates'],
            parent_seconds=call['parent_observed_seconds']))
    total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    require(total==terminal['payload_before_terminal_bytes']+(root/'terminal.json').stat().st_size
            and total<=16*2**20,'full archive capacity/accounting')
    require(not {'torch','numpy','scipy','highspy'}&set(sys.modules),'stdlib audit only')
    return dict(status='PASS_INDEPENDENT_SCALED_ENDPOINT_AUDIT',sources=output,archive_bytes=total,
                seconds=time.monotonic()-start,real_models=0,gpu_calls=0,dense='NOT_ADMITTED')


if __name__=='__main__': print(json.dumps(audit(sys.argv[1]),indent=2))
