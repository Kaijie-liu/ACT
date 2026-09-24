"""Create the finite synthetic freeze only, not an execution command."""
from scoped_proof.io import ROOT, load, save, sha
from source_enclosure.format import identity
from shared_route_residual.fixtures import fixture
from shared_route_residual.study import CONFIG, OUTPUT


if __name__=='__main__':
    old=load(ROOT/'configs/backend_controls/frontier_proof_compare_r1.json')['common']['sources']
    for name,digest in old.items():
        if sha(ROOT/name)!=digest:raise ValueError('historical frozen source changed: '+name)
    control='docs/shared_route_residual_controls_20260924_r1.json'
    if load(ROOT/control)['status']!='PASS':raise ValueError('controls')
    sources=dict(old)
    for path in sorted((ROOT/'shared_route_residual').glob('*.py')):
        sources[str(path.relative_to(ROOT))]=sha(path)
    name='docs/shared_route_residual_protocol_20260924_r1.md';sources[name]=sha(ROOT/name)
    calls=[]
    for kind in ('prunable','tied','random'):
        for repeat in range(3):
            for mode in (('pairwise','shared') if repeat%2==0 else ('shared','pairwise')):
                calls.append({'id':f'{kind}_{repeat}_{mode}','fixture':kind,'repeat':repeat,'mode':mode})
    cfg={'schema':'SHARED_ROUTER_RESIDUAL_SYNTHETIC_R1','sources':sources,
        'controls':{'path':control,'sha256':sha(ROOT/control)},'output':str(OUTPUT),
        'fixture_sha256':{k:identity(fixture(k)) for k in ('prunable','tied','random')},
        'calls':calls,'budget_seconds':30.,'threads':2,'sampled_rss_limit':8*2**30,
        'new_real_requests':0,'native_solver_calls':0,
        'scope':'router-only evidence composition; no complete MoE/production/external claim'}
    print(save(CONFIG,cfg))
