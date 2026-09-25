"""Static expected unpack-site inventory from already checked saved sources.

No checker invocation, cache implementation, timing claim or new bound query.
These sites match the frozen pair-check functions; network/router internals
are deliberately excluded. Identity counts are NOT measured parsing cost.
"""
import argparse
from collections import Counter
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoped_proof.io import load,save,sha
from source_enclosure.format import identity


def inventory(bundle):
    root=bundle['prefix']['input']
    ends={'router':bundle['prefix']['router']['steps'][-1]['state']}
    ends.update({t['name']:t['steps'][-1]['state'] for t in bundle['experts']})
    sites=[]
    for p in bundle['pairs']:
        a,b=p['pair']
        sequence=[('join:input',root),('join:router',ends['router']),
                  ('join:expert_a',ends[f'expert{a}']),('join:expert_b',ends[f'expert{b}']),
                  ('join:target',p['joint']),('guard:source',p['joint']),('guard:target',p['guarded']),
                  ('projection:source',p['guarded']),('projection:target',p['projected']),
                  ('output:source',p['projected'])]
        for site,state in sequence:
            sites.append({'pair':p['pair'],'site':site,'source_sha256':identity(state)})
    counts=Counter(s['source_sha256'] for s in sites)
    return {'expected_unpack_sites':len(sites),'distinct_source_identities':len(counts),
            'repeated_identity_sites':sum(n-1 for n in counts.values()),'sites':sites,
            'identity_multiplicities':dict(sorted(counts.items())),
            'not_measured':['parser self time','cache lookup/hash cost','copy/freeze cost','memory cost'],
            'scope':'static pair-check call sites only; excludes network and router internals'}


def run():
    began=time.monotonic()
    cfg=load(ROOT/'configs/backend_controls/source_cost_supervised_r1.json')
    checked=load(ROOT/'docs/source_cost_supervised_audit_20260925_r1.json')
    if checked['audit']!='PASS' or checked['completed_profiles']!=2:raise ValueError('saved audit prerequisite')
    for name in ('scoped_source/check.py','full_source/check_obligations.py'):
        if sha(ROOT/name)!=cfg['sources'][name]:raise ValueError('call-site implementation drift')
    out=[]
    for r in checked['results']:
        path=Path(cfg['output'])/r['id']/'profile/construction.json'
        b=load(path,r['profile']['files']['construction']['sha256'])
        if b['source_sha256']!=r['profile']['source_sha256']:raise ValueError('source binding')
        out.append({'id':r['id'],'construction_sha256':sha(path),**inventory(b)})
    return {'status':'SAVED_STATIC_INVENTORY','parent_audit_sha256':sha(ROOT/'docs/source_cost_supervised_audit_20260925_r1.json'),
            'analysis_source_sha256':sha(Path(__file__)),'results':out,'new_solves':0,'new_profile_calls':0,
            'separate_analysis_seconds':time.monotonic()-began,
            'conclusion':'repeated exact identities motivate a bounded parse-reuse control, NOT a speedup or parser-dominance claim'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path);a=p.parse_args()
    r=run()
    if a.report:save(a.report,r)
    print([{k:v[k] for k in ('id','expected_unpack_sites','distinct_source_identities','repeated_identity_sites')} for v in r['results']])
