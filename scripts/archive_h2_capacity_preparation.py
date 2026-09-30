"""Fixed source-preparation control roster and independent final accounting."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from scoped_proof.io import load,save,sha
from scoped_source.capacity_prep_audit import audit
from scoped_source.capacity_intake import sources,PROTOCOL_SHA


def derive(root):
    root=Path(root); bindings=load(root/'implementation.json')
    if bindings!=sources(): raise ValueError('control source set differs from current implementation')
    for name,digest in bindings.items():
        if sha(root/'implementation'/name)!=digest: raise ValueError('archived source identity')
    expected={'tests':16,'success':True,'failures':[],'errors':[],'skipped':[],
              'full_size_models':0,'solves':0}
    outcome=load(root/'test_outcome.json')
    if outcome!=expected or type(outcome['tests']) is not int or outcome['success'] is not True:
        raise ValueError('complete passing control suite required')
    roster=[('normal','',30,'IDENTITY_PREPARED'),('exception','exception',30,'ERROR'),
            ('partial','partial',30,'ERROR'),('wrong-recipe','wrong_recipe',30,'ERROR'),
            ('wrong-source','wrong_source',30,'ERROR'),('delay','delay',6,'TIMEOUT'),
            ('late','late',6,'TIMEOUT'),('receive-delay','receive_delay',6,'TIMEOUT'),
            ('resource','',6,'RESOURCE_LIMIT'),('prelaunch','',1e-9,'TIMEOUT'),
            ('final-publication','',6,'TIMEOUT')]
    rows=[]
    for name,control,budget,status in roster:
        path=root/name; call=load(path/'caller_observation.json'); inv=load(path/'invocation.json')
        spec=load(path/'spec.json')
        if (call['status']!=status or inv['budget']!=budget or
            inv['rss_limit']!=(1 if name=='resource' else 2*2**30) or
            spec!={'case':'tiny_control','control':control,'recipe_sha256':PROTOCOL_SHA}):
            raise ValueError('fixed preparation control roster')
        checked=audit(path,call,expected_sources=bindings)
        rows.append({'name':name,'observed':call,'audit':checked,
            'events':{phase:sha(path/(phase+'_events.jsonl')) if (path/(phase+'_events.jsonl')).exists() else None
                      for phase in ('prepare','receive')}})
    for path in root.iterdir():
        if path.is_dir() and (path/'caller_observation.json').is_file() and path.name not in {r[0] for r in roster}:
            if load(path/'caller_observation.json')['root']==str(path): raise ValueError('unexpected extra execution')
    return {'schema':'H2_CAPACITY_PREPARATION_CONTROLS_V1','status':'PASS','controls':outcome,
            'test_outcome_sha256':sha(root/'test_outcome.json'),
            'implementation_manifest_sha256':sha(root/'implementation.json'),
            'recipe_sha256':PROTOCOL_SHA,'calls':rows,'call_count':len(rows),
            'statuses':dict(Counter(row['audit']['status'] for row in rows)),
            'full_size_admitted':False,'real_model_admitted':False,'proof_status':'NOT_A_PROOF'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    option=p.add_mutually_exclusive_group(required=True); option.add_argument('--output',type=Path); option.add_argument('--check',type=Path)
    args=p.parse_args(); result=derive(args.root)
    if args.output:
        if not args.output.resolve().is_relative_to(Path('/data1/Kane/MOE')): raise ValueError('output outside workspace')
        save(args.output,result)
    elif load(args.check)!=result: raise ValueError('changed compact archive')
    print({'status':result['status'],'tests':result['controls']['tests'],'calls':result['call_count'],
           'statuses':result['statuses'],'real_model_admitted':False})
