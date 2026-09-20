"""Regenerate the entire source trace and all new LPs under an owned build window."""
import argparse
import json
from pathlib import Path
import shutil
import time
import resource
from router_source.capture import ROOT,sha
from router_source.build import save
from source_enclosure.build import build as prefix_build
from source_enclosure.format import identity
from source_enclosure.produce import relu,join
from full_source.graph import validate,operator
from full_source.lift import affine
from full_source.delta import encode
from full_source.obligations import build as obligations
from full_source.build import CODE as OLD_CODE
from full_bounds.worker import CODE as BOUND_CODE
from range_pipeline.steps import affine as ranged_affine
from range_pipeline.candidates import propose
from property_ranges.select import select

CODE={**OLD_CODE,'property_ranges/verify_source.py':'verify_full.py',
      'source_ranges/check.py':'range_check.py','range_pipeline/check_step.py':'range_step_check.py'}
CODE.pop('full_source/verify.py')
CODE['property_ranges/check_selection.py']='selection_check.py'


def build(root,job,deadline):
    start=time.monotonic();load=lambda p:json.loads(p.read_bytes())
    for name,h in job['input_files'].items():
        if sha(Path(name))!=h:raise ValueError('frozen upstream bytes changed')
    arm=job['arm']
    if arm not in ('prefix','property'):raise ValueError('fixed arm')
    upstream=root/'upstream';upstream.mkdir();t=time.monotonic()
    prefix_build(upstream,Path(job['prefix_source']),job['prefix_source_hash'])
    prefix_seconds=time.monotonic()-t;prefix=upstream/'relocated';pm=load(prefix/'manifest.json')
    dst=root/'relocated/source';dst.mkdir(parents=True);shutil.copytree(prefix,dst/'prefix')
    source=load(prefix/'router_source.json');doc=load(Path(job['expert_document']))
    pair=validate(doc,source,load(prefix/'experts.json'));save(dst/'full_experts.json',doc)
    policy={'selection':arm,'layer':6,'rows_per_expert':2,'native_seconds':3.}
    range_dir=root/'range_queries';range_dir.mkdir();steps=[];times=[];range_rows=[];ends=[]
    for e in doc['experts']:
        i=e['expert'];shape=operator(source['request']['lower']['shape'],e['layers'][0])[0]
        state=load(prefix/f'expert{i}_relu.state.json')
        for index in ['coordinate']+list(range(2,9)):
            t=time.monotonic();old=state;before=shape;range_seconds=0.;selection_seconds=0.
            if index=='coordinate':kind='CoordinateLift';op=[{j:1} for j in range(len(state['hz']['c']))];bias=[0]*len(op)
            else:kind=e['layers'][index]['kind'];shape,op,bias=operator(shape,e['layers'][index])
            tag=f'expert{i}/full/{index}';row={'expert':i,'layer':index,'kind':kind,'input_shape':before,'output_shape':shape}
            facts=None
            if index==6:
                facts=[None]*len(op);context={'request_id':identity(source['request']),'scope':f'pair{pair}/expert{i}','layer':'6'}
                begin=time.monotonic()
                selection=select(old,op,bias,e['layers'][8],context,source['request']['classes'],
                                 source['request']['clean_prediction'],arm)
                row['selection_file']=f'expert{i}_selection.json'
                save(dst/row['selection_file'],selection)
                # Durable roster precedes all native calls, even if later work is terminated.
                save(root/f'selection_expert{i}.json',{'record':selection,'record_sha256':identity(selection),
                     'seconds_since_build_start':time.monotonic()-start})
                selection_seconds=time.monotonic()-begin;begin=time.monotonic()
                for j in selection['selected']:
                    fact,r=propose(old,op[j],bias[j],context,j,range_dir,f'e{i}_r{j}',deadline,3.)
                    facts[j]=fact
                    range_rows.append({'expert':i,**r})
                range_seconds=time.monotonic()-begin
            prop_start=time.monotonic()
            if kind=='Flatten':row.update(source=identity(state),target=identity(state))
            else:
                if facts is not None:state,proof=ranged_affine(old,op,bias,facts,context,tag)
                elif kind=='ReLU':state,proof=relu(old,tag)
                else:state,proof=affine(old,op,bias,tag)
                row['proof']=proof
            propagation=time.monotonic()-prop_start;serial_start=time.monotonic()
            if kind!='Flatten':
                name=f'expert{i}_{index}.delta.json';save(dst/name,encode(old,state));row['file']=name
            serial=time.monotonic()-serial_start;steps.append(row)
            times.append({'expert':i,'layer':index,'seconds':time.monotonic()-t,'range_seconds':range_seconds,
                'selection_seconds':selection_seconds,'propagation_seconds':propagation,'delta_serialization_seconds':serial,
                'continuous':len(state['continuous_ids']),'binary':len(state['binary_ids']),
                'outputs':len(state['hz']['c'])})
            save(root/f'progress_{len(steps):02d}.json',times[-1])
        if shape!=[1,source['request']['classes']]:raise ValueError('complete expert endpoint')
        ends.append(state)
    t=time.monotonic();joint,proof=join(load(prefix/'guard.state.json'),*ends);save(dst/'joint.state.json',joint)
    common,obs=obligations(joint,pair,source['request']['classes'],source['request']['clean_prediction'])
    save(dst/'lp_base.json',common);save(dst/'obligations.json',obs)
    output_seconds=time.monotonic()-t
    save(dst/'trace.json',{'schema':'SELECTED_COMPLETE_EXPERT_TRACE_V1','steps':steps,'joint_proof':proof,'joint_sha256':identity(joint)})
    for src,name in CODE.items():shutil.copyfile(ROOT/src,dst/name)
    # The source checker imports the pure dual checker from its enclosing proof namespace.
    for src,name in BOUND_CODE.items():
        target=root/'relocated'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/src,target)
    files={str(p.relative_to(dst)):sha(p) for p in sorted(dst.rglob('*')) if p.is_file()}
    save(dst/'manifest.json',{'schema':'SELECTED_COMPLETE_EXPERT_LP_BUNDLE_V1','policy':policy,
        'prefix_manifest_sha256':sha(prefix/'manifest.json'),'request':source['request'],'pair':pair,'files':files,
        'endpoint':'COMPLETE_SOURCE_AND_NEW_LP_CONSTRUCTIONS_NOT_POSITIVITY'})
    save(root/'preparation.json',{'parent_manifest_sha256':sha(dst/'manifest.json'),'request':source['request'],'pair':pair,
        'scope':'Regenerated input/prefix/all expert layers from frozen stored source parameters; checkpoint capture excluded.'})
    save(root/'generation.json',{'arm':arm,'manifest_sha256':sha(dst/'manifest.json'),'prefix_generation_seconds':prefix_seconds,
        'steps':times,'range_rows':range_rows,'range_native_calls':sum(r['calls'] for r in range_rows),
        'joint_and_output_seconds':output_seconds,'build_seconds':time.monotonic()-start,
        'new_output_obligations':len(obs['rows']),'old_LP_certificates_used':0,'network_forward_calls':0,
        'checkpoint_loads':0,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args()
    build(a.root,json.loads((a.root/'job.json').read_bytes()),json.loads((a.root/'build_window.json').read_bytes())['deadline'])
