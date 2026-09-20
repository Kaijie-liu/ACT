"""Extend checked prefixes; produce NEW complete expert state and all output LPs."""
import argparse
import json
from pathlib import Path
import shutil
import time
import resource
from router_source.capture import ROOT,sha
from router_source.build import save,JOB,JOB_HASH
from source_enclosure.format import identity
from source_enclosure.produce import relu,join
from full_source.capture import capture
from full_source.graph import validate,operator
from full_source.lift import affine
from full_source.delta import encode
from full_source.obligations import build as obligations

PREFIX=ROOT/'data/moe/results/source_enclosure_conv98_20260920_v1/relocated'
PREFIX_HASH='01e7a3b62ae49c1ed89eb6ddc3aaaf23ce8f1ca202c062be5483b04b896a2579'
CODE={'full_source/graph.py':'full_graph.py','full_source/delta.py':'delta_format.py',
      'full_source/check_lift.py':'lift_check.py','full_source/check_obligations.py':'obligation_check.py',
      'full_source/verify.py':'verify_full.py'}


def build(root, prefix=PREFIX, prefix_hash=PREFIX_HASH, supplied=None):
    start=time.monotonic();load=lambda p:json.loads(p.read_bytes())
    if sha(prefix/'manifest.json')!=prefix_hash:raise ValueError('prefix identity')
    pm=load(prefix/'manifest.json')
    for name,h in pm['files'].items():
        p=(prefix/name).resolve()
        if not p.is_relative_to(prefix.resolve()) or sha(p)!=h:raise ValueError('prefix file changed')
    dst=root/'relocated';dst.mkdir();t=time.monotonic();shutil.copytree(prefix,dst/'prefix')
    copy_seconds=time.monotonic()-t;source=load(prefix/'router_source.json');pair=pm['pair']
    t=time.monotonic()
    if supplied is None:
        if sha(JOB)!=JOB_HASH:raise ValueError('original request job changed')
        doc=capture(load(JOB),source,pair)
    else:doc=supplied
    capture_seconds=time.monotonic()-t
    validate(doc,source,load(prefix/'experts.json'));save(dst/'full_experts.json',doc)
    steps=[];ends=[];times=[];base=load(prefix/'guard.state.json')
    for e in doc['experts']:
        i=e['expert'];shape=operator(source['request']['lower']['shape'],e['layers'][0])[0]
        state=load(prefix/f'expert{i}_relu.state.json')
        for index in ['coordinate']+list(range(2,9)):
            t=time.monotonic();old=state;before=shape
            if index=='coordinate':
                kind='CoordinateLift';rows=[{j:1} for j in range(len(state['hz']['c']))];bias=[0]*len(rows)
            else:
                kind=e['layers'][index]['kind'];shape,rows,bias=operator(shape,e['layers'][index])
            tag=f'expert{i}/full/{index}';row={'expert':i,'layer':index,'kind':kind,'input_shape':before,'output_shape':shape}
            if kind=='Flatten':
                row.update(source=identity(state),target=identity(state))
            else:
                if kind=='ReLU':state,proof=relu(state,tag)
                else:state,proof=affine(state,rows,bias,tag)
                file=f'expert{i}_{index}.delta.json';save(dst/file,encode(old,state));row.update(file=file,proof=proof)
            steps.append(row);times.append({'expert':i,'layer':index,'seconds':time.monotonic()-t,
                'outputs':len(state['hz']['c']),'continuous':len(state['continuous_ids']),
                'binary':len(state['binary_ids']),'equalities':len(state['hz']['b']),
                'nnz':sum(len(state['hz'][k]['data']) for k in ('Gc','Gb','Ac','Ab','Auc','Aub'))})
            save(root/f'progress_{len(steps):02d}.json',times[-1])
        if shape!=[1,source['request']['classes']]:raise ValueError('expert classification endpoint')
        ends.append(state)
    t=time.monotonic();joint,proof=join(base,*ends);save(dst/'joint.state.json',joint)
    join_seconds=time.monotonic()-t;t=time.monotonic()
    common,obs=obligations(joint,pair,source['request']['classes'],source['request']['clean_prediction'])
    save(dst/'lp_base.json',common);save(dst/'obligations.json',obs);output_seconds=time.monotonic()-t
    save(dst/'trace.json',{'schema':'COMPLETE_EXPERT_TRACE_V1','steps':steps,'joint_proof':proof,
                           'joint_sha256':identity(joint)})
    for src,name in CODE.items():shutil.copyfile(ROOT/src,dst/name)
    files={str(p.relative_to(dst)):sha(p) for p in sorted(dst.rglob('*')) if p.is_file()}
    save(dst/'manifest.json',{'schema':'COMPLETE_EXPERT_OUTPUT_LP_BUNDLE_V1','prefix_manifest_sha256':prefix_hash,
        'request':source['request'],'pair':pair,'files':files,'endpoint':'COMPLETE_EXPERTS_AND_NEW_OUTPUT_LP_CONSTRUCTIONS_NO_BOUNDS'})
    save(root/'generation.json',{'manifest_sha256':sha(dst/'manifest.json'),'bundle_bytes':sum(p.stat().st_size for p in dst.rglob('*') if p.is_file()),
        'copy_prefix_seconds':copy_seconds,'capture_seconds':capture_seconds,'steps':times,
        'join_and_serialization_seconds':join_seconds,'output_construction_serialization_seconds':output_seconds,
        'total_build_before_publication_seconds':time.monotonic()-start,
        'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'checkpoint_loads':int(supplied is None),'network_forward_calls':0,'native_solver_calls':0,
        'old_LP_certificates_used':0,'new_output_obligations':len(obs['rows']),
        'joint_nnz':sum(len(joint['hz'][k]['data']) for k in ('Gc','Gb','Ac','Ab','Auc','Aub')),
        'scope':'Copied stored checked prefix + fresh remaining parameter capture/lowering + new output LP construction; old prefix generation excluded.'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);build(p.parse_args().root.resolve())
