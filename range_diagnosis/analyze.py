"""No-solve localization on hash-bound saved source/LP records; NOT a prover."""
import argparse
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time

from router_source.checker import tensor
from source_enclosure.format import identity
from upstream_source.checker import csr, rational

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'docs/range_pipeline_v1_review.json'
PACKAGE=ROOT/'data/moe/results/range_pipeline_conv98_20260920_v1/range_on/relocated'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def classification(lo,hi):
    if lo>hi:raise ValueError('reversed range')
    return 'active' if lo>=0 else 'inactive' if hi<=0 else 'unstable'


def triangle_gap(lo,hi):
    """Max vertical gap of the scalar continuous ReLU triangle, not whole LP."""
    return -lo*hi/(hi-lo) if classification(lo,hi)=='unstable' else F(0)


def map_outputs(outputs,cids,bids,global_ids):
    if len(set(cids+bids))!=len(cids+bids) or len(set(global_ids))!=len(global_ids):
        raise ValueError('aliased factor identity')
    indices={name:j for j,name in enumerate(global_ids)}
    if any(name not in indices for name in cids+bids):raise ValueError('missing factor identity')
    c=list(map(rational,outputs['c']));gc=csr(outputs['Gc'],[len(c),len(cids)]);gb=csr(outputs['Gb'],[len(c),len(bids)])
    rows=[]
    for x,y in zip(gc,gb,strict=True):
        rows.append({**{indices[cids[j]]:v for j,v in x.items()},**{indices[bids[j]]:v for j,v in y.items()}})
    return c,rows


def values(projection,point):
    c,rows=projection
    return [a+sum((w*point[j] for j,w in row.items()),F(0)) for a,row in zip(c,rows,strict=True)]


def last_relu_terms(pre,post,coeff,bias,final_margin,weight):
    if not len(pre)==len(post)==len(coeff):raise ValueError('last-layer dimension')
    delta=[h-max(a,F(0)) for a,h in zip(pre,post,strict=True)]
    signed=[weight*w*d for w,d in zip(coeff,delta,strict=True)]
    linear=bias+sum((w*h for w,h in zip(coeff,post,strict=True)),F(0))
    repaired=bias+sum((w*max(a,F(0)) for w,a in zip(coeff,pre,strict=True)),F(0))
    lift=weight*(final_margin-linear)
    if weight*final_margin-lift-sum(signed,F(0))!=weight*repaired:raise ValueError('expert decomposition')
    return delta,signed,lift,weight*repaired


def load_reviewed(package,review_path):
    review=json.loads(review_path.read_bytes())
    arms=[a for a in review['arms'] if a['arm']=='range_on']
    if review['status']!='PASS' or review['issues'] or len(arms)!=1 or not arms[0]['complete']:
        raise ValueError('complete reviewed on-arm required')
    arm=arms[0];manifest=json.loads((package/'manifest.json').read_bytes())
    if sha(package/'manifest.json')!=arm['sealed']['manifest_sha256']:raise ValueError('review/package identity')
    for name,h in manifest['files'].items():
        p=(package/name).resolve()
        if not p.is_relative_to(package.resolve()) or sha(p)!=h:raise ValueError('reviewed file changed')
    if manifest['request']['dataset_index']!=98 or manifest['pair']!=[1,2]:raise ValueError('fixed request')
    return arm,manifest


def analyze(package=PACKAGE,review_path=REVIEW):
    start=time.monotonic();arm,manifest=load_reviewed(package,review_path)
    read=lambda n:json.loads((package/n).read_bytes())
    source=read('source/prefix/router_source.json');doc=read('source/full_experts.json')
    base=read('source/lp_base.json');obs=read('source/obligations.json');trace=read('source/trace.json')
    pair=manifest['pair'];label=manifest['request']['clean_prediction'];classes=manifest['request']['classes']
    if (doc['request']!=manifest['request'] or doc['pair']!=pair or [e['expert'] for e in doc['experts']]!=pair or
            obs['source_sha256']!=trace['joint_sha256'] or identity(base)!=obs['base_sha256']):
        raise ValueError('source/LP binding')
    expected=[i for i in range(classes) if i!=label]
    if ([r['competitor'] for r in obs['rows']]!=expected or
            [r['competitor'] for r in manifest['outcomes']]!=expected or
            [r['competitor'] for r in arm['result']['rows']]!=expected):raise ValueError('all properties required')
    ids=base['continuous_ids']+base['binary_ids'];n=base['variables']
    if len(ids)!=n-2:raise ValueError('LP factor count')
    joint=map_outputs(read('source/joint.state.json')['hz'],base['continuous_ids'],base['binary_ids'],ids)
    inventory={x['name']:{k:x[k] for k in ('dtype','shape','sha256')} for x in source['state_inventory']}
    experts=[];neurons=[];census=[]
    prefix_trace=read('source/prefix/trace.json')
    for e in doc['experts']:
        i=e['expert'];prefix=read(f'source/prefix/expert{i}_relu.state.json')
        ci=list(prefix['continuous_ids']);bi=list(prefix['binary_ids']);previous=identity(prefix);layers={}
        proofs={1:prefix_trace['certificates'][f'expert{i}_relu']}
        steps=[r for r in trace['steps'] if r['expert']==i]
        if [s['layer'] for s in steps]!=['coordinate']+list(range(2,9)):raise ValueError('complete layer roster')
        for step in steps:
            layer=step['layer']
            if step['kind']=='Flatten':
                if step['source']!=previous or step['target']!=previous:raise ValueError('flatten identity')
                continue
            d=read('source/'+step['file'])
            if d['source']!=previous or step['proof']['source']!=previous:raise ValueError('delta parent chain')
            ci+=d['continuous_suffix'];bi+=d['binary_suffix'];previous=d['target']
            if layer in (6,7):layers[layer]=map_outputs(d['outputs'],ci,bi,ids)
            if step['kind']=='ReLU':proofs[layer]=step['proof']
        side='left' if i==pair[0] else 'right';maps=trace['joint_proof']['maps']
        if ([base['continuous_ids'][j] for j in maps[side+'_c']]!=ci or
                [base['binary_ids'][j] for j in maps[side+'_b']]!=bi or trace['joint_proof'][side]!=previous):
            raise ValueError('expert/joint factor map')
        last=e['layers'][8];wi,w=tensor(last['weight']);bident,b=tensor(last['bias']);width=len(layers[6][0])
        if (last['kind']!='Linear' or wi!=inventory[last['weight_name']] or bident!=inventory[last['bias_name']] or
                wi['shape']!=[classes,width] or bident['shape']!=[classes] or len(layers[7][0])!=width):
            raise ValueError('final classifier identity/dimensions')
        coefficients={k:[w[label*width+j]-w[k*width+j] for j in range(width)] for k in expected}
        biases={k:b[label]-b[k] for k in expected}
        for layer,p in proofs.items():
            if len(p['branches'])!=len(p['ranges']):raise ValueError('ReLU roster')
            branches=[classification(*map(F,r)) for r in p['ranges']]
            if branches!=p['branches']:raise ValueError('saved range/branch mismatch')
            census.append({'expert':i,'layer':layer,'counts':dict(Counter(branches)),'rows':len(branches)})
        ranges=[tuple(map(F,r)) for r in proofs[7]['ranges']]
        for j,(lo,hi) in enumerate(ranges):
            gap=triangle_gap(lo,hi)
            potential={k:max(F(0),-coefficients[k][j])*gap for k in expected}
            neurons.append({'expert':i,'row':j,'branch':classification(lo,hi),'range':[str(lo),str(hi)],
                'previous_fixed_roster':j<2,'triangle_gap':str(gap),
                'unweighted_harm_potential_by_competitor':{str(k):str(v) for k,v in potential.items()},
                'max_observed_harm':F(0),'worst_observed_competitor':None})
        experts.append({'expert':i,'pre':layers[6],'post':layers[7],'ranges':ranges,'coefficients':coefficients,'biases':biases})
    all_rows=[];joint_checks=arm['result']['source_checks']['joint'];total_binary=sum(c['counts'].get('unstable',0) for c in census)
    if total_binary!=joint_checks['binary']:raise ValueError('ReLU census/binary count')
    by_neuron={(r['expert'],r['row']):r for r in neurons}
    for obligation,outcome,bound in zip(obs['rows'],manifest['outcomes'],arm['result']['rows'],strict=True):
        k=obligation['competitor'];row={'competitor':k,'checked_lower_bound':bound.get('checked_lower_bound'),
            'point_status':'MISSING_UNVERIFIED_POINT','observations':[]}
        if outcome['file'] is None:all_rows.append(row);continue
        record=read(outcome['file'])
        if (record['competitor']!=k or record['pair']!=pair or record['request_id']!=identity(manifest['request']) or
                record['source_sha256']!=obs['source_sha256'] or record['base_sha256']!=obs['base_sha256'] or
                record['lp_sha256']!=bound['lp_sha256'] or record['parent_manifest_sha256']!=manifest['parent_manifest_sha256']):
            raise ValueError('candidate/source/obligation binding')
        raw=record['approximate_primal_not_checked']
        if raw is None:all_rows.append(row);continue
        point=list(map(rational,raw))
        if len(point)!=n:raise ValueError('stored point dimension')
        lam,w=point[-2:];output=values(joint,point)
        margins=[output[s*classes+label]-output[s*classes+k] for s in range(2)]
        objective=F(obligation['offset'])+sum((v*point[j] for j,v in csr(obligation['objective'],[1,n])[0].items()),F(0))
        if objective!=margins[1]+w:raise ValueError('weighted LP objective identity')
        product=lam*(margins[0]-margins[1]);product_gap=product-w
        lifts=F(0);relu_signed=F(0);repaired=F(0);harm=F(0);stable_gap=F(0)
        max_envelope_excess=F(0);max_range_excess=F(0);min_gap=None
        for s,e in enumerate(experts):
            weight=lam if s==0 else 1-lam;pre=values(e['pre'],point);post=values(e['post'],point)
            delta,signed,lift,repair=last_relu_terms(pre,post,e['coefficients'][k],e['biases'][k],margins[s],weight)
            lifts+=lift;relu_signed+=sum(signed,F(0));repaired+=repair;harm+=sum((max(F(0),-v) for v in signed),F(0))
            for j,((lo,hi),a,h,d,contribution) in enumerate(zip(e['ranges'],pre,post,delta,signed,strict=True)):
                node=by_neuron[(e['expert'],j)]
                min_gap=d if min_gap is None else min(min_gap,d)
                max_range_excess=max(max_range_excess,lo-a,a-hi)
                max_envelope_excess=max(max_envelope_excess,d-triangle_gap(lo,hi))
                if node['branch']!='unstable':stable_gap=max(stable_gap,abs(d));continue
                loss=max(F(0),-contribution)
                if loss>node['max_observed_harm']:node.update(max_observed_harm=loss,worst_observed_competitor=k)
                row['observations'].append({'expert':e['expert'],'row':j,'pre_float':float(a),'post_float':float(h),
                    'relu_gap_float':float(d),'weighted_signed_margin_gap_float':float(contribution)})
        if objective+product_gap-lifts-relu_signed!=repaired:raise ValueError('whole point decomposition identity')
        terms={'relaxed_objective':objective,'product_gap':product_gap,'product_only_replaced':objective+product_gap,
            'final_affine_residual':lifts,'weighted_last_relu_signed_gap':relu_signed,'observed_harm_sum':harm,
            'product_and_last_relu_replaced':repaired,'minimum_last_relu_gap':min_gap,
            'max_stable_relu_discrepancy':stable_gap,'max_range_excess':max_range_excess,
            'max_triangle_gap_excess':max_envelope_excess,'gate_value':lam}
        row.update(point_status='UNVERIFIED_POINT_DIAGNOSTIC_ONLY',candidate_sha256=sha(package/outcome['file']),
            exact_terms={a:str(b) for a,b in terms.items()},display_terms={a:float(b) for a,b in terms.items()},
            decomposition_residual='0')
        all_rows.append(row)
    ranking=sorted((r for r in neurons if r['branch']=='unstable'),key=lambda r:(-r['max_observed_harm'],r['expert'],r['row']))
    ranked=[{'expert':r['expert'],'row':r['row'],'worst_competitor':r['worst_observed_competitor'],
             'max_observed_harm':str(r['max_observed_harm']),'display_harm':float(r['max_observed_harm'])} for r in ranking]
    for r in neurons:r['max_observed_harm']=str(r['max_observed_harm'])
    return {'schema':'LAST_RELU_SAVED_LOCALIZATION_V1','review_sha256':sha(review_path),
        'package_manifest_sha256':sha(package/'manifest.json'),'script_sha256':sha(Path(__file__)),
        'request':manifest['request'],'pair':pair,'source_sha256':obs['source_sha256'],
        'relu_census':census,'total_relu_binaries':total_binary,'last_relu_rows':neurons,
        'last_relu_unstable':len(ranking),'ranking_heuristic':'max harmful signed last-ReLU contribution over ALL saved property points',
        'ranked_unstable_rows':ranked,'required_properties':len(expected),'properties':all_rows,
        'new_solver_calls':0,'network_forward_calls':0,'new_source_propagations':0,
        'exact_primal_feasibility_checked':False,'new_lower_bounds_proved':0,'new_complete_safe':0,
        'production_verdict_changed':False,'seconds':time.monotonic()-start,
        'scope':'Saved-point localization only. Product/ReLU replacement holds other LP factors fixed, not a network forward, sound new bound or feasible repaired point. No unique root cause or LP impossibility claim.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=analyze();a.output.mkdir(exist_ok=False)
    with (a.output/'analysis.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ('required_properties','last_relu_unstable','new_solver_calls','seconds')}))
