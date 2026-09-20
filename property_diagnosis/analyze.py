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
REVIEW=ROOT/'docs/property_ranges_v1_review.json'
REVIEW_HASH='0b90c33a8c11d5bbadf2c386812c0189ac78a323a0ea0d1e94dafccfdd112760'
PACKAGE=ROOT/'data/moe/results/property_ranges_conv98_20260920_v1'
COMMON=[1,2,3,4,5,6,8,9]


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


def load_reviewed(package,review_path,arm_name,expected_review_hash=REVIEW_HASH):
    if sha(review_path)!=expected_review_hash:raise ValueError('archived review identity')
    review=json.loads(review_path.read_bytes())
    arms=[a for a in review['arms'] if a['arm']==arm_name]
    if review['status']!='PASS' or review['issues'] or len(arms)!=1 or not arms[0]['complete']:
        raise ValueError('complete reviewed requested arm required')
    arm=arms[0];manifest=json.loads((package/'manifest.json').read_bytes())
    if sha(package/'manifest.json')!=arm['sealed']['manifest_sha256']:raise ValueError('review/package identity')
    for name,h in manifest['files'].items():
        p=(package/name).resolve()
        if not p.is_relative_to(package.resolve()) or sha(p)!=h:raise ValueError('reviewed file changed')
    if manifest['request']['dataset_index']!=98 or manifest['pair']!=[1,2]:raise ValueError('fixed request')
    return arm,manifest


def analyze_arm(package,review_path,arm_name,groups):
    start=time.monotonic();arm,manifest=load_reviewed(package,review_path,arm_name)
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
            neurons.append({'expert':i,'row':j,'branch':classification(lo,hi),'range':[str(lo),str(hi)],
                'group':group_of(i,j,groups)})
        experts.append({'expert':i,'pre':layers[6],'post':layers[7],'ranges':ranges,'coefficients':coefficients,'biases':biases})
    all_rows=[];joint_checks=arm['result']['source_checks']['joint'];total_binary=sum(c['counts'].get('unstable',0) for c in census)
    if total_binary!=joint_checks['binary']:raise ValueError('ReLU census/binary count')
    by_neuron={(r['expert'],r['row']):r for r in neurons}
    for obligation,outcome,bound in zip(obs['rows'],manifest['outcomes'],arm['result']['rows'],strict=True):
        k=obligation['competitor']
        if k not in COMMON:continue # ninth obligation retained only in endpoint ledger; no point analysis
        row={'competitor':k,'checked_lower_bound':bound.get('checked_lower_bound'),
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
        if obligation['gate']!=['0','1']:raise ValueError('registered gate range')
        difference=margins[0]-margins[1]
        product=lam*difference;product_gap=product-w
        contributions={name:F(0) for name in ('prefix_selected','property_selected','other')}
        dual=dual_terms(bound,objective)
        lifts=F(0);relu_signed=F(0);repaired=F(0);harm=F(0);stable_gap=F(0)
        max_envelope_excess=F(0);max_range_excess=F(0);min_gap=None
        for s,e in enumerate(experts):
            weight=lam if s==0 else 1-lam;pre=values(e['pre'],point);post=values(e['post'],point)
            delta,signed,lift,repair=last_relu_terms(pre,post,e['coefficients'][k],e['biases'][k],margins[s],weight)
            lifts+=lift;relu_signed+=sum(signed,F(0));repaired+=repair;harm+=sum((max(F(0),-v) for v in signed),F(0))
            for j,((lo,hi),a,h,d,contribution) in enumerate(zip(e['ranges'],pre,post,delta,signed,strict=True)):
                node=by_neuron[(e['expert'],j)]
                contributions[node['group']]+=contribution
                min_gap=d if min_gap is None else min(min_gap,d)
                max_range_excess=max(max_range_excess,lo-a,a-hi)
                max_envelope_excess=max(max_envelope_excess,d-triangle_gap(lo,hi))
                if node['branch']!='unstable':stable_gap=max(stable_gap,abs(d));continue
                row['observations'].append({'expert':e['expert'],'row':j,'pre_float':float(a),'post_float':float(h),
                    'relu_gap_float':float(d),'weighted_signed_margin_gap_float':float(contribution),
                    'group':node['group'],'exact_relu_gap':str(d),'exact_signed_contribution':str(contribution)})
        if (objective+product_gap-lifts-relu_signed!=repaired or
                sum(contributions.values(),F(0))!=relu_signed):raise ValueError('whole point decomposition identity')
        terms={'relaxed_objective':objective,'product_gap':product_gap,'product_only_replaced':objective+product_gap,
            'final_affine_residual':lifts,'weighted_last_relu_signed_gap':relu_signed,'observed_harm_sum':harm,
            'product_and_last_relu_replaced':repaired,'minimum_last_relu_gap':min_gap,
            'max_stable_relu_discrepancy':stable_gap,'max_range_excess':max_range_excess,
            'max_triangle_gap_excess':max_envelope_excess,'gate_value':lam,
            'last_relu_only_replaced':objective-lifts-relu_signed,
            'difference_value':difference,'relaxed_product':w,**dual}
        row.update(point_status='UNVERIFIED_POINT_DIAGNOSTIC_ONLY',candidate_sha256=sha(package/outcome['file']),
            exact_terms={a:str(b) for a,b in terms.items()},display_terms={a:float(b) for a,b in terms.items()},
            decomposition_residual='0',
            gate_range=obligation['gate'],difference_range=obligation['difference'],
            exact_relu_groups={name:str(v) for name,v in contributions.items()},
            solver_record={k:record['solver'][k] for k in ('status','success','reported_objective_including_offset','native_seconds')})
        all_rows.append(row)
    unstable=sum(r['branch']=='unstable' for r in neurons)
    return {'schema':'PAIRED_SAVED_LOCALIZATION_ARM_V1','arm':arm_name,'review_sha256':sha(review_path),
        'package_manifest_sha256':sha(package/'manifest.json'),'script_sha256':sha(Path(__file__)),
        'request':manifest['request'],'pair':pair,'source_sha256':obs['source_sha256'],
        'relu_census':census,'total_relu_binaries':total_binary,'last_relu_rows':neurons,
        'last_relu_unstable':unstable,'properties':all_rows,'diagnostic_properties':COMMON,
        'required_output_properties':len(expected),
        'endpoint_ledger':[{'competitor':r['competitor'],'status':r['status'],
                           'paired_diagnosis':r['competitor'] in COMMON} for r in arm['result']['rows']],
        'new_solver_calls':0,'network_forward_calls':0,'new_source_propagations':0,
        'exact_primal_feasibility_checked':False,'new_lower_bounds_proved':0,'new_complete_safe':0,
        'production_verdict_changed':False,'seconds':time.monotonic()-start,
        'scope':'Own-matrix unverified point arithmetic only; no causal, primal, optimum or safety upgrade.'}

def group_of(expert,row,groups):
    for name,selected in groups.items():
        if [expert,row] in selected:return name
    return 'other'


def dual_terms(bound,objective):
    lower,constant,correction=[F(bound[k]) for k in ('checked_lower_bound','dual_constant','residual_box_correction')]
    if lower!=constant+correction:raise ValueError('checked dual accounting')
    return {'checked_lower_bound':lower,'dual_constant':constant,'residual_box_correction':correction,
            'residual_l1':F(bound['residual_l1']),
            'nonzero_residual_coordinates':F(bound['nonzero_residual_coordinates']),
            'point_minus_bound':objective-lower}


def paired_terms(left,right):
    if left['competitor']!=right['competitor']:raise ValueError('paired property identity')
    a={k:F(v) for k,v in left['exact_terms'].items()};b={k:F(v) for k,v in right['exact_terms'].items()}
    keys=('checked_lower_bound','relaxed_objective','product_gap','final_affine_residual',
          'weighted_last_relu_signed_gap','product_and_last_relu_replaced','point_minus_bound',
          'dual_constant','residual_box_correction')
    delta={k:b[k]-a[k] for k in keys}
    rhs=(delta['product_and_last_relu_replaced']-delta['product_gap']+delta['final_affine_residual']
         +delta['weighted_last_relu_signed_gap']-delta['point_minus_bound'])
    if delta['checked_lower_bound']!=rhs:raise ValueError('paired decomposition')
    group_delta={k:F(right['exact_relu_groups'][k])-F(left['exact_relu_groups'][k])
                 for k in ('prefix_selected','property_selected','other')}
    if sum(group_delta.values(),F(0))!=delta['weighted_last_relu_signed_gap']:raise ValueError('paired group accounting')
    return {'competitor':left['competitor'],'exact_delta':{k:str(v) for k,v in delta.items()},
            'display_delta':{k:float(v) for k,v in delta.items()},
            'exact_relu_group_delta':{k:str(v) for k,v in group_delta.items()},'decomposition_residual':'0'}


def analyze():
    started=time.monotonic();review=json.loads(REVIEW.read_bytes())
    if sha(REVIEW)!=REVIEW_HASH or [a['arm'] for a in review['arms']]!=['prefix','property']:
        raise ValueError('frozen two-arm review')
    groups={name+'_selected':[[int(i),j] for i,s in review['arms'][slot]['selection_snapshots'].items()
                             for j in s['record']['selected']] for slot,name in enumerate(('prefix','property'))}
    if (any(len(v)!=4 for v in groups.values()) or
            set(map(tuple,groups['prefix_selected']))&set(map(tuple,groups['property_selected']))):
        raise ValueError('fixed disjoint row groups')
    arms=[analyze_arm(PACKAGE/name/'relocated',REVIEW,name,groups) for name in ('prefix','property')]
    if arms[0]['request']!=arms[1]['request']:raise ValueError('paired request identity')
    if any([r['competitor'] for r in a['properties']]!=COMMON for a in arms):raise ValueError('eight common properties')
    paired=[]
    for left,right in zip(arms[0]['properties'],arms[1]['properties'],strict=True):
        if left['point_status']!='UNVERIFIED_POINT_DIAGNOSTIC_ONLY' or right['point_status']!='UNVERIFIED_POINT_DIAGNOSTIC_ONLY':
            paired.append({'competitor':left['competitor'],'status':'MISSING_SAVED_POINT_NO_PAIRED_ACCOUNTING'})
        else:paired.append(paired_terms(left,right))
    return {'schema':'PAIRED_PROPERTY_SAVED_DIAGNOSIS_V1','review_sha256':sha(REVIEW),
            'script_sha256':sha(Path(__file__)),'diagnostic_properties':COMMON,'required_output_properties':9,
            'excluded_from_point_diagnosis':[7],'row_groups':groups,'arms':arms,'paired':paired,
            'new_solver_calls':0,'network_forward_calls':0,'new_source_propagations':0,'new_lower_bounds_proved':0,
            'new_complete_safe':0,'exact_primal_feasibility_checked':False,'exact_LP_optimality_checked':False,
            'production_verdict_changed':False,'seconds':time.monotonic()-started,
            'scope':'Paired saved-point accounting at different own-matrix points, not causal attribution or proof.'}


def no_external_work(event,args):
    if event.startswith(('subprocess.','socket.')) or event in ('os.system','os.exec','os.fork'):
        raise PermissionError('saved-only diagnosis: no external execution')
    if event=='import' and args[0].split('.')[0] in ('torch','numpy','scipy','highspy','gurobipy'):
        raise ImportError('saved-only diagnosis: no model/solver imports')


if __name__=='__main__':
    import sys
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not sys.flags.no_site:raise ValueError('python -S required')
    sys.addaudithook(no_external_work)
    if a.output.exists():raise ValueError('new analysis directory required')
    result=analyze();a.output.mkdir(exist_ok=False)
    with (a.output/'analysis.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps({'properties':COMMON,'arms':len(result['arms']),'new_solver_calls':0,'seconds':result['seconds']}))
