"""Independent stdlib audit of R1 recipes/rows/candidates, not DAG execution.

No imports of the producer, DAG support checker, native bridge or fixture helpers.
This rechecks all fixed coefficients and exact residuals. It does not independently
re-execute the native control or prove network lowering.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import time


def packed(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def ident(value):return sha(packed(value))
def need(ok, message):
    if not ok:raise ValueError(message)


def audit(root):
    start=time.monotonic();deadline=start+300
    def tick():
        if time.monotonic()>=deadline:raise TimeoutError('independent audit deadline')
    def read(name):
        p=root/name;need(p.is_file() and not p.is_symlink() and p.stat().st_size<=128*2**20,'audit input')
        raw=p.read_bytes();tick();return raw,json.loads(raw)
    _,terminal=read('terminal.json');_,execution=read('execution.json')
    need(terminal['status']=='PASS' and len(terminal['calls'])==4,'R1 terminal/call inventory')
    for call in terminal['calls']:
        need(call['status']=='COMPLETE' and call['returncode']==0 and
             0<=call['parent_observed_seconds']<300,'call completion/budget')
    need(terminal['code']==execution['code'],'code inventories')
    for name,h in execution['code'].items():
        need(sha((root/'code'/name).read_bytes())==h,'archived code bytes')
    results=[]
    for name in ('small','dense'):
        raw,doc=read('objects/'+name+'.json');_,receipt=read('objects/'+name+'_receipt.json')
        need(packed(doc)==raw and sha(raw)==receipt['graph_sha256'],'graph bytes')
        need(set(doc)=={'schema','request','source','frame','factors','domains','target','nodes','roots'} and
             doc['schema']=='HZ_COEFFICIENT_DAG_V1','graph inventory')
        need(doc['request']==ident({'control':'coefficient-dag-frozen-20261013'}),'request recipe')
        if name=='dense':
            n=256;width=3072
            need(doc['frame']==23 and doc['target']==0 and doc['domains']==[
                dict(name='given',frame=23,experts=[],guard=None,parents=[])],'dense domain/frame')
            need(doc['factors']==[[f'x/{j}','continuous',-1] for j in range(width)],'dense factor map')
            need(doc['source']==ident(dict(recipe='fixed-dense-duplicate-v1',width=width,rows=n,salt=2,
                rhs='output:(r-2)/16;eq:(3r-1)/32')),'dense declared recipe identity')
            need(len(doc['nodes'])==512 and len(doc['roots'])==512,'dense node/root inventory')
            # Exact direct residual with q=1/8, z=(r%3+1)/16, independent of DAG.
            rnum=[0]*width;checked=0
            for r in range(n):
                tick();node=doc['nodes'][r]
                need(node[0]==f'original/{r}' and node[1]==0 and len(node[2])==width and node[3]==[],
                     'dense original node')
                for c,entry in enumerate(node[2]):
                    numerator=(-1 if (r+c)%2 else 1)*(1+(17*r+7*c+6)%31)
                    need(entry==[c,str(F(numerator,32))],'every dense original coefficient')
                    rnum[c]+=(2+(r%3+1))*numerator;checked+=1
                need(doc['nodes'][n+r]==[f'negative/{r}',0,[],[[r,'-1']]],'every dense negative reference')
                need(doc['roots'][r]==[f'output/{r}',0,'output',str(F(r-2,16)),r],'every output rhs/root')
                need(doc['roots'][n+r]==[f'eq/{r}',0,'eq',str(F(3*r-1,32)),n+r],'every eq rhs/root')
            need(checked==786432,'dense complete coefficient count')
            need(doc['nodes'][0][2]==doc['nodes'][62][2] and
                 F(doc['roots'][256+62][3])-F(doc['roots'][256][3])==F(93,16),'inconsistent equality control')
            expected_stats=dict(nodes=512,roots=512,local_nnz=786432,references=256,active=3072,bytes=len(raw))
            expected_cases=[(False,F(161793,256),[[j,str(F(v,512))] for j,v in enumerate(rnum)],786432),
                            (True,F(-261,2),[[j,'0'] for j in range(width)],0)]
            need(len(receipt['results'])==2,'dense complete candidate inventory')
        else:
            # Original rows are specified directly, never expanded using node refs.
            need(doc['frame']==17 and doc['target']==3 and doc['factors']==[
                ['shared/x','continuous',-1],['shared/y','continuous',-1],
                ['private/0','continuous',0],['private/1','continuous',1],['private/2','binary',2]],
                'small frame/factor inventory')
            need(doc['domains']==[dict(name='common',frame=17,experts=[],guard=None,parents=[]),
                dict(name='expert/0',frame=17,experts=[0],guard=None,parents=[0]),
                dict(name='expert/1',frame=17,experts=[1],guard=None,parents=[0]),
                dict(name='pair/0/1',frame=17,experts=[0,1],guard=[0,1],parents=[0,1,2])],
                'small domain/guard inventory')
            nodes=[['affine',0,[[0,'1'],[1,'2']],[]],['active-alias',0,[],[[0,'1']]],
                ['negative-with-private',1,[[2,'1/2']],[[0,'-1']]],
                ['other-expert',2,[[3,'-1/2']],[[0,'1']]],
                ['diamond',3,[],[[1,'1'],[2,'1']]],['guard',3,[[0,'1']],[]]]
            roots=[['out/0',0,'output','3/4',0],['out/1',0,'output','7/4',1],
                ['out/2',2,'output','1/8',3],['out/3',3,'output','0',4],['relu/eq',1,'eq','-1/4',2],
                ['guard/0',3,'ine','0',5],['guard/1',3,'ine','1',5]]
            original=[('output','3/4',{0:F(1),1:F(2)}),('output','7/4',{0:F(1),1:F(2)}),
                ('output','1/8',{0:F(1),1:F(2),3:F(-1,2)}),('output','0',{2:F(1,2)}),
                ('eq','-1/4',{0:F(-1),1:F(-2),2:F(1,2)}),('ine','0',{0:F(1)}),('ine','1',{0:F(1)})]
            need(doc['nodes']==nodes and doc['roots']==roots,'all small nodes/roots')
            need(doc['source']==ident([[k,c,[[j,str(v)] for j,v in sorted(es.items())]] for k,c,es in original]),
                 'small original row source identity')
            weights=dict(output=list(map(F,['1','-1','2','-3'])),eq=[F(1,4)],ine=[F(-1,2),F(-1,4)])
            positions=dict(output=0,eq=0,ine=0);constant=F(1,8);residual=[F(0)]*4
            for kind,c,es in original:
                w=weights[kind][positions[kind]];positions[kind]+=1;constant+=w*F(c)
                for j,v in es.items():residual[j]+=(w if kind=='output' else -w)*v
            expected_stats=dict(nodes=6,roots=7,local_nnz=5,references=5,active=4,bytes=len(raw))
            expected_cases=[(False,constant,[[j,str(v)] for j,v in enumerate(residual)],5)]
            need(len(receipt['results'])==1,'small complete candidate inventory')
        need(receipt['stats']==expected_stats,'stored component counts')
        for result,(cancel,constant,residual,local) in zip(receipt['results'],expected_cases,strict=True):
            tick();q=result['query'];duals=result['duals']
            need(q['graph']==sha(raw) and q['request']==doc['request'] and q['source']==doc['source']
                and q['target']==doc['target'] and result['query_sha256']==ident(q),'query binding')
            if name=='dense':
                need(q['offset']=='0' and q['weights']==['1/8']*256 and duals==dict(ine=[],
                    eq=['-1/8' if cancel else str(F(r%3+1,16)) for r in range(256)]),'frozen dense candidate')
            else:need(q['offset']=='1/8' and q['weights']==['1','-1','2','-3'] and
                duals==dict(eq=['1/4'],ine=['-1/2','-1/4']),'frozen small candidate')
            lower=constant-sum(abs(F(v)) for _,v in residual)
            need(result['cancel']==cancel and result['constant']==str(constant) and result['lower']==str(lower)
                and result['residual_sha256']==ident(residual),'full independent residual and bound')
            need(result['local_support_visits']==local and result['reference_support_visits']==(256 if name=='dense' else 5),
                 'component accumulation counters')
            results.append(dict(case=name,cancel=cancel,lower=str(lower),local_support_visits=local))
        need(sum(receipt['checker_work'].values())<=8000000,'cumulative checker work')
    tick();return dict(status='PASS',seconds=time.monotonic()-start,results=results,
        dense_original_coefficients_checked=786432,dense_given_equality_system='INFEASIBLE_DUPLICATE_ROW_RHS',
        dense_infeasibility_gap='93/16',guarantee='FIXED_GIVEN_ROW_ALGEBRA_ONLY',
        real_models=0,gpu_calls=0,full_request_certificates=0)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);args=p.parse_args()
    print(json.dumps(audit(args.archive.resolve()),indent=2))
