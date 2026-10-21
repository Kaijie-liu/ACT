"""B2 same-object numerical workers, under the existing owned-session deadline.

No source propagation in the GenBaB path: all unordered pairs are guarded and
covered. The separate legacy CROWN path pays for its own ACT route frontend.
"""
import time
ENTRY=time.monotonic()
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load,save,sha,tick
import genbab_cpu_ready as backend
from act.pipeline.moe.genbab_spec import make_request,clauses,receive,digest,ENGINE


def properties(classes,label,indices=None):
    indices=list(range(classes-1)) if indices is None else indices
    competitors=[i for i in range(classes) if i!=label]
    out=[]
    for i in indices:
        row=[0.]*classes;row[label]=1.;row[competitors[i]]=-1.
        out.append(dict(id=f'property_{i}',q=row,c=0.))
    return out


def obligation_status(status):
    if status not in ('verified','falsified','unknown'):
        raise ValueError('unexpected verifier terminal status: '+str(status))
    # Falsified static obligations are not full-model counterexamples.
    return 'UNSAT' if status=='verified' else 'UNKNOWN'


def load_model(item):
    from act.pipeline.moe.external_pair_worker import load as load_same
    # This loader checks checkpoint, model state, exact represented tensors,
    # epsilon box and clean label. It does not load another arm's conclusions.
    return load_same(item)


def full_replay(model,tensors,candidate,label):
    import torch
    from act.pipeline.moe.experiment1 import _forward_validate
    shape=tuple(tensors['center'].shape[1:])
    if candidate.shape!=shape or not torch.isfinite(candidate).all():
        return dict(valid=False,reason='SHAPE_OR_NONFINITE')
    lo,hi=tensors['lower'][0],tensors['upper'][0]
    if not bool(((candidate>=lo)&(candidate<=hi)).all()):
        return dict(valid=False,reason='OUTSIDE_SAVED_BOX')
    with torch.no_grad():output=model(candidate.unsqueeze(0))
    if not torch.isfinite(output).all():return dict(valid=False,reason='NONFINITE_OUTPUT')
    return _forward_validate(model,candidate,lower=tensors['lower'],upper=tensors['upper'],clean_prediction=label)


def verify_graph(graph,lower,upper,property_count,deadline):
    from api import ABCrownSolver,VerificationSpec,default_config
    from act.back_end.moe.genbab_pair import scalar_bad_set_clauses
    raw=scalar_bad_set_clauses(property_count)
    spec=VerificationSpec.build_from_input_bounds(lower,upper,[raw])
    config=default_config()
    for key,value in backend.CONFIG.items():backend.put_config(config,key,value)
    if (config['bab']['timeout_scale']!=1 or config['bab']['override_timeout'] is not None
            or config['bab']['cut']['enabled'] or config['solver']['beta-crown']['all_node_split_LP']):
        raise ValueError('unchanged shared-time non-LP/non-cut configuration')
    tick(deadline);config['bab']['timeout']=max(.001,deadline-time.monotonic()-2)
    start=time.monotonic()
    solver=ABCrownSolver(spec=spec,computing_graph=graph,config=config)
    result=solver.verify(return_reference=True)
    tick(deadline)
    obligation_status(result.status)
    stats=result.stats
    # A single aggregate BaB bound is not a per-property certificate table.
    record=dict(status=result.status,success=result.success,
        stats={k:backend.json_value(stats.get(k)) for k in ('elapsed','bab','pgd')},
        stage='nonlinear_bab' if stats.get('bab') else 'incomplete_only',
        seconds=time.monotonic()-start,actual_raw_spec=raw,
        bound_propagation='alpha-crown',complete_verifier='bab',branching='nonlinear',
        guarantee='GENBAB_NUMERICAL_FILTER_NOT_INDEPENDENT_PROOF')
    net=getattr(result.reference.get('model'),'net',None)
    record['split_activations']={} if net is None else {
        str(key):[dict(name=node.name,kind=type(node).__name__,input_index=index) for node,index in values]
        for key,values in getattr(net,'split_activations',{}).items()}
    return record,stats.get('attack_examples')


def multi_property_control(folder,deadline):
    """One new API-shape control, not a rerun of the earlier two smoke arms."""
    save(folder/'environment.json',backend.probe())
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64)
    class Graph(torch.nn.Module):
        def forward(self,x):
            f=.5+x+torch.sigmoid(2*x)*(-2*x)
            guard=-.5-x+torch.relu(2*x)
            values=torch.cat((f,f+.25),dim=1)
            return values+torch.relu(guard-values)
    graph=Graph().eval();lo=torch.tensor([[-1.]]);hi=torch.tensor([[1.]])
    with torch.no_grad():
        import math
        for x in (-1.,-.5,0.,.5,1.):
            actual=graph(torch.tensor([[x]]))[0]
            f=.5-x*math.tanh(x);g=abs(x)-.5
            expected=torch.tensor([max(f,g),max(f+.25,g)])
            if not torch.allclose(actual,expected,atol=1e-14,rtol=1e-14):raise ValueError('vector guard graph')
    result,attack=verify_graph(graph,lo,hi,2,deadline)
    if result['status']=='falsified':raise ValueError('contradiction on fixed safe vector fixture')
    return dict(status='API_COMPLETED',verifier=result,real_calls=0,native_calls=1,
                acceptance='spec shape and ordinary completion, no new positive gate')


def genbab(folder,query,deadline):
    save(folder/'environment.json',backend.probe())
    import torch
    from act.util.typing_compat import install_typing_override
    install_typing_override()
    from act.back_end.moe.genbab_pair import GenBaBGuardedProperties
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    item=query['item'];model,tensors=load_model(item)
    center,lo,hi=[tensors[k] for k in ('center','lower','upper')]
    with torch.no_grad():classes=int(model(center).shape[-1])
    props=properties(classes,item['sample']['label'],query.get('property_indices'))
    graph_id=digest(dict(adapter_sha256=sha(ROOT/'act/back_end/moe/genbab_pair.py'),
                        checkpoint=item['subject']['checkpoint_sha256'],properties=props,
                        encoding='RELU_MAX_EACH_PROPERTY_ALL_PAIR_GUARDS'))
    request=make_request(model_sha256=item['subject']['checkpoint_sha256'],
        input_sha256=digest(item['sample']),graph_sha256=graph_id,experts=model.spec.num_experts,
        classes=classes,properties=props,mode='guarded',deadline=deadline)
    all_clauses=clauses(request,deadline=deadline)
    pairs=request['pairs'] if query['kind']=='genbab_aggregate' else [query['pair']]
    save(folder/'spec.json',dict(request=request,clause_roster=all_clauses,
        requested_pairs=pairs,query=query,deadline=deadline,encoding='scalar-max',
        OR='one single-row unsafe alternative for each property; AND only inside scalar max',
        configuration=backend.CONFIG))
    receipts=[];results=[];witness=None
    for pair in pairs:
        if time.monotonic()+3>=deadline:break
        graph=GenBaBGuardedProperties(model,tuple(pair),props).eval()
        selected=[c for c in all_clauses if c['pair']==pair]
        n=len(results)
        save(folder/f'pair_{n:02d}_started.json',dict(pair=pair,clock=time.monotonic(),
            clause_sha256=[digest(c) for c in selected],graph_sha256=graph_id,deadline=deadline))
        outcome,attacks=verify_graph(graph,lo,hi,len(props),deadline)
        replay=[]
        if attacks is not None:
            image_shape=tuple(center.shape[1:]);pixels=center[0].numel()
            if attacks.numel()%pixels:raise ValueError('external candidate input dimensions')
            for candidate in attacks.detach().cpu().double().reshape(-1,*image_shape):
                tick(deadline)
                row=full_replay(model,tensors,candidate,item['sample']['label'])
                with torch.no_grad():scores=graph(candidate.unsqueeze(0))[0]
                row['scalar_properties']=backend.json_value(scores)
                row['guarded_static_violation']=bool(torch.isfinite(scores).all() and (scores<=0).any())
                replay.append(row)
                if row['valid'] and witness is None:
                    torch.save(candidate,folder/'witness.pt');witness=sha(folder/'witness.pt')
        # No real request has to enter BaB if alpha-CROWN already proves it.
        # The smoke's actual branching gate is not repeated for every query.
        status=obligation_status(outcome['status'])
        for clause in selected:
            receipts.append(dict(request_sha256=digest(request),clause_sha256=digest(clause),
                engine=ENGINE,stage=outcome['stage'],status=status))
        row=dict(pair=pair,outcome=outcome,replay=replay,clause_sha256=[digest(c) for c in selected],
                 clock=time.monotonic())
        save(folder/f'pair_{n:02d}_result.json',row);results.append(row)
        if witness is not None or outcome['status']!='verified':break
        del graph
    if query['kind']=='genbab_aggregate':
        aggregate=receive(request,receipts,deadline=deadline)
        status=aggregate['status']
    else:
        aggregate=dict(scope='SELECTED_GUARDED_PAIR_DIAGNOSTIC_NOT_COMPLETE_REQUEST',
            expected=len(props),received=len(receipts),formal_SAFE=False)
        status='POSITIVE' if len(receipts)==len(props) and all(r['status']=='UNSAT' for r in receipts) else 'UNKNOWN'
    if witness is not None:status='UNSAFE_REPLAYED'
    tick(deadline)
    return dict(status=status,aggregate=aggregate,receipts=receipts,pairs=results,
        witness_sha256=witness,expected_pairs=len(pairs),completed_pairs=len(results),
        evidence_grade='GENBAB_NUMERICAL_FILTER',formal_SAFE=False,
        selection='B1_FAILURE_SELECTED_DEVELOPMENT_NOT_FAIR_POPULATION_COMPARISON')


def crown(folder,query,deadline):
    from act.util.typing_compat import install_typing_override
    install_typing_override()
    from act.util.options import PerformanceOptions
    from act.pipeline.moe.external_pair_worker import frontend
    PerformanceOptions.set_debug_output_file(str(folder/'act_debug.log'))
    request=dict(query['item'],method='external_crown',head=query['code_commit'])
    save(folder/'request.json',request)
    # Legacy frontend creates an independent child in the SAME owned session.
    # Its original 300-second clock starts at the outer launch, not here.
    os.chdir(ROOT);frontend(folder,query['started_at'])
    tick(deadline)
    output=load(folder/'external.json',limit=2**20)
    if output['status']=='UNSAFE':
        import torch
        model,tensors=load_model(query['item'])
        point=torch.load(folder/'witness.pt',map_location='cpu',weights_only=True)
        if point.shape==tensors['center'].shape:point=point[0]
        replay=full_replay(model,tensors,point,query['item']['sample']['label'])
        output['full_model_replay']=replay
        output['status']='UNSAFE_REPLAYED' if replay['valid'] else 'UNKNOWN'
    output['evidence_grade']='CROWN_NUMERICAL_FILTER';output['formal_SAFE']=False
    return output


def worker(folder):
    launch=load(folder/'launch.json');deadline=launch['worker_deadline'];query=launch['query']
    backend.limits(deadline)
    # Per-file native writes cannot consume another full stage allowance. The
    # outer receiver still checks cumulative files and rejects any overrun;
    # RLIMIT_FSIZE is not represented as a filesystem-wide quota.
    file_limit=launch['remaining_file_limit']
    if type(file_limit) is not int or not 0<file_limit<=128*2**20:
        raise ValueError('remaining artifact limit')
    resource.setrlimit(resource.RLIMIT_FSIZE,(file_limit,file_limit))
    def cutoff(*_):os.write(2,b'B2_AUTHORIZED_CUTOFF\n');os._exit(124)
    signal.signal(signal.SIGALRM,cutoff)
    with (folder/'native.log').open('xb',buffering=0) as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            if query['kind']=='synthetic_vector_control':result=multi_property_control(folder,deadline)
            elif query['kind']=='plain_crown':result=crown(folder,query,deadline)
            else:result=genbab(folder,query,deadline)
            tick(deadline)
            save(folder/'outcome.json',dict(query_sha256=launch['query_sha256'],result=result,
                                          worker_deadline=deadline,clock=time.monotonic()))
            tick(deadline)
        except BaseException as exc:
            traceback.print_exc()
            if time.monotonic()<deadline:save(folder/'error.json',dict(kind=type(exc).__name__,error=repr(exc)))
            if isinstance(exc,TimeoutError):os.write(2,b'B2_AUTHORIZED_CUTOFF\n');return 124
            return 1
        finally:sys.stdout.flush();sys.stderr.flush()
    return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path,required=True)
    args=parser.parse_args();raise SystemExit(worker(args.worker))
