"""Frozen, analytic three-arm multilayer semantics control; no trained model.

One existing owned-session supervisor, one cumulative 300-second control batch.
Numerical results remain conditional on their backend/source policies.
"""
import time
ENTRY = time.monotonic()
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import resource
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from scoped_proof.io import load, save, sha
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
import genbab_cpu_ready as backend

CASES = ('safe', 'unsafe', 'tie')
ARMS = ('act', 'genbab_naive', 'genbab_pruned')
CODE = ('scripts/multilayer_stage0.py', 'act/back_end/moe/multilayer.py',
        'act/back_end/moe/multilayer_external.py', 'act/back_end/moe/multilayer_audit.py',
        'scripts/b2_observed_worker.py', 'scripts/genbab_cpu_ready.py',
        'scoped_proof/owned_session.py', 'scoped_source/development_work.py')


def affine(weights, bias):
    import torch
    from torch import nn
    layer = nn.Linear(len(weights[0]), len(weights), dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weights, dtype=torch.float64))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float64))
    return layer.eval()


def fixture(case):
    import torch
    from act.back_end.moe.multilayer import RoutedLayer
    from act.back_end.moe.multilayer_external import SequentialTop2
    if case not in CASES:
        raise ValueError('registered fixture only')
    r1 = affine([[0.],[0.],[0.]], [0.,0.,0.]) if case == 'tie' else affine([[1.],[-1.],[0.]], [0.,0.,-.5])
    first = RoutedLayer(r1, [affine([[1.]], [1.]), affine([[1.]], [-1.]), affine([[1.]], [0.])],
                        k=2, mode='selected_softmax')
    r2 = affine([[0.],[0.],[0.]], [0.,0.,0.]) if case == 'tie' else affine([[1.],[-1.],[0.]], [0.,0.,-4.])
    offset = -.25 if case == 'unsafe' else 3.
    third = 3. if case == 'tie' else -100.
    second = RoutedLayer(r2, [affine([[1.],[0.]], [offset,0.]),
        affine([[-1.],[0.]], [offset,0.]), affine([[0.],[0.]], [third,0.])], k=2, mode='selected_softmax')
    model = SequentialTop2([first, second]).eval()
    center = torch.zeros((1,1), dtype=torch.float64)
    return model, dict(center=center, lower=center-1, upper=center+1,
        rows=torch.tensor([[1.,-1.]], dtype=torch.float64), thresholds=torch.zeros(1,dtype=torch.float64))


def point_reference(model, tensors, case):
    """Finite differential evidence; analytic bounds in freeze supply the proof."""
    import torch
    from act.back_end.moe.multilayer import HistoryPlan
    from act.back_end.moe.multilayer_external import StaticPrefix, GuardedHistory, excluded_pair
    plan = HistoryPlan(model,tensors['center'],weighted_matmul=True)
    minimum = math.inf
    ties = set()
    legal = set()
    matches = 0
    # Rational dyadic points, covering both half-margin boundaries and endpoints.
    grid = torch.arange(-128,129,dtype=torch.float64).reshape(-1,1)/128
    with torch.no_grad():
        for history in plan.histories():
            compiled, score_rows = plan.compile(history)
            separate = StaticPrefix(model,history).eval()
            guarded = GuardedHistory(model,history,tensors['rows'],tensors['thresholds']).eval()
            for x in grid.split(1):
                joint = compiled(x)
                direct = separate(x)
                if not torch.allclose(joint[:,:2],direct,atol=1e-12,rtol=1e-12):
                    raise ValueError('different static-history functions')
                guards = [float(joint[0,indices[j]]-joint[0,indices[i]])
                          for pair,indices in zip(history,score_rows)
                          for i in pair for j in range(len(indices)) if j not in pair]
                expected = max(float(direct[0,0]-direct[0,1]),*guards)
                if abs(float(guarded(x)[0,0])-expected)>1e-10:
                    raise ValueError('guard AND/property scalar-max direction')
                if max(guards)<=0:
                    legal.add(history); minimum=min(minimum,float(direct[0,0]-direct[0,1]))
                    if any(g==0 for g in guards):ties.add(history)
                matches += 1
        if excluded_pair((0,1),3,{(0,2):0.,(1,2):0.}) is not None:
            raise ValueError('tie incorrectly pruned')
        if excluded_pair((0,1),3,{(0,2):float('nan')}) is not None:
            raise ValueError('nonfinite incorrectly pruned')
        if excluded_pair((0,1),3,{(0,2):-1.}) is None:
            raise ValueError('strict excluded subtree not recognized')
        center_margin = float((model(tensors['center']) @ tensors['rows'].T)[0,0])
    if case in ('safe','tie') and minimum <= 0:
        raise ValueError('analytic-positive fixture contradiction')
    if case=='unsafe' and center_margin>=0:
        raise ValueError('analytic center witness contradiction')
    return dict(points=257, history_point_comparisons=matches, observed_legal_histories=len(legal),
                observed_tie_histories=len(ties), sampled_minimum=minimum, center_margin=center_margin,
                guarantee='SAMPLING_IS_NOT_UNIVERSAL_PROOF')


def check_coverage(value):
    """Independent finite tree audit of external controls, including exclusions."""
    universe=set(itertools.product(tuple(itertools.combinations(range(3),2)),repeat=2))
    covered=set()
    observed=set()
    for rec in value['histories']:
        h=tuple(map(tuple,rec['history']))
        if h not in universe or h in observed:raise ValueError('invalid/duplicate history')
        observed.add(h)
        if rec['outcome']['status']=='verified':covered.add(h)
    for rec in value['exclusions']:
        prefix=tuple(map(tuple,rec['prefix']))
        matching={h for h in universe if h[:len(prefix)]==prefix}
        if not matching or len(matching)!=rec['covered'] or matching&covered:
            raise ValueError('overlap/incomplete excluded subtree')
        if rec['reason']['upper'] >= -rec['reason']['tolerance']:
            raise ValueError('non-strict exclusion')
        reason=rec['reason']
        if reason['selected'] not in prefix[-1] or reason['outside'] in prefix[-1]:
            raise ValueError('wrong exclusion direction')
        bound_records=[r for r in value['prefix_bounds'] if tuple(map(tuple,r['prefix']))==prefix[:-1]]
        if (len(bound_records)!=1 or [reason['selected'],reason['outside'],reason['upper']]
                not in bound_records[0]['upper']):raise ValueError('unbound exclusion upper')
        covered |= matching
    if value['status']=='POSITIVE' and covered!=universe:
        raise ValueError('missing history in positive aggregate')
    if len(covered)!=value['verified_histories']+value['pruned_histories']:
        raise ValueError('coverage count mismatch')


def worker(folder):
    log = os.open(folder/'native.log', os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
    os.dup2(log,1);os.dup2(log,2);os.close(log)
    request=load(folder/'request.json')
    deadline=request['worker_deadline']
    backend.limits(deadline)
    if request['arm'] != 'act':
        save(folder/'environment.json',backend.probe())
    import torch
    from act.util.typing_compat import install_typing_override
    install_typing_override()
    from act.util.device_manager import initialize_device
    from act.util.options import PerformanceOptions
    from act.back_end.moe.multilayer import verify_multilayer_box
    from act.back_end.moe.multilayer_audit import audit_multilayer_result
    from act.back_end.moe.multilayer_external import verify_histories
    from act.back_end.moe.class_separated_top1 import validate_replay
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    initialize_device('cpu','float64')
    PerformanceOptions.set_debug_output_file(str(folder/'act_debug.log'))
    model,tensors=fixture(request['case'])
    def fingerprint(t):
        t=t.detach().cpu().contiguous()
        return hashlib.sha256(str((tuple(t.shape),str(t.dtype))).encode()+t.numpy().tobytes()).hexdigest()
    state={k:fingerprint(v) for k,v in model.state_dict().items()}
    tensor_ids={k:fingerprint(v) for k,v in tensors.items()}
    reference=point_reference(model,tensors,request['case'])
    save(folder/'reference.json',reference)
    publications=iter(range(1000000))
    publish=lambda r:save(folder/f'progress_{next(publications):06d}.json',r)
    if request['arm']=='act':
        value=verify_multilayer_box(model,**tensors,total_seconds=deadline-time.monotonic(),
            max_histories=9,weighted_matmul=True,progress=publish)
        replay=lambda _:value.get('witness') is not None and validate_replay(model,
            torch.tensor(value['witness'],dtype=torch.float64),tensors['lower'],tensors['upper'],
            tensors['rows'],tensors['thresholds'])
        audit=audit_multilayer_result(value,expected_request_id=value.get('request_id'),replay=replay)
        if audit['status']!='PASS':raise ValueError('ACT aggregate audit: '+str(audit))
        if (value['identity']['state']!=state or value['identity']['tensors']!=tensor_ids
                or value['identity']['weighted_matmul'] is not True):
            raise ValueError('ACT result differs from independently bound fixture')
    else:
        from b2_observed_worker import verify_graph
        value=verify_histories(model,**tensors,pruning=request['arm']=='genbab_pruned',deadline=deadline,
            verify_graph=verify_graph,publish=publish)
        check_coverage(value)
        audit={'status':'PASS','kind':'finite history coverage, not numerical reproof'}
        if value['status']=='UNSAFE_REPLAYED' and not validate_replay(model,
            torch.tensor(value['witness'],dtype=torch.float64),tensors['lower'],tensors['upper'],
            tensors['rows'],tensors['thresholds']):raise ValueError('invalid dynamic-model witness')
    if request['case']=='tie' and value['status']=='POSITIVE':
        excluded=(sum(r['status']=='EXCLUDED' for r in value['records']) if request['arm']=='act'
                  else value['pruned_histories'])
        if excluded:raise ValueError('tie-legal histories excluded')
    if time.monotonic()>=deadline:raise TimeoutError('control reception deadline')
    save(folder/'outcome.json',dict(query_sha256=sha(folder/'request.json'),clock=time.monotonic(),
        case=request['case'],arm=request['arm'],state=state,tensors=tensor_ids,result=value,audit=audit))


def main_batch(folder,manifest):
    from b1_observed_batch import session_ok
    from b2_observed_batch import sizes
    spec=load(manifest)
    if ROOT==Path('/data1/Kane/MOE/ACT') or not (ROOT/'.frozen_commit').is_file():
        raise ValueError('immutable execution copy required')
    if folder.exists():raise ValueError('new run directory required')
    if any(sha(ROOT/p)!=h for p,h in spec['source_files'].items()):raise ValueError('frozen source mismatch')
    if backend.CONFIG!=spec['genbab_config']:raise ValueError('fixed backend configuration')
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    prior=spec.get('retained_tooling_failure')
    prior_seconds=0.;prior_bytes=0
    if prior:
        previous=load(prior['complete'],prior['sha256'])
        prior_seconds=previous.get('cumulative_seconds',previous['seconds'])
        prior_bytes=sizes(Path(prior['complete']).parent)+previous.get('prior_bytes',0)
    folder.mkdir();deadline=ENTRY+300-prior_seconds
    save(folder/'start.json',dict(commit=(ROOT/'.frozen_commit').read_text().strip(),
        manifest_sha256=sha(manifest),entry=ENTRY,deadline=deadline,prior_seconds=prior_seconds,prior_bytes=prior_bytes))
    rows=[]
    for query in spec['queries']:
        target=folder/query['id'];target.mkdir();(target/'tmp').mkdir();(target/'cache').mkdir()
        request=dict(query,worker_deadline=deadline-6,deadline=deadline)
        save(target/'request.json',request)
        env=dict(PATH='/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
            CUDA_VISIBLE_DEVICES='',TMPDIR=str(target/'tmp'),XDG_CACHE_HOME=str(target/'cache'))
        interpreter=backend.ACT if query['arm']=='act' else str(backend.ENVIRONMENT/'bin/python')
        began=time.monotonic()
        receipt=execute([interpreter,'-I','-B',str(ROOT/'scripts/multilayer_stage0.py'),
            '--worker',str(target)],run_deadline=deadline-4,cleanup_deadline=deadline-2,
            env=env,rss_limit=None,development_policy=POLICY)
        accepted=False;actual='ERROR'
        if session_ok(receipt) and (target/'outcome.json').is_file():
            value=load(target/'outcome.json')
            accepted=(value['query_sha256']==sha(target/'request.json') and value['clock']<deadline-6
                      and time.monotonic()<deadline)
            actual=value['result']['status'] if accepted else 'ERROR'
        row=dict(**query,actual=actual,accepted=accepted,session=receipt,seconds=time.monotonic()-began,
            artifact_bytes=sizes(folder))
        row['passed']=accepted and actual==query['expected']
        rows.append(row);save(target/'terminal.json',row)
        print(json.dumps({k:row[k] for k in ('id','actual','passed','seconds')}),flush=True)
        if not row['passed'] or sizes(folder)+prior_bytes>128*2**20-2**20:break
    save(folder/'complete.json',dict(status='PASS' if len(rows)==len(spec['queries']) and all(r['passed'] for r in rows) else 'STOP',
        rows=rows,seconds=time.monotonic()-ENTRY,cumulative_seconds=prior_seconds+time.monotonic()-ENTRY,
        real_requests=0,training_calls=0,prior_bytes=prior_bytes,
        artifact_bytes_before_final=sizes(folder),unexecuted=[q for q in spec['queries'][len(rows):]],
        guarantee='NUMERICAL_SEMANTICS_CONTROL_NOT_SOURCE_COMPLETE_PROOF'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',type=Path);parser.add_argument('--run',type=Path);parser.add_argument('--manifest',type=Path)
    args=parser.parse_args()
    if args.worker:
        from b1_observed_batch import finish_worker_io
        try:
            worker(args.worker)
        except BaseException as exc:
            save(args.worker/'error.json',dict(kind=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            raise
        finally:
            finish_worker_io()
    else:main_batch(args.run,args.manifest)
