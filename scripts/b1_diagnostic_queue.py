"""The registered B1 offline query list on the existing owned-session runner.

No queue expansion, checkpoint selection, repeated propagation or online repair.
Missing capture ordinals stay missing. All dependency/result files are bound.
"""
import time
ENTRY = time.monotonic()
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import statistics
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'scripts')]
from scoped_proof.io import load, save, sha, tick, PYTHON
from scoped_proof.owned_session import execute
from scoped_source.development_work import DevelopmentWork, POLICY
from b1_observed_batch import sizes, session_ok, require

PGD = dict(steps=20,restarts=3,step_size_fraction=.25,seed_base=20261021,
           objective='cross_entropy',projection='saved_lower_upper',dtype='float64',device='cpu')


class ExtendedDiagnosticWork(DevelopmentWork):
    """Only the separately authorized offline1800s query; no online gate change."""
    def __init__(self,deadline):
        require(type(deadline) in (int,float) and math.isfinite(deadline)
                and 0<deadline-time.monotonic()<=1800,'one authorized extended deadline')
        self.deadline=deadline
        self.counts=dict(row_visits=0,coefficient_visits=0,arithmetic=0)
        self._continued=False
        self.tick()


def query_work(deadline,query):
    if query['kind']=='extended_milp':
        require(query['seconds']==1800 and query['observation']['eligibility']['eligible'],
                'registered offline extension only')
        return ExtendedDiagnosticWork(deadline)
    return DevelopmentWork(deadline,policy=POLICY)


def observation_tags(value):
    """Descriptive, overlapping labels; never turn a negative bound into UNSAFE."""
    tags=[];native=value.get('native_result',{});decision=value.get('decision',{})
    finite=lambda x:isinstance(x,(int,float)) and math.isfinite(x)
    lower=decision.get('minimum');raw=value.get('mip_dual_bound_with_offset')
    candidate=value.get('raw_candidate_objective');point=value.get('point')
    if native.get('status')==1:tags.append('SOLVER_LIMIT')
    if native.get('status')==1 and finite(raw) and raw>0:
        tags.append('RAW_POSITIVE_DUAL_NOT_ACCEPTED_BY_STATUS_POLICY')
    if finite(lower) and lower<=0:tags.append('NONPOSITIVE_POLICY_LOWER')
    if finite(candidate) and candidate<0:
        tags.append('NEGATIVE_CANDIDATE_NOT_CHECKED_FEASIBLE')
        if native.get('status')==0:tags.append('STATUS0_NEGATIVE_CANDIDATE')
    if extension_eligibility(value)['eligible']:tags.append('EXTENDED_SOLVE_SIGN_CONDITION')
    if point is None:tags.append('MISSING_JOINT_POINT')
    else:
        # Magnitudes are retained, not thresholded into unique causal claims.
        if point.get('gate_consistency') not in (None,0.):tags.append('POINT_GATE_DISCREPANCY')
        if point.get('product_gap') not in (None,0.):tags.append('POINT_PRODUCT_DISCREPANCY')
    if value.get('replay',{}):
        if value['replay'].get('valid'):tags.append('SAVED_FULL_MODEL_REPLAY_VALID')
    return tags


def call_statistics(observations):
    """One observation per native invocation, not per covered pair/property.

    Selected full captures are still the same returned invocation. Elapsed is
    the saved high-level solver-call duration, not exclusive native solver time.
    """
    from collections import Counter
    finite=lambda x:type(x) in (int,float) and math.isfinite(x)
    def distribution(values):
        values=[v for v in values if finite(v)]
        return dict(count=len(values),minimum=min(values) if values else None,
            maximum=max(values) if values else None,
            median=statistics.median(values) if values else None,sum=sum(values))
    calls={}
    for observation in observations:
        key=(observation['scope']['request'],observation['scope']['ordinal'])
        if key in calls:
            require(calls[key]==observation,'one bound observation per call')
        else:calls[key]=observation
    native=[v for v in calls.values() if v.get('native_called')]
    limited=[v for v in native if v.get('native_result',{}).get('status')==1]
    positive=[v for v in limited if finite(v.get('mip_dual_bound_with_offset'))
              and v['mip_dual_bound_with_offset']>0]
    return dict(unique_observations=len(calls),returned_native_calls=len(native),
        native_statuses=dict(Counter(str(v.get('native_result',{}).get('status')) for v in native)),
        paths=dict(Counter(v['scope']['path'] for v in native)),
        returned_elapsed=distribution(v.get('decision',{}).get('elapsed') for v in native),
        status1_positive_raw_dual=dict(calls=len(positive),
            not_accepted_certificates=True,
            elapsed=distribution(v.get('decision',{}).get('elapsed') for v in positive),
            raw_dual_with_offset=distribution(v['mip_dual_bound_with_offset'] for v in positive),
            raw_mip_gap=distribution(v.get('native_result',{}).get('mip_gap') for v in positive)),
        time_scope='LOWERING_SOLVE_AND_CANDIDATE_HANDLING_EXCLUDES_LATER_CAPTURE_NOT_PURE_NATIVE_TIME',
        unavailable_fields=['raw_termination_message','exact_per_call_time_limit_in_timeout_requests',
                            'native_dual_multipliers','measured_HiGHS_thread_count'],
        partial_elapsed_is_not_total_request_cost=True,does_not_upgrade_terminals=True)


def summarize(online,freeze):
    """Saved-only prefix report: preserve all requests and all obligation states."""
    from collections import Counter
    spec=load(freeze);start=load(online/'start.json')
    require(start['freeze_sha256']==sha(freeze),'same prefix request freeze')
    requests=[];status_counts=Counter();tags_count=Counter();states=Counter()
    registered=returned=0;all_calls=[]
    for item in spec['requests']:
        folder=online/item['job_id'];reception=folder/'reception.json'
        if not reception.exists():
            requests.append(dict(job_id=item['job_id'],status='RUNNING_OR_NOT_STARTED',obligations=None));continue
        terminal=load(folder/'terminal.json');receipt=load(reception);inv=last_inventory(folder)
        status=receipt['status'];status_counts[status]+=1
        row=dict(job_id=item['job_id'],status=status,accepted=receipt['accepted'],
                 seconds=receipt['through_terminal_ledger_seconds'],
                 terminal_sha256=sha(folder/'terminal.json'),obligations=[])
        if inv is not None:
            observations={p.name.split('_')[-1][:-5]:p for p in (folder/'diagnostics').glob('*_observation_*.json')}
            seen_calls={}
            returned+=inv.get('native_calls',0);registered+=inv['registered_f0_calls']
            for obligation in inv['obligations']:
                entry=dict(obligation);states[entry['state']]+=1
                digest=entry.get('observation_sha256');p=observations.get(digest)
                if p is not None:
                    observed=load(p,digest=digest);entry['tags']=observation_tags(observed)
                    seen_calls[digest]=observed
                    entry['observation_path']=str(p);entry['observation_sha256']=digest
                    for key in ('decision','native_result','raw_candidate_objective','point','point_missing','ranges'):
                        if key in observed:entry[key]=observed[key]
                    tags_count.update(entry['tags'])
                else:entry['tags']=['NO_SAVED_RETURN_OBSERVATION']
                row['obligations'].append(entry)
            row['coverage_complete']=inv.get('roster',{}).get('coverage_complete',False)
            row['native_call_statistics']=call_statistics(seen_calls.values())
            row['unreferenced_observation_hashes']=sorted(set(observations)-set(seen_calls))
            all_calls.extend(seen_calls.values())
        else:row['obligations']=None
        requests.append(row)
    return dict(online_root=str(online),completed=sum(status_counts.values()),expected=41,
        statuses=dict(status_counts),registered_f0_calls=registered,returned_native_f0_calls=returned,
        obligation_states=dict(states),descriptive_obligation_tags=dict(tags_count),requests=requests,
        native_call_statistics=call_statistics(all_calls),
        obligation_tags_unit='MODEL_INPUT_PAIR_PROPERTY_NOT_INDEPENDENT_NATIVE_CALL',
        grade='NUMERICAL_DEVELOPMENT_DIAGNOSIS_NOT_UNIQUE_CAUSES_OR_SOURCE_COMPLETE_CERTIFICATES',
        attribution_queries_added=0)


def last_inventory(folder):
    paths = sorted((folder/'diagnostics').glob('*_inventory_*.json'))
    return None if not paths else load(paths[-1],limit=2**20)


def extension_eligibility(observation):
    """No unshifted native objective, missing zero, or primal-as-lower shortcut."""
    lower = observation.get('decision',{}).get('minimum')
    kind = 'POLICY_MINIMUM_WITH_OFFSET'
    if lower is None:
        lower = observation.get('mip_dual_bound_with_offset')
        kind = 'RAW_SOLVER_DUAL_WITH_OFFSET_NOT_POLICY_ACCEPTED'
    candidate = observation.get('raw_candidate_objective')
    finite = lambda v: isinstance(v,(int,float)) and math.isfinite(v)
    return dict(eligible=finite(lower) and finite(candidate) and lower<0 and candidate>=0,
                lower=lower,lower_kind=kind,candidate=candidate,
                candidate_exact_feasibility_checked=False)


def make_plan(online, freeze):
    """After online completion, instantiate the already frozen ordering rules."""
    spec = load(freeze); complete = load(online/'online_complete.json'); start = load(online/'start.json')
    require(start['freeze_sha256']==sha(freeze),'same frozen online queue')
    prior_seconds = start['prior_seconds']
    require(math.isclose(complete['cumulative_online_seconds'],
                        complete['online_seconds']+prior_seconds,abs_tol=1e-5),
            'retained tooling prefix cost')
    require(complete['completed_calls']==41 and complete['status']=='COMPLETE',
            'finish registered online queue before offline execution')
    sources=[]; observations=[]; full_jobs=[]
    for item in spec['requests']:
        folder=online/item['job_id']; terminal=load(folder/'terminal.json')
        launch=load(folder/'launch.json')
        require(launch['item']==item and launch['freeze_sha256']==sha(freeze),
                'same materialized online request')
        reception=load(folder/'reception.json')
        inventory=last_inventory(folder)
        files={p.name.split('_')[-1][:-5]:p for p in (folder/'diagnostics').glob('*_native_*.json')}
        binding = None if inventory is None else inventory['binding']
        if binding is not None:
            require(binding['model']==item['subject']['model_state']['sha256'],'same request model')
        full_jobs.append(dict(item=item,binding=binding,online_status=terminal['status'],
            online_accepted=reception['accepted'],terminal_sha256=sha(folder/'terminal.json')))
        if inventory is None: continue
        for event in inventory.get('full_capture_events',[]):
            digest=event.get('sha256'); path=files.get(digest)
            source=dict(slot=len(sources),request_ordinal=item['ordinal'],job_id=item['job_id'],
                event=event,binding=binding,online_status=terminal['status'],online_accepted=reception['accepted'],
                path=None if path is None else str(path),sha256=digest)
            if path is not None: require(sha(path)==digest,'captured source identity')
            sources.append(source)
        for p in sorted((folder/'diagnostics').glob('*_observation_*.json')):
            digest=p.name.split('_')[-1][:-5]
            value=load(p,digest=digest,limit=2**20)
            require(all(value['scope'][k]==v for k,v in binding.items()),'same observation request')
            observations.append(dict(request_ordinal=item['ordinal'],job_id=item['job_id'],
                path=str(p),sha256=digest,scope=value['scope'],
                eligibility=extension_eligibility(value)))
    require(len(sources)<=10,'first ten source ordinals')
    while len(sources)<10:
        sources.append(dict(slot=len(sources),request_ordinal=None,path=None,sha256=None,
                            event={'status':'NOT_REACHED'},online_accepted=False))
    queries=[]
    def add(kind,seconds,**kwargs):
        queries.append(dict(id=f'q{len(queries):03d}',kind=kind,seconds=seconds,**kwargs))
    for job in full_jobs:
        add('replay_pgd',60,job=job,sources=[s for s in sources if s['request_ordinal']==job['item']['ordinal']])
    for s in sources: add('same_lp',120,source=s)
    dependencies={}
    for s in sources:
        dependencies[s['slot']]=[f'q{len(queries):03d}',f'q{len(queries)+1:03d}']
        add('difference_lower',60,source=s);add('difference_upper',60,source=s)
    for s in sources: add('replace_difference',60,source=s,dependencies=dependencies[s['slot']])
    for s in sources[:2]:
        for k in range(16):
            for endpoint in (0,1): add('slice_endpoint',30,source=s,slice_index=k,endpoint=endpoint)
    require(len(queries)==145 and sum(q['seconds'] for q in queries)==7380,'unchanged diagnostic catalog')
    # One native problem per observation. Monolithic copies do not create
    # additional eligible extended solves; an unavailable source is not replaced.
    extended=[]
    for observation in sorted(observations,key=lambda r:(r['request_ordinal'],r['scope']['ordinal'])):
        if not observation['eligibility']['eligible']: continue
        matches=[s for s in sources if s['request_ordinal']==observation['request_ordinal']
                 and s['event'].get('query')==observation['scope']['ordinal']]
        source=matches[0] if matches else dict(path=None,sha256=None,event={'status':'MATRIX_NOT_RETAINED'})
        extended.append(dict(id=f'e{len(extended):02d}',kind='extended_milp',seconds=1800,
                             source=source,observation=observation))
        if len(extended)==8: break
    return dict(schema='B1_FIXED_OFFLINE_QUEUE_V1',freeze_sha256=sha(freeze),
        online_root=str(online),online_complete_sha256=sha(online/'online_complete.json'),
        sources=sources,queries=queries,extended=extended,pgd=PGD,
        local_seconds_cap=21600-complete['stage_elapsed']-prior_seconds,
        prior_online_seconds=complete['stage_elapsed']+prior_seconds,
        extended_seconds_cap=14400,repairs_online=False)


def bind_source(problem, captured, job=None):
    scope=problem.value['scope']; event=captured['event']
    require(scope==event['receipt']['scope'],'same captured obligation')
    require(all(scope[k]==v for k,v in captured['binding'].items()),'same source request')
    require(scope['ordinal']==event['query'] and scope['property_index']==event['property_index']
            and event['pair'] in scope['pairs'],'same pair/property/query')
    if job is not None:
        require(captured['binding']==job['binding'] and
                scope['model']==job['item']['subject']['model_state']['sha256'],
                'saved point belongs to this model/input/domain request')


def mapped_candidate(problem, shape):
    import numpy as np
    if problem.value['candidate']!='RAW_UNCHECKED': return None,'MISSING_NATIVE_CANDIDATE'
    point=problem.array('candidate'); native=problem.native
    if point.ndim!=1 or point.size!=native['variables']: return None,'NATIVE_CANDIDATE_SHAPE'
    if not np.isfinite(point).all(): return None,'NONFINITE_NATIVE_CANDIDATE'
    if problem.value['scope']['path']=='pair':
        nc=native['n_cont']; nb=native['n_bin']; hz=problem.hz('input'); local=point
        source=problem.hz('f0')
    else:
        offset=native['selector_offset']; branches=native['branches']; selectors=point[offset:offset+len(branches)]
        selected=np.flatnonzero(selectors==1.)
        if len(selected)!=1 or np.count_nonzero(selectors)!=1: return None,'NONUNIT_SELECTOR_NOT_PROJECTED'
        j=int(selected[0]); branch=branches[j]; nc=branch['n_cont'];nb=branch['n_bin']
        start=native['local_offsets'][j]; local=point[start:start+nc+branch['n_bin']]
        hz=problem.hz(f'input_{j}')
        source=problem.hz(f'f0_{j}')
    if (hz.n_cont>nc or hz.n_bin>nb or hz.frame_id!=source.frame_id
            or local.size!=nc+nb): return None,'INPUT_FACTOR_MAPPING'
    if hz.n_out!=math.prod(shape): return None,'INPUT_MAPPING_SHAPE'
    value=hz.c+hz.Gc@local[:hz.n_cont]+hz.Gb@(2*local[nc:nc+hz.n_bin]-1)
    return np.asarray(value).reshape(shape),None


def source_network_point(problem, model, point, work):
    """Same numerical assignment versus native forward, not feasible/optimal evidence."""
    import numpy as np
    import torch
    p=problem; x=p.array('candidate');meta=p.value['source_metadata']
    nc=p.native['n_cont'] if p.value['scope']['path']=='pair' else None
    if nc is None:
        j=int(np.flatnonzero(x[p.native['selector_offset']:
                              p.native['selector_offset']+len(p.native['branches'])]==1.)[0])
        branch=p.native['branches'][j];nc=branch['n_cont'];start=p.native['local_offsets'][j]
        if branch['pair']!=meta['selected_pair']:
            return dict(missing='CAPTURED_CONTEXT_NOT_ACTIVE_BRANCH',
                        captured_pair=meta['selected_pair'],active_pair=branch['pair'])
        local=x[start:start+nc+branch['n_bin']]
    else:local=x
    joint=p.hz('upstream_expert_joint');router=p.hz('upstream_conditioned_router')
    beta=2*local[nc:]-1
    def output(hz):
        work.visit('coefficient_visits',hz.n_out+hz.Gc.nnz+hz.Gb.nnz)
        return hz.c+hz.Gc@local[:hz.n_cont]+hz.Gb@beta[:hz.n_bin]
    y,r=output(joint),output(router);q=np.asarray(p.value['scope']['q'])
    a,b=meta['selected_pair'];constant=p.value['scope']['constant']
    hz_b=float(q@y[meta['b_rows']])+constant
    hz_d=float(q@(y[meta['a_rows']]-y[meta['b_rows']]))
    hz_m=float(r[a]-r[b]);hz_s=(1/(1+math.exp(-hz_m)) if hz_m>=0 else math.exp(hz_m)/(1+math.exp(hz_m)))
    with torch.no_grad():
        scores=model.router(point.unsqueeze(0))[0]
        ea=model.experts[a](point.unsqueeze(0))[0];eb=model.experts[b](point.unsqueeze(0))[0]
        tq=torch.from_numpy(q.copy());native_b=float(tq@eb)+constant
        native_d=float(tq@(ea-eb));native_m=float(scores[a]-scores[b])
        native_s=float(torch.sigmoid(scores[a]-scores[b]))
        guard=max((float(scores[k]-scores[i]) for i in (a,b)
                   for k in range(len(scores)) if k not in (a,b)),default=0.)
    finite=lambda v:float(v) if math.isfinite(float(v)) else None
    return dict(pair=[a,b],source_b=finite(hz_b),source_d=finite(hz_d),source_m=finite(hz_m),
        source_exact_gate_substitution=finite(hz_b+hz_s*hz_d),native_b=finite(native_b),
        native_d=finite(native_d),native_m=finite(native_m),native_static_pair_property=finite(native_b+native_s*native_d),
        source_minus_native_b=finite(hz_b-native_b),source_minus_native_d=finite(hz_d-native_d),
        source_minus_native_m=finite(hz_m-native_m),maximum_guard_violation=finite(guard),
        native_lp_residuals=p.residuals(x,relaxed=False),
        interpretation='POINTWISE_NUMERICAL_DIAGNOSTIC_NOT_EXACT_FEASIBILITY_OR_UNIQUE_CAUSE')


def replay_pgd(query, deadline, output, work):
    import torch
    import torch.nn.functional as F
    from act.util.device_manager import initialize_device
    from act.back_end.moe.factory import load_output_moe_checkpoint
    from act.pipeline.moe.staged_verifier import _tensor_identity,_model_state_identity
    from act.pipeline.moe.experiment1 import _forward_validate
    from b1_saved_diagnostics import StoredProblem
    torch.set_num_threads(1); initialize_device('cpu','float64')
    item=query['job']['item']; subject=item['subject']; sample=item['sample']
    require(sha(subject['checkpoint'])==subject['checkpoint_sha256'],'checkpoint identity')
    model,_=load_output_moe_checkpoint(subject['checkpoint'],map_location='cpu');model.cpu().double().eval()
    require(_model_state_identity(model)==subject['model_state'],'same model state')
    source=item['tensors'];require(sha(source['path'])==source['sha256'],'materialized input file')
    saved=torch.load(source['path'],map_location='cpu',weights_only=True)
    center,lower,upper=[saved[k] for k in ('center','lower','upper')]
    for k in ('center','lower','upper'): require(_tensor_identity(saved[k])==sample[k],'same input '+k)
    label=int(sample['label']); records=[]; winner=None
    def replay(point, origin):
        nonlocal winner
        tick(deadline)
        point=point.detach().cpu().double()
        finite=bool(torch.isfinite(point).all())
        inside=finite and point.shape==center.shape[1:] and bool(((point>=lower[0]) & (point<=upper[0])).all())
        if not inside:
            r=dict(origin=origin,valid=False,strict_domain=False,finite=finite)
        else:
            with torch.no_grad(): logits=model(point.unsqueeze(0))
            finite=bool(torch.isfinite(logits).all())
            if not finite: r=dict(origin=origin,valid=False,strict_domain=True,finite=False)
            else:
                r=_forward_validate(model,point,lower=lower,upper=upper,clean_prediction=label)
                other=logits[0].clone();other[label]=-torch.inf
                r.update(origin=origin,strict_domain=True,finite=True,margin=float(logits[0,label]-other.max()))
                if r['valid'] and winner is None: winner=point.clone()
        records.append(r); tick(deadline)
        return r['valid']
    replay(center[0],'center')
    for captured in query['sources']:
        if captured.get('path') is None:
            records.append(dict(origin='stored_candidate',missing=captured['event']['status'],valid=False));continue
        p=StoredProblem(captured['path'],captured['sha256'],deadline,work=work)
        bind_source(p,captured,query['job'])
        point,missing=mapped_candidate(p,tuple(center.shape))
        if point is None: records.append(dict(origin='stored_candidate',missing=missing,valid=False))
        else:
            tensor=torch.from_numpy(point.copy())[0]
            replay(tensor,f"stored_slot_{captured['slot']}")
            # A wrong/unselected static pair can explain a relaxation point,
            # but only the dynamic full-model replay above establishes UNSAFE.
            if records[-1].get('strict_domain') and records[-1].get('finite'):
                records[-1]['source_network']=source_network_point(p,model,tensor,work)
        del p
    generator=torch.Generator(device='cpu').manual_seed(PGD['seed_base']+item['ordinal'])
    completed_steps=0;restarts=0;cutoff=False
    for restart in range(PGD['restarts']):
        if winner is not None: break
        if time.monotonic()+2>=deadline: cutoff=True;break
        point=lower+(upper-lower)*torch.rand(center.shape,generator=generator,dtype=center.dtype)
        restarts+=1
        for step in range(PGD['steps']):
            if time.monotonic()+2>=deadline: cutoff=True;break
            point.requires_grad_(True)
            loss=F.cross_entropy(model(point),torch.tensor([label]))
            gradient,=torch.autograd.grad(loss,point)
            with torch.no_grad():
                point=torch.maximum(lower,torch.minimum(upper,
                    point+item['epsilon']*PGD['step_size_fraction']*gradient.sign())).detach()
            completed_steps+=1
            if replay(point[0],f'pgd_{restart}_{step}'): break
        if cutoff: break
    witness=None
    if winner is not None:
        torch.save({'input':winner,'model_sha256':subject['checkpoint_sha256'],
                    'sample':sample,'grade':'NATIVE_FLOAT64_FULL_MODEL_REPLAY'},output/'witness.pt')
        witness=dict(path=str(output/'witness.pt'),sha256=sha(output/'witness.pt'))
    tick(deadline)
    return dict(status='UNSAFE_REPLAYED' if winner is not None else 'NO_COUNTEREXAMPLE_FOUND',
        records=records,pgd=PGD,completed_steps=completed_steps,started_restarts=restarts,
        budget_cutoff=cutoff,witness=witness,native_solver_calls=0,
        guarantee='NUMERICAL_FULL_MODEL_REPLAY_NOT_NETWORK_PROOF',repairs_online=False)


def worker(folder):
    launch=load(folder/'launch.json');deadline=launch['worker_deadline'];query=launch['query']
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE,(128*2**20,128*2**20))
    def cutoff(*_):
        os.write(2,b'B1_DIAGNOSTIC_AUTHORIZED_CUTOFF\n');os._exit(124)
    signal.signal(signal.SIGALRM,cutoff)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.monotonic()))
    work=query_work(deadline,query)
    with (folder/'native.log').open('xb',buffering=0) as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        try:
            from act.util.typing_compat import install_typing_override
            install_typing_override()
            if query['kind']=='replay_pgd': result=replay_pgd(query,deadline,folder,work)
            else:
                from b1_saved_diagnostics import StoredProblem
                source=query['source'];p=StoredProblem(source['path'],source['sha256'],deadline,work=work)
                bind_source(p,source)
                kind=query['kind'];limit=deadline-time.monotonic()-2
                save(folder/'prepared.json',dict(source_sha256=source['sha256'],scope=p.value['scope'],
                    work=work.counts,clock=time.monotonic(),native_call_started=False))
                if kind in ('same_lp','extended_milp'):
                    result=p.solve_same_matrix(relaxed=kind=='same_lp',time_limit=limit)
                elif kind in ('difference_lower','difference_upper'):
                    result=p.difference_extreme(upper=kind=='difference_upper',time_limit=limit)
                elif kind=='slice_endpoint':
                    result=p.slice_endpoint(slice_index=query['slice_index'],endpoint=query['endpoint'],time_limit=limit)
                elif kind=='replace_difference':
                    deps=[load(d['path'],digest=d['sha256'])['result'] for d in launch['dependencies']]
                    result=p.replace_difference_range(lower_record=deps[0],upper_record=deps[1],time_limit=limit)
                else: raise ValueError('unknown registered diagnostic')
            tick(deadline)
            save(folder/'outcome.json',dict(query_id=query['id'],query_sha256=launch['query_sha256'],
                result=result,worker_deadline=deadline,clock=time.monotonic(),work=work.counts,
                work_scope='stored coefficients/rows and reception; native solver and torch arithmetic opaque'))
            tick(deadline)
        except BaseException as exc:
            traceback.print_exc()
            if time.monotonic()<deadline:
                save(folder/'error.json',dict(kind=type(exc).__name__,error=repr(exc),clock=time.monotonic(),work=work.counts))
            if isinstance(exc,TimeoutError):
                os.write(2,b'B1_DIAGNOSTIC_AUTHORIZED_CUTOFF\n');return 124
            return 1
        finally: sys.stdout.flush();sys.stderr.flush()
    return 0


def cutoff_terminal(session, folder):
    if session['status']=='TIMEOUT': return True
    if session.get('returncode')!=124 or not (folder/'native.log').exists(): return False
    with (folder/'native.log').open('rb') as stream:
        stream.seek(max(0,(folder/'native.log').stat().st_size-256))
        return b'B1_DIAGNOSTIC_AUTHORIZED_CUTOFF' in stream.read()


def run(root,online,freeze):
    require(not root.exists() and root.parent==Path('/data1/Kane/MOE/results'),'new diagnostic directory')
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    require(ROOT!=Path('/data1/Kane/MOE/ACT') and (ROOT/'.frozen_commit').exists(),
            'independent committed diagnostic copy')
    commit=(ROOT/'.frozen_commit').read_text().strip()
    plan=make_plan(online,freeze);root.mkdir();save(root/'plan.json',plan)
    save(root/'start.json',dict(commit=commit,script_sha256=sha(__file__),entry=ENTRY,
        helper_sha256=sha(ROOT/'scripts/b1_saved_diagnostics.py'),pgd=PGD))
    extended_used=0.;results=[]
    prefix=Path('/data1/Kane/MOE/results/b1_observed_20261021_r1')
    for q in plan['queries']+plan['extended']:
        local_used=time.monotonic()-ENTRY-extended_used
        extended=q['kind']=='extended_milp';remaining=(14400-extended_used if extended else plan['local_seconds_cap']-local_used)
        started=time.monotonic();end=started+q['seconds'];folder=root/q['id'];folder.mkdir()
        missing=None;deps=[]
        if remaining<q['seconds']:missing='UNEXECUTED_STAGE_BUDGET'
        elif q['kind']!='replay_pgd' and not q['source'].get('path'):missing='UNEXECUTED_SOURCE_'+q['source']['event']['status']
        elif sizes(root)+sizes(online)+sizes(prefix)+2**20>2*2**30:missing='UNEXECUTED_ARTIFACT_BUDGET'
        if q['kind']=='replace_difference' and missing is None:
            for key in q['dependencies']:
                receipt_path=root/key/'receipt.json';outcome_path=root/key/'outcome.json'
                if not receipt_path.exists() or not load(receipt_path)['accepted']:
                    missing='UNEXECUTED_MISSING_RANGE';break
                outcome=load(outcome_path)
                if outcome['result'].get('numerical_bound') is None: missing='UNEXECUTED_MISSING_RANGE';break
                deps.append(dict(path=str(outcome_path),sha256=sha(outcome_path)))
        row=dict(query_id=q['id'],kind=q['kind'],accepted=False,status=missing,native_calls=None)
        if missing is None:
            digest=hashlib.sha256(json.dumps(q,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            save(folder/'launch.json',dict(query=q,query_sha256=digest,worker_deadline=end-5,
                deadline=end,dependencies=deps,commit=commit))
            env=dict(PATH='/usr/bin:/bin',PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                     NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='')
            try:
                session=execute([PYTHON,'-I','-B',__file__,'--worker',str(folder)],
                    run_deadline=end-3,cleanup_deadline=end-1,env=env,rss_limit=None,development_policy=POLICY)
                row.update(session=session,status=session['status'])
                if cutoff_terminal(session,folder):row['status']='TIMEOUT'
                if (folder/'outcome.json').exists():
                    outcome=load(folder/'outcome.json',limit=2**20)
                    require(outcome['query_id']==q['id'] and outcome['query_sha256']==digest
                            and outcome['worker_deadline']==end-5,'same diagnostic output identity')
                    work=query_work(end,q);work.continue_from(outcome['work'])
                    work.visit('coefficient_visits',((folder/'outcome.json').stat().st_size+4095)//4096)
                    row.update(outcome_sha256=sha(folder/'outcome.json'),received_work=work.counts,
                        accepted=session_ok(session) and time.monotonic()<end and outcome['clock']<end-5,
                        native_calls=outcome['result'].get('native_calls',outcome['result'].get('native_solver_calls')))
                    if row['accepted']:row['status']=outcome['result']['status']
            except BaseException as exc:
                row.update(status='ERROR',accepted=False,reception_error=repr(exc))
        else:row['native_calls']=0
        row['seconds']=time.monotonic()-started
        results.append(row)
        local_used=time.monotonic()-ENTRY-extended_used-(time.monotonic()-started if extended else 0.)
        save(root/f'ledger_{len(results):03d}.json',dict(local_seconds=local_used,extended_seconds=extended_used,
            local_attempts=sum(r['kind']!='extended_milp' and 'session' in r for r in results),
            extended_attempts=sum(r['kind']=='extended_milp' and 'session' in r for r in results),
            catalog_entries=len(results),last=row,repairs_online=False))
        # The final compact receipt is the acceptance boundary; its own write
        # tail is charged to the stage, not claimed as measured inside itself.
        total_bytes=sizes(root)+sizes(online)+sizes(prefix)
        within=time.monotonic()<end and total_bytes+2**16<=2*2**30
        row['accepted']=row['accepted'] and within
        if not within:row['status']='RECEPTION_LIMIT'
        row['seconds']=time.monotonic()-started
        row['artifact_bytes_before_receipt']=total_bytes
        save(folder/'receipt.json',row)
        if extended:extended_used+=time.monotonic()-started
        local_used=time.monotonic()-ENTRY-extended_used
        print(json.dumps(dict(query=q['id'],status=row['status'],accepted=row['accepted'],
                              local_seconds=local_used,extended_seconds=extended_used)),flush=True)
        if (not within or row['status']=='ERROR' or ('session' in row and
                row['session']['cleanup_status']!='LEADER_REAPED_NO_LIVE_SESSION')):
            break
    local_used=time.monotonic()-ENTRY-extended_used
    save(root/'complete.json',dict(status='COMPLETE' if len(results)==len(plan['queries'])+len(plan['extended']) else 'PREFIX_STOP',
        entries=len(results),expected=len(plan['queries'])+len(plan['extended']),
        local_seconds=local_used,extended_seconds=extended_used,results=results,
        seconds_through_result=time.monotonic()-ENTRY,repairs_online=False))
    return 0 if len(results)==len(plan['queries'])+len(plan['extended']) else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path)
    parser.add_argument('--run',type=Path);parser.add_argument('--online',type=Path)
    parser.add_argument('--freeze',type=Path);parser.add_argument('--summary',action='store_true');args=parser.parse_args()
    if args.summary:
        result=summarize(args.online,args.freeze)
        print(json.dumps({k:v for k,v in result.items() if k!='requests'},sort_keys=True));raise SystemExit(0)
    if args.worker:raise SystemExit(worker(args.worker))
    if args.run:raise SystemExit(run(args.run,args.online,args.freeze))
    parser.error('explicit operation required')
