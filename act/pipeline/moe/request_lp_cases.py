"""Fresh request-to-rational-LP generation; no historical proof answers reused.

This is a bounded evidence experiment, not a replacement production SAFE gate.
``generate`` also accepts small in-memory models for portable correctness tests.
"""
import argparse
from dataclasses import asdict
from fractions import Fraction
import itertools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from act.back_end.solver.lp_certificate import identity
from act.pipeline.moe.check_request_lp import (
    RATIONAL_TRUSTED, check_directory, order_envelope, property_row,
)


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, path)


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(model, tensors, request, root, *, query_seconds=10):
    """Fresh lowering, coverage and LP proposals, with a complete inventory.

    Model/input-to-HZ, guard lowering and route exclusion remain trusted.
    All arithmetic after the shared HZ source is stored is checked independently.
    Construction/checker errors propagate; a rejected solver proposal is UNKNOWN.
    The caller must supply the outer process deadline.
    """
    from act.back_end.moe import build_act_moe_program, condition_topk_set
    from act.back_end.moe.hz_routing import (
        analyze_topk_sets, guarded_input_domain, guarded_input_topk_set,
    )
    from act.back_end.solver.hz_lp_export import export
    from act.back_end.solver.lp_certificate import propose
    from act.back_end.solver.check_hz_lp_export import check_export
    from act.back_end.solver.rational_mccormick import build
    from act.back_end.solver.check_rational_mccormick import check_construction
    from act.config.config import HybridZConfig
    from act.front_end.specs import OutputSpec, OutKind
    from act.pipeline.moe.experiment1 import _propagate_component, shared_input_pair_propagation

    root = Path(root); pred = request['clean_prediction']
    classes, experts = request['classes'], request['experts']
    if request['top_k'] != 2 or request['tie_policy'] != 'ANY_LEGAL_TOPK':
        raise ValueError('only tie-inclusive top-2 supported')
    rid = identity(request)
    manifest = {'schema':'request_lp_rational_v3', 'request':request, 'request_id':rid,
        'trusted_base':RATIONAL_TRUSTED, 'positive_threshold':1e-7,
        'proofs':{}, 'obligations':[], 'routes':{'feasible':[], 'infeasible':[],
        'unresolved':list(itertools.combinations(range(experts),2)), 'exact':False}}
    flush = lambda: save(root/'manifest.json', manifest)
    flush()
    program = build_act_moe_program(model, **tensors,
        output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[pred]))
    config = HybridZConfig(max_input_dim=1024, guarded_support_enabled=False,
                          expert_property_solver_backend='scipy')
    router = _propagate_component(program.router, hybridz_config=config)
    routes = analyze_topk_sets(router.output_hz, 2, time_limit_per_set=query_seconds,
                              router_exact=router.output_hz.exact)
    manifest['routes'] = asdict(routes)
    manifest['obligations'] = [{'pair':list(p), 'property_index':i, 'kind':'unknown',
        'reason':'NOT_YET_CHECKED'} for p in routes.feasible for i in range(classes-1)]
    flush()
    if not routes.exact or routes.unresolved:
        return
    values = {}

    def certify(key, record, kind, scope, prop, checker):
        started = time.monotonic()
        ep = root/(key+'.export.json'); save(ep, record)
        ref = lambda p: {'file':p.name, 'sha256':sha(p)}
        item = {'request_id':rid, 'kind':kind, 'scope':scope, 'property_index':prop,
            'status':'PENDING', 'export':ref(ep), 'certificate':None,
            'hz_sha256':record['source_sha256']}
        manifest['proofs'][key] = item; flush()
        checker(record, None)  # Bad construction is an ERROR, not a solver limit.
        try:
            cert = propose(record['lp'], time_limit=query_seconds)
        except ValueError as exc:
            item.update(status='UNKNOWN', error=str(exc))
            value = None
        else:
            cp = root/(key+'.certificate.json'); save(cp, cert)
            item['certificate'] = ref(cp); flush()
            checked = checker(record, cert)
            value = Fraction(checked['bound']['checked_lower_bound'])
            item.update(status='CHECKED', checked_lower_bound=str(value))
        values[key] = value
        item['proposal_and_check_seconds'] = time.monotonic()-started; flush()
        print(key, item['status'], str(value), flush=True)
        return value

    def support(key, hz, q, kind, scope, prop):
        record = export(hz, q, sparse=True)
        return certify(key, record, kind, scope, prop,
            lambda r,c: check_export(r,c,expected_source_sha256=record['source_sha256']))

    for expert in sorted({i for p in routes.feasible for i in p}):
        entry = guarded_input_domain(router.input_hz,router.output_hz,expert,2).hz
        hz = _propagate_component(program.experts[expert],entry_hz=entry,hybridz_config=config).output_hz
        for i in range(classes-1):
            support(f'e{expert}_p{i}', hz, property_row(classes,pred,i), 'expert', {'membership':expert}, i)
    for p in routes.feasible:
        pair = list(p); pending = []
        for row in [r for r in manifest['obligations'] if r['pair']==pair]:
            i = row['property_index']; keys = [f'e{e}_p{i}' for e in p]
            if all(values[k] is not None and values[k]>Fraction.from_float(1e-7) for k in keys):
                row.update(kind='reused', sources=keys)
            else:
                pending.append(row)
        flush()
        if not pending:
            continue
        entry = guarded_input_topk_set(router.input_hz,router.output_hz,p).hz
        joint = shared_input_pair_propagation(program.experts[p[0]],program.experts[p[1]],
            entry_hz=entry,hybridz_config=config).joint.output_hz
        conditioned = condition_topk_set(router.output_hz,p).hz
        prefix = f's{p[0]}_{p[1]}'
        qm = [0]*experts; qm[p[0]]=1; qm[p[1]]=-1
        gkeys = [prefix+'_gate_lo', prefix+'_gate_hi']
        gl = support(gkeys[0],conditioned,qm,'router_order',{'pair':pair},None)
        gu = support(gkeys[1],conditioned,[-v for v in qm],'router_order',{'pair':pair},None)
        gate = order_envelope(gl,gu)  # Universal or dyadic order range, no sigmoid search.
        for row in pending:
            i=row['property_index']; q=property_row(classes,pred,i); qd=q+[-v for v in q]
            base=prefix+f'_p{i}'; low_key=base+'_lo'; high_key=base+'_hi'
            lo=support(low_key,joint,qd,'difference',{'pair':pair},i)
            neg=support(high_key,joint,[-v for v in qd],'difference',{'pair':pair},i)
            if lo is None or neg is None:
                row['reason']='DIFFERENCE_LP_UNCHECKED'; flush(); continue
            if lo > -neg:
                raise ValueError('inconsistent checked difference range')
            bounds=[str(lo),str(-neg)]
            original=json.loads((root/manifest['proofs'][low_key]['export']['file']).read_text())['source']
            record=build(original,q,0,gate,bounds)
            key=base+'_rational'
            row.update(kind='residual', difference_lower=low_key,difference_upper=high_key,
                gate_lower=gkeys[0],gate_upper=gkeys[1],lambda_bounds=gate,difference_bounds=bounds,source=key)
            certify(key,record,'rational_weighted',{'pair':pair},i,
                lambda r,c: check_construction(r,c,source_hash=record['source_sha256'],
                                              q=q,offset=0,gate=gate,difference=bounds))
            flush()


def selected_request(config, case, project):
    selection_path = project/config['selection']
    if sha(selection_path) != config['selection_sha256']:
        raise ValueError('selection drift')
    selection=json.loads(selection_path.read_text())
    sample=selection['samples'][case['rank']]; subject=selection['models'][case['model']]
    if sample['dataset_index'] != case['dataset_index']:
        raise ValueError('case index drift')
    request={'selection_sha256':config['selection_sha256'], 'dataset_index':case['dataset_index'],
        'checkpoint_sha256':subject['checkpoint_sha256'], 'epsilon':selection['request']['epsilon'],
        **{k:sample[k] for k in ('center','lower','upper')},
        'clean_prediction':sample['clean_predictions'][case['model']], 'classes':10,
        'experts':8,'top_k':2,'tie_policy':'ANY_LEGAL_TOPK'}
    return request, subject


def worker(root):
    import torch
    from act.back_end.moe import load_output_moe_checkpoint
    from act.pipeline.moe.staged_verifier import _tensor_identity
    from act.util.device_manager import initialize_device
    job=json.loads((root/'job.json').read_text()); request=job['request']
    torch.set_num_threads(1); initialize_device('cpu','float64')
    for field in ('checkpoint','tensors'):
        if sha(job[field]['path']) != job[field]['sha256']:
            raise ValueError(field+' drift')
    model,_=load_output_moe_checkpoint(job['checkpoint']['path'],map_location='cpu')
    model=model.cpu().double().eval()
    tensors=torch.load(job['tensors']['path'],map_location='cpu',weights_only=True)
    if set(tensors) != {'center','lower','upper'}:
        raise ValueError('unexpected materialized input fields')
    for key,tensor in tensors.items():
        if _tensor_identity(tensor) != request[key]: raise ValueError('tensor identity drift')
    with torch.no_grad():
        if int(model(tensors['center']).argmax()) != request['clean_prediction']:
            raise ValueError('prediction drift')
    generate(model,tensors,request,root,query_seconds=job['query_seconds'])


def run(config_path, root):
    from act.pipeline.moe.experiment1 import PROJECT_ROOT,WRITE_ROOT,_inside,_git_value
    from act.pipeline.moe.paired_followup import source_identity
    if _git_value('branch','--show-current')!='feat/moe-route-verification' or _git_value('status','--porcelain'):
        raise ValueError('clean feature branch required')
    config=json.loads(config_path.read_text())
    if config['protocol']!='ACT_ONLY_RATIONAL_REQUESTS_R1' or config['outer_seconds']!=1800 or config['query_seconds']!=10:
        raise ValueError('unregistered protocol')
    expected=[('seed0',9,4029),('seed1',7,4018),('seed2',5,4014)]
    if [(c['model'],c['rank'],c['dataset_index']) for c in config['cases']] != expected:
        raise ValueError('fixed cases changed')
    root=_inside(root,WRITE_ROOT); root.mkdir(exist_ok=False)
    head=_git_value('rev-parse','HEAD'); code=source_identity()
    runtime={'head':head,'source_sha256':code,'config':config,'config_sha256':sha(config_path),
        'borrowed_computed_proof_facts':0,'cases':[]}
    save(root/'runtime.json',runtime)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    for case in config['cases']:
        if source_identity()!=code or _git_value('status','--porcelain'):
            raise ValueError('execution source drift')
        started=time.monotonic()
        request,subject=selected_request(config,case,PROJECT_ROOT)
        tensor=PROJECT_ROOT/config['input_root']/f"{case['rank']}.pt"
        if sha(tensor)!=case['tensor_sha256']:raise ValueError('frozen tensor file drift')
        directory=root/f"{case['model']}_{case['dataset_index']}"; directory.mkdir()
        job={'request':request,'request_id':identity(request),'query_seconds':config['query_seconds'],
            'checkpoint':{'path':subject['checkpoint'],'sha256':subject['checkpoint_sha256']},
            'tensors':{'path':str(tensor),'sha256':case['tensor_sha256']}}
        save(directory/'job.json',job)
        with (directory/'worker.log').open('x') as log:
            p=subprocess.Popen([sys.executable,'-m','act.pipeline.moe.request_lp_cases','--worker',str(directory)],
                cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                rc=p.wait(timeout=max(.001,config['outer_seconds']-(time.monotonic()-started)))
                terminal={'status':'COMPLETED' if rc==0 else 'ERROR','returncode':rc}
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait(); terminal={'status':'TIMEOUT','returncode':p.returncode}
        terminal['generation_seconds']=time.monotonic()-started
        save(directory/'terminal.json',terminal)
        checked=None; check_started=time.monotonic()
        # Check the preserved partial inventory too; timeout is never promoted.
        if (directory/'manifest.json').exists() and terminal['status']!='ERROR':
            checked=check_directory(directory,expected_request_id=job['request_id'])
            save(directory/'check.json',checked)
        terminal['independent_check_seconds']=time.monotonic()-check_started
        terminal['evidence_bytes']=sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        terminal['request_id']=job['request_id']; terminal['check']=checked
        save(directory/'terminal.json',terminal)
        runtime['cases'].append({'case':case,**terminal}); save(root/'runtime.json',runtime)
        print(case,terminal,flush=True)
        if terminal['status']=='ERROR':raise RuntimeError('worker error preserved; no replacement')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',type=Path); group.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.run:
        if args.config is None:parser.error('--run requires --config')
        run(args.config,args.run)
    else:worker(args.worker)
