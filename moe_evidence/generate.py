"""Generic pre-F0 capture/proposal path, opt-in and separate from ACT policy."""
from dataclasses import asdict
from fractions import Fraction
import itertools
from pathlib import Path
import time

from act.back_end.solver.lp_certificate import identity, rational
from moe_evidence.schema import (TRUSTED,THRESHOLD,validate_request,route_pairs,
                                 pair_key,interval_lp,interval_certificate,gate_envelope)
from scripts.optional_evidence_dev_contract import read,save
from portable_proof.runtime import digest


def reference(root,name): return {'file':name,'sha256':digest((root/name).read_bytes())}


def initial_manifest(request):
    rid=validate_request(request)
    return {'schema':'WEIGHTED_TOP2_EVIDENCE_V1','request':request,'request_id':rid,
            'trusted_base':TRUSTED,'positive_threshold':THRESHOLD,'generation_complete':False,
            'common_facts':None,'contexts':{},'supports':{},'obligations':[],
            'routes':{'feasible':[],'infeasible':[],
                      'unresolved':[list(p) for p in itertools.combinations(range(request['experts']),2)],'exact':False}}


def capture(model,tensors,request,config,root,budget):
    """Capture every required pair. Common facts are recomputed, never borrowed.

    V2's common TOP1 prelude is used only for routing and source intervals;
    requested linear properties are projected independently here. No V2 SAFE
    verdict or its classification facts are adopted as evidence.
    """
    import act.pipeline.moe.paired_monolithic as mono
    import torch
    from scripts.budget_contract_v2 import verify_v2
    from act.back_end.solver.hz_lp_export import snapshot,export
    from act.back_end.moe import condition_topk_set,guarded_input_topk_set
    from act.pipeline.moe.experiment1 import shared_input_pair_propagation
    from act.config.config import HybridZConfig
    from act.pipeline.moe.staged_verifier import _model_state_identity,_tensor_identity
    root=Path(root);m=initial_manifest(request);rid=m['request_id'];start=time.monotonic()
    if config.get('comparison_method') != 'monolithic_f0':
        raise ValueError('evidence capture requires the matched common prelude, not staged solving')
    if (root/'manifest.json').exists(): raise FileExistsError('no resume')
    if model.training or model.spec.top_k!=2 or len(model.experts)!=request['experts']:
        raise ValueError('wrong model semantics/dimensions')
    if _model_state_identity(model)!=request['model_state'] or any(_tensor_identity(tensors[k])!=request[k] for k in ('center','lower','upper')):
        raise ValueError('model/input identity mismatch')
    if any(str(t.dtype)!='torch.float64' or t.device.type!='cpu' for t in tensors.values()):
        raise ValueError('CPU float64 required')
    for key,sign in (('lower',-1),('upper',1)):
        if not torch.equal(tensors[key],(tensors['center']+sign*request['epsilon']).clamp(0,1)):
            raise ValueError('represented domain differs from V2 input box')
    with torch.no_grad():
        if int(model(tensors['center']).argmax())!=request['clean_prediction']:
            raise ValueError('clean prediction mismatch')
    def flush(): save(root/'manifest.json',m)
    flush();snapshots=[]
    class Captured(BaseException): pass
    def construct(*,internal,config,reuse=None,budget=None,**_):
        m['routes']=asdict(internal['route_sets'])
        # asdict preserves tuples; canonicalize the JSON-shaped protocol.
        m['routes']={**m['routes'],**{k:[list(p) for p in m['routes'][k]] for k in ('feasible','infeasible','unresolved')}}
        pairs=route_pairs(request,m['routes']);program=internal['program'];router=internal['router']
        if not m['routes']['exact'] or m['routes']['unresolved']: flush();raise Captured()
        if len(program.experts)!=request['experts'] or program.output_width!=request['classes']:
            raise ValueError('lowered dimension mismatch')
        if len(snapshots)!=1: raise ValueError('missing charged common fact prelude')
        original=snapshots[0]['payload']
        scope={'request_id':rid,'frame_id':router.output_hz.frame_id,'gate':request['gate'],'tie_policy':request['tie_policy']}
        common={'request_id':rid,'identity':{k:request[k] for k in ('model_state','center','lower','upper')},
                'scope':scope,'pairs':[list(p) for p in pairs],
                'branches':[{'expert':b['candidate'],'interval':b['proof_output_bounds']} for b in original['branches']]}
        save(root/'common_facts.json',common);m['common_facts']=reference(root,'common_facts.json')
        bounds={b['expert']:b['interval'] for b in common['branches']}
        for pair in pairs:
            for i,prop in enumerate(request['properties']):
                row={'pair':list(pair),'property_index':i,'property':prop,'kind':'pending'}
                facts=[]
                for expert in pair:
                    lp=interval_lp(bounds[expert],prop,request['classes']);cert=interval_certificate(lp)
                    if rational(cert['claimed_lower_bound'])>rational(THRESHOLD):
                        facts.append({'expert':expert,'request_id':rid,'scope':scope,'property_index':i,
                                      'guard':'TOP2_MEMBERSHIP','interval':bounds[expert],'certificate':cert})
                if len(facts)==2: row.update(kind='reused',facts=facts)
                m['obligations'].append(row)
        flush()
        support=config['support'];solver=config['solver']
        hc=HybridZConfig(max_input_dim=1024,guarded_support_enabled=True,
            guarded_support_lp_neurons=int(support['lp_neurons']),guarded_support_milp_neurons=int(support['milp_neurons']),
            guarded_support_lp_time_limit=float(support['lp_time_limit']),guarded_support_milp_time_limit=float(support['milp_time_limit']),
            guarded_support_solver_backend=str(support.get('solver_backend','scipy')),
            expert_property_solver_backend=str(solver.get('backend','scipy')))
        for pair in pairs:
            pending=[v for v in m['obligations'] if v['pair']==list(pair) and v['kind']=='pending']
            if not pending:continue
            key=pair_key(pair);budget.check('evidence_pair_capture')
            hc.guarded_support_lp_time_limit=budget.limit('evidence_support_lp',float(support['lp_time_limit']))
            hc.guarded_support_milp_time_limit=budget.limit('evidence_support_milp',float(support['milp_time_limit']))
            conditioned=condition_topk_set(router.output_hz,pair).hz
            entry=guarded_input_topk_set(router.input_hz,router.output_hz,pair).hz
            joint=shared_input_pair_propagation(program.experts[pair[0]],program.experts[pair[1]],
                entry_hz=entry,hybridz_config=hc,expert_relation='shared_input').joint.output_hz
            for name,hz in ((key+'_joint.json',joint),(key+'_router.json',conditioned)):save(root/name,snapshot(hz))
            m['contexts'][key]={'request_id':rid,'pair':list(pair),'expert_order':list(pair),
                'joint_source':reference(root,key+'_joint.json'),'router_source':reference(root,key+'_router.json')}
            def support_export(name,kind,index,hz,q):
                budget.check('evidence_export');record=export(hz,q,sparse=True)
                file=name+'.export.json';save(root/file,record)
                m['supports'][name]={'request_id':rid,'pair':list(pair),'property_index':index,'kind':kind,
                    'source_sha256':record['source_sha256'],'export':reference(root,file),'certificate':None,'status':'PENDING'}
            q=[0]*request['experts'];q[pair[0]]=1;q[pair[1]]=-1
            support_export(key+'_gate_lo','router_order',None,conditioned,q)
            support_export(key+'_gate_hi','router_order',None,conditioned,[-v for v in q])
            for row in pending:
                i=row['property_index'];q=row['property']['q'];qd=q+[str(-rational(v)) for v in q]
                support_export(key+f'_p{i}_lo','difference',i,joint,qd)
                support_export(key+f'_p{i}_hi','difference',i,joint,[str(-rational(v)) for v in qd])
                row.update(kind='residual',weighted_status='RANGE_UNAVAILABLE',weighted=None,certificate=None)
            flush()
        budget.check('evidence_capture_complete');m['generation_complete']=True;flush();raise Captured()
    original=mono._run_monolithic;mono._run_monolithic=construct
    try:
        try:
            report=verify_v2(model,tensors['center'],request['epsilon'],config,
                journal_path=root/'budget_journal.jsonl',started=budget.started,
                identity={'conditional_evidence_request':rid},expected_clean_prediction=request['clean_prediction'],
                common_fact_callback=snapshots.append)
        except Captured: pass
        else:
            m['capture_terminal']=report.status
            # A production return cannot replace the independent evidence path.
            m['generation_complete']=False;flush()
    finally: mono._run_monolithic=original
    save(root/'generation.json',m)
    save(root/'capture.json',{'seconds':time.monotonic()-start,'complete':m['generation_complete'],
                             'weighted_property_solver_called':False,'floating_F0_called':False})


def propose_all(root,budget,*,cap=60,reserve=80):
    from act.back_end.solver.lp_certificate import propose
    from act.back_end.solver.check_hz_lp_export import check_export
    from act.back_end.solver.rational_mccormick import build
    from act.back_end.solver.check_rational_mccormick import check_construction
    root=Path(root);m=read(root/'manifest.json');calls=[];values={}
    if not m['generation_complete']:return
    def solve(key,record):
        grant=budget.grant(cap,reserve);start=time.monotonic()
        calls.append({'key':key,'entered_seconds':start-budget.started,'granted_seconds':grant,'status':'PENDING'})
        save(root/'query_log.json',calls)
        try:cert=propose(record['lp'],time_limit=budget.grant(grant,reserve))
        except ValueError as exc:
            if str(exc)!='proposal solver did not complete':raise
            cert=None
        budget.remaining(2)
        if cert:save(root/(key+'.certificate.json'),cert)
        calls[-1].update(status='PROPOSED' if cert else 'UNAVAILABLE',seconds=time.monotonic()-start)
        save(root/'query_log.json',calls)
        return cert,reference(root,key+'.certificate.json') if cert else None
    order=[]
    for pair in sorted(tuple(p) for p in m['routes']['feasible']):
        prefix=pair_key(pair)
        pending=sorted((v for v in m['obligations'] if v['pair']==list(pair) and v['kind']=='residual'),key=lambda v:v['property_index'])
        if not pending:continue
        order.extend((prefix+'_gate_lo',prefix+'_gate_hi'))
        for row in pending:order.extend((prefix+f"_p{row['property_index']}_lo",prefix+f"_p{row['property_index']}_hi"))
    if set(order)!=set(m['supports']):raise ValueError('support schedule mismatch')
    for key in order:
        item=m['supports'][key]
        # Preserve the check reserve and return a checkable incomplete record.
        if budget.deadline-budget.clock() <= reserve+.01:return
        budget.remaining(2);record=read(root/item['export']['file'])
        check_export(record,None,expected_source_sha256=item['source_sha256'])
        cert,ref=solve(key,record)
        checked=check_export(record,cert,expected_source_sha256=item['source_sha256'])
        values[key]=rational(checked['bound']['checked_lower_bound']) if cert else None
        item.update(status='PROPOSED' if cert else 'UNAVAILABLE',certificate=ref);save(root/'manifest.json',m)
    for row in m['obligations']:
        if row['kind']!='residual':continue
        if budget.deadline-budget.clock() <= reserve+.01:return
        budget.remaining(2);key=pair_key(row['pair']);i=row['property_index'];base=key+f'_p{i}'
        lo,neg=values[base+'_lo'],values[base+'_hi']
        if lo is None or neg is None:continue
        if lo>-neg:raise ValueError('inconsistent checked difference')
        gate=gate_envelope(values[key+'_gate_lo'],values[key+'_gate_hi']);difference=[str(lo),str(-neg)]
        src=read(root/m['contexts'][key]['joint_source']['file']);prop=row['property']
        record=build(src,prop['q'],prop['constant'],gate,difference)
        check_construction(record,source_hash=identity(src),q=prop['q'],offset=prop['constant'],gate=gate,difference=difference)
        name=base+'_weighted';save(root/(name+'.export.json'),record);cert,ref=solve(name,record)
        check_construction(record,cert,source_hash=identity(src),q=prop['q'],offset=prop['constant'],gate=gate,difference=difference)
        row.update(weighted_status='PROPOSED' if cert else 'UNAVAILABLE',weighted=reference(root,name+'.export.json'),
                   certificate=ref,gate_bounds=gate,difference_bounds=difference)
        save(root/'manifest.json',m)
