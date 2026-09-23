"""New-identity BN edge repair control; same physical request, zero solving."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_csr_execution import supervise
from metamoe_expert_diagnostic import require_clean
from metamoe_paired_execution_r2 import validate as validate_environment

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'configs/recent_moe/metamoe_protected_smoke_r1.json'
OLD_HEAD='22b4b2f60fdbde32bcae0cfda871ccaab3e196a6'
CONVERTER='act/pipeline/verification/torch2act.py'
CONFIG=ROOT/'configs/recent_moe/metamoe_bn_corrected_control_r1.json'
OUTPUT=Path('/data1/Kane/MOE/baseline_runs/metamoe_bn_corrected_control_20260923_r1')
ADDED=('scripts/metamoe_bn_corrected_control.py','scripts/metamoe_assignment_layers.py',
       'scripts/audit_metamoe_bn_corrected_control.py',
       'act/back_end/solver/current_assignment.py','tests/test_bn_expansion_edges.py',
       'tests/test_bn_expansion_diagnosis.py','docs/metamoe_bn_repair_protocol_20260923_r1.md')


def changed_parent(parent):
    """Explicit new identity, never mutate/rebind the historical manifest."""
    updated={}
    for p,h in parent['files'].items():
        actual=sha256(p)
        if actual!=h and p!=CONVERTER:raise ValueError('unregistered source change: '+p)
        updated[p]=actual
    if updated[CONVERTER]==parent['files'][CONVERTER]:raise ValueError('missing converter repair')
    old=subprocess.check_output(['git','show',OLD_HEAD+':'+CONVERTER],cwd=ROOT)
    if hashlib.sha256(old).hexdigest()!=parent['files'][CONVERTER]:raise ValueError('old converter identity')
    candidate={**parent,'files':updated}
    validate_environment(candidate)
    return candidate


def validate(cfg):
    if (cfg['protocol']!='bn_edges_same_mnist0_r1' or cfg['seconds']!=30. or cfg['assignment_seconds']!=3.
            or cfg['output_root']!=str(OUTPUT) or cfg['native_queries']!=0 or cfg['proposals']!=1
            or cfg['parent_sha256']!=sha256(PARENT) or cfg['expert']!=1 or cfg['point_tolerance']!=1e-9):
        raise ValueError('new identity/point-control contract')
    parent=changed_parent(json.loads(PARENT.read_text()))
    if cfg['explicit_source_rebinding']!={CONVERTER:{'old':json.loads(PARENT.read_text())['files'][CONVERTER],
                                                  'new':parent['files'][CONVERTER]}}:
        raise ValueError('undeclared source rebinding')
    for p,h in cfg['files'].items():
        if sha256(p)!=h:raise ValueError('control source drift: '+p)
    expected={**parent['files'],str(PARENT):sha256(PARENT),**{n:sha256(ROOT/n) for n in ADDED}}
    if (cfg['files']!=expected or cfg['python']!=parent['python']['act'] or
            cfg['group_rss_limit_bytes']!=parent['group_rss_limit_bytes']):raise ValueError('incomplete bindings')
    return parent


def freeze():
    require_clean()
    if CONFIG.exists() or OUTPUT.exists():raise FileExistsError('new identity required')
    old=json.loads(PARENT.read_text());parent=changed_parent(old)
    cfg={'protocol':'bn_edges_same_mnist0_r1','seconds':30.,'assignment_seconds':3.,'expert':1,
         'native_queries':0,'proposals':1,'point_tolerance':1e-9,'parent_sha256':sha256(PARENT),
         'output_root':str(OUTPUT),'python':parent['python']['act'],'group_rss_limit_bytes':parent['group_rss_limit_bytes'],
         'explicit_source_rebinding':{CONVERTER:{'old':old['files'][CONVERTER],'new':parent['files'][CONVERTER]}},
         'files':{**parent['files'],str(PARENT):sha256(PARENT),**{n:sha256(ROOT/n) for n in ADDED}},
         'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
    validate(cfg);write(CONFIG,cfg)


def build_request(parent,expert=1):
    """Build current physical request; caller owns budget and query policy."""
    import torch
    from act.back_end.moe.factory import build_act_moe_program
    from act.back_end.moe.class_separated_top1 import classification_rows
    from act.front_end.specs import OutputSpec,OutKind
    from metamoe_paired_model import load_full
    from metamoe_functional_intake import adapted_class_separated
    request=parent['requests'][0]
    with np.load(request['tensor_file'],allow_pickle=False) as z:
        center,lower,upper=(torch.from_numpy(z[k].copy()) for k in ('center','lower','upper'))
    original=load_full(parent['repo'],parent['checkpoint'],parent['files'][parent['checkpoint']])
    adapted,_=adapted_class_separated(original,center);surrogate=adapted.reduced_components(center)
    rows=classification_rows(original.total_classes,request['label'])
    thresholds=torch.full((len(rows),),parent['margin'],dtype=torch.float64)
    program=build_act_moe_program(surrogate,center=center,lower=lower,upper=upper,
        output_spec=OutputSpec(kind=OutKind.LINEAR_LE,c=-rows,d=-thresholds))
    return request,center,lower,upper,original,surrogate,rows,thresholds,program


def worker(cfg):
    began=time.monotonic();parent=validate(cfg)
    import torch
    from act.util.device_manager import initialize_device
    from act.util.stats import VerifyResult,VerifyStatus
    from act.config.config import HybridZConfig
    from act.back_end.solver import solver_hz as sh
    from act.back_end.solver.current_assignment import AssignmentScope,propose_current_assignment,check_current_assignment,model_fingerprint
    from act.back_end.solver.isolated_feasibility import save_npz,csr_arrays
    from act.back_end.moe.route_a import HybridzTF,_analyze_router,set_transfer_function,set_solver_mode,verify_once
    from act.back_end.moe.hz_routing import guarded_input_domain
    from metamoe_assignment_layers import concrete_layer,evaluate_factors
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(100);initialize_device('cpu','float64')
    request,center,lower,upper,original,surrogate,rows,thresholds,program=build_request(parent)
    tf=HybridzTF(config=HybridZConfig(**parent['hybridz']));set_transfer_function(tf);set_solver_mode('hybridz')
    captured={}
    def capture(self,output_hz,out_spec,*,input_hz=None,input_shape=None,**unused):
        if captured:raise ValueError('multiple expert evaluations')
        captured.update(hz=output_hz,input_hz=input_hz,input_shape=input_shape)
        return [VerifyResult(VerifyStatus.UNKNOWN,metadata={'reason':'capture_only'})]
    with patch.object(sh,'milp',side_effect=AssertionError('native query forbidden')),patch.object(sh.HZSolver,'evaluate_spec',capture):
        router_hz,input_hz=_analyze_router(program.router,tf)
        branch=guarded_input_domain(input_hz,router_hz,1,1).hz
        tf.set_entry_hz(branch)
        try:verify_once(program.experts[1],model_fn=surrogate.experts[1],timelimit=30.)
        finally:tf.clear_entry_hz()
    model=sh._lower_hz_milp(captured['hz']);folder=OUTPUT/'worker'
    scope=AssignmentScope(sha256(CONFIG),sha256(request['tensor_file']),uuid.uuid4().hex)
    start=time.monotonic();deadline=min(began+30.,start+cfg['assignment_seconds'])
    proposal=propose_current_assignment(model,scope,deadline)
    point,check=check_current_assignment(model,proposal,scope,deadline)
    assignment_seconds=time.monotonic()-start
    save_npz(folder/'base_model.npz',**csr_arrays(model.A),row_lb=model.row_lb,row_ub=model.row_ub,
             var_lb=model.var_lb,var_ub=model.var_ub,integrality=model.integrality,value_center=model.value_center,
             **{'value_'+k:v for k,v in csr_arrays(model.value_matrix).items()})
    save_npz(folder/'proposal.npz',point=proposal.point)
    write(folder/'assignment.json',{'scope':asdict(scope),'check':check,'model_sha256':model_fingerprint(model),
        'construction_seconds':proposal.construction_seconds,'proposal_and_check_seconds':assignment_seconds,
        'free_prefix':proposal.free_prefix,'relu_rows':proposal.relu_rows})
    if point is None:raise ValueError('fresh assignment did not validate')
    recovered=sh.HZSolver._recover_input(model,point,captured['input_hz'],captured['input_shape'],0).reshape_as(center)
    if not torch.equal(recovered,center) or not ((recovered>=lower)&(recovered<=upper)).all():raise ValueError('point identity')
    values={};records=[];stored={};net=program.experts[1]
    for layer in net.layers:
        if layer.kind=='ASSERT':continue
        preds=net.preds[layer.id]
        if len(preds)>1:raise ValueError('fixed model unexpectedly branched')
        with torch.no_grad():actual=concrete_layer(layer,values.get(preds[0]) if preds else None,recovered)
        values[layer.id]=actual;hz=tf.get_sparse_hz(layer.id)
        abstract=evaluate_factors(hz,point,model.n_cont)
        record={'layer':layer.id,'kind':layer.kind,'preds':preds,'n_out':hz.n_out,
                'max_abs_difference':float(np.max(np.abs(actual.numpy()-abstract)))}
        if layer.params.get('is_batchnorm_decomposition'):
            record['input_variables_match_predecessor']=bool(len(preds)==1 and
                list(net.layers[preds[0]].out_vars)==list(layer.in_vars))
        records.append(record);stored[f'{layer.id}_ir']=actual.numpy();stored[f'{layer.id}_hz']=abstract
    with torch.no_grad():source,scores=original(recovered);padded=surrogate.experts[1](recovered)
    represented=model.value_center+model.value_matrix@point;ir=values[records[-1]['layer']].numpy()
    stored.update(point=recovered.numpy(),source=source.numpy().reshape(-1),padded=padded.numpy().reshape(-1),
        represented=represented,center=center.numpy(),lower=lower.numpy(),upper=upper.numpy(),
        rows=rows.numpy(),thresholds=thresholds.numpy())
    save_npz(folder/'values.npz',**stored)
    ih=captured['input_hz']
    save_npz(folder/'input_map.npz',center=ih.c,**{'Gc_'+k:v for k,v in csr_arrays(ih.Gc).items()},
             **{'Gb_'+k:v for k,v in csr_arrays(ih.Gb).items()})
    errors={'layer_max':max(r['max_abs_difference'] for r in records),
            'source_vs_padded':float(np.max(np.abs(stored['source']-stored['padded']))),
            'source_vs_ir':float(np.max(np.abs(stored['source']-ir))),
            'source_vs_hz':float(np.max(np.abs(stored['source']-represented)))}
    ok=all(v<=cfg['point_tolerance'] for v in errors.values()) and all(r.get('input_variables_match_predecessor',True) for r in records)
    write(folder/'result.json',{'status':'POINT_CONFORMANCE_PASS' if ok else 'POINT_CONFORMANCE_FAIL',
        'errors':errors,'layers':records,'model_sha256':model_fingerprint(model),'scope':asdict(scope),
        'n_cont':model.n_cont,'n_bin':model.n_bin,'constraint_rows':model.A.shape[0],'constraint_nnz':model.A.nnz,
        'minimum_source_margin':float((rows@source.reshape(-1)).min()),
        'minimum_hz_margin':float(np.min(rows.numpy()@represented)),
        'original_route':int(scores.argmax(1)),'original_prediction':int(source.argmax(1)),
        'new_native_queries':0,'fresh_proposals':1,'source_complete':False,'robustness_proved':False,
        'proposal_and_check_seconds':assignment_seconds,'worker_through_result_seconds':time.monotonic()-began,
        'artifacts':{p.name:sha256(p) for p in folder.glob('*.npz')}})


def run(cfg):
    began=time.monotonic();validate(cfg);require_clean();OUTPUT.mkdir(parents=True,exist_ok=False)
    write(OUTPUT/'launch.json',{'config_sha256':sha256(CONFIG),'mode':'ONE_NEW_MATRIX_POINT_NO_NATIVE_QUERY'})
    receipt=supervise([cfg['python'],str(Path(__file__).resolve()),'--worker'],str(ROOT),OUTPUT/'worker',cfg['seconds'],cfg['group_rss_limit_bytes'])
    validate(cfg);p=OUTPUT/'worker/result.json';result=json.loads(p.read_text()) if p.exists() else None
    write(OUTPUT/'terminal.json',{'outer_status':receipt['status'],'receipt':receipt,'result':result,
        'status':result['status'] if receipt['status']=='COMPLETED' and result else receipt['status'],
        'config_sha256':sha256(CONFIG),'total_seconds':time.monotonic()-began,'historical_relabelled':False})
    print(receipt['status'],result['status'] if result else None)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    for flag in ('freeze','run','worker'):g.add_argument('--'+flag,action='store_true')
    a=p.parse_args()
    if a.freeze:freeze()
    else:
        cfg=json.loads(CONFIG.read_text());worker(cfg) if a.worker else run(cfg)
