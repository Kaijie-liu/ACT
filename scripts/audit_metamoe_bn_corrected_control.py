"""Independent saved-row/point comparison, not a global source proof or solve."""
import argparse
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import metamoe_bn_corrected_control as control
from audit_metamoe_current_assignment import require,read,inventory,check_point,recover_and_compare
from audit_metamoe_protected_smoke_r1 import arrays
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def audit():
    began=time.monotonic();cfg=read(control.CONFIG);parent=control.validate(cfg)
    root=control.OUTPUT;folder=root/'worker'
    result,terminal,receipt=(read(p) for p in (folder/'result.json',root/'terminal.json',folder/'receipt.json'))
    assignment=read(folder/'assignment.json')
    require(terminal['receipt']==receipt and terminal['result']==result,'terminal binding')
    require(read(root/'launch.json')['config_sha256']==terminal['config_sha256']==sha256(control.CONFIG),'config identity')
    require(receipt['status']==terminal['outer_status']=='COMPLETED' and receipt['exit_code']==0 and
            receipt['error'] is None and terminal['status']==result['status'],'outer precedence')
    require(receipt['command']==[cfg['python'],str(control.ROOT/'scripts/metamoe_bn_corrected_control.py'),'--worker'],
            'worker invocation')
    require(receipt['deadline_seconds']==cfg['seconds'] and
            0<result['worker_through_result_seconds']<=receipt['execution_including_preflight_seconds']<cfg['seconds'] and
            receipt['execution_including_preflight_seconds']<=receipt['total_with_postflight_seconds']<=terminal['total_seconds'],
            'cost accounting')
    require(receipt['group_rss_limit_bytes']==cfg['group_rss_limit_bytes'] and
            receipt['peak_sampled_group_rss_bytes']<=cfg['group_rss_limit_bytes'],'resource bound')
    for stream in ('stdout','stderr'):
        require(sha256(folder/f'{stream}.txt')==receipt[stream+'_sha256'],'stream identity')
    require(set(result['artifacts'])=={'base_model.npz','proposal.npz','values.npz','input_map.npz'},'artifact completeness')
    for name,h in result['artifacts'].items():require(sha256(folder/name)==h,'artifact identity')
    z=arrays(folder/'base_model.npz');point=arrays(folder/'proposal.npz')['point'];v=arrays(folder/'values.npz')
    base=check_point(z,point)
    comparison=recover_and_compare(z,point,arrays(folder/'input_map.npz'),v)
    require(result['scope']==assignment['scope'] and assignment['scope']['request_sha256']==sha256(control.CONFIG)
            and assignment['scope']['input_sha256']==sha256(parent['requests'][0]['tensor_file'])
            and bool(assignment['scope']['evaluation_nonce']),'current scope')
    require(assignment['check']['accepted'] and
            0<assignment['construction_seconds']<=assignment['proposal_and_check_seconds']<cfg['assignment_seconds'] and
            result['proposal_and_check_seconds']==assignment['proposal_and_check_seconds'],'assignment budget/check')
    require(result['model_sha256']==assignment['model_sha256'] and
            result['model_sha256']!='38538a4242c941a1221d7372956e9fcfee8bbf5c117f23c37fb9efa27eee51d3','new matrix required')
    tensors=arrays(Path(parent['requests'][0]['tensor_file']))
    require(all(np.array_equal(v[k],tensors[k]) for k in ('center','lower','upper')),'physical tensor identity')
    # Original source/row arrays must remain identical to the sealed point replay.
    oldroot=Path('/data1/Kane/MOE/baseline_runs/metamoe_assignment_replay_20260923_r1/worker')
    previous=arrays(oldroot/'replay.npz')
    require(all(np.array_equal(v[k],previous[k]) for k in ('source','padded','point','rows','thresholds')),
            'source/property/point drift')
    require([r['layer'] for r in result['layers']]==list(range(26)),'layer completeness')
    layer_errors=[]
    for row in result['layers']:
        i=row['layer'];a,b=v[f'{i}_ir'],v[f'{i}_hz']
        require(a.shape==b.shape==(row['n_out'],),'layer dimensions')
        delta=float(np.max(np.abs(a-b)))
        require(delta==row['max_abs_difference'],'layer error record')
        require(row.get('input_variables_match_predecessor',True),'BN edge mismatch')
        layer_errors.append(delta)
    errors={'layer_max':max(layer_errors),'source_vs_padded':float(np.max(np.abs(v['source']-v['padded']))),
            'source_vs_ir':float(np.max(np.abs(v['source']-v['25_ir']))),
            'source_vs_hz':float(np.max(np.abs(v['source']-v['represented'])))}
    require(errors==result['errors'] and np.allclose(v['represented'],v['25_hz'],atol=1e-12,rtol=0),'aggregate error')
    ok=all(e<=cfg['point_tolerance'] for e in errors.values())
    require(result['status']==('POINT_CONFORMANCE_PASS' if ok else 'POINT_CONFORMANCE_FAIL'),'conclusion')
    require(result['new_native_queries']==0 and result['fresh_proposals']==1 and not result['robustness_proved']
            and not result['source_complete'] and not terminal['historical_relabelled'],'scientific boundary')
    return {'audit':'PASS','issues':0,'status':result['status'],'errors':errors,'base_check':base,
            'point_comparison':comparison,'model_sha256':result['model_sha256'],
            'config_sha256':sha256(control.CONFIG),'receipt':receipt,'total_seconds':terminal['total_seconds'],
            'assignment_seconds':assignment['proposal_and_check_seconds'],'files':inventory(root),
            'source_forward_reexecuted_by_auditor':False,'new_native_queries':0,
            'claim':'Point conformance on the same object only; not all-domain conversion proof, SAFE or relaxation attribution.',
            'audit_seconds':time.monotonic()-began,'audit_source_sha256':sha256(Path(__file__))}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'output exists');value=audit();write(a.output,value)
    print(value['audit'],value['status'],value['errors'])
