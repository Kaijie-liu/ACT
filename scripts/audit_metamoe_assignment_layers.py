"""Saved-only layer point comparison; no original forward/propagation/solve."""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import metamoe_assignment_layers_r2 as r2
from audit_metamoe_current_assignment import require, read, inventory
from audit_metamoe_protected_smoke_r1 import arrays
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def compare(observations, stored):
    require([r['layer'] for r in observations]==list(range(26)), 'incomplete/reordered fixed expert layers')
    required={'source','final_represented'}|{f'{i}_{side}' for i in range(26) for side in ('ir','hz')}
    require(set(stored)==required,'missing/extra layer arrays')
    result=[]
    for info in observations:
        i=info['layer'];a,b=stored[f'{i}_ir'],stored[f'{i}_hz']
        require(a.shape==b.shape==(info['n_out'],) and np.isfinite(a).all() and np.isfinite(b).all(),
                'nonfinite/layer dimension')
        delta=float(np.max(np.abs(a-b),initial=0.))
        require(delta==info['max_abs_difference'] and (delta>1e-9)==info['point_disagrees_at_1e_9'],
                'layer difference mismatch')
        result.append({'layer':i,'kind':info['kind'],'max_abs_difference':delta})
    require(np.isfinite(stored['source']).all() and stored['source'].shape==stored['25_ir'].shape,
            'source dimensions/nonfinite')
    require(np.allclose(stored['final_represented'],stored['25_hz'],atol=1e-12,rtol=0),'final HZ mismatch')
    return {'layers':result,'maximum_layer_error':max(r['max_abs_difference'] for r in result),
        'first_disagreeing_layer':next((r['layer'] for r in result if r['max_abs_difference']>1e-9),None),
        'final_ir_vs_source':float(np.max(np.abs(stored['25_ir']-stored['source']))),
        'final_ir_vs_hz':float(np.max(np.abs(stored['25_ir']-stored['final_represented'])))}


def audit():
    began=time.monotonic();cfg=read(r2.CONFIG);r2.validate(cfg)
    root=r2.OUTPUT;folder=root/'worker'
    terminal,receipt=read(root/'terminal.json'),read(folder/'receipt.json')
    result,layer=read(folder/'result.json'),read(folder/'layers.json')
    require(terminal['result']==layer and terminal['receipt']==receipt,'terminal/result binding')
    require(read(root/'launch.json')['config_sha256']==terminal['config_sha256']==sha256(r2.CONFIG),'config')
    require(receipt['status']==terminal['outer_status']=='COMPLETED' and receipt['exit_code']==0 and
            receipt['error'] is None and layer['identity_validated_at_end'],'outer precedence')
    require(receipt['command']==[cfg['python'],str(r2.ROOT/'scripts/metamoe_assignment_layers_r2.py'),'--worker'],
            'worker command')
    require(receipt['deadline_seconds']==cfg['seconds'] and
            0<result['worker_through_result_seconds']<=receipt['execution_including_preflight_seconds']<cfg['seconds']
            and receipt['execution_including_preflight_seconds']<=receipt['total_with_postflight_seconds']<=terminal['total_seconds'],
            'budget accounting')
    require(receipt['group_rss_limit_bytes']==cfg['group_rss_limit_bytes'] and
            receipt['peak_sampled_group_rss_bytes']<=cfg['group_rss_limit_bytes'],'resource policy')
    for name in ('stdout','stderr'):
        require(sha256(folder/f'{name}.txt')==receipt[name+'_sha256'],'stream binding')
    for name in ('replay','input_map'):
        require(sha256(folder/f'{name}.npz')==result[name+'_sha256'],'replay binding')
    require(sha256(folder/'layers.npz')==layer['arrays_sha256'],'layer arrays binding')
    previous=read(r2.r1.replay.OUTPUT/'worker/result.json')
    require(result['model_sha256']==result['saved_model_sha256']==previous['model_sha256'] and
            result['fresh_full_matrix_check']['accepted'],'matrix/point identity')
    stored=arrays(folder/'layers.npz');comparison=compare(layer['layers'],stored)
    for k in ('final_ir_vs_source','final_ir_vs_hz','first_disagreeing_layer'):
        require(comparison[k]==layer[k], 'aggregate '+k)
    fresh=arrays(folder/'replay.npz');old=arrays(r2.r1.replay.OUTPUT/'worker/replay.npz')
    for k in ('point','source','padded','center','lower','upper','rows','thresholds'):
        require(np.array_equal(fresh[k],old[k]),'source/input/property drift '+k)
    require(np.array_equal(stored['source'],fresh['source']) and
            np.array_equal(stored['final_represented'],fresh['represented']), 'layer/replay cross binding')
    progress=sorted(folder.glob('layer_progress_*.json'))
    require(len(progress)==26,'partial progress missing')
    for i,p in enumerate(progress):
        value=read(p)
        require(p.name==f'layer_progress_{i:03d}.json' and value['layers']==layer['layers'][:i+1]
                and not value['identity_validated_at_end'],'progress order/precedence')
    failure=read(r2.r1.OUTPUT/'terminal.json')
    require(failure['outer_status']=='ERROR' and failure['result'] is None,'failed R1 changed')
    require(not terminal['historical_result_relabelled'] and not terminal['opens_formal_cohort'] and
            layer['new_native_queries']==result['new_native_queries']==0 and
            layer['new_proposals']==result['new_assignment_proposals']==0,'scientific boundary')
    return {'audit':'PASS','issues':0,'comparison':comparison,'receipt':receipt,
        'total_seconds':terminal['total_seconds'],'R1_failed_total_seconds':failure['total_seconds'],
        'config_sha256':sha256(r2.CONFIG),'fresh_matrix_sha256':result['model_sha256'],
        'R2_files':inventory(root),'failed_R1_files':inventory(r2.r1.OUTPUT),
        'new_native_queries':0,'original_forward_reexecuted_by_auditor':False,
        'claim':'Pointwise HZ and concrete IR agree; both differ from the recorded original output. Conversion requires investigation; not a global proof or SAFE.',
        'audit_source_sha256':sha256(Path(__file__)),'audit_seconds':time.monotonic()-began}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'output exists');value=audit();write(a.output,value)
    print(value['audit'],value['comparison'])
