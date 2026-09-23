"""Independent forward-only replay of raw-order selection and physical boxes.

No HZ, routing analysis, support or verifier imports. Does not call the
selector's eligibility or tensor-construction helper.
"""
import json
from pathlib import Path
import sys
import time
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def review(plan_path, root):
    began=time.monotonic();plan=json.loads(plan_path.read_text())
    smoke_path=Path(__file__).resolve().parents[1]/'configs/recent_moe/metamoe_checked_paired_smoke_r1.json'
    cfg=json.loads(smoke_path.read_text());selection=json.loads((root/'selection.json').read_text())
    if sha256(smoke_path)!=plan['smoke_sha256'] or selection['plan_sha256']!=sha256(plan_path):
        raise ValueError('review identity')
    for file,h in {**plan['raw_data_files'],**plan['history_files'],**plan['sources'],**cfg['files']}.items():
        if sha256(file)!=h:raise ValueError('review source/data drift: '+file)
    import numpy as np
    import torch
    from metamoe_paired_model import load_full
    torch.set_num_threads(2);torch.set_num_interop_threads(2);torch.manual_seed(100)
    model=load_full(cfg['repo'],cfg['checkpoint'],cfg['files'][cfg['checkpoint']])
    sys.path.insert(0,str(Path(cfg['repo'])/'src/Formal_Neural_Network_Verification/alpha-beta-crown'))
    from create_vnnlib_specs import load_dataset
    reviewed=[]
    for ds in plan['datasets']:
        records=selection['scans'][ds]
        if not 1<=len(records)<=plan['scan_cap']:raise ValueError('scan size')
        images,labels,_=load_dataset(ds,plan['data_root'],len(records))
        expected=[]
        for i,(image,label) in enumerate(zip(images,labels)):
            x=image.unsqueeze(0).double()
            with torch.no_grad():out=model(x)[0]
            if not torch.isfinite(out).all():raise ValueError('nonfinite review')
            label=int(label)+plan['global_offsets'][ds];prediction=int(out.argmax(1))
            if records[i]!={'index':i,'label':label,'prediction':prediction,'finite':True}:
                raise ValueError('saved clean selection differs')
            if i in plan['excluded'][ds] or label!=prediction:continue
            expected.append(i)
            request=next(r for r in selection['requests'] if r['dataset']==ds and r['index']==i)
            if request['label']!=label or request['clean_prediction']!=prediction:raise ValueError('label')
            if sha256(request['tensor_file'])!=selection['tensor_hashes'][request['tensor_file']]:raise ValueError('tensor hash')
            with np.load(request['tensor_file'],allow_pickle=False) as a:
                original=x.numpy()
                if (a['center'].dtype!=np.float64 or not np.array_equal(a['center'],original) or
                    not np.array_equal(a['lower'],np.maximum(-10,np.minimum(10,original-cfg['epsilon']))) or
                    not np.array_equal(a['upper'],np.maximum(-10,np.minimum(10,original+cfg['epsilon'])))):
                    raise ValueError('original materialization mismatch')
            reviewed.append(request['id'])
        actual=[r['index'] for r in selection['requests'] if r['dataset']==ds]
        if len(expected)!=plan['per_dataset'] or expected!=actual or expected[-1]!=len(records)-1:
            raise ValueError('not exact first eligible prefix')
    if len(reviewed)!=len(selection['requests']) or len(set(reviewed))!=len(reviewed):raise ValueError('coverage')
    return {'status':'INDEPENDENT_SOURCE_SELECTION_PASS','plan_sha256':sha256(plan_path),
        'selection_sha256':sha256(root/'selection.json'),'reviewed':reviewed,
        'verification_calls':0,'seconds':time.monotonic()-began,
        'trust':'Pinned source dataset transform and original model float64 forward; not source network proof'}


if __name__=='__main__':
    root=Path(sys.argv[2]);write(root/'review'/'result.json',review(Path(sys.argv[1]),root))
