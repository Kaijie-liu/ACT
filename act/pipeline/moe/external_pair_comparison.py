"""Frozen complete-cost ACT adaptive / ACT-fronted CROWN observed follow-up."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from act.pipeline.moe.experiment1 import PROJECT_ROOT,WRITE_ROOT,_inside,_sha256,_git_value
from act.pipeline.moe.paired_followup import save,source_identity
from act.pipeline.moe.external_compatibility import COMMITS,ENV,git
from act.pipeline.moe.request_lp_control import SELECTION,SELECTION_HASH

ACT='/data1/Kane/miniconda3/envs/act-py312/bin/python'
CFG=PROJECT_ROOT/'act/pipeline/moe/configs/relation_shared_r1.json'
CFG_HASH='26e9ffef88ad257fbc079bdd50df75b78c9625e379a3adcd20b396b495acb4c7'
ARMS=('adaptive','crown')
DEFAULT=PROJECT_ROOT/'act/pipeline/moe/configs/external_pair_comparison_r1.json'


def artifacts():
    if _sha256(SELECTION)!=SELECTION_HASH or _sha256(CFG)!=CFG_HASH:raise ValueError('frozen input/method drift')
    s=json.loads(SELECTION.read_text());s['samples']=s['samples'][:10]
    for path,head in COMMITS.items():
        if git(path,'rev-parse','HEAD')!=head or git(path,'status','--porcelain'):raise ValueError('external source drift')
    if _sha256(Path(s['dataset']['raw_test_batch']))!=s['dataset']['raw_test_batch_sha256']:raise ValueError('dataset drift')
    for v in s['models'].values():
        if _sha256(Path(v['checkpoint']))!=v['checkpoint_sha256']:raise ValueError('checkpoint drift')
    return s


def jobs(selection,smoke):
    result=[];models=sorted(selection['models'])
    for rank,s in enumerate(selection['smoke_samples' if smoke else 'samples']):
        for k in range(3):
            model=models[(rank+k)%3]
            for p in range(2):
                method=ARMS[(rank+models.index(model)+p)%2]
                result.append({'rank':rank,'model':model,'method':method,'position':p,
                    'dataset_index':s['dataset_index'],'job_id':f'rank{rank}_{model}_{method}'})
    return result


def materialize(root,selection,smoke):
    """Shared immutable raw-input preparation, not a cached verification answer.

    Excluded equally from both request budgets; loading these tensors is charged.
    Each arm (and external environment) must read exactly this file.
    """
    import torch
    from act.util.device_manager import initialize_device
    from act.pipeline.moe.train import _load_dataset
    from act.pipeline.moe.staged_verifier import _tensor_identity
    from act.back_end.moe.factory import load_output_moe_checkpoint
    start=time.monotonic();initialize_device('cpu','float64');torch.set_num_threads(1)
    os.environ['ACT_TORCHVISION_DATA_ROOT']=str(PROJECT_ROOT/'data/torchvision')
    _,payload=load_output_moe_checkpoint(selection['models']['seed0']['checkpoint'],map_location='cpu')
    dataset=_load_dataset(payload['dataset'],False,download=False);result={}
    (root/'inputs').mkdir()
    for rank,s in enumerate(selection['smoke_samples' if smoke else 'samples']):
        x,label=dataset[s['dataset_index']];x=x.unsqueeze(0).double();eps=selection['request']['epsilon']
        tensors={'center':x,'lower':(x-eps).clamp(0,1),'upper':(x+eps).clamp(0,1)}
        if label!=s['label'] or any(_tensor_identity(v)!=s[k] for k,v in tensors.items()):raise ValueError('frozen tensor identity differs')
        path=root/'inputs'/f'{rank}.pt';torch.save(tensors,path)
        result[str(rank)]={'path':str(path),'sha256':_sha256(path)}
    return result,time.monotonic()-start


def execute(command,log,started,env,budget=300):
    """Kill the entire owned request process group, including cross-env child."""
    expired=False
    with log.open('x') as handle:
        process=subprocess.Popen(command,cwd=PROJECT_ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=process.wait(timeout=max(.001,budget-(time.monotonic()-started)))
        except subprocess.TimeoutExpired:
            expired=True
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            code=process.wait()
    return code,expired or time.monotonic()-started>budget


def run(config_path,smoke):
    if _git_value('branch','--show-current')!='feat/moe-route-verification' or _git_value('status','--porcelain'):
        raise ValueError('clean feature checkout required')
    if Path(sys.executable).resolve()!=Path(ACT).resolve():raise ValueError('act-py312 required')
    config_path=_inside(config_path,PROJECT_ROOT);cfg=json.loads(config_path.read_text())
    if cfg['protocol']!='EXTERNAL_PAIR_COMPLETE_COST_R1' or cfg['budget_seconds']!=300 or cfg['sample_count']!=10:
        raise ValueError('protocol changed')
    from act.pipeline.moe.review_external_pair_comparison import audit
    selection=artifacts();source=source_identity();gate=None
    if not smoke:
        prior=Path(cfg['smoke_output']);rt=json.loads((prior/'runtime.json').read_text())
        if not rt['smoke'] or rt['source_sha256']!=source or rt['config_sha256']!=_sha256(config_path):raise ValueError('smoke mismatch')
        checked=audit(prior)
        if checked!=json.loads((prior/'audit.final.json').read_text()) or not all(checked['complete_per_arm'][a]>0 for a in ARMS):raise ValueError('smoke conformance gate failed')
        gate={'root':str(prior),'audit_sha256':_sha256(prior/'audit.final.json')}
    root=_inside(Path(cfg['smoke_output' if smoke else 'output']),WRITE_ROOT);root.mkdir(exist_ok=False)
    tensorfiles,prep=materialize(root,selection,smoke)
    runtime={'config_path':str(config_path),'config_sha256':_sha256(config_path),'config':{**cfg,'methods':{'adaptive':{'path':str(CFG),'sha256':CFG_HASH}}},
        'source_sha256':source,'git_head':_git_value('rev-parse','HEAD'),'smoke':smoke,'smoke_gate':gate,
        'selection_sha256':SELECTION_HASH,'selection':selection,'commits':COMMITS,'external_python':ENV,
        'tensors':tensorfiles,'input_preparation_seconds_excluded_equally':prep,'started_unix':time.time()}
    save(root/'runtime.json',runtime)
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''}
    schedule=jobs(selection,smoke)
    for number,job in enumerate(schedule,1):
        if source_identity()!=source or _git_value('status','--porcelain'):raise ValueError('source drift during frozen execution')
        artifacts();directory=root/job['job_id'];directory.mkdir()
        sample=selection['smoke_samples' if smoke else 'samples'][job['rank']]
        request={'subject':selection['models'][job['model']],'sample':sample,'epsilon':selection['request']['epsilon'],
            'method':job['method'],'tensors':tensorfiles[str(job['rank'])],'head':runtime['git_head'],
            'config':{'path':str(CFG),'sha256':CFG_HASH}}
        started=time.monotonic();save(directory/'request.json',request)
        command=[ACT,'-m','act.pipeline.moe.external_pair_worker','--root',str(directory),'--started',repr(started)]
        code,expired=execute(command,directory/'worker.log',started,env)
        row={**job,'budget_seconds':300,'wall_seconds':time.monotonic()-started,'outer_timeout':expired,
            'return_code':code,'status':'TIMEOUT' if expired else 'ERROR','package':None,'snapshot_sha256':None,
            'request_sha256':_sha256(directory/'request.json'),'load_average_after':list(os.getloadavg())}
        if (directory/'common_facts.json').exists():row['snapshot_sha256']=_sha256(directory/'common_facts.json')
        if (directory/'routes.json').exists():row['routes_sha256']=_sha256(directory/'routes.json')
        if (directory/'external.json').exists():row['external_sha256']=_sha256(directory/'external.json')
        if not expired and code==0:
            if job['method']=='adaptive':
                package=directory/'package';m=json.loads((package/'manifest.json').read_text())
                row.update(status=m['status'],package=str(package),manifest_sha256=_sha256(package/'manifest.json'))
            else:row['status']=json.loads((directory/'external.json').read_text())['status']
        row['wall_seconds']=time.monotonic()-started
        if row['wall_seconds']>300:row.update(status='TIMEOUT',outer_timeout=True,package=None)
        row['evidence_level']='HZ_POLICY_ACCEPTED' if job['method']=='adaptive' else 'CROWN_NUMERICAL_FILTER'
        save(directory/'terminal.json',row)
        with (root/'rows.jsonl').open('a') as handle:
            handle.write(json.dumps(row,sort_keys=True)+'\n');handle.flush();os.fsync(handle.fileno())
        print(f"{'smoke' if smoke else 'full'} {number}/{len(schedule)} {job['job_id']} {row['status']} {row['wall_seconds']:.2f}s",flush=True)
        if row['status']=='ERROR':raise RuntimeError('worker error retained, no replacement')
    save(root/'audit.final.json',audit(root))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=DEFAULT)
    p.add_argument('--smoke',action='store_true');p.add_argument('--pipeline',action='store_true');a=p.parse_args()
    with (PROJECT_ROOT/'data/moe/results/route_complexity_pairing.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.pipeline:run(a.config,True);run(a.config,False)
        else:run(a.config,a.smoke)
