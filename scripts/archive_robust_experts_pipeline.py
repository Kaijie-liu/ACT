"""Re-read exact saved weights, epoch roster, evaluation sources and costs."""
import argparse
import json
from pathlib import Path
import sys
import time
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def archive(path):
    started=time.monotonic()
    cfg=json.loads(path.read_text());root=Path(cfg['output_root'])
    for file,digest in cfg['files'].items():
        if sha256(file)!=digest:raise ValueError('input identity')
    summary=json.loads((root/'summary.json').read_text())
    if summary['config_sha256']!=sha256(path) or not summary['accepted']:raise ValueError('incomplete batch')
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,cfg['repo'])
    from robust_experts_supervised_pipeline import state_digest
    rows=[]
    for arm in ['dense','convmoe']:
        folder=root/arm
        term=json.loads((folder/'terminal.json').read_text())
        if term not in summary['records'] or not term['accepted'] or term['execution_seconds']>cfg['seconds_per_arm']:
            raise ValueError('terminal budget/binding')
        receipt=json.loads((folder/'receipt.json').read_text())
        if receipt['status']!='COMPLETED':raise ValueError('outer not completed')
        for kind in ['stdout','stderr']:
            if sha256(folder/(kind+'.txt'))!=receipt[kind+'_sha256']:raise ValueError('outer log changed')
        tr=json.loads((folder/'train.json').read_text());ev=json.loads((folder/'evaluate.json').read_text())
        check=json.loads((folder/'audit.json').read_text())
        checkpoint=folder/'final_epoch.ckpt';digest=sha256(checkpoint)
        for r in [tr,ev,check]:
            if r['config_sha256']!=sha256(path) or r['arm']!=arm or r['checkpoint_sha256']!=digest:
                raise ValueError('final-weight binding')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['epoch']!=tr['epochs'] or saved['global_step']!=tr['global_step']:
            raise ValueError('final training cursor')
        if saved['reproduction_binding']!={'arm':arm,'config_sha256':sha256(path)}:
            raise ValueError('checkpoint provenance')
        if state_digest(saved['state_dict'])!=tr['state_digest']:
            raise ValueError('saved state mismatch')
        if len(tr['epoch_checkpoints'])!=tr['epochs'] or len(list(folder.glob('epoch*.json')))!=tr['epochs']:
            raise ValueError('epoch roster')
        for name,h in tr['epoch_checkpoints'].items():
            if sha256(folder/'checkpoints'/name)!=h:raise ValueError('epoch hash')
        if [r['kind'] for r in ev['records']]!=['clean','PGD20','APGD20']:raise ValueError('test roster')
        for r in ev['records']:
            expected=(640 if r['kind']=='clean' else 256) if cfg['mode']=='control' else 10000
            if r['examples']!=expected or r['state_digest']!=tr['state_digest'] or r['checkpoint_sha256']!=digest:
                raise ValueError('test state/denominator')
        isolated=(folder/'logging.json').exists()
        warning='Previous log files in this directory will be deleted' in (folder/'evaluate.stderr').read_text()
        if isolated:
            log=json.loads((folder/'logging.json').read_text())
            if warning or len(log['files'])!=2 or any(sha256(p)!=h for p,h in log['files'].items()):
                raise ValueError('stage log preservation')
        elif not warning:
            raise ValueError('unexpected R1 log status')
        costs={s:json.loads((folder/(s+'_finished.json')).read_text())['seconds'] for s in ['train','evaluate','audit']}
        if sum(costs.values())>term['execution_seconds']+.05:raise ValueError('stage accounting')
        rows.append({'arm':arm,'terminal':term,'epochs':tr['epochs'],'global_step':tr['global_step'],
            'checkpoint_sha256':digest,'stage_seconds':costs,'evaluations':ev['records'],
            'stage_metrics_preserved':isolated,'CSV_train_log_overwrite_observed':warning,
            'saved_hashes':{str(p):sha256(p) for p in [folder/'train.json',folder/'evaluate.json',folder/'audit.json',
                folder/'prepared.json',folder/'receipt.json',folder/'terminal.json']}})
    return {'audit':'INDEPENDENT_SAVED_WORKFLOW_REVIEW_PASS','config_sha256':sha256(path),
        'summary_sha256':sha256(root/'summary.json'),'rows':rows,'summary':summary,
        'logging_preservation_pass':all(r['stage_metrics_preserved'] for r in rows),
        'separate_audit_seconds':time.monotonic()-started,'paper_scale_training':cfg['mode']=='training',
        'formal_SAFE':False,'exact_gpu_resume':False,
        'scope':'saved full-workflow control; metrics from limited batches are NOT paper accuracy estimates'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();write(a.output,archive(a.config))
