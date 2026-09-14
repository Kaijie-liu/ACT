"""Compact immutable training archive, after the independent landing audit."""
import argparse
import json
from pathlib import Path

from act.pipeline.moe.conv_training import atomic_json,sha,selected_epoch


def archive(root,output):
    if output.exists():raise FileExistsError(output)
    launch=json.loads((root/'launch.json').read_text())
    config=json.loads((root/'config.json').read_text())
    landed=json.loads((root/'CONV_LANDED_summary.json').read_text())
    audit=json.loads((root/'audit.json').read_text())
    supervisor=json.loads((root/'supervisor.json').read_text())
    if (landed['status']!='LANDED_AUDITED' or supervisor['status']!='LANDED_AUDITED'
        or audit['status']!='PASS' or audit['issues']):raise ValueError('training is not audited and landed')
    for name,key in [('audit.json','audit_sha256'),('launch.json','launch_sha256'),('split.json','split_sha256')]:
        if sha(root/name)!=landed[key]:raise ValueError('landing artifact drift')
    if sha(root/'CONV_LANDED_summary.json')!=supervisor['landed_sha256']:raise ValueError('supervisor landing drift')
    if sha(root/'config.json')!=launch['config_sha256'] or sha(root/'source.tar')!=launch['source_tar_sha256']:
        raise ValueError('recipe/source identity drift')
    rows=[json.loads((root/'epochs'/f'{e:03d}.json').read_text()) for e in range(1,config['epochs']+1)]
    if selected_epoch(rows)!=landed['best_epoch']:raise ValueError('selection reconstruction mismatch')
    for e,row in enumerate(rows,1):
        if (row['epoch']!=e or sha(root/'checkpoints'/f'epoch_{e:03d}.pt')!=row['checkpoint_sha256']
                or row['train']['samples']!=45000 or row['validation']['samples']!=5000):
            raise ValueError('checkpoint or denominator drift')
    result=dict(status='PASS',issues=[],run_root=str(root),launch=launch,config=config,landed=landed,
        supervisor=supervisor,epochs=rows,hashes={str(p.relative_to(root)):sha(p) for p in
            [root/'launch.json',root/'config.json',root/'split.json',root/'audit.json',
             root/'training_summary.json',root/'CONV_LANDED_summary.json',root/'supervisor.json',
             root/'training.log',root/'audit.log',*sorted((root/'epochs').glob('*.json'))]},
        scope='second architecture training/selection and independent concrete metric replay; no robustness or SAFE claim')
    atomic_json(output,result)
    print(json.dumps({'status':'PASS','epochs':len(rows),'best_epoch':landed['best_epoch'],'test':landed['test']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    archive(a.root.resolve(),a.output.resolve())
