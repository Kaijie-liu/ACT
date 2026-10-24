"""Read-only, standard-library record/identity audit; no tensor deserialization."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RUN=Path('/data1/Kane/MOE/results/multilayer_training_20261022_r1')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())


def audit(manifest=None,run=None):
    manifest=Path(manifest) if manifest else ROOT/'docs/multilayer_training_freeze_20261022.json'
    run=Path(run) if run else RUN
    spec=read(manifest);epochs_count=spec['epochs']
    start=read(run/'start.json');complete=read(run/'complete.json');split=read(run/'split.json')
    assert complete['status']=='COMPLETE' and len(complete['models'])==len(spec['models'])
    assert start['manifest_sha256']==digest(manifest)
    assert start['split_sha256']==digest(run/'split.json')
    if 'original_split_sha256' in spec:assert digest(run/'split.json')==spec['original_split_sha256']
    a,b=set(split['train']),set(split['validation'])
    assert len(a)==45000 and len(b)==5000 and not a&b and a|b==set(range(50000))
    code=Path('/data1/Kane/MOE/executions')/run.name
    assert (code/'.frozen_commit').read_text().strip()==start['commit']
    for p,h in spec['source_files'].items():assert digest(code/p)==h,p
    for p,h in spec['data_files'].items():assert digest(p)==h,p
    rows=[]
    for expected,actual in zip(spec['models'],complete['models']):
        assert actual['id']==expected['id']
        folder=run/actual['id']; epochs=[read(folder/f'epoch_{i:03d}.json') for i in range(1,epochs_count+1)]
        assert len(list(folder.glob('epoch_*.json')))==epochs_count
        best=max(epochs,key=lambda r:r['validation']['accuracy'])
        assert [r['epoch'] for r in epochs]==list(range(1,epochs_count+1))
        assert best['epoch']==actual['best_epoch'] and best['validation']==actual['validation']
        assert digest(actual['checkpoint'])==actual['checkpoint_sha256']
        assert read(folder/'complete.json')==actual
        for key,count in [('validation',5000),('test',10000)]:
            metric=actual[key]
            assert metric['total']==count and metric['accuracy']==metric['correct']/count
            assert all(sum(c)==2*count for c in metric['route_counts'])
        counts=actual['test']['route_counts']
        rows.append(dict(id=actual['id'],best_epoch=actual['best_epoch'],
            validation=actual['validation']['accuracy'],test=actual['test']['accuracy'],
            zero_use_experts=[sum(v==0 for v in c) for c in counts],
            top_two_selection_fraction=[sum(sorted(c)[-2:])/sum(c) for c in counts],
            seconds=actual['seconds'],checkpoint_sha256=actual['checkpoint_sha256']))
    assert not list(run.rglob('error.json')) and not list(run.rglob('*.partial'))
    return dict(status='PASS',models=rows,epochs=epochs_count*len(spec['models']),seconds=complete['seconds'],
        complete_sha256=digest(run/'complete.json'),split_sha256=digest(run/'split.json'),
        bytes=sum(p.stat().st_size for p in run.rglob('*') if p.is_file()),
        controller_log_bytes=run.with_suffix('.controller.log').stat().st_size,
        scope='frozen source/data/weight identities and selection records; no accuracy rerun or proof')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path);parser.add_argument('--run',type=Path)
    parser.add_argument('--output',type=Path);args=parser.parse_args();result=audit(args.manifest,args.run)
    if args.output:
        with args.output.open('x') as stream:json.dump(result,stream,sort_keys=True,indent=2);stream.write('\n')
    else:print(json.dumps(result,sort_keys=True,indent=2))
