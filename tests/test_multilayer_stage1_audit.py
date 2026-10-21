"""New offline schema/union controls; no trained model, solver or new telemetry."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scoped_proof.io import load,save,sha
from multilayer_stage1_audit import audit


class AuditControls(unittest.TestCase):
    def test_all_timeout_catalog_partial_pgd_union_and_mutation(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE/results',prefix='stage1_audit_control_') as name:
            root=Path(name);models=[]
            for label,layers,experts in [('L1E8',1,8),('L2E8',2,8),('L3E8',3,8),('L2E16',2,16)]:
                for seed in range(3):models.append(dict(id=f'{label}_seed{seed}',architecture=dict(layers=layers,experts=experts)))
            spec=dict(models=models,excluded_ranges=[],history_union_sha256=hashlib.sha256(b'[]').hexdigest(),
                census_multipliers=[.5,1.,2.],budgets=dict(artifacts=128*2**20,stage=52*3600))
            manifest=root/'manifest.json';save(manifest,spec);save(root/'start.json',dict(manifest_sha256=sha(manifest)))
            samples=[dict(rank=i,index=1256+i) for i in range(30)]
            (root/'selection').mkdir()
            with gzip.open(root/'selection/outcome.json.gz','wt') as f:json.dump(dict(value=dict(samples=samples)),f)
            queries=[]
            for rank,sample in enumerate(samples[:20]):
                for position,model in enumerate(models):
                    arms=['genbab_pruned'] if model['architecture']['layers']==1 else ['act','genbab_naive','genbab_pruned']
                    shift=(rank+position)%len(arms);arms=arms[shift:]+arms[:shift]
                    for arm in arms:queries.append(dict(kind='verification',model=model,sample=sample,arm=arm,
                        selection_root=str(root/'selection')))
            save(root/'catalog.json',dict(manifest_sha256=sha(manifest),queries=queries,
                selection_outcome_sha256=sha(root/'selection/outcome.json.gz')))
            ordinal=0
            def terminal(key,query,extra):
                nonlocal ordinal
                target=root/key;target.mkdir(exist_ok=True)
                request=dict(**query,manifest_sha256=sha(manifest),worker_deadline=100.)
                save(target/'request.json',request)
                t=dict(status='TIMEOUT',accepted=False,through_reception_seconds=.9,hard_budget=300,
                    files={},partial=None)
                save(target/'terminal.json',t)
                row=dict(id=key,kind=query['kind'],**t,**extra,seconds=1.,terminal_sha256=sha(target/'terminal.json'))
                ordinal+=1;save(root/f'ledger_{ordinal:04d}.json',row)
                return target
            terminal('selection',dict(kind='selection'),{})
            for model in models:
                folder=terminal('pgd_'+model['id'],dict(kind='pgd',model=model,samples=samples[:20]),dict(model=model['id']))
                if model['id']=='L3E8_seed0':
                    save(folder/'domain_001.json',dict(index=1256,multiplier=1.,request_sha256=sha(folder/'request.json'),
                        accepted=True,clock=10.,deadline=11.,result=dict(status='UNSAFE_REPLAYED',
                            witness=[[.1]],guarantee='FULL_MODEL_REPLAY')))
            for model in models:terminal('census_'+model['id'],dict(kind='census',model=model,samples=samples),dict(model=model['id']))
            for i,q in enumerate(queries):terminal(f'verify_{i:03d}',q,
                dict(model=q['model']['id'],index=q['sample']['index'],rank=q['sample']['rank'],arm=q['arm']))
            save(root/'complete.json',dict(status='COMPLETE',full_calls=600,auxiliary_calls=25,terminals=625,seconds=1000.))
            value=audit(root,manifest)
            self.assertEqual(value['status'],'PASS_COMPLETE')
            self.assertEqual(value['endpoints']['L3E8']['adjusted']['n'],59)
            self.assertEqual(value['endpoints']['L3E8']['adjusted']['k'],59)
            self.assertEqual(value['endpoints']['L3E8']['raw']['n'],60)
            self.assertEqual(value['unsafe_source_counts'],dict(public_pgd=1))
            # Local synthetic fixture mutation only; never alter a real receipt.
            path=root/'ledger_0026.json';row=load(path);row['status']='POSITIVE'
            path.write_text(json.dumps(row))
            with self.assertRaises(AssertionError):audit(root,manifest)


if __name__=='__main__':unittest.main()
