"""Six-model catalog/terminal audit controls, no model/data/solver calls."""
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scoped_proof.io import load,save,sha
from multilayer_stage1 import request_catalog
from multilayer_stage1_audit import audit


class ResidualQueueControls(unittest.TestCase):
    def test_two_arm_240_full_ledger_and_unsafe_union(self):
        self._full_ledger(pruned_only=False)

    def test_pruned_only_120_full_ledger_and_unsafe_union(self):
        self._full_ledger(pruned_only=True)

    def _full_ledger(self,*,pruned_only):
        slots=120 if pruned_only else 240
        arms=['genbab_pruned'] if pruned_only else ['act','genbab_pruned']
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE/results',prefix='stage1b_ledger_control_') as name:
            root=Path(name)
            models=[dict(id=f'R{depth}E8_seed{seed}',architecture=dict(layers=depth,experts=8))
                    for depth in (2,3) for seed in range(3)]
            spec=dict(models=models,excluded_ranges=[[98,98]],
                history_union_sha256=hashlib.sha256(b'[98]').hexdigest(),
                selection=dict(count=30,verification_count=20),
                study=dict(arms=arms,endpoint_configs=['R3E8'],collapsed_models=[]),
                census_multipliers=[.5,1.,2.],budgets=dict(slots=slots,artifacts=128*2**20,
                    stage=(10 if pruned_only else 20)*3600))
            manifest=root/'manifest.json';save(manifest,spec)
            save(root/'start.json',dict(manifest_sha256=sha(manifest)))
            samples=[dict(rank=i,index=2000+i) for i in range(30)]
            (root/'selection').mkdir()
            with gzip.open(root/'selection/outcome.json.gz','wt') as stream:
                json.dump(dict(value=dict(samples=samples)),stream)
            queries=request_catalog(spec,samples,root/'selection')
            self.assertEqual(len(queries),slots)
            self.assertEqual([q['arm'] for q in queries[:4]],['genbab_pruned']*4 if pruned_only else
                ['act','genbab_pruned','genbab_pruned','act'])
            if pruned_only:
                self.assertEqual({q['arm'] for q in queries},{'genbab_pruned'})
                self.assertEqual([q['model']['id'] for q in queries[:6]],[m['id'] for m in models])
            self.assertEqual({q['sample']['index'] for q in queries},set(range(2000,2020)))
            save(root/'catalog.json',dict(manifest_sha256=sha(manifest),queries=queries,
                selection_outcome_sha256=sha(root/'selection/outcome.json.gz')))
            ordinal=0
            def terminal(key,query,extra):
                nonlocal ordinal
                target=root/key;target.mkdir(exist_ok=True)
                save(target/'request.json',dict(**query,manifest_sha256=sha(manifest),worker_deadline=100.))
                value=dict(status='TIMEOUT',accepted=False,through_reception_seconds=.9,
                    hard_budget=300,files={},partial=None)
                save(target/'terminal.json',value)
                ordinal+=1
                save(root/f'ledger_{ordinal:04d}.json',dict(id=key,kind=query['kind'],**value,**extra,
                    seconds=1.,terminal_sha256=sha(target/'terminal.json')))
                return target
            terminal('selection',dict(kind='selection'),{})
            for model in models:
                target=terminal('pgd_'+model['id'],dict(kind='pgd',model=model,samples=samples[:20]),dict(model=model['id']))
                if model['id']=='R3E8_seed0':
                    save(target/'domain_001.json',dict(index=2000,multiplier=1.,
                        request_sha256=sha(target/'request.json'),accepted=True,clock=10.,deadline=11.,
                        result=dict(status='UNSAFE_REPLAYED',witness=[[.1]],guarantee='FULL_MODEL_REPLAY')))
            for model in models:
                terminal('census_'+model['id'],dict(kind='census',model=model,samples=samples),dict(model=model['id']))
            for i,query in enumerate(queries):
                terminal(f'verify_{i:03d}',query,dict(model=query['model']['id'],arm=query['arm'],
                    index=query['sample']['index'],rank=query['sample']['rank']))
            save(root/'complete.json',dict(status='COMPLETE',full_calls=slots,auxiliary_calls=13,terminals=slots+13,seconds=1000.))
            value=audit(root,manifest)
            self.assertEqual(value['status'],'PASS_COMPLETE')
            self.assertEqual(value['endpoints']['R3E8']['adjusted']['n'],59)
            self.assertEqual(value['endpoints']['R3E8']['raw']['n'],60)
            self.assertEqual(set(value['endpoints']),{'R3E8'})
            self.assertNotIn('excluding_collapsed_seed_descriptive',value['endpoints']['R3E8'])
            self.assertIsNone(value['costs_L3_over_L1'])
            if pruned_only:
                self.assertEqual(value['paired'],{})
                self.assertEqual(value['expected_calls'],120)
                self.assertEqual(value['summaries']['R2E8/genbab_pruned']['n'],60)
                self.assertEqual(value['summaries']['R3E8/genbab_pruned']['n'],60)
            else:
                self.assertEqual(value['paired']['R2E8']['pairs'],60)
                self.assertEqual(value['paired']['R3E8']['pairs'],60)
            self.assertEqual(value['unsafe_source_counts'],dict(public_pgd=1))
            # Missing full-call receipt must not become a complete success.
            last=root/f'ledger_{slots+13:04d}.json';held=root/'last_held.json';last.rename(held)
            with self.assertRaises(AssertionError):audit(root,manifest)
            held.rename(last)
            # Intentional corruption of an isolated synthetic receipt. The
            # production save() correctly forbids overwriting real evidence.
            row=load(last);row['arm']='genbab_naive';last.write_text(json.dumps(row))
            with self.assertRaises(AssertionError):audit(root,manifest)

    def test_old_catalog_and_new_arm_budget_limits(self):
        samples=[dict(rank=i,index=3000+i) for i in range(30)]
        models=[dict(id=f'L{depth}E{experts}_seed{s}',architecture=dict(layers=depth,experts=experts))
            for depth,experts in ((1,8),(2,8),(3,8),(2,16)) for s in range(3)]
        spec=dict(models=models,budgets=dict(slots=600))
        got=request_catalog(spec,samples,'/fixed/selection')
        expected=[]
        for rank,sample in enumerate(samples[:20]):
            for pos,model in enumerate(models):
                arms=['genbab_pruned'] if model['architecture']['layers']==1 else ['act','genbab_naive','genbab_pruned']
                shift=(rank+pos)%len(arms)
                for arm in arms[shift:]+arms[:shift]:
                    expected.append(dict(kind='verification',model=model,sample=sample,arm=arm,selection_root='/fixed/selection'))
        self.assertEqual(got,expected)
        for arms in ([],['act','act'],['not_registered']):
            with self.assertRaises(ValueError):request_catalog({**spec,'study':dict(arms=arms)},samples,'/fixed/selection')
        with self.assertRaises(ValueError):request_catalog({**spec,'budgets':dict(slots=240)},samples,'/fixed/selection')


if __name__=='__main__':unittest.main()
