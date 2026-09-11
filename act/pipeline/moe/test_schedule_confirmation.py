import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from act.pipeline.moe import schedule_confirmation as runner
from act.pipeline.moe.schedule_confirmation_selection import index_fields, source_indices, verify_exclusions
from act.pipeline.moe.common_fact_snapshot import publish_snapshot
from act.pipeline.moe.test_route_complexity_schedule import model
from act.pipeline.moe.staged_verifier import verify_staged_linf, write_evidence_package
from act.pipeline.moe.experiment1 import _sha256


class ConfirmationTests(unittest.TestCase):
    def setup_config(self):
        cfg = json.loads(runner.DEFAULT.read_text())
        selection, configs = runner.artifacts(cfg)
        return cfg, selection, configs

    def test_frozen_30_new_inputs_and_balanced_three_arms(self):
        cfg, selection, configs = self.setup_config()
        verify_exclusions(selection)
        self.assertEqual(len(runner.jobs(selection, False)), 270)
        self.assertEqual(len(runner.jobs(selection, True)), 9)
        self.assertTrue(set(r['dataset_index'] for r in selection['samples']).isdisjoint(selection['excluded_indices']))
        self.assertEqual(selection['smoke_samples'][0]['dataset_index'], 3000)
        jobs = runner.jobs(selection, False)
        for m in selection['models']:
            for arm in runner.ARMS:
                for pos in range(3):
                    self.assertEqual(sum(j['model']==m and j['method']==arm and j['position']==pos for j in jobs),10)
        self.assertEqual({k for k in configs['adaptive'] if configs['adaptive'][k]!=configs['matched'][k]}, {'comparison_method'})
        self.assertNotIn('route_complexity_schedule', configs['legacy'])
        self.assertEqual(configs['legacy']['f0']['solver']['property_seconds'],300)

    def test_config_and_comparator_drift_rejected(self):
        cfg, _, _ = self.setup_config()
        for key, value in [('sample_count',31),('budget_seconds',301),('primary_comparator','legacy'),
                           ('selection_sha256','bad'),('selection_audit_sha256','bad')]:
            bad = copy.deepcopy(cfg);bad[key] = value
            with self.assertRaises(ValueError): runner.artifacts(bad)
        bad = copy.deepcopy(cfg);bad['methods']['legacy']['sha256']='bad'
        with self.assertRaises(ValueError): runner.artifacts(bad)
        retired=json.loads((Path(__file__).parent/'configs/schedule_confirmation_r1.json').read_text())
        with self.assertRaisesRegex(ValueError,'retired preprocessing'):runner.artifacts(retired)

    def test_totensor_initialization_order_and_retained_smoke_identity(self):
        import numpy as np
        from torchvision.transforms.functional import to_tensor
        original=torch.get_default_dtype()
        try:
            image=np.full((2,2,3),129,dtype=np.uint8)
            torch.set_default_dtype(torch.float32);cast=to_tensor(image).double()
            torch.set_default_dtype(torch.float64);direct=to_tensor(image)
            self.assertFalse(torch.equal(cast,direct))
        finally: torch.set_default_dtype(original)
        _,selection,configs=self.setup_config()
        root=runner.PROJECT_ROOT/'data/moe/results/schedule_confirmation_smoke_20260912_r1'
        row=json.loads((root/'rows.jsonl').read_text().splitlines()[0])
        evidence=json.loads((Path(row['package'])/'evidence.json').read_text())
        expected=runner.expected_identity(selection,row,configs['adaptive'],True)
        self.assertEqual(expected,evidence['identity'])
        old=json.loads((Path(__file__).parent/'configs/schedule_confirmation_selection_r1.json').read_text())
        self.assertEqual([s['dataset_index'] for s in old['samples']],[s['dataset_index'] for s in selection['samples']])
        self.assertNotEqual(old['smoke_samples'][0]['center'],selection['smoke_samples'][0]['center'])

    def test_missing_smoke_and_changed_source_rejected(self):
        cfg, _, _ = self.setup_config()
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            cfg['smoke_output']=tmp
            with self.assertRaisesRegex(ValueError,'audited old-input smoke'): runner.smoke_gate(runner.DEFAULT,cfg,'s')
            (Path(tmp)/'runtime.json').write_text(json.dumps({'smoke':True,'config_sha256':_sha256(runner.DEFAULT),'source_sha256':'old'}))
            (Path(tmp)/'audit.final.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'identity differs'): runner.smoke_gate(runner.DEFAULT,cfg,'new')

    def test_exclusion_parser_does_not_confuse_expert_indices(self):
        self.assertEqual(index_fields({'indices':[99,100],'samples':[{'dataset_index':17,'pair':[1,2]}]}),{17})
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            path=Path(tmp)/'sample_indices.json';path.write_text('{"indices":[4,5]}')
            self.assertEqual(source_indices(path),{4,5})

    def fixture(self, directory):
        cfg, _, configs = self.setup_config()
        net = model(((2.,)+ (0.,)*9, (3.,)+(0.,)*9, (4.,)+(0.,)*9))
        snapshots = []
        checkpoint = {'path':'/data1/Kane/MOE/toy_not_a_training_checkpoint.pt','sha256':'toy'}
        report = verify_staged_linf(net,torch.full((1,2),.5,dtype=torch.float64),.1,configs['adaptive'],
                                   checkpoint_identity=checkpoint,common_fact_callback=snapshots.append)
        identity = report.evidence['identity']
        sample = {'dataset_index':42,'label':0,**{k:identity[k] for k in ('center','lower','upper')}}
        selection = {'models':{'seed0':{'checkpoint':checkpoint['path'],'checkpoint_sha256':'toy',
                                       'model_state':identity['model_state']}},
                     'samples':[sample],'smoke_samples':[sample],'request':{'epsilon':.1}}
        report.evidence['execution']={'git_head':'test','dataset_index':42,'config_sha256':runner.METHOD_HASHES['adaptive']}
        job = {'job_id':'rank0_seed0_adaptive','rank':0,'method':'adaptive','model':'seed0','dataset_index':42}
        d = directory/job['job_id'];d.mkdir()
        publish_snapshot(d/'common_facts.json',snapshots[0]);write_evidence_package(report,d/'package')
        row = {**job,'budget_seconds':300,'wall_seconds':299.,'outer_timeout':False,'return_code':0,
               'status':'SAFE','package':str(d/'package'),'manifest_sha256':_sha256(d/'package/manifest.json'),
               'snapshot_sha256':_sha256(d/'common_facts.json')}
        return row, {'config':cfg,'git_head':'test','smoke':True}, selection, configs

    def test_durable_timeout_snapshot_and_identity_mutations(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);row,rt,selection,configs=self.fixture(root)
            self.assertTrue(runner.inspect_row(root,row,rt,selection,configs)['package'])
            killed={**row,'outer_timeout':True,'status':'TIMEOUT','package':None,'return_code':None,'wall_seconds':300.1}
            info=runner.inspect_row(root,killed,rt,selection,configs)
            self.assertFalse(info['package']);self.assertIsNotNone(info['facts'])
            for key,value in [('status','SAFE'),('snapshot_sha256',None),('snapshot_sha256','wrong')]:
                bad={**killed,key:value}
                with self.assertRaises(ValueError): runner.inspect_row(root,bad,rt,selection,configs)
            altered=copy.deepcopy(selection);altered['models']['seed0']['checkpoint_sha256']='different'
            with self.assertRaises(ValueError): runner.inspect_row(root,killed,rt,altered,configs)

    def test_three_arm_summary_is_input_clustered_and_conflict_closed(self):
        cfg,selection,_=self.setup_config();rows=[];details={}
        for job in runner.jobs(selection,False):
            status='SAFE' if job['method']=='adaptive' and job['rank']==0 else 'UNKNOWN'
            rows.append({**job,'status':status,'wall_seconds':2.})
            details[job['model'],job['rank'],job['method']]={'facts':None,'pair_count':2,'package':True}
        result=runner.summarize(rows,details,selection,False)
        self.assertAlmostEqual(result['input_clustered_contrasts']['matched']['SAFE']['mean_input_cluster_difference'],1/30)
        self.assertEqual(result['common_fact_pairs_unavailable'],90)
        self.assertEqual(result['models']['seed1']['contrasts']['legacy']['SAFE']['gained'],[0])
        next(r for r in rows if r['rank']==0 and r['method']=='legacy')['status']='UNSAFE'
        with self.assertRaisesRegex(ValueError,'conflict'): runner.summarize(rows,details,selection,False)

    def test_incomplete_or_reordered_stream_fails_before_package_checks(self):
        cfg,selection,configs=self.setup_config()
        expected=runner.jobs(selection,True)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp)
            (root/'runtime.json').write_text(json.dumps({'config':cfg,'config_path':str(runner.DEFAULT),
                'config_sha256':_sha256(runner.DEFAULT),'smoke':True}))
            for rows in (expected[:-1],list(reversed(expected))):
                (root/'rows.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
                with patch.object(runner,'artifacts',return_value=(selection,configs)), patch.object(
                        runner,'inspect_row',side_effect=AssertionError('stream should fail first')):
                    with self.assertRaisesRegex(ValueError,'incomplete, duplicate or reordered'):runner.audit(root)

    def test_legacy_static_weighted_control_still_matches_scheduled_arms(self):
        _,_,configs=self.setup_config()
        # One expert violates a class margin, yet the actual weighted output
        # is safe. A legacy reference must keep variable weighted obligations.
        net=model(((0.,1.)+(-2.,)*8,(3.,0.)+(-2.,)*8))
        values=[]
        for arm in runner.ARMS:
            r=verify_staged_linf(net,torch.full((1,2),.5,dtype=torch.float64),.1,configs[arm])
            values.append(r.status)
        self.assertEqual(values,['SAFE','SAFE','SAFE'])


if __name__=='__main__': unittest.main()
