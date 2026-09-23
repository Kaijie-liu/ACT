import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import metamoe_receipt_reserve as control
import audit_metamoe_receipt_reserve as review
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


class ProtocolControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir='/data1/Kane/MOE',prefix='reserve-protocol-')
        self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()

    def test_frozen_selection_order_and_no_new_parameters(self):
        parent=json.loads(control.PARENT.read_text())
        rows=control.selected_requests(parent)
        self.assertEqual([r['id'] for r in rows],['cifar10_1','cifar10_2','mnist_1','mnist_3'])
        self.assertEqual(control.VARIANTS,{'full_native':1.,'receipt_reserve':.8})
        self.assertEqual(len(control.roster(rows)),8)
        self.assertEqual(control.roster(rows)[:4],[['cifar10_1','full_native'],['cifar10_1','receipt_reserve'],
            ['cifar10_2','receipt_reserve'],['cifar10_2','full_native']])
        with self.assertRaises(ValueError):control.view(parent,'unregistered')

    def synthetic_batch(self, fail):
        cfg={'protocol':'CONTROL_ONLY','output_root':str(self.root/'batch'),'seconds':.12,
            'group_rss_limit_bytes':8*2**30,'execution_commit':'HEAD','act_options':{},
            'requests':[{'id':'synthetic','label':0,'tensor_file':'no-data'}],
            'files':{'no-data':'synthetic-hash'},'margin':1e-7}
        cfg['roster']=control.roster(cfg['requests'])
        path=self.root/'config.json';write(path,cfg)
        code="raise RuntimeError('synthetic')" if fail else (
            "import sys,json,time; from pathlib import Path; p=Path(sys.argv[1]); "
            "(p/'result.json').write_text('{'); time.sleep(2)")
        def command(cfg,path,rid,variant):
            return [sys.executable,'-c',code,str(Path(cfg['output_root'])/variant/f'{rid}_act')]
        with patch.object(control,'validate'),patch.object(control,'require_clean'),patch.object(control,'command',side_effect=command):
            control.run(path)
            replay=self.root/'replay.json';write(replay,review.replay(path))
            result=review.audit(path,replay)
            self.assertEqual(result['audit'],'PASS')
            self.assertGreater(result['cost']['batch_wall_through_summary_seconds'],result['cost']['charged_request_seconds'])
            # Removing/reordering a terminal may never manufacture a smaller denominator.
            summary_path=Path(cfg['output_root'])/'summary.json'
            summary=json.loads(summary_path.read_text());summary['rows'].pop();summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError,'denominator'):review.audit(path,replay)
        return result

    def test_outer_deadline_partial_evidence_and_complete_cost(self):
        r=self.synthetic_batch(False)
        self.assertEqual([x['status'] for x in r['rows']],['TIMEOUT','TIMEOUT'])
        self.assertTrue(all(x['seconds']>=.12 for x in r['rows']))
        self.assertTrue(all(x['grade']=='NONE' for x in r['rows']))

    def test_exception_failstop_keeps_full_denominator(self):
        r=self.synthetic_batch(True)
        self.assertEqual([x['status'] for x in r['rows']],['ERROR','NOT_STARTED_AFTER_ERROR'])

    def test_native_budget_mutations_rejected(self):
        q=self.root/'protected/evaluation_000/query_000';q.mkdir(parents=True)
        req={'token':'t','model_sha256':'m','query_sha256':'q','scope':{'phase':'expanded'},'deadline_monotonic':5.}
        budget={**req,'applied_fraction':.8,'remaining_before_budget_publication':2.,'proposed_native_seconds':1.6}
        write(q/'request.json',req);write(q/'native_budget.json',budget)
        self.assertEqual(review.native_caps(self.root,.8)['native_cost_missing'],1)
        for field,value in (('token','wrong'),('applied_fraction',1.),('deadline_monotonic',6.),
                            ('proposed_native_seconds',2.),('remaining_before_budget_publication',float('inf'))):
            bad={**budget,field:value};(q/'native_budget.json').write_text(json.dumps(bad))
            with self.subTest(field=field),self.assertRaises(ValueError):review.native_caps(self.root,.8)
        (q/'native_budget.json').write_text(json.dumps(budget))

    def test_replay_must_cover_variant_not_just_input(self):
        cfg={'requests':[{'id':'a','label':0}],'margin':1e-7}
        rows=[{'id':'a','variant':'receipt_reserve','status':'UNSAFE_REPLAYED','result_sha256':'r'}]
        record={'audit':'INDEPENDENT_ORIGINAL_MODEL_REPLAY_PASS','config_sha256':'h','summary_sha256':'s',
            'separate_audit_seconds':0.,'rows':[]}
        with self.assertRaisesRegex(ValueError,'replay'):review.check_replay(cfg,'s','h',rows,record)
        record['rows']=[{'id':'a','variant':'full_native','result_sha256':'r'}]
        with self.assertRaisesRegex(ValueError,'replay'):review.check_replay(cfg,'s','h',rows,record)


if __name__=='__main__':unittest.main()
