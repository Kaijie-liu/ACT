"""Full fixed-roster and validation-selection checks without trained models."""
import copy
import os
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from architecture_s2_study import ARMS,all_requests,selected_epoch,primary_report


class StudyControls(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('S2_CONTROL_ROOT'),'requires dedicated synthetic record directory')
    def test_primary_record_aggregation_and_missing_measurement(self):
        import numpy as np
        from architecture_s1_execution import Artifacts,read_record,digest
        from scoped_proof.io import sha
        from act.pipeline.moe.architecture_s2_evaluation import pack_arrays
        root=Path(os.environ['S2_CONTROL_ROOT'])/'primary_records'
        store=Artifacts.create(root);count=10000
        pixels=dict(pixels=np.zeros((count,3,32,32),dtype=np.uint8),labels=np.zeros(count,dtype=np.int64))
        input_sha=store.write(root/'inputs.json.gz',pack_arrays(pixels))
        models=[dict(id=f'{a}_seed{s}',arm=a,seed=s,family='moe' if a in ('M','M-reg') else 'dense',
                     checkpoint_sha256='synthetic checkpoint') for s in range(5) for a in ARMS]
        spec=dict(models=models,source_files={},inputs=dict(path=str(root/'inputs.json.gz'),sha256=input_sha),
                  bootstrap=dict(resamples=10000,seed=20261031),request_catalog_sha256=digest(all_requests(models)))
        manifest=root/'manifest.json';store.write(manifest,spec)
        # A missing accepted measurement must not produce an aggregate verdict.
        with self.assertRaises(FileNotFoundError):primary_report(root)
        self.assertFalse((root/'primary_summary.json.gz').exists())
        for m in models:
            good=m['arm']=='M';moe=m['family']=='moe';ident='primary_'+m['id'];folder=root/ident;folder.mkdir()
            arrays=dict(prediction=np.zeros(count,dtype=np.int64),label=pixels['labels'],
                clean=np.ones(count,dtype=bool),certified=np.full(count,good,dtype=bool),
                lower_margin=np.full((count,9),1 if good else -1,dtype=np.float32),
                upper_margin=np.full((count,9),2,dtype=np.float32),
                route_count=np.full(count,1 if moe else 0,dtype=np.uint8))
            if moe:
                router=np.zeros((count,8),dtype=np.float32);router[:,:2]=1
                mask=np.zeros((count,4),dtype=np.uint8);mask[:,0]=1
                arrays.update(router_lower=router,router_upper=router,pair_mask=mask)
            ah=store.write(folder/'arrays.json.gz',pack_arrays(arrays))
            q=dict(manifest_sha256=sha(manifest),model=m['id'],mode='primary',radius=2,worker_deadline=10)
            qh=store.write(folder/'request.json.gz',q)
            value=dict(checkpoint_sha256=m['checkpoint_sha256'],input_sha256=input_sha,
                family=m['family'],count=count,properties=9,array_sha256=ah,clean=count,
                certified=count if good else 0,batches=[])
            oh=store.write(folder/'outcome.json.gz',dict(request_sha256=qh,clock=1,value=value))
            store.write(folder/'published.json.gz',dict(sha256=oh,clock=2))
            th=store.write(folder/'terminal.json.gz',dict(accepted=True,status='AUXILIARY_COMPLETE'))
            store.write(root/(ident+'.ledger.json.gz'),dict(accepted=True,status='AUXILIARY_COMPLETE',
                terminal_sha256=th,seconds=1))
        primary_report(root)
        result=read_record(root/'primary_summary.json.gz')
        self.assertEqual(result['decision'],'ENTER_S2B')
        self.assertEqual(result['arms']['M']['certified'],50000)
        self.assertEqual(result['comparisons']['M_minus_D-act']['interval95'],[1.,1.])
        self.assertFalse(result['secondary_stage_complete'])

    def test_full_roster_no_filter_or_extra_calls(self):
        models=[dict(id=f'{a}_seed{s}',arm=a,seed=s) for a in ARMS for s in range(5)]
        rows=all_requests(models)
        self.assertEqual(len(rows),24020)
        self.assertEqual(len({r['id'] for r in rows}),len(rows))
        self.assertEqual(rows,all_requests(list(reversed(models))))
        for mode,count in [('primary',1),('incomplete',1000),('complete',200)]:
            selected=[r for r in rows if r['mode']==mode]
            self.assertEqual(len(selected),20*count)
            for m in models:
                observed=[r for r in selected if r['model']==m['id']]
                self.assertEqual(len(observed),count)
                if mode!='primary':self.assertEqual([r['index'] for r in observed],list(range(count)))
        with self.assertRaises(ValueError):all_requests(models[:-1])
        bad=copy.deepcopy(models);bad[0]['seed']=4
        with self.assertRaises(ValueError):all_requests(bad)

    def test_epoch_selection_and_missing_denominator(self):
        epochs=[dict(epoch=i,total=45000,validation=dict(total=5000,clean=2000,certified=100))
                for i in range(1,101)]
        self.assertEqual(selected_epoch(epochs)['epoch'],1)
        epochs[4]['validation']['certified']=101
        epochs[5]['validation']['certified']=101
        self.assertEqual(selected_epoch(epochs)['epoch'],5)
        epochs[5]['validation']['clean']=2001
        self.assertEqual(selected_epoch(epochs)['epoch'],6)
        with self.assertRaises(ValueError):selected_epoch(epochs[:-1])
        epochs[0]['validation']['total']=4999
        with self.assertRaises(ValueError):selected_epoch(epochs)


if __name__=='__main__':unittest.main()
