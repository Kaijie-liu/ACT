import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import freeze_metamoe_checked_small as selection
from recent_moe_deployment import sha256


class SelectionControls(unittest.TestCase):
    def test_exclude_failed_component_and_intake_indices(self):
        c=[{'requests':[{'dataset':'MNIST','dataset_index':0},{'id':'cifar10_test_0','index':0}]},
           {'requests':[{'dataset':'MNIST','index':4}]}]
        self.assertEqual(selection.historical_indices(c),{'CIFAR10':[0],'MNIST':[0,4]})
        with self.assertRaises(ValueError):selection.historical_indices([{'requests':[{'dataset':'MNIST'}]}])

    def test_first_correct_raw_prefix_ignores_route_and_bounds(self):
        rec=[{'index':i,'label':1,'prediction':int(i!=1),'finite':True,'route_count':100-i} for i in range(4)]
        self.assertEqual(selection.choose_prefix(rec,[0],2),[2,3])
        for r in rec:r['route_count']=0;r['bound']=1e10
        self.assertEqual(selection.choose_prefix(rec,[0],2),[2,3])

    def test_no_extension_undefined_and_nonordered_rejected(self):
        r=[{'index':0,'label':0,'prediction':0,'finite':True}]
        for bad,count in ((r,2),(r*2,1),([{**r[0],'index':1}],1),([{**r[0],'finite':False}],1)):
            with self.assertRaises(ValueError):selection.choose_prefix(bad,[],count)

    def test_selection_identity_dimensions_tamper_and_order(self):
        tmp=tempfile.TemporaryDirectory(prefix='selection-control-',dir='/data1/Kane/MOE');self.addCleanup(tmp.cleanup)
        root=Path(tmp.name);p=root/'plan.json';p.write_text('{}')
        plan={'datasets':['CIFAR10','MNIST'],'per_dataset':5,'excluded':{'CIFAR10':[0],'MNIST':[0]}}
        rr=[];scans={};hashes={}
        for ds in plan['datasets']:
            scans[ds]=[{'index':i,'label':1,'prediction':1,'finite':True} for i in range(6)]
            for i in range(1,6):
                f=root/f'{ds.lower()}_{i}.npz';x=np.zeros((1,3,32,32),dtype=np.float64)
                np.savez(f,center=x,lower=x-2/255,upper=x+2/255);hashes[str(f)]=sha256(f)
                rr.append({'id':f'{ds.lower()}_{i}','dataset':ds,'index':i,'label':1,'clean_prediction':1,'tensor_file':str(f)})
        s={'plan_sha256':sha256(p),'verification_calls':0,'requests':rr,'scans':scans,'tensor_hashes':hashes}
        with patch.object(selection,'PLAN',p),patch.object(selection,'SELECTION',root):
            selection.check_selection(plan,s)
            bad=copy.deepcopy(s);bad['requests'][0]['label']=2
            with self.assertRaises(ValueError):selection.check_selection(plan,bad)
            bad=copy.deepcopy(s);bad['requests']=bad['requests'][::-1]
            with self.assertRaises(ValueError):selection.check_selection(plan,bad)
            np.savez(rr[0]['tensor_file'],center=x,lower=x,upper=x)
            with self.assertRaises(ValueError):selection.check_selection(plan,s)

    def test_plan_gate_and_source_mutation(self):
        plan=selection.build_plan();selection.validate_plan(plan)
        for key,value in [('per_dataset',6),('scan_cap',2000),('excluded',{}),('seconds',600)]:
            bad=copy.deepcopy(plan);bad[key]=value
            with self.assertRaises(ValueError):selection.validate_plan(bad)


if __name__=='__main__':unittest.main()
