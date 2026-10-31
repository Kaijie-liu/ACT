"""S2 primary semantics, complete denominators and two-level resampling."""
import copy
import unittest
import numpy as np
import torch
from torch import nn
from act.back_end.moe.multilayer import RoutedLayer
from act.back_end.moe.multilayer_external import SequentialTop2
from act.pipeline.moe.architecture_s2_evaluation import (
    input_batch,secondary_domain,primary_batch,check_primary_arrays,pack_arrays,unpack_arrays,check_inputs,
)
from act.pipeline.moe.architecture_s2_statistics import crossed_differences,primary_summary
from act.pipeline.moe.architecture_s2_training import evaluate_ibp
from torch.utils.data import DataLoader,TensorDataset


def constant(values,inputs=2):
    m=nn.Linear(inputs,len(values))
    with torch.no_grad():m.weight.zero_();m.bias.copy_(torch.tensor(values,dtype=m.weight.dtype))
    return nn.Sequential(nn.Flatten(),m)


class PrimaryControls(unittest.TestCase):
    def test_lossless_archive_and_invalid_dtype(self):
        arrays=dict(x=np.array([-.0,np.nextafter(np.float32(0),np.float32(1)),1.],dtype=np.float32),
                    bits=np.array([[1,0,255]],dtype=np.uint8),labels=np.array([0,9],dtype=np.int64))
        restored=unpack_arrays(pack_arrays(arrays))
        for k,v in arrays.items():
            self.assertEqual(restored[k].dtype,v.dtype)
            self.assertEqual(restored[k].tobytes(),v.tobytes())
        with self.assertRaises(ValueError):pack_arrays(dict(bad=np.array([{}],dtype=object)))
        with self.assertRaises(ValueError):unpack_arrays(dict(format='other',data=''))
        with self.assertRaises(ValueError):check_inputs(dict(pixels=np.zeros((1,3,32,32),dtype=np.float32),labels=np.array([0])),count=1)

    def test_physical_input_matches_totensor_and_secondary_lift(self):
        from torchvision.transforms import ToTensor
        image=np.arange(3*4*4,dtype=np.uint8).reshape(4,4,3)
        pixels=torch.from_numpy(image.transpose(2,0,1).copy())[None]
        center,lo,hi=input_batch(pixels)
        self.assertTrue(torch.equal(center[0],ToTensor()(image)))
        secondary=secondary_domain(pixels,4)
        for k,v in (('center',center),('lower',lo),('upper',hi)):
            self.assertTrue(torch.equal(secondary[k],v.double()))
        self.assertTrue((secondary['rows'][:,4]==1).all())
        self.assertTrue((secondary['rows'].sum(1)==0).all())

    def test_primary_matches_training_evaluation_and_keeps_clean_errors(self):
        pixels=torch.tensor([[0,255],[100,200],[23,57]],dtype=torch.uint8)
        y=torch.tensor([0,1,2])
        model=constant([2.,0.,-1.])
        arrays,metrics=primary_batch(model,pixels,y)
        check_primary_arrays(arrays,family='dense',count=3,classes=3)
        reference=evaluate_ibp(model,DataLoader(TensorDataset(input_batch(pixels)[0],y),batch_size=3),2/255)
        self.assertEqual(metrics['certified'],reference['certified'])
        self.assertEqual(metrics['clean'],1);self.assertEqual(metrics['count'],3)
        self.assertEqual(arrays['certified'].tolist(),[True,False,False])

    def test_tie_cover_roundtrip_and_corruption(self):
        # Every equal-score pair is legal; the zero-penalty tie is not stable.
        model=SequentialTop2([RoutedLayer(constant([0.]*8),[constant([2.,0.,-1.]) for _ in range(8)],
                                         k=2,mode='selected_softmax')])
        pixels=torch.tensor([[0,255],[100,200]],dtype=torch.uint8);y=torch.tensor([0,0])
        arrays,metrics=primary_batch(model,pixels,y)
        check_primary_arrays(arrays,family='moe',count=2,classes=3)
        self.assertEqual(arrays['route_count'].tolist(),[28,28]);self.assertEqual(metrics['single_route'],0)
        for change in ('missing','mask','label','property','nan','padding'):
            bad=copy.deepcopy(arrays)
            if change=='missing':del bad['upper_margin']
            elif change=='mask':bad['pair_mask'][0,0]^=1
            elif change=='label':bad['label'][0]=1
            elif change=='property':bad['lower_margin']=bad['lower_margin'][:,:1]
            elif change=='nan':bad['lower_margin'][0,0]=np.nan
            else:bad['pair_mask'][0,-1]|=128
            with self.assertRaises(ValueError):check_primary_arrays(bad,family='moe',count=2,classes=3)

    def test_manual_crossed_bootstrap_and_chunk_differential(self):
        delta=np.array([[[1,0,-1],[0,1,0]],[[0,-1,1],[1,0,0]]])
        result=crossed_differences(delta,resamples=31,seed=21,chunk=7)
        sr,ir=(np.random.default_rng(s) for s in np.random.SeedSequence(21).spawn(2))
        expected=[]
        for _ in range(31):
            s=sr.integers(0,2,size=2);i=ir.integers(0,3,size=3)
            expected.append(delta[:,s][:,:,i].mean(axis=(1,2)))
        np.testing.assert_array_equal(result,np.array(expected))
        np.testing.assert_array_equal(result,crossed_differences(delta,resamples=31,seed=21,chunk=1))
        np.testing.assert_array_equal(result,crossed_differences(delta,resamples=31,seed=21,chunk=31))

    def test_seed_uncertainty_not_fixed_seed_input_only(self):
        delta=np.array([[[1]*10,[-1]*10]])
        results=crossed_differences(delta,resamples=100,seed=3)
        self.assertEqual(results.min(),-1);self.assertEqual(results.max(),1)
        # An input-only bootstrap of this example is identically zero.

    def test_all_gates_and_exact_boundaries(self):
        n=100;cert=np.zeros((4,5,n),dtype=bool);clean=np.ones_like(cert)
        cert[:,:,:20]=True;cert[2,:,20:22]=True
        routes=np.zeros((4,5,n),dtype=np.uint8);routes[2:]=1;routes[2:,:,-10:]=2
        r=primary_summary(cert,clean,routes,resamples=2000,expected_inputs=n)
        self.assertTrue(r['gates']['m_minus_active_at_least_2pp'])
        self.assertTrue(r['gates']['m_single_route_at_least_90pct'])
        cert[2,:,22:40]=True
        r=primary_summary(cert,clean,routes,resamples=200,expected_inputs=n)
        self.assertEqual(r['decision'],'ENTER_S2B')
        # Single-route guard is independent of a strong accuracy gain.
        routes[2,:,0]=2
        self.assertEqual(primary_summary(cert,clean,routes,resamples=200,expected_inputs=n)['decision'],'STOP_S2A_WRITE_RESULTS')
        routes[2,:,0]=1;cert[1,:,40:61]=True
        self.assertFalse(primary_summary(cert,clean,routes,resamples=200,expected_inputs=n)['gates']['m_minus_total_nonnegative'])

    def test_missing_rows_and_inconsistent_positive_rejected(self):
        cert=np.zeros((4,5,10),dtype=bool);clean=np.zeros_like(cert)
        routes=np.zeros((4,5,10),dtype=np.uint8);routes[2:]=1
        with self.assertRaises(ValueError):primary_summary(cert,clean,routes,expected_inputs=11)
        cert[0,0,0]=True
        with self.assertRaises(ValueError):primary_summary(cert,clean,routes,expected_inputs=10)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
