"""Lifetime and retained-diagnostic controls; no trained model or LP solve."""
import ast
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from act.back_end.moe.multilayer import RoutedLayer, HistoryPlan, _lower, verify_multilayer_box
from act.back_end.moe.route_a import MissingComponentHZ, _component_output_hz
from act.back_end.hybridz_tf import HybridzTF
from act.back_end.core import Bounds, Fact, ConSet
from act.config.config import HybridZConfig
from act.back_end.transfer_functions import get_transfer_function, set_transfer_function
from act.util.device_manager import initialize_device
import act.pipeline.moe.experiment1 as component
import act.back_end.solver.solver_hz as solver

OLD = Path('/data1/Kane/MOE/executions/multilayer_stage1_20261023_r6/act/pipeline/moe/experiment1.py')


def affine(w, b):
    layer=nn.Linear(len(w[0]),len(w),dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(w,dtype=torch.float64))
        layer.bias.copy_(torch.tensor(b,dtype=torch.float64))
    return layer.eval()


def model(depth):
    layers=[]
    for _ in range(depth):
        layers.append(RoutedLayer(affine([[.1],[-.1],[0.]],[.1,-.1,0.]),
            [affine([[.1]],[1.]),affine([[-.1]],[1.]),affine([[.2]],[1.])],
            k=2,mode='selected_softmax'))
    return nn.Sequential(*layers,affine([[1.],[0.]],[0.,0.])).eval()


def net(depth):
    x=torch.zeros(1,1,dtype=torch.float64);plan=HistoryPlan(model(depth),x,weighted_matmul=True)
    graph,_=plan.compile(((0,1),)*depth)
    return _lower(graph,x,x-1,x+1,2+3*depth)


class LifetimeControls(unittest.TestCase):
    def setUp(self):
        initialize_device('cpu','float64');torch.set_num_threads(1)
        set_transfer_function(None)
    def tearDown(self):set_transfer_function(None)

    def test_restore_unset_and_existing_on_success_and_exception(self):
        graph=net(2)
        for prior in (None,HybridzTF()):
            for fails in (False,True):
                with self.subTest(prior=prior is not None, fails=fails):
                    refs=[]
                    def factory(*args,**kwargs):
                        value=HybridzTF(*args,**kwargs);refs.append(weakref.ref(value));return value
                    set_transfer_function(prior)
                    with patch.object(component,'HybridzTF',side_effect=factory):
                        if fails:
                            with patch.object(component,'analyze',side_effect=ValueError('fixed failure')):
                                with self.assertRaisesRegex(ValueError,'fixed failure'):
                                    component._propagate_component(graph,hybridz_config=HybridZConfig(max_input_dim=1))
                        else:component._propagate_component(graph,hybridz_config=HybridZConfig(max_input_dim=1))
                    if prior is None:
                        with self.assertRaises(RuntimeError):get_transfer_function()
                    else:self.assertIs(get_transfer_function(),prior)
                    gc.collect();self.assertIsNone(refs[0]())

    def test_old_retention_and_exact_coefficient_differential(self):
        tree=ast.parse(OLD.read_text())
        node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_propagate_component')
        scope=dict(vars(component))
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(OLD),'exec'),scope)
        for depth in (2,3):
            graph=net(depth);set_transfer_function(None)
            old=scope['_propagate_component'](graph,hybridz_config=HybridZConfig(max_input_dim=1))
            retained=weakref.ref(get_transfer_function())
            self.assertIsNotNone(retained())
            set_transfer_function(None);gc.collect();self.assertIsNone(retained())
            new=component._propagate_component(graph,hybridz_config=HybridZConfig(max_input_dim=1))
            for name in ('output_hz','input_hz'):
                a,b=getattr(old,name),getattr(new,name)
                self.assertEqual((a.frame_id,a.exact),(b.frame_id,b.exact))
                for field in ('c','b','ub'):np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
                for field in ('Gc','Gb','Ac','Ab','Auc','Aub'):
                    left,right=getattr(a,field),getattr(b,field)
                    self.assertEqual(left.shape,right.shape)
                    for member in ('data','indices','indptr'):
                        np.testing.assert_array_equal(getattr(left,member),getattr(right,member))

    def test_previous_history_released_before_next_propagation(self):
        refs=[];objectives=[];count=0
        native=component._propagate_component;linear=solver.sparse_hz_linear
        def propagate(*args,**kwargs):
            nonlocal count
            gc.collect()
            self.assertTrue(all(r() is None for r in refs+objectives))
            value=native(*args,**kwargs);refs.append(weakref.ref(value));count+=1;return value
        def project(*args,**kwargs):
            value=linear(*args,**kwargs);objectives.append(weakref.ref(value));return value
        def feasible(*args,**kwargs):return SimpleNamespace(status='feasible' if count==1 else 'infeasible')
        unknown=SimpleNamespace(status='unknown',minimum=None,solver_status=1,solver_bound_kind='none',elapsed=0.,candidate_input=None)
        x=torch.zeros(1,1,dtype=torch.float64)
        # Plain scoped replacements do not retain call arguments as Mock does;
        # retaining those arguments would itself keep the observed HZ alive.
        with patch.object(component,'_propagate_component',new=propagate), \
             patch.object(solver,'sparse_hz_linear',new=project), \
             patch.object(solver,'hz_check_feasibility',new=feasible), \
             patch.object(solver,'hz_minimize_output',new=lambda *a,**kw:unknown):
            result=verify_multilayer_box(model(1),center=x,lower=x-1,upper=x+1,
                rows=torch.tensor([[1.,-1.]],dtype=torch.float64),thresholds=torch.zeros(1,dtype=torch.float64),
                weighted_matmul=True,total_seconds=30)
        self.assertEqual(count,3);self.assertEqual(result['status'],'UNKNOWN')
        self.assertEqual([r['status'] for r in result['records']],['UNKNOWN','EXCLUDED','EXCLUDED'])

    def test_sparse_exception_reason_retained_not_box_substituted(self):
        import act.back_end.hybridz_tf.tf_mlp as mlp
        for exc in (MemoryError('fixed'), ValueError('fixed')):
            with self.subTest(exception=type(exc).__name__):
                tf=HybridzTF(config=HybridZConfig(max_input_dim=1))
                layer=SimpleNamespace(id=1,kind='RELU')
                tf._net=SimpleNamespace(by_id={1:layer})
                bounds=Bounds(torch.tensor([[-1.]]),torch.tensor([[1.]]))
                tf._sparse_hz_cache[1]=solver.sparse_hz_from_bounds(bounds,frame_id=0)
                with patch.object(mlp,'sparse_hz_apply_layer',side_effect=exc):
                    tf._propagate_sparse_hz(layer,bounds,Fact(bounds=bounds,cons=ConSet()))
                self.assertIsNone(tf.get_sparse_hz(1))
                row=tf.sparse_drop_diagnostics()[0]
                self.assertEqual(row,dict(layer=1,kind='RELU',reason='sparse_op_failed:RELU:'+type(exc).__name__))
                with patch('act.back_end.moe.route_a.get_assert_layer',return_value=SimpleNamespace(id=2)):
                    with self.assertRaises(MissingComponentHZ) as caught:
                        _component_output_hz(SimpleNamespace(preds={2:[1]}),tf)
                self.assertEqual(caught.exception.diagnostics,[row])


if __name__=='__main__':unittest.main()
