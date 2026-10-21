"""Frozen synthetic alignment controls; no trained data, native solve or GPU."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.solver import solver_hz as hz
from act.back_end.solver.lp_first import solve,LPFirstDeadline
from act.back_end.moe import monolithic_f0 as mono
from act.pipeline.moe import pgd_first as pgd,staged_verifier as staged
from act.pipeline.moe.request_budget import RequestBudget,BudgetExhausted
from act.back_end.moe import GateKind


def result(value,status=0,success=True,dual=None):
    return NS(fun=value,status=status,success=success,mip_dual_bound=dual,
              x=np.array([.5]),mip_gap=0.,mip_node_count=0)


class Clock:
    def __init__(self):self.now=0.
    def __call__(self):return self.now


class Toy(torch.nn.Module):
    def __init__(self,safe=False):
        super().__init__();self.anchor=torch.nn.Parameter(torch.tensor([.6],dtype=torch.float64))
        self.safe=safe;self.spec=NS(gate=GateKind.SELECTED_SOFTMAX,top_k=2)
    def forward(self,x):
        a=(x*0+1 if self.safe else self.anchor-x)
        return torch.cat((a,-a),dim=1)
    def forward_with_routing(self,x):
        ids=torch.tensor([[0,1] if float(x[0,0])<.5 else [1,2]])
        return self(x),NS(indices=ids)


class AlignmentControls(unittest.TestCase):
    def fake_solve(self,values,offset=0.,tolerance=1e-7,clock=None,deadline=10.):
        clock=clock or Clock();calls=[];attempts=[]
        objective=np.array([1.]);flags=np.array([1]);bounds=object();constraints=object()
        def native(**kw):
            calls.append(kw);clock.now+=1
            r=values[len(calls)-1]
            if isinstance(r,BaseException):raise r
            return r
        r,actual=solve(solver=native,objective=objective,integrality=flags,bounds=bounds,
            constraints=constraints,offset=offset,tolerance=tolerance,lp_seconds=2.,deadline=deadline,
            attempts=attempts,clock=clock)
        for c in calls:
            self.assertIs(c['c'],objective);self.assertIs(c['bounds'],bounds);self.assertIs(c['constraints'],constraints)
        np.testing.assert_array_equal(flags,[1]);self.assertIsNot(calls[0]['integrality'],flags)
        return r,actual,calls,attempts

    def test_lp_positive_same_matrix_no_milp(self):
        r,flags,calls,trace=self.fake_solve([result(1.)])
        self.assertEqual(len(calls),1);self.assertEqual(flags.tolist(),[0]);self.assertTrue(trace[0]['accepted_positive'])

    def test_nonpositive_nonoptimal_nonfinite_and_offset_fallback(self):
        for first,offset,tol in ((result(-1.),0.,1e-7),(result(1.,1,False),0.,1e-7),
            (result(float('nan')),0.,1e-7),(result(1.),-2.,1e-7),(result(1e-7),0.,1e-7),
            (result(1.),0.,2.),(ValueError('native failure'),0.,1e-7)):
            r,flags,calls,trace=self.fake_solve([first,result(3.,dual=3.)],offset,tol)
            self.assertEqual(len(calls),2);self.assertEqual(flags.tolist(),[1])
            self.assertEqual(calls[1]['options']['time_limit'],9.)
            self.assertFalse(trace[0]['accepted_positive'])

    def test_deadline_no_late_positive_or_fresh_fallback(self):
        with self.assertRaises(LPFirstDeadline):self.fake_solve([result(1.)],deadline=.5)
        with self.assertRaises(LPFirstDeadline):self.fake_solve([result(-1.)],deadline=1.)
        with self.assertRaises(TimeoutError):self.fake_solve([TimeoutError('outer deadline')])

    def lowered(self):
        return hz._HZMILP(np.array([1.]),sp.csr_matrix([[1.]]),sp.csr_matrix((0,1)),
            np.array([]),np.array([]),np.array([0.]),np.array([1.]),np.array([1]),0,1)

    def test_hz_default_off_and_lp_candidate_not_recovered(self):
        model=self.lowered()
        with patch.object(hz,'_lower_hz_milp',return_value=model),patch.object(hz,'milp',return_value=result(.5,dual=.5)) as native:
            off=hz.hz_minimize_output(object(),0,input_hz=None,input_shape=None,time_limit=10.)
            self.assertEqual(native.call_count,1);self.assertIs(native.call_args.kwargs['integrality'],model.integrality)
            self.assertEqual(off.native_solve_attempts,())
            on=hz.hz_minimize_output(object(),0,input_hz=None,input_shape=None,time_limit=10.,lp_first_seconds=2.)
            self.assertEqual(on.solver_bound_kind,'lp_status0_optimum')
            self.assertIsNone(on.candidate_input);self.assertIsNone(on.candidate_objective)
            self.assertGreater(on.minimum,1.4);self.assertEqual(len(on.native_solve_attempts),1)

    def test_local_deadline_keeps_solver_limit_and_no_new_call(self):
        with patch.object(hz,'_lower_hz_milp',return_value=self.lowered()),patch.object(hz,'milp') as native:
            d=hz.hz_minimize_output(object(),0,input_hz=None,input_shape=None,time_limit=0.,lp_first_seconds=2.)
        self.assertEqual(d.status,'timeout');self.assertEqual(d.solver_status,1);native.assert_not_called()
        model=mono._MonolithicMILP(np.array([1.]),np.array([1]),np.array([0.]),np.array([1.]),
            sp.csr_matrix((0,1)),np.array([]),np.array([]),(0,),0,())
        with patch.object(mono,'_build_disjunction',return_value=model),patch.object(mono,'milp') as native:
            d=mono.solve_monolithic_weighted_top2_f0([object()],input_shape=(1,1),time_limit=0.,lp_first_seconds=2.)
        self.assertEqual(d.reason,mono.UNKNOWN_MONOLITHIC_SOLVER_LIMIT);native.assert_not_called()

    def test_monolithic_fractional_selector_is_not_witness(self):
        model=mono._MonolithicMILP(np.array([1.]),np.array([1]),np.array([0.]),np.array([1.]),
            sp.csr_matrix((0,1)),np.array([]),np.array([]),(0,),0,())
        with patch.object(mono,'_build_disjunction',return_value=model),patch.object(mono,'milp',return_value=result(.5)):
            d=mono.solve_monolithic_weighted_top2_f0([object()],input_shape=(1,1),time_limit=10.,lp_first_seconds=2.)
        self.assertEqual(d.status,'SAFE');self.assertIsNone(d.active_pair);self.assertIsNone(d.candidate_input)
        self.assertEqual(d.solver_bound_kind,'lp_status0_optimum')

    def test_lp_label_and_diagnostic_exception_preserved(self):
        from act.pipeline.moe.request_diagnostics import DeferredCapture,compact_observation
        owner=NS(ledger=NS(tick=lambda:None));holder=DeferredCapture(owner,{},[],[])
        with patch.object(hz,'_lower_hz_milp',return_value=self.lowered()),patch.object(hz,'milp',return_value=result(.5)):
            d=hz.hz_minimize_output(object(),0,input_hz=None,input_shape=None,time_limit=10.,lp_first_seconds=2.,diagnostic_capture=holder)
        record=compact_observation(holder,d,None,owner.ledger)
        self.assertEqual(record['actual_integrality'],'ALL_CONTINUOUS')
        self.assertEqual(record['point_missing'],'CONTINUOUS_FACTOR_POINT_NOT_RECOVERED')
        broken=NS(lp_relaxation=lambda **kw:(_ for _ in ()).throw(ValueError('identity')))
        with patch.object(hz,'_lower_hz_milp',return_value=self.lowered()),patch.object(hz,'milp',return_value=result(.5)):
            with self.assertRaisesRegex(ValueError,'identity'):
                hz.hz_minimize_output(object(),0,input_hz=None,input_shape=None,time_limit=10.,lp_first_seconds=2.,diagnostic_capture=broken)

    def setup_attack(self,safe=False):
        m=Toy(safe).eval();x=torch.tensor([[.5]],dtype=torch.float64)
        return m,x,x-.3,x+.3,dict(enabled=True,recipe=pgd.RECIPE,seed=20261021,seconds=5.)

    def test_pgd_dynamic_witness_and_no_parameter_or_rng_change(self):
        m,x,l,u,cfg=self.setup_attack();state=torch.random.get_rng_state().clone();weight=m.anchor.detach().clone()
        r,w=pgd.attack(m,x,l,u,label=0,epsilon=.3,cfg=cfg,budget=RequestBudget(300))
        self.assertEqual(r['status'],'UNSAFE_REPLAYED');self.assertTrue(((w>=l[0])&(w<=u[0])).all())
        self.assertTrue(torch.equal(state,torch.random.get_rng_state()));self.assertTrue(torch.equal(weight,m.anchor))
        self.assertIsNone(m.anchor.grad);self.assertFalse(m.training)

    def test_route_flip_alone_not_unsafe_and_all_fixed_steps(self):
        m,x,l,u,cfg=self.setup_attack(True)
        r,w=pgd.attack(m,x,l,u,label=0,epsilon=.3,cfg=cfg,budget=RequestBudget(300))
        self.assertEqual(r['status'],'NO_COUNTEREXAMPLE_FOUND');self.assertIsNone(w)
        self.assertEqual((r['completed_steps'],r['started_restarts']),(60,3))

    def test_strict_domain_and_nonfinite_rejection(self):
        m,x,l,u,cfg=self.setup_attack();b=RequestBudget(300)
        for point in (u[0]+1e-9,torch.tensor([float('nan')]),torch.tensor([[.7]])):
            self.assertFalse(pgd.strict_replay(m,point,lower=l,upper=u,label=0,budget=b)['valid'])
        with patch.object(m,'forward',return_value=torch.tensor([[float('nan'),0.]])):
            self.assertFalse(pgd.strict_replay(m,torch.tensor([.7]),lower=l,upper=u,label=0,budget=b)['valid'])

    def test_deadline_and_gradient_error_never_unsafe(self):
        m,x,l,u,cfg=self.setup_attack();clock=Clock();b=RequestBudget(300,clock=clock);clock.now=300
        r,w=pgd.attack(m,x,l,u,label=0,epsilon=.3,cfg=cfg,budget=b)
        self.assertEqual(r['status'],'REQUEST_BUDGET_EXHAUSTED');self.assertIsNone(w)
        with patch.object(m,'forward',side_effect=RuntimeError('no gradient')):
            r,w=pgd.attack(m,x,l,u,label=0,epsilon=.3,cfg=cfg,budget=RequestBudget(300))
        self.assertEqual(r['status'],'CANDIDATE_GENERATION_ERROR');self.assertIsNone(w)

    def test_replay_deadline_and_attack_subbudget(self):
        m,x,l,u,cfg=self.setup_attack();clock=Clock();b=RequestBudget(300,clock=clock)
        original=m.forward_with_routing
        def late(point):
            out=original(point);clock.now=300.;return out
        with patch.object(m,'forward_with_routing',side_effect=late):
            with self.assertRaises(BudgetExhausted):
                pgd.strict_replay(m,torch.tensor([.7],dtype=torch.float64),lower=l,upper=u,label=0,budget=b)
        def ticking():clock.now+=1.;return clock.now
        clock.now=0.;b=RequestBudget(300,clock=ticking);cfg['seconds']=.5
        r,w=pgd.attack(m,x,l,u,label=0,epsilon=.3,cfg=cfg,budget=b)
        self.assertEqual(r['status'],'PGD_SUBBUDGET_EXHAUSTED');self.assertEqual(r['completed_steps'],0);self.assertIsNone(w)

    def config(self):
        cfg=json.loads(staged.DEFAULT_CONFIG.read_text())
        cfg.update(scoped_proof_reuse=True,route_complexity_schedule=dict(version=1,total_seconds=300,multi_pair_tier1_fraction=.25))
        return cfg

    def test_staged_pgd_terminal_skips_propagation_and_has_no_coverage(self):
        m,x,l,u,attack=self.setup_attack();cfg=self.config();cfg['pgd_first']=attack
        with patch('act.pipeline.moe.route_complexity_schedule.prepare',side_effect=AssertionError('must not propagate')):
            d=staged.verify_staged_linf(m,x,.3,cfg,expected_clean_prediction=0)
        self.assertEqual(d.status,'UNSAFE');self.assertFalse(d.evidence['route_coverage']['coverage_complete'])
        self.assertFalse(d.evidence['verdict']['certificate_complete']);self.assertFalse(d.evidence['tier1']['invoked'])
        self.assertEqual(d.evidence['verdict']['decision_tier'],'PGD_FIRST_FULL_FORWARD')

    def test_config_and_default_off_preserve_old_path(self):
        cfg=self.config();self.assertIsNone(pgd.validate(cfg))
        for value in (-1.,float('nan'),True):
            bad=copy.deepcopy(cfg);bad['f0']['solver']['lp_first_seconds']=value
            with self.assertRaises(ValueError):staged._validate_config(bad)
        m,x,l,u,_=self.setup_attack(True)
        tier=dict(status='UNKNOWN',reason='TEST_STOP',branches=[],full_model_witness_valid=False,schedule={})
        with (patch('act.pipeline.moe.route_complexity_schedule.prepare',return_value=(tier,None,None,None)) as prepare,
              patch.object(pgd,'attack',side_effect=AssertionError('default must not attack'))):
            d=staged.verify_staged_linf(m,x,.3,cfg)
        self.assertEqual(d.status,'UNKNOWN');self.assertEqual(prepare.call_count,1);self.assertNotIn('pgd_first',d.evidence)

    def test_enabled_no_hit_uses_original_path_and_null_is_off(self):
        m,x,l,u,attack=self.setup_attack(True);cfg=self.config()
        tier=dict(status='UNKNOWN',reason='TEST_STOP',branches=[],full_model_witness_valid=False,schedule={})
        for setting in (None,attack):
            cfg['pgd_first']=setting
            with patch('act.pipeline.moe.route_complexity_schedule.prepare',return_value=(tier,None,None,None)) as prepare:
                d=staged.verify_staged_linf(m,x,.3,cfg)
            self.assertEqual(d.status,'UNKNOWN');self.assertEqual(prepare.call_count,1)
            self.assertEqual('pgd_first' in d.evidence,setting is not None)

    def test_final_reception_cannot_return_late_witness(self):
        m,x,l,u,attack=self.setup_attack();cfg=self.config();cfg['pgd_first']=attack
        clock=Clock();b=RequestBudget(300,clock=clock)
        def progress(value):
            if value['active_stage']=='VERDICT_COMPLETE':clock.now=300.
        with (patch.object(staged,'RequestBudget',return_value=b),
              patch('act.pipeline.moe.route_complexity_schedule.prepare',side_effect=AssertionError('no propagation'))):
            with self.assertRaises(BudgetExhausted):staged.verify_staged_linf(m,x,.3,cfg,progress_callback=progress)


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
