"""Finite S2a objective, selection and schedule controls; no CIFAR access."""
import copy
import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from act.pipeline.moe.architecture_s2_training import (
    build, schedule, objective, evaluate_ibp, selection_key, tune_items, selected_followups,
)
from act.pipeline.moe.route_envelope_ibp import margin_rows, model_bounds, certified_loss


def fixture():
    moe = dict(input_dim=4, layers=1, experts=3, hidden_output=4,
               router_hidden=[4], expert_hidden=[6, 3], classes=3)
    dense = dict(widths=[4, 6, 3, 3])
    return dict(epochs=100, warmup_epochs=5, ramp_epochs=40, epsilon=2/255,
                kappa_end=.5, seeds=list(range(5)), arms=[
        dict(id=arm, family='dense' if arm.startswith('D') else 'moe',
             architecture=dense if arm.startswith('D') else moe, parameter_count=0,
             configurations=[dict(lr=lr, route_weight=rw) for lr,rw in configs])
        for arm,configs in [('D-act',[(.001,0),(.0005,0)]),('D-tot',[(.001,0),(.0005,0)]),
                            ('M',[(.001,.1),(.001,1.)]),('M-reg',[(.001,0),(.0005,0)])]])


class S2TrainingControls(unittest.TestCase):
    def test_schedule_all_arms_fixed_and_full_radius_validation(self):
        spec=fixture()
        for epoch in range(1,6):
            self.assertEqual(schedule(epoch, 4, 5, spec), (0.,1.))
        self.assertEqual(schedule(6,0,5,spec),(0.,1.))
        self.assertEqual(schedule(26,0,5,spec),(1/255,.75))
        self.assertEqual(schedule(46,0,5,spec),(2/255,.5))
        self.assertEqual(schedule(100,4,5,spec),(2/255,.5))

    def test_24_runs_reuse_seed0_and_no_test_selection(self):
        spec=fixture();items=tune_items(spec)
        results=[dict(item=i,validation=dict(certified=10+i['configuration'],clean=30),
                      checkpoint='not_loaded',test=dict(certified=1000-999*i['configuration'])) for i in items]
        picked, followups=selected_followups(spec,results)
        self.assertEqual(len(items)+len(followups),24)
        self.assertTrue(all(r['item']['configuration']==1 for r in picked.values()))
        self.assertEqual({i['seed'] for i in followups},{1,2,3,4})
        self.assertEqual(len({i['id'] for i in items+followups}),24)
        for r in results:r['validation']['certified']=10
        tied,_=selected_followups(spec,results)
        self.assertTrue(all(r['item']['configuration']==0 for r in tied.values()))
        with self.assertRaises(ValueError): selected_followups(spec,results[:-1])

    def test_same_objective_and_balanced_warmup(self):
        x=torch.arange(16,dtype=torch.float32).reshape(4,4)/16;y=torch.tensor([0,1,2,0])
        for arm in ('D-act','M','M-reg'):
            item=next(i for i in tune_items(fixture()) if i['arm']==arm)
            model=build(item)
            for kappa,eps in [(1.,0.),(.5,2/255)]:
                loss,stats=objective(model,x,y,epsilon=eps,kappa=kappa,
                                     route_weight=item['route_weight'],family=item['family'])
                expected=kappa*stats['natural']+(1-kappa)*stats['robust']+item['route_weight']*stats['route_penalty']+.1*stats['balance']
                self.assertAlmostEqual(float(loss.detach()),expected,places=5)
                self.assertGreater(stats['balance'],0) if arm.startswith('M') else self.assertEqual(stats['balance'],0)
                if kappa==1:self.assertEqual(stats['route_penalty'],0)
                else:
                    lb=model_bounds(model,(x-eps).clamp(0,1),(x+eps).clamp(0,1),margin_rows(y,3,dtype=x.dtype))[0]
                    self.assertAlmostEqual(stats['robust'],float(certified_loss(lb).detach()),places=6)
                model.zero_grad();loss.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_validation_uses_full_box_and_clean_errors_unaccepted(self):
        item=tune_items(fixture())[0];model=build(item)
        with torch.no_grad():
            for p in model.parameters():p.zero_()
            model[-1].bias.copy_(torch.tensor([2.,0.,-1.]))
        x=torch.ones(4,4)/2;y=torch.tensor([0,0,1,2])
        result=evaluate_ibp(model,DataLoader(TensorDataset(x,y),batch_size=2),2/255)
        self.assertEqual((result['clean'],result['certified'],result['total']),(2,2,4))
        self.assertIsNone(result['single_route'])

    def test_resume_objective_optimizer_and_shuffle(self):
        item=next(i for i in tune_items(fixture()) if i['arm']=='M')
        m=build(item);opt=torch.optim.AdamW(m.parameters(),lr=item['lr'],weight_decay=.0001)
        gen=torch.Generator().manual_seed(9);x=torch.rand(8,4,generator=gen);y=torch.arange(8)%3
        def step(net,optimizer,order):
            optimizer.zero_grad();loss,_=objective(net,x[order],y[order],epsilon=2/255,kappa=.5,
                route_weight=.1,family='moe');loss.backward();optimizer.step()
        step(m,opt,torch.randperm(8,generator=gen))
        restored=copy.deepcopy(m);resumed=torch.optim.AdamW(restored.parameters(),lr=item['lr'],weight_decay=.0001)
        resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
        gen2=torch.Generator();gen2.set_state(gen.get_state())
        order=torch.randperm(8,generator=gen);self.assertTrue(torch.equal(order,torch.randperm(8,generator=gen2)))
        step(m,opt,order);step(restored,resumed,order)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.parameters(),restored.parameters())))

    def test_full_shape_synthetic_objective_and_parameter_binding(self):
        # Untrained architecture controls, not CIFAR training or test selection.
        from act.pipeline.moe.architecture_dense import parameter_count
        configs=[('dense',dict(widths=[3072,602,301,10])),
                 ('dense',dict(widths=[3072,1758,879,10])),
                 ('moe',dict(input_dim=3072,layers=1,experts=8,hidden_output=128,
                             router_hidden=[128],expert_hidden=[256,128],classes=10))]
        x=torch.linspace(0,1,2*3072).reshape(2,3,32,32);y=torch.tensor([0,9])
        for family,architecture in configs:
            model=build(dict(family=family,architecture=architecture,seed=0))
            count=sum(p.numel() for p in model.parameters())
            self.assertEqual(count,6961368 if family=='moe' else parameter_count(architecture['widths']))
            loss,_=objective(model,x,y,epsilon=2/255,kappa=.5,
                             route_weight=.1 if family=='moe' else 0.,family=family)
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
            metrics=evaluate_ibp(model,DataLoader(TensorDataset(x,y),batch_size=2),2/255)
            self.assertEqual(metrics['total'],2)
            del model,loss


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
