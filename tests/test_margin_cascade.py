"""Whole-history margin phases, exact coverage and minus-M-A semantics."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
import torch
import gate_cascade as C
import margin_cascade as M
from act.back_end.moe import gate_decoupling as G
from test_gate_cascade import tensors, outcome
from test_gate_decoupling import fixture
from test_moe_margin_branch import domain


class MarginPhase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1); torch.set_default_dtype(torch.float64)

    def records(self, decouple):
        d = domain(2); widths = [hi-lo for lo, hi in d.intervals]
        p = dict(source_id=d.source_id, parent_domain_sha256=G.domain_digest(d),
            heuristic_only=True, certified_error_bound=False, widths=widths,
            difference_absolute_amplitudes=[1., 2.], layer_lipschitz=[1., 1.], property_head_norm=1.,
            scores=[widths[0], 2*widths[1]], selected_layer=1, tie_rule='lowest layer index')
        cfg = dict(pieces=2, decouple=decouple, guards=False)
        rows = []
        for i, cell in enumerate(M.D.cells_for(d, 'MC_2', layer=1)):
            sd = M.scoped(d, cell, 1, decouple)
            for v in (('V2_LO', 'V2_HI') if decouple else ('COUPLED',)):
                rows.append(dict(cell_index=i, layer=1, cell=cell, variant=v, source_id=sd.source_id,
                    domain_sha256=G.domain_digest(sd), parent_domain_sha256=G.domain_digest(d),
                    initial_lower=[1., 2.], guard_support=None, outcome=None))
        return d, cfg, p, rows

    def test_all_cells_properties_and_component_binding(self):
        for enabled in (True, False):
            d, cfg, p, rows = self.records(enabled); end=time.monotonic()+5
            done = M.aggregate(d, cfg, p, rows, 2, [3, 3], end)
            self.assertTrue(done['closed']); self.assertEqual(done['selected_layer'], 1)
            self.assertEqual(done['expected_nodes'], 4 if enabled else 2)
            self.assertFalse(M.aggregate(d, cfg, p, rows[:-1], 2, [3, 3], end)['closed'])
            bad=copy.deepcopy(rows); bad[0]['initial_lower'][1]=0.
            self.assertFalse(M.aggregate(d, cfg, p, bad, 2, [3, 3], end)['closed'])
            for field, value in [('source_id', 'c'*64), ('layer', 0), ('variant', 'foreign')]:
                bad=copy.deepcopy(rows); bad[0][field]=value
                with self.assertRaises(ValueError): M.aggregate(d, cfg, p, bad, 2, [3, 3], end)
            with self.assertRaises(ValueError):
                M.aggregate(d, dict(cfg, decouple=not enabled), p, rows, 2, [3, 3], end)
            with self.assertRaises(TimeoutError): M.aggregate(d, cfg, p, rows, 2, [3, 3], time.monotonic()-1)
        self.assertNotEqual(C.identity({}, [], [], 'a'*64, dict(pieces=2, decouple=True, guards=False)),
            C.identity({}, [], [], 'a'*64, dict(pieces=2, decouple=False, guards=False)))

    def test_partial_phase_falls_back_without_resetting_deadline(self):
        model, rows, th=fixture(2, classes=3); data=tensors(rows, th)
        cfg=dict(pieces=2, decouple=False, guards=True); end=time.monotonic()+5; calls=[]
        def phase(*args, **kwargs):
            self.assertEqual(kwargs['margin_branch'], cfg); self.assertEqual(args[6], end)
            return dict(status='TIMEOUT', accepted=False, evidence=None, source_id='a'*64, seconds=.1)
        def coupled(*args):
            self.assertEqual(args[-1], end); calls.append(1)
            return dict(outcome(2), status='unknown'), None
        callback=C.GateCascade(model, data, dict(model={},sample={},selection_root='synthetic'),
            ROOT.parent/'results','b'*64,lambda:C.W.CAP,phase_runner=phase,coupled=coupled,margin_branch=cfg)
        value,_=callback(((0,1),(0,1)),[],end)
        self.assertEqual(value['status'],'unknown'); self.assertEqual(calls,[1])
        self.assertTrue(value['cascade']['used_fallback'])
        self.assertEqual(value['cascade']['margin_branch'],cfg)

    def test_cell_guard_support_stays_conditional_and_partial_residual_is_unknown(self):
        from test_moe_aware_frontend import stub_support
        model,rows,th=fixture(2,classes=3);data=tensors(rows,th);d=domain(2)
        def support(graph,*args,**kwargs):
            result=stub_support(graph,*args,**kwargs)
            result['cells']=[dict(layer=c.layer,lower=c.lower,upper=c.upper) for c in graph.cells]
            return result
        def forbidden(*args):self.fail('checked positive conditional bounds need no extra solve')
        end=time.monotonic()+8
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            folder=Path(tmp);journal=C.W.Journal(folder,end,C.W.CAP)
            config=dict(pieces=2,decouple=True,guards=True)
            value=M.solve(model,data,d,config,end,journal,lambda *a:[],bound=forbidden,verify=forbidden,support=support)
            self.assertTrue(value['closed']);self.assertEqual(value['operation_counts']['guard_support_calls'],4)
            self.assertTrue(all(r['initial_lower'] is None for r in value['endpoints']))
            request=dict(margin_branch=config,properties=2,source_request=dict(model=dict(architecture=dict(experts=3))))
            # Actual transfer uses JSON, as does the ordinary phase receiver.
            value=json.loads(json.dumps(value));M.receive(folder,request,value,d,end)
            bad=copy.deepcopy(value['endpoints']);bad[0]['guard_support']['cells'][0]['layer']=9
            with self.assertRaises(ValueError):M.aggregate(d,config,value['priority'],bad,2,[3,3],end)
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            config=dict(pieces=2,decouple=False,guards=False)
            value=M.solve(model,data,d,config,end,C.W.Journal(Path(tmp),end,C.W.CAP),lambda *a:[],
                bound=lambda *a:[-1.,1.],verify=lambda *a:(dict(outcome(2),status='unknown'),None))
            self.assertFalse(value['closed']);self.assertEqual(value['operation_counts']['residual_bab_calls'],2)
            self.assertEqual(value['variants'],['COUPLED'])

    @unittest.skipUnless(os.environ.get('MC_NATIVE')=='1','isolated fixed native whole requests')
    def test_native_complete_margin_phases_and_minus_ma(self):
        from act.pipeline.moe.multilayer_training import build
        from multilayer_stage1 import fingerprint
        import multilayer_stage1 as stage
        from scoped_proof.io import sha,load
        cfg=dict(input_dim=1,layers=2,experts=3,classes=3,hidden_output=1,router_hidden=[],expert_hidden=[])
        model=build(cfg,0).double().eval()
        with torch.no_grad():
            for depth,layer in enumerate(model.layers):
                layer.router[-1].weight.copy_(torch.tensor([[.2],[-.2],[0.]]))
                layer.router[-1].bias.copy_(torch.tensor([2.,2.,-4.]))
                for i,expert in enumerate(layer.experts):
                    expert[-1].weight.fill_(.1 if depth==0 else 0.)
                    expert[-1].bias.fill_(.3 if depth==0 else 0.)
                    if depth==1:expert[-1].bias[0]=2.+float(i==1)
        data=tensors(torch.tensor([[1.,-1.,0.],[1.,0.,-1.]]),torch.tensor([0.,.1]))
        end=float(os.environ['MC_CONTROL_DEADLINE'])
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as tmp:
            root=Path(tmp);checkpoint=root/'synthetic.pt'
            torch.save(dict(format='act-sequential-top2-v1',architecture=cfg,seed=0,
                state_dict=model.state_dict(),synthetic_control=True),checkpoint)
            torch.save(data,root/'input.pt')
            request=dict(model=dict(id='untrained_synthetic',checkpoint=str(checkpoint),
                checkpoint_sha256=sha(checkpoint),architecture=cfg,seed=0),
                sample=dict(file='input.pt',sha256=sha(root/'input.pt'),tensors={k:fingerprint(v) for k,v in data.items()}),
                selection_root=str(root),arm='genbab_pruned',margin_branch_pieces=2)
            stage.ARTIFACT_ROOT,stage.ARTIFACT_LIMIT=root,C.W.CAP
            for decouple in (True,False):
                folder=root/('decoupled' if decouple else 'coupled');folder.mkdir()
                req=dict(request,gate_cascade=decouple)
                completed=stage.verify(folder,req,model,data,end,torch);r=completed['result']
                self.assertEqual((r['status'],r['verified_histories'],r['pruned_histories']),('POSITIVE',1,8))
                self.assertFalse(r['formal_SAFE'])
                phase=r['histories'][0]['outcome']['cascade'];self.assertFalse(phase['used_fallback'])
                child=folder/'v2_history_0000';pr=load(child/'phase_request.json');v=load(child/'outcome.json')['value']
                self.assertEqual(v['selected_layer'],1);self.assertEqual(v['expected_nodes'],4 if decouple else 2)
                d=C.scoped_domain(G,pr['source_request'],pr['history'],pr['prefix_bounds'],
                    pr['source_id'],pr['backend_identity'],end)
                bad=copy.deepcopy(v);bad['endpoints'].pop()
                with self.assertRaises(ValueError):C.receive_phase(child,pr,bad,d,end)
                print('M-C native request',json.dumps(dict(decouple=decouple,phase=phase,
                    selected_layer=v['selected_layer'],nodes=v['expected_nodes'],counts=v['operation_counts'])),flush=True)


if __name__=='__main__':unittest.main()
