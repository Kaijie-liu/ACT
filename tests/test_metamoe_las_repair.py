import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metamoe_las_repair import install, active_ancestors


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.events=[]
        x=NS(name='x',inputs=[])
        r=NS(name='r',inputs=[x],output_shape=(1,2),lA=None)
        e=NS(name='e',inputs=[x],output_shape=(1,2),lA=None)
        ro=NS(name='ro',inputs=[r],output_shape=(1,1))
        eo=NS(name='eo',inputs=[e],output_shape=(1,1))
        BoundConcat=type('BoundConcat',(),{})
        final=BoundConcat();final.name='y';final.inputs=[ro,eo];final.axis=1
        class Net:
            output_name=['y']
            def nodes(self):return [x,r,e,ro,eo,final]
            def __getitem__(self,key):return next(n for n in self.nodes() if n.name==key)
            def get_splittable_activations(self):return [r,e]
        class Solver:
            def __init__(self):
                self.net=Net();self.c=torch.tensor([[[1.,0.]]],dtype=torch.float64)
                self.returned={'r':torch.ones(3,1,2,dtype=torch.float64)}
            def get_lA(self,*args,**kwargs):return self.returned
        class Domains:
            def __init__(self,ret,lAs,net):self.net=net;self.all_lAs=lAs
            def add(self,bounds,d=None,**kwargs):
                assert len(self.all_lAs)==len(bounds['lAs'])
                self.last=bounds
                return 'native_called'
            def __len__(self):return 1
        install(Domains,Solver,lambda k,d:self.events.append((k,d)))
        self.s=Solver();self.Domains=Domains
        self.initial={'r':torch.ones(1,1,2,dtype=torch.float64),
                      'e':torch.zeros(1,1,2,dtype=torch.float64)}

    def init(self):return self.Domains({},self.initial,net=self.s)
    def get(self):return self.s.get_lA(None,3,device='cpu',transpose=True)

    def test_native_no_schema_and_no_missing_unchanged(self):
        self.assertIs(self.get(),self.s.returned)
        self.init();self.s.returned['e']=torch.zeros(3,1,2,dtype=torch.float64)
        self.assertIs(self.get(),self.s.returned)

    def test_restore_only_disconnected_zero_not_bounds(self):
        domain=self.init();before=self.s.returned
        result=self.get()
        self.assertIs(result['r'],before['r'])
        self.assertNotIn('e',before)
        self.assertEqual(result['e'].shape,(3,1,2))
        self.assertEqual(int(result['e'].count_nonzero()),0)
        lb=torch.tensor([[-.2]],dtype=torch.float64)
        bounds={'lAs':result,'lower_bounds':lb}
        self.assertEqual(domain.add(bounds),'native_called')
        self.assertIs(domain.last['lower_bounds'],lb)
        self.assertEqual(self.events[-1][0],'DOMAIN_ADD_COMPLETE')

    def test_missing_live_layer_refused(self):
        self.init();self.s.returned={'e':torch.zeros(3,1,2,dtype=torch.float64)}
        with self.assertRaisesRegex(ValueError,'active'):self.get()

    def test_tiny_nonzero_is_not_dropped(self):
        self.s.c[0,0,1]=1e-300
        self.init()
        with self.assertRaisesRegex(ValueError,'active'):self.get()

    def test_multi_spec_union(self):
        self.s.c=torch.tensor([[[1.,0.]],[[0.,1.]]],dtype=torch.float64)
        self.init()
        with self.assertRaisesRegex(ValueError,'active'):self.get()

    def test_nonzero_reference_missing_refused(self):
        self.initial['e'].fill_(1.)
        self.init()
        with self.assertRaisesRegex(ValueError,'active'):self.get()

    def test_changed_scope_refused(self):
        self.init();self.s.c[0,0,0]=2
        with self.assertRaisesRegex(ValueError,'scope'):self.get()

    def test_changed_graph_refused(self):
        self.init();self.s.net['ro'].inputs=[self.s.net['e']]
        with self.assertRaisesRegex(ValueError,'scope'):self.get()

    def test_unknown_key_refused(self):
        self.init();self.s.returned['bad']=torch.zeros(3,1,2,dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,'unexpected'):self.get()

    def test_shape_mismatch_refused(self):
        self.init();self.s.returned['r']=torch.zeros(3,1,3,dtype=torch.float64)
        with self.assertRaisesRegex(ValueError,'shape'):self.get()

    def test_existing_node_coefficient_not_overwritten(self):
        self.init();self.s.net['e'].lA=torch.ones(1)
        with self.assertRaisesRegex(ValueError,'active'):self.get()

    def test_domain_mismatch_and_unsupported_layout_refused(self):
        d=self.init()
        with self.assertRaisesRegex(ValueError,'schema'):d.add({'lAs':{}})
        with self.assertRaisesRegex(ValueError,'restoration'):self.s.get_lA(None,3,device='cpu',transpose=False)

    def test_independent_event_audit_and_mutations(self):
        import copy
        from audit_metamoe_las_paired import check_events
        d=self.init();values=self.get();d.add({'lAs':values})
        records=[{'event':'INSTALLED','repair':'disconnected_zero_branching_metadata_v1','bounds_unchanged':True}]
        records += [{'event':k,**v} for k,v in self.events]
        for i,r in enumerate(records):r.update(seq=i,monotonic=float(i),request_config_sha256='h')
        self.assertEqual(check_events(records,'h')['restorations'],1)
        self.assertEqual(check_events(records,'h')['domain_add_completions'],1)
        for field in ('active','shape','zero','seq'):
            bad=copy.deepcopy(records)
            if field=='active':bad[1]['C'][0][0][1]=1e-300
            elif field=='shape':bad[2]['shapes']['e'][0]=4
            elif field=='zero':bad[1]['initial_zero_keys']=[]
            else:bad[-1]['seq']=0
            with self.assertRaises(ValueError):check_events(bad,'h')


if __name__=='__main__':unittest.main()
