import sys
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from metamoe_las_observe import install


class ObservationTests(unittest.TestCase):
    def test_preserves_original_values_and_assertion(self):
        events=[]
        class Tensor:
            shape=(1,2);dtype='float64'
            def count_nonzero(self): return 1
            def detach(self): return self
            def cpu(self): return self
            def tolist(self): return [[1.,0.]]
        value=Tensor()
        node=NS(name='a',used=False,perturbed=True,lA=None,output_shape=(1,2))
        graph=NS(get_splittable_activations=lambda:[node])
        class Solver:
            net=graph;c=value
            def get_lA(self,*args,**kwargs): return {'a':value}
        class Domains:
            def __init__(self,ret,lAs): self.all_lAs=lAs;self.net=Solver()
            def add(self,bounds):
                assert len(self.all_lAs)==len(bounds['lAs'])
                return bounds
        install(Domains,Solver,lambda kind,data:events.append((kind,data)))
        d=Domains({}, {'a':value})
        out={'lAs':{'a':value}}
        self.assertIs(d.add(out),out)
        self.assertIs(d.net.get_lA()['a'],value)
        with self.assertRaises(AssertionError): d.add({'lAs':{}})
        self.assertEqual(events[-1][1]['stored_keys'],['a'])
        self.assertEqual(events[-1][1]['returned_lAs'],{})
        self.assertIsNone(node.lA)


if __name__=='__main__':unittest.main()
