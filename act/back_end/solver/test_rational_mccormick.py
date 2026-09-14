import copy
import unittest
from fractions import Fraction
from act.back_end.solver.rational_mccormick import build, csr
from act.back_end.solver.check_rational_mccormick import check_construction
from act.back_end.solver.lp_certificate import identity, propose


def source():
    return {'c':[2.,1.], 'Gc':csr([{0:1},{0:1}],1), 'Gb':csr([{},{}],0),
            'Ac':csr([],1),'Ab':csr([],0),'Auc':csr([{0:1}],1),'Aub':csr([{}],0),
            'b':[],'ub':[1.], 'frame_id':7,'exact':True}


class RationalMcCormickTests(unittest.TestCase):
    def checked(self, r, gate=(0,1), difference=(1,1), certificate=None):
        return check_construction(r,certificate,source_hash=identity(source()),q=[1],offset=0,
                                  gate=gate,difference=difference)

    def test_exact_and_degenerate_rectangles(self):
        for gate in [(0,1),('1/2','1/2')]:
            r=build(source(),[1],0,gate,[1,1]);cert=propose(r['lp'])
            result=self.checked(r,gate=gate,certificate=cert)
            self.assertGreaterEqual(Fraction(result['bound']['checked_lower_bound']),0)

    def test_rectangle_planes_cover_concrete_products(self):
        src=source();src['Gc']=csr([{0:2},{0:1}],1)
        r=build(src,[1],0,[0,'1/2'],[0,2])
        from act.back_end.solver.sparse_lp_certificate import rows
        for x in [-1,Fraction(-1,2),0,1]:
            for lam in [0,Fraction(1,4),Fraction(1,2)]:
                values=[x,lam,lam*(1+x)]
                for row,rhs in zip(rows(r['lp']['A'],3),r['lp']['b']):
                    self.assertLessEqual(sum(v*values[j] for j,v in row),Fraction(rhs))

    def test_mutations_rejected_without_using_builder(self):
        r=build(source(),[1],0,[0,1],[1,1])
        mutations=[lambda x:x['lp']['A']['data'].__setitem__(-1,'1'),
                   lambda x:x['lp']['b'].__setitem__(-1,'100'),
                   lambda x:x['lp']['c'].__setitem__(0,'0'),
                   lambda x:x['lp']['lower'].__setitem__(-1,'1'),
                   lambda x:x['q'].__setitem__(0,'-1'),
                   lambda x:x['source']['c'].reverse(),
                   lambda x:x['source'].__setitem__('frame_id',8),
                   lambda x:x['gate'].__setitem__(1,'1/2'),
                   lambda x:x['difference'].__setitem__(0,'2')]
        # Last coefficient of upper plane is already +1: mutate another term.
        mutations[0]=lambda x:x['lp']['A']['data'].__setitem__(1,'-10')
        for mutate in mutations:
            changed=copy.deepcopy(r);mutate(changed)
            with self.subTest(changed=changed),self.assertRaises(ValueError):self.checked(changed)

    def test_inverted_ranges(self):
        for gate,diff in [([1,0],[0,1]),([0,1],[1,0]),([-1,1],[0,1])]:
            with self.assertRaises(ValueError):build(source(),[1],0,gate,diff)

    def test_projection_before_rounding(self):
        src=source();src['c']=[0.1,0.2]
        r=build(src,[3],'1/7',[0,1],[-1,1])
        self.assertEqual(Fraction(r['d']['constant']),3*(Fraction(0.1)-Fraction(0.2)))
        check_construction(r,source_hash=identity(src),q=[3],offset='1/7',gate=[0,1],difference=[-1,1])


if __name__=='__main__':unittest.main()
