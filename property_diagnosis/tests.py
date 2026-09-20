from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest

from property_diagnosis.analyze import (classification, triangle_gap, map_outputs, values,
    last_relu_terms, load_reviewed, sha, dual_terms, paired_terms, no_external_work)
from property_diagnosis.summarize import summarize, check_point, COMMON
from source_enclosure.format import sparse


def point(k,second=False):
    J,P,R,L,D,C=(-8,7,-5,-9,-7,-2) if second else (-10,8,-6,-11,-8,-3)
    t={'relaxed_objective':J,'product_gap':P,'final_affine_residual':0,
       'weighted_last_relu_signed_gap':R,'product_and_last_relu_replaced':J+P-R,
       'product_only_replaced':J+P,'last_relu_only_replaced':J-R,'checked_lower_bound':L,
       'dual_constant':D,'residual_box_correction':C,'point_minus_bound':J-L,
       'gate_value':F(1,2),'difference_value':4,'relaxed_product':2-P}
    return {'competitor':k,'point_status':'UNVERIFIED_POINT_DIAGNOSTIC_ONLY',
        'checked_lower_bound':str(L),'decomposition_residual':'0',
        'exact_terms':{v:str(x) for v,x in t.items()},'gate_range':['0','1'],'difference_range':['-20','20'],
        'exact_relu_groups':{'prefix_selected':'0','property_selected':str(R+3),'other':'-3'},
        'observations':[{'expert':e,'row':0} for e in (1,2)]}


def fixture():
    flags={'new_solver_calls':0,'network_forward_calls':0,'new_source_propagations':0,
           'new_lower_bounds_proved':0,'new_complete_safe':0,'exact_primal_feasibility_checked':False,
           'production_verdict_changed':False}
    groups={'prefix_selected':[[1,0],[1,1],[2,0],[2,1]],
            'property_selected':[[1,10],[1,30],[2,17],[2,31]]}
    nodes=[{'expert':e,'row':j,'branch':'unstable' if j==0 else 'inactive',
            'range':['-1','1'] if j==0 else ['-2','-1'],
            'group':next((n for n,v in groups.items() if [e,j] in v),'other')}
           for e in (1,2) for j in range(64)]
    arms=[]
    for second,name in enumerate(('prefix','property')):
        arms.append({**flags,'arm':name,'diagnostic_properties':COMMON,'required_output_properties':9,
            'properties':[point(k,second) for k in COMMON],'last_relu_rows':deepcopy(nodes),
            'last_relu_unstable':2,'total_relu_binaries':2,'relu_census':[{'counts':{'unstable':2}}],
            'endpoint_ledger':[{'competitor':k,'status':'MISSING' if k==7 and second else 'NONPOSITIVE',
                                'paired_diagnosis':k in COMMON} for k in range(1,10)],
            'package_manifest_sha256':name,'source_sha256':name,'request':{'dataset_index':98},'pair':[1,2],'seconds':0})
    return {**flags,'schema':'PAIRED_PROPERTY_SAVED_DIAGNOSIS_V1','diagnostic_properties':COMMON,
        'required_output_properties':9,'excluded_from_point_diagnosis':[7],'exact_LP_optimality_checked':False,
        'row_groups':groups,'arms':arms,'paired':[paired_terms(a,b) for a,b in zip(arms[0]['properties'],arms[1]['properties'])],
        'script_sha256':'synthetic','review_sha256':'synthetic','seconds':0}


class SavedDiagnosisControls(unittest.TestCase):
    def summary(self,d):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as t:
            p=Path(t)/'analysis.json';p.write_text(json.dumps(d));return summarize(p)

    def test_full_eight_pairs_and_ninth_ledger(self):
        s=self.summary(fixture());self.assertEqual(len(s['paired']),8)
        self.assertEqual(s['arms'][1]['endpoint_ledger'][6]['status'],'MISSING')
        self.assertEqual(s['arms'][0]['group_counts'],{'prefix_selected':4,'property_selected':4,'other':120})

    def test_missing_point_retained_not_zero(self):
        d=fixture();d['arms'][1]['properties'][0]={'competitor':1,'point_status':'MISSING_UNVERIFIED_POINT'}
        d['paired'][0]={'competitor':1,'status':'MISSING_SAVED_POINT_NO_PAIRED_ACCOUNTING'}
        self.assertEqual(self.summary(d)['arms'][1]['points_present'],7)

    def test_roster_and_factor_omissions_rejected(self):
        for key in ('properties','last_relu_rows','endpoint_ledger'):
            with self.subTest(key=key):
                d=fixture();d['arms'][0][key].pop()
                with self.assertRaises(ValueError):self.summary(d)

    def test_seventh_point_and_wrong_groups_rejected(self):
        for mutate in (lambda d:d['arms'][1]['properties'].append(point(7)),
                       lambda d:d['row_groups']['property_selected'].__setitem__(0,[1,11])):
            d=fixture();mutate(d)
            with self.assertRaises(ValueError):self.summary(d)

    def test_exact_not_rounded_accounting(self):
        d=fixture();d['arms'][0]['properties'][0]['exact_terms']['product_and_last_relu_replaced']='40000000000000001/10000000000000000'
        with self.assertRaisesRegex(ValueError,'point accounting'):self.summary(d)

    def test_paired_and_group_tampering(self):
        for field in ('exact_delta','exact_relu_group_delta'):
            d=fixture();key=next(iter(d['paired'][0][field]));d['paired'][0][field][key]='999'
            with self.assertRaises(ValueError):self.summary(d)

    def test_point_does_not_authorize_proof(self):
        for k in ('new_complete_safe','exact_primal_feasibility_checked','exact_LP_optimality_checked','production_verdict_changed'):
            d=fixture();d[k]=1
            with self.assertRaises(ValueError):self.summary(d)

    def test_dual_box_correction_not_all_roundoff(self):
        b={'checked_lower_bound':'-11','dual_constant':'-8','residual_box_correction':'-3',
           'residual_l1':'7/3','nonzero_residual_coordinates':3}
        t=dual_terms(b,F(-10));self.assertEqual(t['point_minus_bound'],1)
        self.assertEqual(t['residual_box_correction'],-3)
        b['dual_constant']='-7'
        with self.assertRaises(ValueError):dual_terms(b,F(-10))

    def test_wrong_competitor_binding(self):
        with self.assertRaises(ValueError):paired_terms(point(1),point(2,True))
        r=point(1);r['exact_terms']['relaxed_product']='0'
        with self.assertRaisesRegex(ValueError,'product accounting'):check_point(r)

    def test_triangle_degeneracy_and_signed_lift(self):
        self.assertEqual(triangle_gap(F(-2),F(1)),F(2,3))
        self.assertEqual(classification(F(0),F(0)),'active')
        self.assertEqual(triangle_gap(F(-1),F(0)),0)
        with self.assertRaises(ValueError):triangle_gap(F(1),F(0))
        _,signed,lift,repaired=last_relu_terms([F(-1),F(1,4)],[F(1,2),F(1,2)],
            [F(-2),F(3)],F(1,10),F(61,100),F(1,5))
        self.assertEqual(lift,F(1,500));self.assertEqual(repaired,F(17,100))
        self.assertEqual(signed,[-F(1,5),F(3,20)])

    def test_own_factor_ids_not_cross_arm_indices(self):
        output={'c':['1/3'],'Gc':sparse([{0:F(2),1:F(3)}],2),'Gb':sparse([{0:F(-1)}],1)}
        p=map_outputs(output,['shared','private'],['sign'],['unused','sign','shared','private'])
        self.assertEqual(values(p,[F(999),F(1),F(2),F(4)]),[F(46,3)])
        for ids in (['shared','sign'],['shared','sign','private','shared']):
            with self.assertRaises(ValueError):map_outputs(output,['shared','private'],['sign'],ids)

    def test_manifest_review_and_saved_file_binding(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as t:
            root=Path(t);package=root/'proof';package.mkdir();data=package/'data.json';data.write_text('{}')
            m={'request':{'dataset_index':98},'pair':[1,2],'files':{'data.json':sha(data)}}
            (package/'manifest.json').write_text(json.dumps(m));review=root/'review.json'
            r={'status':'PASS','issues':[],'arms':[{'arm':'property','complete':True,
                'sealed':{'manifest_sha256':sha(package/'manifest.json')}}]}
            review.write_text(json.dumps(r));h=sha(review)
            load_reviewed(package,review,'property',h)
            with self.assertRaises(ValueError):load_reviewed(package,review,'prefix',h)
            with self.assertRaisesRegex(ValueError,'review identity'):load_reviewed(package,review,'property','wrong')
            data.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'file changed'):load_reviewed(package,review,'property',h)

    def test_no_solver_or_external_work_hook(self):
        for event,args in [('subprocess.Popen',()),('socket.connect',()),('os.system',())]:
            with self.assertRaises(PermissionError):no_external_work(event,args)
        for name in ('numpy','scipy.optimize','torch','highspy','gurobipy'):
            with self.assertRaises(ImportError):no_external_work('import',(name,))
        no_external_work('import',('fractions',))


if __name__=='__main__':unittest.main()
