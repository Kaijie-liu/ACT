from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest

from range_diagnosis.analyze import classification,triangle_gap,map_outputs,values,last_relu_terms,load_reviewed,sha
from source_enclosure.format import sparse
from range_diagnosis.summarize import summarize


class LocalizationControls(unittest.TestCase):
    def test_triangle_zero_and_tie_ranges(self):
        self.assertEqual(triangle_gap(F(-2),F(1)),F(2,3))
        for a,b,kind in [(0,1,'active'),(-1,0,'inactive'),(0,0,'active'),(2,3,'active')]:
            self.assertEqual(classification(F(a),F(b)),kind);self.assertEqual(triangle_gap(F(a),F(b)),0)
        with self.assertRaises(ValueError):triangle_gap(F(1),F(-1))

    def test_triangle_convex_combinations_bound_and_attainment(self):
        lo,hi=F(-2),F(1);gap=triangle_gap(lo,hi);seen=[]
        # Convex hull of graph vertices (lo,0),(0,0),(hi,hi).
        for i in range(13):
            for j in range(13-i):
                a,b=F(i,12),F(j,12);z=a*lo+b*hi;h=b*hi
                d=h-max(z,F(0));self.assertGreaterEqual(d,0);self.assertLessEqual(d,gap);seen.append(d)
        self.assertIn(gap,seen)

    def test_signed_property_contributions_and_exact_lift_residual(self):
        pre=[F(-1),F(1,4)];post=[F(1,2),F(1,2)];coeff=[F(-2),F(3)]
        delta,signed,lift,repaired=last_relu_terms(pre,post,coeff,F(1,10),F(61,100),F(1,5))
        self.assertEqual(delta,[F(1,2),F(1,4)]);self.assertEqual(signed,[-F(1,5),F(3,20)])
        self.assertEqual(lift,F(1,500));self.assertEqual(repaired,F(17,100))
        self.assertEqual(F(1,5)*F(61,100)-lift-sum(signed),repaired)
        with self.assertRaises(ValueError):last_relu_terms(pre,post,coeff[:1],0,0,1)

    def test_factor_identity_mapping_not_raw_indices(self):
        outputs={'c':['1/3'],'Gc':sparse([{0:F(2),1:F(3)}],2),'Gb':sparse([{0:F(-1)}],1)}
        mapped=map_outputs(outputs,['shared','expert1/private'],['expert1/sign'],
            ['expert2/private','expert1/sign','shared','expert1/private'])
        self.assertEqual(values(mapped,[F(999),F(1),F(2),F(4)]),[F(46,3)])
        for c,b,g in [(['shared','shared'],['expert1/sign'],['shared','expert1/sign']),
                      (['shared','expert1/private'],['expert1/sign'],['shared','expert1/sign']),
                      (['shared','expert1/private'],['expert1/sign'],['shared','shared','expert1/private','expert1/sign'])]:
            with self.assertRaises(ValueError):map_outputs(outputs,c,b,g)

    def test_hash_bound_input_and_partial_review_reject(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);package=root/'proof';package.mkdir();data=package/'data.json';data.write_text('{}')
            m={'request':{'dataset_index':98},'pair':[1,2],'files':{'data.json':sha(data)}}
            (package/'manifest.json').write_text(json.dumps(m));review=root/'review.json'
            r={'status':'PASS','issues':[],'arms':[{'arm':'range_on','complete':True,'sealed':{'manifest_sha256':sha(package/'manifest.json')}}]}
            review.write_text(json.dumps(r));load_reviewed(package,review)
            data.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'file changed'):load_reviewed(package,review)
            r['arms'][0]['complete']=False;review.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'reviewed'):load_reviewed(package,review)

    def test_summary_retains_missing_points_and_rejects_claim_upgrade(self):
        self.summary_control('missing_point')
        self.summary_control('false_safe')

    def test_summary_rejects_incomplete_rosters(self):
        self.summary_control('missing_property');self.summary_control('missing_rank')

    def test_summary_exact_accounting_not_rounded_sign(self):
        self.summary_control('valid');self.summary_control('bad_identity')

    def summary_control(self,mode):
        nodes=[{'expert':e,'row':j,'branch':'unstable' if j==0 else 'inactive'} for e in (1,2) for j in range(64)]
        d={'schema':'LAST_RELU_SAVED_LOCALIZATION_V1','required_properties':9,
            'new_solver_calls':0,'network_forward_calls':0,'new_source_propagations':0,'new_lower_bounds_proved':0,
            'new_complete_safe':0,'exact_primal_feasibility_checked':False,'production_verdict_changed':False,
            'last_relu_rows':nodes,'last_relu_unstable':2,'total_relu_binaries':2,
            'ranked_unstable_rows':[{'expert':e,'row':0,'worst_competitor':1,'max_observed_harm':'1'} for e in (1,2)],
            'relu_census':[{'counts':{'unstable':2}}],'script_sha256':'control','review_sha256':'control',
            'package_manifest_sha256':'control','source_sha256':'control','request':{'dataset_index':98},'pair':[1,2],
            'seconds':0,'scope':'synthetic accounting control, not a network proof','properties':[]}
        for k in range(1,10):
            d['properties'].append({'competitor':k,'point_status':'UNVERIFIED_POINT_DIAGNOSTIC_ONLY',
                'checked_lower_bound':'-2','candidate_sha256':'control','decomposition_residual':'0',
                'exact_terms':{'relaxed_objective':'-2','product_gap':'3','product_only_replaced':'1',
                    'final_affine_residual':'0','weighted_last_relu_signed_gap':'-1','product_and_last_relu_replaced':'2'}})
        if mode=='missing_point':d['properties'][0]={'competitor':1,'checked_lower_bound':None,'point_status':'MISSING_UNVERIFIED_POINT'}
        if mode=='missing_property':d['properties'].pop()
        if mode=='missing_rank':d['ranked_unstable_rows'].pop()
        if mode=='false_safe':d['new_complete_safe']=1
        if mode=='bad_identity':d['properties'][0]['exact_terms']['product_and_last_relu_replaced']='20000000000000001/10000000000000000'
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            p=Path(tmp)/'analysis.json';p.write_text(json.dumps(d))
            if mode in ('valid','missing_point'):
                s=summarize(p);self.assertEqual(s['new_complete_safe'],0);self.assertEqual(s['points_present'],8 if mode=='missing_point' else 9)
            else:
                with self.assertRaises(ValueError):summarize(p)


if __name__=='__main__':unittest.main()
