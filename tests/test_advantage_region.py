"""Ordinary selection/diagnostic controls; no native library or model calls."""
import importlib.util
import inspect
from pathlib import Path
import unittest
from act.pipeline.moe.external_pair_worker import diagnostic_pairs

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('advantage_region',ROOT/'scripts/advantage_region.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


class AdvantageControls(unittest.TestCase):
    def test_selection_has_fixed_groups_and_sealed_exclusion(self):
        rows=[dict(method='adaptive',status='SAFE',dataset_index=5000+i,rank=i,
                   model='seed0',job_id=str(i)) for i in range(65)]
        coverage={str(i):[[0,1],[0,2]] if i<51 else [[0,1]] for i in range(65)}
        rows.append(dict(method='adaptive',status='SAFE',dataset_index=4088,rank=-1,model='seed0',job_id='sealed'))
        selected=m.select(list(reversed(rows)),coverage)
        self.assertEqual([r['rank'] for _,r,_ in selected],list(range(60)))
        self.assertEqual([g for _,_,g in selected],['multi_pair']*51+['single_pair_control']*9)
        with self.assertRaises(ValueError):m.select(rows[1:],coverage)

    def test_unknown_pair_diagnosis_is_not_route_coverage(self):
        q=dict(id='a',historical={'group':'multi_pair','pairs':[[0,1]]},item={'identity':'same'})
        row=dict(status='UNKNOWN',accepted=True,outcome={'pairs':[
            {'pair':[0,1],'outcome':{'status':'verified'}},
            {'pair':[0,2],'outcome':{'status':'unknown'}}]})
        found=m.local_queries([q],[row]);self.assertEqual(len(found),1)
        self.assertEqual(found[0]['diagnostic_pair'],[0,2]);self.assertFalse(found[0]['in_historical_roster'])
        self.assertFalse(found[0]['full_slot']);self.assertEqual(found[0]['seconds'],300)
        for bad in ({**row,'accepted':False},{**row,'status':'POSITIVE'},
                    {**row,'outcome':{'pairs':[]}}):self.assertEqual(m.local_queries([q],[bad]),[])

    def test_decision_counts_requests_not_negative_rows(self):
        qs=[dict(parent_query=str(i),diagnostic_pair=[0,1],in_historical_roster=i!=0) for i in range(3)]
        rows=[dict(accepted=True,outcome={'pairs':[{'pair':[0,1],'lower':[-1.]*9}]}) for _ in qs]
        d=m.decision(qs,rows);self.assertEqual(d['pattern_requests'],3)
        self.assertEqual(d['next'],'GUARD_CONDITIONED_PROPAGATION_RESEARCH_CANDIDATE')
        self.assertIn('NOT_GUARD_NECESSITY',d['guarantee'])
        rows[0]['outcome']['pairs'][0]['lower']=[0.]*9
        self.assertEqual(m.decision(qs,rows)['pattern_requests'],2)
        self.assertEqual(m.decision(qs,rows)['next'],'STOP_B3_MECHANISM_SELECTION_REPORT')
        rows[1]={'accepted':False,'status':'TIMEOUT'}
        self.assertEqual(m.decision(qs,rows)['missing'],['1'])
        self.assertEqual(m.decision(qs,rows)['reason'],'INSUFFICIENT_DIAGNOSTIC_COVERAGE')
        rows[1]={'accepted':True,'status':'UNSAFE_REPLAYED','outcome':{'pairs':[]}}
        self.assertEqual(m.decision(qs,rows)['replayed_violation_conflicts'],['1'])

    def test_wrong_pair_or_missing_property_rejected(self):
        q=[dict(parent_query='x',diagnostic_pair=[0,1],in_historical_roster=True)]
        for pair,lower in [([0,2],[-1.]*9),([0,1],[-1.]*8),([0,1],[float('nan')]*9)]:
            with self.assertRaises(ValueError):m.decision(q,[dict(accepted=True,outcome={'pairs':[dict(pair=pair,lower=lower)]})])

    def test_local_scope_cannot_be_complete_coverage(self):
        q=dict(scope='SELECTED_WHOLE_BOX_PAIR_DIAGNOSIS_NOT_COMPLETE_REQUEST',diagnostic_pairs=[[0,7]])
        self.assertEqual(diagnostic_pairs(q,8),[[0,7]])
        self.assertIsNone(diagnostic_pairs({},8))
        for bad in ({**q,'scope':'ALL_REGISTERED_FEASIBLE_PAIRS'},
                    *({**q,'diagnostic_pairs':p} for p in ([],[[0,7],[1,2]],[[1,1]],[[7,0]],[[0,8]],[[False,2]]))):
            with self.assertRaises(ValueError):diagnostic_pairs(bad,8)

    def test_reuses_completed_numerical_api_without_native_smoke(self):
        self.assertIn('verify_graph',m.unchanged_numerical_worker()['functions'])

    def test_existing_supervisor_is_reused_without_native_import(self):
        self.assertIn('batch.invoke(',inspect.getsource(m.run))
        self.assertIn('time.monotonic()+300',inspect.getsource(m.run))
        self.assertIn('all_artifacts=artifacts',inspect.getsource(m.run))
        self.assertIn('source_identity()',inspect.getsource(m.run))
        self.assertIn("query.get('selection'",inspect.getsource(m.worker.genbab))
        self.assertIn("'diagnostic_pair' in query",inspect.getsource(m.worker.crown))


if __name__=='__main__':unittest.main()
