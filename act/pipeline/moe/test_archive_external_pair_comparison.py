import json
from pathlib import Path
import unittest
from act.pipeline.moe.archive_external_pair_comparison import OUTPUT,describe


class ExternalArchiveTests(unittest.TestCase):
    def test_complete_denominators_and_evidence_classes(self):
        r=json.loads(OUTPUT.read_text());full=r['experiments']['full']['summary']
        self.assertEqual(full['rows'],60);self.assertEqual(full['complete_per_arm'],{'adaptive':25,'crown':30})
        self.assertEqual(full['unsafe_replayed'],9);self.assertTrue(r['exact_reaudit_equals_saved'])
        self.assertEqual(r['full_costs']['adaptive']['states'],{'SAFE':11,'UNSAFE':9,'UNKNOWN':5,'TIMEOUT':5})
        self.assertEqual(r['full_costs']['crown']['states'],{'POSITIVE':13,'UNKNOWN':17})
        for row in full['models'].values():
            self.assertEqual(row['methods']['adaptive']['evidence_level'],'HZ_POLICY_ACCEPTED')
            self.assertEqual(row['methods']['crown']['evidence_level'],'CROWN_NUMERICAL_FILTER')

    def test_both_sides_of_non_dominance_retained(self):
        r=json.loads(OUTPUT.read_text());d=r['positive_discordances']
        a=[p for p in d if p['positive_only']=='adaptive'];b=[p for p in d if p['positive_only']=='crown']
        self.assertEqual(r['positive_intersection'],8);self.assertEqual((len(a),len(b)),(3,5))
        self.assertEqual(sum(x['pair_count']>1 for x in a),2)
        self.assertTrue(all(x['crown_minimum_obligation']['lower']<0 for x in a))
        self.assertTrue(all('SOLVER_LIMIT' in x['reason'] and x['adaptive']=='UNKNOWN' for x in b))
        self.assertEqual({(x['model'],x['dataset_index']) for x in a},
                         {('seed0',4029),('seed1',4018),('seed2',4014)})

    def test_all_costs_include_timeouts(self):
        rows=[{'method':arm,'status':state,'wall_seconds':seconds,'outer_timeout':state=='TIMEOUT'}
              for arm in ('adaptive','crown') for state,seconds in [('UNKNOWN',1),('TIMEOUT',300)]]
        d=describe(rows)
        for arm in d:
            self.assertEqual(d[arm]['mean_observed_seconds'],150.5)
            self.assertEqual(d[arm]['total_observed_seconds'],301)
            self.assertEqual(d[arm]['outer_timeouts'],1)

    def test_complete_input_strata(self):
        r=json.loads(OUTPUT.read_text());s=r['route_strata']
        self.assertEqual((s['single']['model_input_pairs'],s['multiple']['model_input_pairs']),(16,14))
        self.assertEqual(s['unavailable']['model_input_pairs'],0)
        self.assertEqual(len(r['full_requests']),30)
        self.assertEqual(len({p['dataset_index'] for p in r['full_requests']}),10)


if __name__=='__main__':unittest.main()
