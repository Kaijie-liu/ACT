"""Post-run accounting regression/mutation checks; no LP/native queries."""
from copy import deepcopy
import io
from pathlib import Path
import time
import unittest
from single_check_portable.execution import ROOT,read,save_new
from portable_proof.runtime import digest
from scripts.review_modular_diagnostic_results import C,ARCHIVE,derive


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved=read(ARCHIVE)
        old={r['job_id']:r for r in read(C.PRIOR)['rows']}
        cls.records=[]
        for row in cls.saved['rows']:
            root=C.OUTPUT/row['job_id']
            cls.records.append((row,read(root/'construction.json'),
                [read(p) for p in sorted((root/'arithmetic_events').glob('*.json'))],
                old[row['job_id']]['arithmetic_detail']['construction']))

    def test_all_four_saved_records_derive_without_new_candidate(self):
        rows=[derive(*r) for r in self.records]
        self.assertEqual([r['job_id'] for r in rows],C.POLICY['job_ids'])
        self.assertEqual([r['primes_merged'] for r in rows],[80,77,49,49])
        self.assertTrue(all(r['checked_feasible_upper_bounds']==0 for r in rows))

    def test_repeated_phase_costs_are_summed_not_last_value(self):
        for record in self.records:
            result=derive(*record);c=record[1]
            parts=[s['seconds'] for s in c['journal']['segments'] if s['phase']=='CRT']
            self.assertGreater(len(parts),1)
            self.assertEqual(result['arithmetic_phase_windows']['CRT'],len(parts))
            self.assertAlmostEqual(result['arithmetic_phase_seconds']['CRT'],sum(parts))
            self.assertGreater(result['arithmetic_phase_seconds']['CRT'],parts[-1])

    def test_unsupported_progress_cost_and_point_mutations_reject(self):
        for mode in ('cap','point','check','round','modulus','residual_phase','unreached_zero','sum'):
            row,c,events,old=deepcopy(self.records[0])
            if mode=='cap':c['arithmetic']['first_limit']['cap']+=1
            elif mode=='point':c['bundle']={}
            elif mode=='check':row['complete_independent_check']=True
            elif mode=='round':c['stats']['rounds'][0]['status']='EXACT_SYSTEM_RESIDUAL_ZERO'
            elif mode=='modulus':c['stats']['max_modulus_bits']+=1
            elif mode=='residual_phase':events[0]['phase']='exact_residual'
            elif mode=='unreached_zero':row['costs']['phases']['check']['seconds']=0.
            else:c['journal']['observed_phase_seconds']+=1
            with self.assertRaises(ValueError,msg=mode):derive(row,c,events,old)

    def test_counterexample_and_network_verdict_not_inferred(self):
        for record in self.records:
            r=derive(*record)
            self.assertFalse(r['network_SAFE'] or r['network_UNSAFE'])
            self.assertEqual(r['complete_original_equation_checks'],0)
            self.assertNotIn('upper_bound',r)


if __name__=='__main__':
    log=io.StringIO();start=time.monotonic()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    n=1
    while (ROOT/f'docs/modular_execution_review_controls_attempt{n:03}.json').exists():n+=1
    dest=ROOT/f'docs/modular_execution_review_controls_attempt{n:03}.json'
    save_new(dest,{'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
        'log':log.getvalue(),'seconds':time.monotonic()-start,'archive':C.ref(ARCHIVE),
        'sources':{str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in
            (Path(__file__).resolve(),ROOT/'scripts/review_modular_diagnostic_results.py')},
        'native_or_reconstruction_calls':0,'scope':'post-run accounting/mutation controls, no altered acceptance or new solve'})
    print(log.getvalue());print(dest,flush=True)
    if not result.wasSuccessful():raise SystemExit(1)
