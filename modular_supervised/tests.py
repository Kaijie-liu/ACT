"""Analytic real-native path plus synthetic owned-boundary faults; no network LPs."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from single_check_portable.execution import ROOT, ACT, save_new, read
from portable_proof.runtime import digest, original_bytes
from lp_sandwich.check import identity
from lp_sandwich.tests import csr
from exact_basis.tests import case
from modular_supervised.flow import (supervise, audit, costs, review_candidate, sources,
                                    owned_stage, inventory, semantic_stop, phase_state)
from modular_supervised.batch import loop, summarize

OBSERVATIONS = []
ARTIFACT_ROOT = None


def overwrite(path, value):
    path.write_bytes(original_bytes(value))


def resign_candidate(root):
    c = read(root / 'candidate.json')
    c['artifact_sha256'] = inventory(root)
    overwrite(root / 'candidate.json', c)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ARTIFACT_ROOT
        ARTIFACT_ROOT = Path(tempfile.mkdtemp(prefix='modular_supervision_controls_', dir=ROOT / 'data/moe/results'))
        cls.base = ARTIFACT_ROOT
        lp, s, _, _ = case()
        lp['c'] = [-1, 0]
        s['lp_sha256'] = identity(lp)
        cls.lp, cls.statement = lp, s
        cls.input = cls.base / 'input.json'
        save_new(cls.input, {'lp': lp, 'statement': s})
        cls.spec = {'job_id': 'analytic', 'input': {'path': str(cls.input), 'sha256': digest(cls.input.read_bytes())},
                    'statement': s, 'statement_sha256': identity(s)}
        cls.good = cls.base / 'good'
        cls.start = time.monotonic()
        v = supervise(cls.spec, cls.good, started=cls.start)
        if v['status'] != 'CHECKED_LP_DIAGNOSTIC':
            raise AssertionError((v, (cls.good / 'driver.log').read_text(), str(cls.good)))
        OBSERVATIONS.append({'case': 'complete_analytic', 'terminal': v, 'costs': costs(cls.good),
                             'upper_bound': read(cls.good / 'check.log')['upper_bound']})

    def clone(self, name):
        path = self.base / name
        shutil.copytree(self.good, path)
        return path

    def fault_root(self, name, elapsed):
        root = self.base / name
        root.mkdir()
        plan = read(self.good / 'plan.json')
        plan['started'] = time.monotonic() - elapsed
        save_new(root / 'plan.json', plan)
        shutil.copyfile(self.good / 'prepared.json', root / 'prepared.json')
        return root, plan['started']

    def test_full_flow_original_clock_and_complete_cost(self):
        v = audit(self.good)
        c, t = review_candidate(self.good), costs(self.good)
        self.assertTrue(v['complete_independent_check'])
        self.assertEqual([r['name'] for r in c['stages']], ['load', 'capture', 'map', 'construct', 'package', 'check'])
        self.assertEqual(read(self.good / 'native/capture.json')['deadline_monotonic'], self.start + 218)
        self.assertEqual(read(self.good / 'construction.json')['deadline_monotonic'], self.start + 218)
        self.assertEqual(t['native_calls'], 1)
        self.assertGreater(t['phases']['capture']['seconds'], t['native_seconds'])
        self.assertGreater(t['phases']['construct']['seconds'], t['construction_seconds'])
        self.assertAlmostEqual(t['whole_supplied_LP_seconds'], t['observed_phase_sum_seconds'] + t['residual_seconds'])
        result = read(self.good / 'check.log')
        self.assertEqual(result['upper_bound'], '-4/3')
        self.assertEqual(result['primal_status'], 'EXACT_FEASIBLE')
        self.assertFalse(result['network_UNSAFE'])
        self.assertFalse(result['network_SAFE'])

    def test_portable_isolated_recheck(self):
        moved = self.base / 'moved'
        shutil.copytree(self.good / 'portable', moved)
        pack = read(self.good / 'packing.json')
        r = subprocess.run([ACT, '-I', '-S', str(moved / 'verify.py'), str(moved / 'bundle.json'),
                            '--bundle-sha256', pack['bundle_sha256'], '--statement-sha256', pack['statement_sha256'],
                            '--timeout-seconds', '10'], cwd=moved, capture_output=True, text=True, check=True, timeout=15)
        out = json.loads(r.stdout)
        original = read(self.good / 'check.log')
        self.assertEqual({k:v for k,v in out.items() if k != 'seconds'}, {k:v for k,v in original.items() if k != 'seconds'})
        self.assertFalse(out['solver_or_model_imported'])

    def test_redundant_equalities_supported_without_row_deletion(self):
        lp = {'matrix_format': 'csr_v1', 'c': [-1,0], 'offset': 0, 'lower': [0,0], 'upper': [1,1],
              'E': csr([[1,1],[2,2]], 2), 'h': [1,2], 'A': csr([], 2), 'b': []}
        s = {**self.statement, 'lp_sha256': identity(lp)}
        path = self.base / 'redundant_input.json'
        save_new(path, {'lp': lp, 'statement': s})
        spec = {'input': {'path': str(path), 'sha256': digest(path.read_bytes())}, 'statement': s, 'statement_sha256': identity(s)}
        root = self.base / 'redundant'
        v = supervise(spec, root, started=time.monotonic())
        self.assertEqual(v['status'], 'CHECKED_LP_DIAGNOSTIC', v)
        self.assertEqual(audit(root)['status'], 'CHECKED_LP_DIAGNOSTIC')
        self.assertTrue((root / 'native/capture.json').exists())
        self.assertTrue((root / 'construction.json').exists())
        self.assertTrue(v['complete_independent_check'])
        self.assertEqual(read(root / 'check.log')['primal_status'], 'EXACT_FEASIBLE')
        self.assertEqual(len(read(root / 'portable/bundle.json')['lp']['h']), 2)
        OBSERVATIONS.append({'case': 'redundant_E_supported', 'root': str(root), 'costs': costs(root), 'status': v['status']})

    def test_bad_input_errors_no_retry_no_native(self):
        spec = deepcopy(self.spec)
        spec['input']['sha256'] = '0' * 64
        root = self.base / 'error'
        v = supervise(spec, root, started=time.monotonic())
        self.assertEqual(v['status'], 'ERROR')
        self.assertEqual(audit(root)['status'], 'ERROR')
        self.assertIsNone(costs(root)['native_calls'])
        self.assertFalse((root / 'native').exists())
        with self.assertRaises(FileExistsError):
            supervise(spec, root, started=time.monotonic())

    def test_expired_and_reserve_deadline_not_reset(self):
        for name, elapsed in [('expired', 301), ('reserve', 219)]:
            root = self.base / name
            v = supervise(self.spec, root, started=time.monotonic() - elapsed)
            self.assertEqual(v['status'], 'TIMEOUT')
            self.assertEqual(audit(root)['status'], 'TIMEOUT')
            t = costs(root)
            self.assertIsNone(t['native_seconds'])
            self.assertTrue(all(p['seconds'] is None for p in t['phases'].values()))
            self.assertFalse((root / 'native').exists())

    def test_outer_owned_cutoff(self):
        root = self.base / 'outer_cutoff'
        popen = subprocess.Popen
        def launch(cmd, *args, **kwargs):
            return popen([ACT, '-m', 'modular_supervised.fault_worker', 'driver_stall', str(root)], *args, **kwargs)
        with patch('modular_supervised.flow.subprocess.Popen', launch):
            v = supervise(self.spec, root, started=time.monotonic() - 297)
        self.assertEqual(v['status'], 'TIMEOUT')
        self.assertTrue(v['outer_process']['killed'])
        self.assertFalse(audit(root)['complete_independent_check'])
        self.assertIsNone(costs(root)['native_seconds'])

    def test_native_boundary_cutoff_preserves_partial(self):
        root, start = self.fault_root('native_stall', 215)
        row = owned_stage(root, 'capture', [ACT, '-m', 'modular_supervised.fault_worker', 'native_stall', str(root)], start, 218)
        self.assertEqual(row['state'], 'TIMEOUT', (row, (root / 'capture.log').read_text()))
        self.assertTrue(row['process']['killed'])
        self.assertTrue((root / 'fault_entered.json').exists())
        self.assertTrue((root / 'native/submission.json').exists())
        self.assertEqual((root / 'native/capture.json').read_bytes(), b'{"interrupted":')
        self.assertFalse((root / 'construction.json').exists())
        OBSERVATIONS.append({'case': 'synthetic_native_boundary_stall', 'phase': row, 'actual_native_solves': 0})

    def test_exact_elimination_cutoff_preserves_partial(self):
        root, start = self.fault_root('construct_stall', 216)
        shutil.copyfile(self.good / 'mapping.json', root / 'mapping.json')
        row = owned_stage(root, 'construct', [ACT, '-m', 'modular_supervised.fault_worker', 'construct_stall', str(root)], start, 218)
        self.assertEqual(row['state'], 'TIMEOUT', (row, (root / 'construct.log').read_text()))
        self.assertTrue(row['process']['killed'])
        self.assertTrue((root / 'fault_entered.json').exists())
        self.assertEqual((root / 'construction.json').read_bytes(), b'{"interrupted":')
        self.assertFalse((root / 'bundle.json').exists())
        OBSERVATIONS.append({'case': 'synthetic_exact_elimination_stall', 'phase': row})

    def test_postcapture_exception_preserves_raw(self):
        root, start = self.fault_root('map_exception', 0)
        shutil.copytree(self.good / 'native', root / 'native')
        sha = digest((root / 'native/capture.json').read_bytes())
        row = owned_stage(root, 'map', [ACT, '-m', 'modular_supervised.fault_worker', 'map_exception', str(root)], start, 218)
        self.assertEqual(row['state'], 'ERROR')
        self.assertFalse(row['process']['killed'])
        self.assertEqual(digest((root / 'native/capture.json').read_bytes()), sha)
        self.assertFalse((root / 'mapping.json').exists())

    def test_full_outer_faults_terminal_and_costs(self):
        # Only the test replaces the driver's command. Production has no fault option.
        original_popen = subprocess.Popen
        for mode, phase, elapsed, expected in (
                ('native_stall', 'capture', 215, 'TIMEOUT'),
                ('construct_stall', 'construct', 215, 'TIMEOUT'),
                ('readback_stall', 'capture', 215, 'TIMEOUT'),
                ('readback_exception', 'capture', 0, 'ERROR'),
                ('map_exception', 'map', 0, 'ERROR')):
            root = self.base / ('full_' + mode)
            def launch(cmd, *args, **kwargs):
                self.assertEqual(cmd[1:3], ['-m', 'modular_supervised.flow'])
                return original_popen([ACT, '-m', 'modular_supervised.fault_worker', 'drive_' + mode, str(root)], *args, **kwargs)
            with patch('modular_supervised.flow.subprocess.Popen', launch):
                result = supervise(self.spec, root, started=time.monotonic() - elapsed)
            self.assertEqual(result['status'], expected, (result, (root / 'driver.log').read_text()))
            self.assertEqual(audit(root)['status'], expected)
            self.assertFalse(result['complete_independent_check'])
            t = costs(root)
            self.assertEqual(t['phases'][phase]['state'], expected)
            self.assertIsNone(t['phases']['package']['seconds'])
            self.assertEqual(t['native_calls'], None if mode == 'native_stall' else 1)
            self.assertAlmostEqual(t['whole_supplied_LP_seconds'], t['observed_phase_sum_seconds'] + t['residual_seconds'])
            if mode.endswith('stall'):
                self.assertTrue(t['phases'][phase]['censored'])
                if mode != 'readback_stall':
                    self.assertTrue(t['unreadable_partial_records'])
            if mode.startswith('readback'):
                self.assertFalse((root / 'native/capture.json').exists())
                self.assertEqual(t['native_cost_evidence'], 'native/raw_native.json')
                self.assertGreaterEqual(t['native_seconds'], 0)
                self.assertFalse((root / 'mapping.json').exists())
            OBSERVATIONS.append({'case': 'full_outer_' + mode, 'terminal': result, 'costs': t,
                                 'synthetic_stall_or_exception': True})

    def test_checker_deadline_partial_output_not_accepted(self):
        root, start = self.fault_root('check_stall', 297)
        row = owned_stage(root, 'check', [ACT, '-m', 'modular_supervised.fault_worker', 'check_stall', str(root)], start, 298)
        self.assertEqual(row['state'], 'TIMEOUT')
        self.assertTrue(row['process']['killed'])
        with self.assertRaises(ValueError):
            read(root / 'check.log')

    def test_missing_or_tampered_evidence_rejected(self):
        for i, name in enumerate(('native/capture.json', 'native/raw_native.json', 'construction.json', 'portable/bundle.json', 'check.log', 'map_stage.json')):
            root = self.clone('mutate_' + str(i))
            path = root / name
            path.write_bytes(path.read_bytes() + b' ')
            with self.assertRaises(ValueError):
                audit(root)
        root = self.clone('missing_check')
        (root / 'check.log').rename(root / 'removed_check.log')
        with self.assertRaises(ValueError):
            audit(root)

    def test_resigned_clock_and_checker_flags_rejected(self):
        root = self.clone('bad_clock')
        c = read(root / 'candidate.json')
        c['stages'][0]['elapsed_seconds'] = 0
        overwrite(root / 'load_stage.json', c['stages'][0])
        overwrite(root / 'candidate.json', c)
        resign_candidate(root)
        with self.assertRaises(ValueError):
            review_candidate(root)
        for name, value in [('isolated', False), ('solver_or_model_imported', True), ('network_SAFE', True)]:
            root = self.clone('bad_flag_' + name)
            out = read(root / 'check.log')
            out[name] = value
            overwrite(root / 'check.log', out)
            resign_candidate(root)
            with self.assertRaises(ValueError):
                review_candidate(root)

    def test_resigned_mapping_and_original_lp_drift_rejected(self):
        root = self.clone('bad_mapping')
        m = read(root / 'mapping.json')
        m['candidate']['x'][0] = 0
        overwrite(root / 'mapping.json', m)
        resign_candidate(root)
        with self.assertRaises(ValueError):
            review_candidate(root)

    def test_resigned_component_cost_overclaim_rejected(self):
        root = self.clone('bad_component_cost')
        r = read(root / 'native/capture.json')
        r['seconds'] = 100
        overwrite(root / 'native/capture.json', r)
        resign_candidate(root)
        with self.assertRaises(ValueError):
            review_candidate(root)
        root = self.clone('bad_lp')
        b = read(root / 'prepared.json')
        b['lp']['offset'] = 100
        overwrite(root / 'prepared.json', b)
        resign_candidate(root)
        with self.assertRaises(ValueError):
            review_candidate(root)

    def test_late_publication_revokes_complete_check(self):
        root = self.clone('late_publication')
        pub = read(root / 'publication.json')
        pub['observed_seconds'] = 301
        overwrite(root / 'publication.json', pub)
        self.assertEqual(audit(root)['status'], 'TIMEOUT')
        self.assertFalse(audit(root)['complete_independent_check'])

    def test_partial_records_censored_missing_is_null(self):
        # Synthetic interrupted-driver fixture; no guessed native count or duration.
        root = self.base / 'partial_terminal'
        supervise(self.spec, root, started=time.monotonic() - 301)
        save_new(root / 'capture_entered.json', {'phase': 'capture', 'seconds': 200})
        (root / 'capture_stage.json').write_bytes(b'{"partial":')
        (root / 'native').mkdir()
        (root / 'native/capture.json').write_bytes(b'{"partial":')
        v = read(root / 'outer.json')
        v['artifact_sha256'] = inventory(root)
        overwrite(root / 'outer.json', v)
        pub = read(root / 'publication.json')
        pub['outer_sha256'] = digest((root / 'outer.json').read_bytes())
        overwrite(root / 'publication.json', pub)
        t = costs(root)
        self.assertIsNone(t['native_seconds'])
        self.assertIsNone(t['native_calls'])
        self.assertIsNone(t['phases']['capture']['seconds'])
        self.assertTrue(t['phases']['capture']['censored'])
        self.assertGreater(t['phases']['capture']['observed_window_seconds'], 100)
        self.assertEqual(t['unreadable_partial_records'], ['capture_stage.json', 'native/capture.json'])

    def test_unresolved_semantics_and_future_clock(self):
        root = self.base / 'semantics'
        root.mkdir()
        for status in ('UNRESOLVED_MODULAR_RECONSTRUCTION', 'LIMIT'):
            overwrite(root / 'construction.json', {'status': status})
            self.assertEqual(semantic_stop(root, 'construct'), status)
        overwrite(root / 'construction.json', {'status': 'FAKE_SAFE'})
        with self.assertRaises(ValueError):
            semantic_stop(root, 'construct')
        with self.assertRaises(ValueError):
            supervise(self.spec, self.base / 'future', started=time.monotonic() + 10)
        self.assertEqual(phase_state({'return_code': 0, 'killed': False}, 298, 298), 'TIMEOUT')

    def test_roster_stops_only_on_error_and_counts_all(self):
        jobs = [{'job_id': str(i)} for i in range(5)]
        states = iter(['UNSUPPORTED_MAPPING', 'TIMEOUT', 'CHECKED_LP_DIAGNOSTIC', 'ERROR'])
        rows = loop(jobs, lambda j: {'status': next(states)}, lambda r: None)
        self.assertEqual([r['status'] for r in rows], ['UNSUPPORTED_MAPPING', 'TIMEOUT', 'CHECKED_LP_DIAGNOSTIC', 'ERROR', 'NOT_RUN_AFTER_ERROR'])
        root = self.base / 'batch_timeout'
        timeout = supervise(self.spec, root, started=time.monotonic() - 301)
        rows = [{'job_id': 'good', **audit(self.good)}, {'job_id': 'timeout', **timeout},
                {'job_id': 'error', 'status': 'ERROR', 'complete_independent_check': False},
                {'job_id': 'unrun', 'status': 'NOT_RUN_AFTER_ERROR', 'complete_independent_check': False}]
        summary = summarize(rows, {'good': self.good, 'timeout': root})
        self.assertEqual(summary['denominator'], 4)
        self.assertEqual(summary['costed_requests'], 2)
        self.assertEqual(summary['uncosted_requests'], 2)
        self.assertIsNone(summary['records'][2]['costs'])
        with self.assertRaises(ValueError):
            summarize(rows, {'good': self.good})
        OBSERVATIONS.append({'case': 'mixed_roster', 'summary': summary})

    def run_lp(self, name, lp):
        s = {**self.statement, 'lp_sha256': identity(lp)}
        path = self.base / (name + '_input.json')
        save_new(path, {'lp': lp, 'statement': s})
        spec = {'input': {'path': str(path), 'sha256': digest(path.read_bytes())},
                'statement': s, 'statement_sha256': identity(s)}
        root = self.base / name
        terminal = supervise(spec, root, started=time.monotonic())
        self.assertEqual(audit(root)['status'], terminal['status'])
        return root, terminal

    def test_large_native_full_clock(self):
        from sparse_basis.tests import sparse_rows
        n = 4096
        lp = {'matrix_format': 'csr_v1', 'c': [-1]*n, 'offset': 0,
              'lower': [0]*n, 'upper': [1]*n,
              'E': sparse_rows([[(i % n, 3)] for i in range(2*n)], n), 'h': [1]*(2*n),
              'A': sparse_rows([], n), 'b': []}
        root, v = self.run_lp('large', lp)
        self.assertEqual(v['status'], 'CHECKED_LP_DIAGNOSTIC')
        self.assertEqual(read(root / 'check.log')['upper_bound'], '-4096/3')
        self.assertEqual(len(read(root / 'mapping.json')['hint']['rows']), 8192)
        t = costs(root)
        self.assertEqual(t['native_calls'], 1)
        self.assertAlmostEqual(t['whole_supplied_LP_seconds'], t['observed_phase_sum_seconds'] + t['residual_seconds'])
        OBSERVATIONS.append({'case': 'large_native_supplied_LP', 'root': str(root), 'costs': t,
                             'variables': n, 'E_rows': 8192, 'upper': '-4096/3', 'synthetic': True})

    def test_complete_check_can_reject_exact_feasibility(self):
        from fractions import Fraction as F
        lp = {'matrix_format':'csr_v1', 'c':[-1], 'offset':0, 'lower':[0], 'upper':[1],
              'E':csr([[1],[1]],1), 'h':['1/3',str(F(1,3)+F(1,2**60))], 'A':csr([],1),'b':[]}
        root, v = self.run_lp('inexact', lp)
        self.assertEqual(v['status'], 'CHECKED_LP_DIAGNOSTIC')
        out = read(root / 'check.log')
        self.assertEqual(out['primal_status'], 'NOT_EXACTLY_FEASIBLE')
        self.assertIsNone(out['upper_bound'])
        self.assertFalse(v['network_SAFE'] or v['network_UNSAFE'])

    def test_component_limit_is_not_error_or_infeasibility(self):
        n = 16385
        lp = {'matrix_format':'csr_v1', 'c':[0]*n, 'offset':0, 'lower':[0]*n,'upper':[1]*n,
              'E':csr([],n),'h':[],'A':csr([],n),'b':[]}
        root, v = self.run_lp('size_limit', lp)
        self.assertEqual(v['status'], 'LIMIT', v)
        self.assertFalse(v['complete_independent_check'])
        self.assertIsNone(costs(root)['native_calls'])

    def test_export_decoder_all_bindings(self):
        from modular_supervised.inputs import load
        ex = {'lp': self.lp, 'source': {'analytic': True}, 'q': [1,0], 'offset': 0}
        ex['source_sha256'] = identity(ex['source'])
        path = self.base / 'analytic.export.json'
        save_new(path, ex)
        s = {**self.statement, 'export_sha256': digest(path.read_bytes()),
             'source_sha256': ex['source_sha256'], 'property': {'q':[1,0], 'constant':0}}
        spec = {'export': {'path':str(path),'sha256':digest(path.read_bytes())},
                'statement':s,'statement_sha256':identity(s)}
        self.assertEqual(load(spec, lambda:None), {'lp':self.lp,'statement':s})
        for key in ('source_sha256', 'export_sha256', 'property'):
            bad = deepcopy(spec)
            bad['statement'][key] = {'q':[0,1],'constant':0} if key=='property' else '0'*64
            bad['statement_sha256'] = identity(bad['statement'])
            with self.assertRaises(ValueError):
                load(bad, lambda:None)
        with self.assertRaises(ValueError):
            load({**spec,'input':self.spec['input']}, lambda:None)

    def test_raw_only_cost_binding_rejects_wrong_request(self):
        from modular_supervised.flow import raw_cost
        raw = read(self.good / 'native/raw_native.json')
        self.assertEqual(raw_cost(self.good, raw, 300), raw['native_seconds'])
        for key,value in [('lp_sha256','0'*64),('statement_sha256','1'*64),('native_seconds',301)]:
            with self.assertRaises(ValueError):
                raw_cost(self.good, {**raw,key:value}, 300)

    def test_raw_capture_mismatch_even_resigned(self):
        root = self.clone('bad_raw')
        raw = read(root / 'native/raw_native.json')
        raw['native_objective'] += 1
        overwrite(root / 'native/raw_native.json', raw)
        resign_candidate(root)
        with self.assertRaises(ValueError):
            review_candidate(root)

    def test_new_plan_policy_and_old_version_rejected(self):
        from modular_supervised.flow import verify_sources
        p = read(self.good / 'plan.json')
        for name in ('schema', 'component_policy', 'sources', 'proposal_seconds'):
            bad = deepcopy(p)
            if name == 'schema':
                bad[name] = 'BASIS_SUPERVISION_PLAN_V1'
            elif name == 'component_policy':
                bad[name]['variables'] = 64
            elif name == 'sources':
                bad[name]['sparse_basis/engine.py'] = '0'*64
            else:
                bad[name] = 250
            with self.assertRaises(ValueError):
                verify_sources(bad)

    def test_study_ordered_error_roster(self):
        from modular_supervised.batch import loop as study_loop
        states = iter(['LIMIT','TIMEOUT','ERROR'])
        jobs = [{'job_id':str(i)} for i in range(4)]
        rows = study_loop(jobs, lambda j:{'status':next(states)}, lambda row:None)
        self.assertEqual([r['status'] for r in rows], ['LIMIT','TIMEOUT','ERROR','NOT_RUN_AFTER_ERROR'])
        self.assertEqual([r['job_id'] for r in rows], ['0','1','2','3'])

    def test_import_policy_bound_to_plan(self):
        from modular_supervised.flow import verify_sources
        p=read(self.good/'plan.json');p['native_options']['small_matrix_value']=1e-9
        with self.assertRaises(ValueError):verify_sources(p)

    def test_small_real_scale_coefficient_through_whole_clock(self):
        lp=deepcopy(self.lp)
        lp['A']={'shape':[1,2],'indptr':[0,2],'indices':[0,1],'data':[3,-3.6294188569593725e-10]}
        root,v=self.run_lp('tiny_full',lp)
        self.assertEqual(v['status'],'CHECKED_LP_DIAGNOSTIC')
        self.assertEqual(read(root/'native/import.json')['status'],'HighsStatus.kOk')
        self.assertEqual(read(root/'check.log')['primal_status'],'EXACT_FEASIBLE')
        self.assertEqual(costs(root)['native_calls'],1)

    def test_arithmetic_and_serialization_full_faults(self):
        popen=subprocess.Popen
        for mode in ('prime_schedule_stall','finite_field_stall','CRT_stall','reconstruction_stall','exact_residual_stall',
                     'arithmetic_exception','arithmetic_limit','serialization_stall','serialization_exception'):
            root=self.base/('full_'+mode)
            def launch(cmd,*args,**kwargs):
                return popen([ACT,'-m','modular_supervised.fault_worker','drive_'+mode,str(root)],*args,**kwargs)
            with patch('modular_supervised.flow.subprocess.Popen',launch):
                v=supervise(self.spec,root,started=time.monotonic()-(215 if mode.endswith('stall') else 0))
            expected='TIMEOUT' if mode.endswith('stall') else 'LIMIT' if mode.endswith('limit') else 'ERROR'
            self.assertEqual(v['status'],expected,(v,(root/'driver.log').read_text()))
            self.assertEqual(audit(root)['status'],expected)
            t=costs(root)
            self.assertEqual(t['native_calls'],1)
            self.assertFalse(v['complete_independent_check'])
            self.assertFalse((root/'portable').exists())
            self.assertGreater(t['arithmetic_progress']['event_count'],0)
            self.assertAlmostEqual(t['whole_supplied_LP_seconds'],t['observed_phase_sum_seconds']+t['residual_seconds'])
            if mode.endswith('stall') or mode.startswith('serialization'):
                self.assertTrue(t['arithmetic_progress']['censored'])
                self.assertIsNone(t['arithmetic_progress']['current_phase_seconds'])
            if mode.startswith('serialization'):
                self.assertTrue(t['construction_serialization']['censored'])
                self.assertIsNone(t['construction_serialization']['seconds'])
            OBSERVATIONS.append({'case':mode,'terminal':v,'costs':t,'synthetic':True})

    def test_instrumentation_exact_differential(self):
        from modular_basis.propose import propose
        m=read(self.good/'mapping.json')
        plain=propose(self.lp,self.statement,m['candidate'],m['hint'],identity(self.statement),identity(m['hint']),deadline=time.monotonic()+30)
        observed=read(self.good/'construction.json')
        for field in ('bundle','status','stats','arithmetic','operations','row_residuals','assembled_system_sha256'):
            self.assertEqual(plain[field],observed[field],field)
        t=costs(self.good)
        self.assertTrue(t['arithmetic_progress']['complete'])
        self.assertGreater(t['construction_serialization']['seconds'],0)
        self.assertLessEqual(t['arithmetic_progress']['event_count'],2048)

    def test_journal_identity_omission_and_clock_rejected(self):
        from modular_supervised.journal import journal_costs
        for i,change in enumerate(('binding','clock','omit','prefix','partial')):
            root=self.clone('journal_bad_'+change)
            paths=sorted((root/'arithmetic_events').glob('*.json'))
            r=read(paths[0])
            if change=='binding':r['bindings']['hint_sha256']='0'*64
            elif change=='clock':r['request_seconds']+=1
            elif change=='prefix':r['segments']=[{'phase':'setup','start_seconds':1,'end_seconds':2,'seconds':1}]
            if change=='omit':paths[0].rename(root/'removed_event.json')
            elif change=='partial':paths[0].write_bytes(b'{')
            else:overwrite(paths[0],r)
            with self.assertRaises(ValueError):journal_costs(root,allow_partial=True)
        root=self.clone('journal_partial_tail')
        paths=sorted((root/'arithmetic_events').glob('*.json'))
        paths[-1].write_bytes(b'{')
        p=journal_costs(root,allow_partial=True)
        self.assertTrue(p['censored'])
        self.assertIsNone(p['current_phase_seconds'])
        with self.assertRaises(ValueError):journal_costs(root,allow_partial=False,construction=read(root/'construction.json'))

    def test_multiple_modular_rounds_full_clock_and_exact_differential(self):
        from modular_basis.propose import propose
        lp={'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[1073741791],
            'E':csr([[1]],1),'h':[1073741790],'A':csr([],1),'b':[]}
        root,v=self.run_lp('multi_round',lp)
        self.assertEqual(v['status'],'CHECKED_LP_DIAGNOSTIC')
        observed=read(root/'construction.json');prepared=read(root/'prepared.json');m=read(root/'mapping.json')
        plain=propose(lp,prepared['statement'],m['candidate'],m['hint'],identity(prepared['statement']),
                      identity(m['hint']),deadline=time.monotonic()+30)
        for key in ('bundle','stats','arithmetic','operations','status','assembled_system_sha256'):
            self.assertEqual(observed[key],plain[key],key)
        self.assertEqual(observed['stats']['primes_used'],3)
        self.assertGreater(observed['stats']['residual_rejections'],0)
        self.assertEqual(read(root/'check.log')['upper_bound'],'-1073741790')
        segments=costs(root)['arithmetic_progress']['segments']
        self.assertEqual(sum(s['phase']=='prime_schedule' for s in segments),3)
        self.assertEqual(sum(s['phase']=='CRT' for s in segments),3)
        OBSERVATIONS.append({'case':'three_round_full_clock','costs':costs(root),'synthetic':True})

    def test_modular_exhaustion_full_terminal_is_unresolved(self):
        root=self.base/'full_modular_exhausted';popen=subprocess.Popen
        def launch(cmd,*args,**kwargs):
            return popen([ACT,'-m','modular_supervised.fault_worker','drive_modular_exhausted',str(root)],*args,**kwargs)
        with patch('modular_supervised.flow.subprocess.Popen',launch):
            v=supervise(self.spec,root,started=time.monotonic())
        self.assertEqual(v['status'],'UNRESOLVED_MODULAR_RECONSTRUCTION')
        self.assertFalse(v['complete_independent_check'])
        self.assertEqual(audit(root)['status'],v['status'])
        self.assertIsNone(costs(root)['phases']['check']['seconds'])
        self.assertFalse((root/'portable').exists())

    def test_modular_round_progress_and_illegal_cycle_mutations(self):
        from modular_supervised.journal import journal_costs,validate_progress
        original=read(self.good/'construction.json')['stats']
        for key,value in [('primes_tried',129),('max_field_product_bits',61),('primes_used',100)]:
            with self.assertRaises(ValueError):validate_progress({**original,key:value})
        wrong=deepcopy(original);wrong['rounds'][0]['prime']=0
        with self.assertRaises(ValueError):validate_progress(wrong)
        wrong=deepcopy(original);wrong['rounds'][0]['status']='FAKE_SAFE'
        with self.assertRaises(ValueError):validate_progress(wrong)
        wrong=deepcopy(original);wrong['rounds'][0]['prime']-=2
        with self.assertRaises(ValueError):validate_progress(wrong,original)
        root=self.clone('journal_bad_cycle')
        paths=sorted((root/'arithmetic_events').glob('*.json'))
        r=read(paths[1]);r['phase']='CRT'
        overwrite(paths[1],r)
        with self.assertRaises(ValueError):journal_costs(root,allow_partial=True)

    def test_bounded_journal_cycles_and_operation_snapshots(self):
        from modular_supervised.journal import TimedBudget,JOURNAL_POLICY,journal_costs
        root,start=self.fault_root('bounded_cycles',0)
        shutil.copyfile(self.good/'mapping.json',root/'mapping.json')
        m=read(root/'mapping.json')
        b=TimedBudget(start+218,root,begin=time.monotonic(),original_started=start,
            bindings={'lp_sha256':identity(self.lp),'statement_sha256':identity(self.statement),
                      'hint_sha256':identity(m['hint'])})
        b.where('assembly','test')
        # Test journal state-machine storage, not 128 new arithmetic solves.
        for i in range(128):
            for phase in ('prime_schedule','finite_field','CRT','reconstruction','exact_residual'):
                b.where(phase,'synthetic_cycle',prime=1073741789-i*2)
        b.operations=99999;b.visit()
        b.finish()
        j=journal_costs(root,allow_partial=True)
        self.assertGreater(j['event_count'],256)
        self.assertLess(j['event_count'],JOURNAL_POLICY['max_events'])
        self.assertTrue(j['censored'])  # Journal alone is never a completed candidate.
        with self.assertRaises(ValueError):b.where('setup','invalid_backward_transition')
        with patch.dict(JOURNAL_POLICY,max_events=b.sequence):
            with self.assertRaises(ValueError):b.snapshot('over_budget')


if __name__ == '__main__':
    unittest.main()
