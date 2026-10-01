"""Fixed CPU supervision controls; no model/data/native solver/GPU execution."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import unittest
from unittest.mock import Mock, patch

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_proof import owned_bounded
from source_enclosure.format import identity
from scripts import hz_propagation_supervised as api


def rewrite(path, value):
    # Controlled mutation of a private copied test artifact, never an execution.
    path.write_text(json.dumps(value, sort_keys=True, separators=(',',':'), allow_nan=False))


def roster():
    cfg = api.protocol()
    return [(name, name, '', cfg['normal_control_seconds'], api.DONE) for name in cfg['cases']]+[
        (fault, 'retained_guard', fault, data[0], data[1]) for fault, data in cfg['faults'].items()]


class PropagationSupervisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(os.environ['HZ_PROPAGATION_CONTROL_ROOT'])
        cls.calls = {}
        for name, case, fault, budget, expected in roster():
            spec = api.specification(case, fault)
            save(cls.root/(name+'_started.json'), {'spec': spec, 'budget': budget, 'expected': expected})
            begin = time.monotonic()
            result = api.supervise(cls.root/name, spec, expected_request_sha256=identity(spec),
                                   budget=budget, rss_limit=1 if fault=='low_rss' else 2*2**30)
            observation = {'begin': begin, 'end': time.monotonic(), 'result': result}
            save(cls.root/(name+'_observed.json'), observation)
            cls.calls[name] = observation
            stages = load(cls.root/name/'terminal.json')['stages']
            if result['status'] == 'CLEANUP_INCOMPLETE' or any(s['cleanup_status']=='CLEANUP_INCOMPLETE' for s in stages):
                raise RuntimeError('unresolved cleanup; no further launch admitted')
        save(cls.root/'calls.json', cls.calls)

    def test_all_fixed_calls_and_reached_faults(self):
        self.assertEqual(set(self.calls), {r[0] for r in roster()})
        for name, _, fault, _, expected in roster():
            with self.subTest(name=name):
                self.assertEqual(self.calls[name]['result']['status'], expected)
                root = self.root/name
                api.audit(root, observation=self.calls[name])
                if not fault:
                    continue
                if fault == 'late_publish':
                    self.assertEqual(load(root/'late_publish_reached.json')['invocation'],
                                     load(root/'invocation.json')['invocation'])
                elif fault == 'launch_failure':
                    self.assertIn('No such file', load(root/'produce_stage.json')['error'])
                elif fault == 'low_rss':
                    self.assertGreater(load(root/'produce_stage.json')['sampled_peak_rss'], 1)
                else:
                    records = [json.loads(line) for p in root.glob('*_events.jsonl') for line in p.read_text().splitlines()]
                    self.assertTrue(any(r.get('event')=='FAULT_REACHED' and r.get('name')==fault for r in records))
                    if fault=='descendant':
                        stage = load(root/'produce_stage.json')
                        created = [r for r in records if r.get('event')=='DESCENDANT_STARTED']
                        self.assertEqual(len(created), 1)
                        self.assertEqual(created[0]['pgid'], stage['pid'])
                        self.assertNotEqual(created[0]['pid'], stage['pid'])
                        self.assertGreater(created[0]['pid'], 0)
                        self.assertTrue(stage['descendant_on_leader_exit'])
                        self.assertEqual(stage['cleanup_status'], 'LEADER_REAPED_NO_LIVE_GROUP')

    def test_independent_recheck_and_reference_differential(self):
        ref = load(ROOT/'docs/hz_checked_propagation_20261001_r5.json')
        old = load(Path(ref['archive'])/'observations.json', ref['observations_sha256'])
        checked = 0
        for name in api.protocol()['cases']:
            result = api.audit(self.root/name, observation=self.calls[name], recheck=True)
            self.assertTrue(result['budget_acceptance_observed'])
            checked += result['bounds_rechecked']
            new = load(self.root/name/'payload.json')['propagation']
            self.assertEqual(new['scope'], old[name]['scope'])
            self.assertEqual(new['output'], old[name]['output'])
            for a,b in zip(new['events'], old[name]['events']):
                self.assertEqual(a['status'], b['status'])
                if a['package'] is not None:
                    self.assertEqual(a['package']['accepted'], b['package']['accepted'])
        self.assertEqual(checked, 10)

    def test_partial_evidence_and_missing_output_not_accepted(self):
        for name in ('partial_package','wrong_scope','missing_check','native_wait_stub','serialization_delay'):
            term = load(self.root/name/'terminal.json')
            self.assertIsNone(term['accepted'])
            self.assertGreater(load(self.root/name/'cost.json')['seconds'], 0)
        self.assertTrue((self.root/'serialization_delay'/'prefix_layer3.json').is_file())
        self.assertFalse((self.root/'serialization_delay'/'payload.json').exists())
        self.assertTrue(load(self.root/'missing_check'/'check_stage.json')['seconds'] > 0)

    def test_used_bounds_source_and_property_mutations_rejected(self):
        root = self.root/'retained_guard'
        spec, inv, original = load(root/'spec.json'), load(root/'invocation.json'), load(root/'payload.json')
        for mode in ('missing_layer','missing_package','missing_side','wrong_side','property','source','bound'):
            p = copy.deepcopy(original)
            e = p['propagation']['events'][0]; pkg = e['package']
            if mode == 'missing_layer': p['propagation']['events'].clear()
            elif mode == 'missing_package': e['package'] = None
            elif mode == 'missing_side': pkg['accepted']['results'].pop()
            elif mode == 'wrong_side': pkg['batch']['queries'][0]['side'] = 'max'
            elif mode == 'property': pkg['batch']['queries'][0]['q'][0] = '2'
            elif mode == 'source': p['propagation']['scope']['expert'] = 1
            else: pkg['accepted']['results'][0]['bound'] = '999999999'
            with self.subTest(mode=mode), self.assertRaises((ValueError, KeyError)):
                api.exact_check(p, spec, inv, time.monotonic()+30)

    def clone(self, case, suffix):
        dest = self.root/('mutation_'+suffix)
        shutil.copytree(self.root/case, dest)
        return dest, copy.deepcopy(self.calls[case])

    def rebind(self, dest, observation, term, cost):
        rewrite(dest/'terminal.json', term)
        cost['terminal_sha256'] = sha(dest/'terminal.json')
        rewrite(dest/'cost.json', cost)
        observation['result'].update(terminal_sha256=sha(dest/'terminal.json'), cost_sha256=sha(dest/'cost.json'))

    def test_rebound_cost_and_environment_mutations(self):
        for mode in ('cost','cuda','threads','cutoff','cleanup','missing_anchor','fake_success','pid','descendant','late_exit'):
            case = 'partial_package' if mode=='fake_success' else 'retained_guard'
            dest, obs = self.clone(case, mode)
            term, cost = load(dest/'terminal.json'), load(dest/'cost.json')
            stage = term['stages'][0]
            if mode=='cost': cost['parent_seconds'] += 1
            elif mode=='cuda': stage['cuda_visible_devices'] = '0'
            elif mode=='threads': stage['cpu_threads'] = 4
            elif mode=='cutoff': stage['run_deadline'] += 1
            elif mode=='cleanup': stage['cleanup_status'] = 'CLEANUP_INCOMPLETE'
            elif mode=='missing_anchor': stage.pop('output_sha256')
            elif mode=='pid': stage['pid'] = None
            elif mode=='descendant': stage['descendant_on_leader_exit'] = True
            elif mode=='late_exit': stage['exit_observed_at'] = stage['run_deadline']+0.01
            else: term['status'] = cost['status'] = api.DONE
            rewrite(dest/'produce_stage.json', stage)
            self.rebind(dest, obs, term, cost)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                api.audit(dest, observation=obs)

    def test_failed_prefix_and_payload_anchor_are_mandatory(self):
        for mode in ('prefix','log','payload','checker'):
            case = 'serialization_delay' if mode=='prefix' else 'retained_guard'
            dest, obs = self.clone(case, 'delete_'+mode)
            if mode=='prefix': (dest/'prefix_layer3.json').unlink()
            elif mode=='log': (dest/'produce_events.jsonl').unlink()
            elif mode=='payload': rewrite(dest/'payload.json', {})
            else: rewrite(dest/'check.json', {})
            with self.subTest(mode=mode), self.assertRaises((ValueError, FileNotFoundError)):
                api.audit(dest, observation=obs)
        # A produced/anchored payload whose parent reception was late is still
        # part of the failed evidence prefix, not deletable because status changed.
        dest, obs = self.clone('retained_guard', 'late_anchored_prefix')
        term, cost = load(dest/'terminal.json'), load(dest/'cost.json')
        term['stages'] = term['stages'][:1]
        stage = term['stages'][0]; stage['status'] = 'TIMEOUT'
        term['status'] = cost['status'] = obs['result']['status'] = 'TIMEOUT'
        term['accepted'] = None
        cost['stage_seconds'] = stage['seconds']; cost['parent_seconds'] = cost['seconds']-stage['seconds']
        rewrite(dest/'produce_stage.json',stage)
        (dest/'check_stage.json').unlink(); (dest/'receive_stage.json').unlink()
        self.rebind(dest,obs,term,cost)
        api.audit(dest,observation=obs)
        (dest/'payload.json').unlink()
        with self.assertRaises(FileNotFoundError): api.audit(dest,observation=obs)

    def test_external_return_observation_required(self):
        root = self.root/'retained_guard'
        self.assertFalse(api.audit(root)['budget_acceptance_observed'])
        obs = copy.deepcopy(self.calls['retained_guard'])
        obs['end'] = load(root/'invocation.json')['deadline']+1
        with self.assertRaises(ValueError): api.audit(root, observation=obs)
        self.assertEqual(self.calls['late_publish']['result']['status'], 'TIMEOUT')
        obs = copy.deepcopy(self.calls['retained_guard'])
        obs['result']['seconds'] = float('nan')
        with self.assertRaises(ValueError): api.audit(root, observation=obs)
        dest, obs = self.clone('late_publish', 'publication_marker')
        (dest/'late_publish_reached.json').unlink()
        with self.assertRaises(FileNotFoundError): api.audit(dest, observation=obs)

    def test_parent_request_and_source_hash(self):
        spec = api.specification()
        with self.assertRaises(ValueError):
            api.supervise(self.root/'never_launched', spec, expected_request_sha256='0'*64)
        self.assertFalse((self.root/'never_launched').exists())
        with self.assertRaises(ValueError):
            api.bind(self.root/'retained_guard', spec, '0'*64)

    def test_cost_includes_cleanup_and_nested_cost_not_added(self):
        for name, obs in self.calls.items():
            root = self.root/name; inv = load(root/'invocation.json')
            cost, term = load(root/'cost.json'), load(root/'terminal.json')
            self.assertAlmostEqual(cost['seconds'], cost['stage_seconds']+cost['parent_seconds'])
            self.assertLessEqual(cost['seconds'], obs['result']['seconds'])
            self.assertLessEqual(obs['result']['seconds'], obs['end']-inv['start'])
            for stage in term['stages']:
                self.assertAlmostEqual(stage['seconds'], stage['execution_seconds']+stage['cleanup_seconds'])
                if stage['pid'] is not None:
                    self.assertEqual(stage['cleanup_status'], 'LEADER_REAPED_NO_LIVE_GROUP')
                    self.assertEqual(stage['remaining_group']['live'], [])

    def test_unrelated_sentinel_survives(self):
        sentinel = subprocess.Popen([PYTHON, '-S','-c','import time; time.sleep(30)'])
        try:
            now = time.monotonic()
            record = owned_bounded.execute([PYTHON, '-S','-c','pass'], self.root/'executor_sentinel.log',
                run_deadline=now+2, cleanup_deadline=now+2.25, env=dict(os.environ), rss_limit=2*2**30)
            self.assertEqual(record['status'], 'COMPLETED')
            self.assertIsNone(sentinel.poll())
        finally:
            sentinel.kill(); sentinel.wait(timeout=2)

    def test_bounded_reap_never_waits_without_timeout(self):
        p = Mock(); p.wait.side_effect = subprocess.TimeoutExpired('mock', .01)
        end = time.monotonic()+.01
        self.assertFalse(owned_bounded.bounded_reap(p, end))
        self.assertGreater(p.wait.call_args.kwargs['timeout'], 0)
        self.assertLessEqual(p.wait.call_args.kwargs['timeout'], .01)
        p.reset_mock()
        self.assertFalse(owned_bounded.bounded_reap(p, time.monotonic()-1))
        p.wait.assert_not_called()

    def test_late_exit_observation_and_cleanup_fail_closed(self):
        self.assertEqual(api.deadline_status('CLEANUP_INCOMPLETE', 100, 1), 'CLEANUP_INCOMPLETE')
        class Process:
            pid = 123456789
            returncode = 0
            def wait(self, **kwargs): return 0
        state = {'late': False}
        def observed(pid):
            state['late'] = True
            return {'live': [], 'zombies': [], 'rss': 0}
        def clock(): return 2. if state['late'] else 0.
        for mode in ('late','unreaped'):
            state['late'] = False
            with patch.object(owned_bounded.subprocess, 'Popen', return_value=Process()), \
                 patch.object(owned_bounded, 'observe', side_effect=observed), \
                 patch.object(owned_bounded, 'parent_rss', return_value=0), \
                 patch.object(owned_bounded, 'exited_unreaped', return_value=Mock(si_code=os.CLD_EXITED,si_status=0)), \
                 patch.object(owned_bounded.os, 'killpg'), patch.object(owned_bounded.time,'monotonic', side_effect=clock), \
                 patch.object(owned_bounded, 'bounded_reap', return_value=mode!='unreaped'):
                # cleanup deadline <= the simulated observation also makes failed
                # cleanup terminate immediately; no fake forever loop.
                result = owned_bounded.execute(['mock'], self.root/('mock_'+mode+'.log'),
                    run_deadline=1, cleanup_deadline=3 if mode=='late' else 2,
                    env={}, rss_limit=100)
            self.assertEqual(result['status'], 'TIMEOUT' if mode=='late' else 'CLEANUP_INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
