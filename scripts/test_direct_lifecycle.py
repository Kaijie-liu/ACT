"""Frozen V2 lifecycle controls, standard library and synthetic sources only."""
import copy
from fractions import Fraction as F
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_source.hz_request_store import Work, POLICY, identity, compact, digest, decode
from scoped_source.scaled_endpoint_controls import fixture, expanded, scalar_bound
from scoped_source.direct_source_build import build
from scoped_source.check_direct_source import CheckedDirectForest
from scoped_source.scaled_endpoints import contract, Session
from scoped_source.direct_candidates import Driver
from scoped_source.direct_method import method, validate
from scoped_source.direct_lifecycle import Lifecycle, prepare, revoke, FixedProposer
from scoped_source.direct_lifecycle_execution import execute
from scoped_source.direct_lifecycle_package import check_package, check_manifest
from scoped_source.direct_supervised import supervise, spec, audit, fixed_source, read_json, ROOT, PYTHON
from scripts.test_bounded_runtime import candidates


def work():
    return Work(time.monotonic() + 300)


class Controls(unittest.TestCase):
    differentials = []
    supervised = []

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='hz-lifecycle-controls-', dir='/data1/Kane/MOE')
        cls.root = Path(cls.temp.name)
        cls.cases = {}
        for name in ('mixed', 'deep', 'half', 'guarded'):
            root, source = fixture(cls.root / name, name)
            raw, _ = build(root, source, work=work(), enabled=True)
            cls.cases[name] = root, source, raw

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def path(self, prefix):
        return self.root / (prefix + str(time.monotonic_ns()))

    def driver(self, name='deep', arm='pair_scoped', w=None):
        root, source, raw = self.cases[name]
        w = w or work()
        f = CheckedDirectForest(raw, root, source, expected=digest(raw), work=w, enabled=True)
        c = contract(f)
        d = Driver(f, c, expected=identity(c), work=w, pair_scoped=arm == 'pair_scoped')
        self.addCleanup(d.abort)
        return d

    def package(self, name='half', arm='pair_scoped', profile=0, fault='', w=None):
        source_root, source, _ = self.cases[name]
        root = self.path('run-')
        result = execute(root, source_root, source, method(arm, profile), 'a' * 64, ROOT,
                         work=w or work(), enabled=True, fault=fault)
        return root, result

    def test_frozen_fixture_identity(self):
        for name in ('half', 'guarded'):
            self.assertEqual(fixed_source(self.path('fixture-'), name), self.cases[name][1])

    def test_default_off_and_closed_registry(self):
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            supervise(self.path('no-'), spec('half', 'dense'))
        with self.assertRaisesRegex(ValueError, 'opt-in'):
            execute(self.path('no-'), *self.cases['half'][:2], method('dense'), 'a' * 64, ROOT, work=work())
        for arm, profile in (('gpu', 0), ('dense', True), ('pair_scoped', 2)):
            with self.assertRaises(ValueError):
                method(arm, profile)
        for field in ('producer', 'production', 'replay'):
            m = method('dense'); m[field]['kind'] = 'other'
            with self.assertRaises(ValueError):
                validate(m)
        m = method('dense'); m['candidate_width'] = True
        with self.assertRaises(ValueError):
            validate(m)
        self.assertEqual(POLICY['visits'], 8000000)
        self.assertEqual(POLICY['total_bytes'], 128 * 2**20)

    def test_roots_readonly_values_no_alias(self):
        d = self.driver()
        tasks, lease, token = prepare(d)
        center, mapping = lease[0]
        self.assertGreater(len(mapping), 0)
        key = next(iter(mapping)); value = mapping[key]
        original_center, original_value = str(center), str(value)
        object.__setattr__(center, '_numerator', 999)
        object.__setattr__(value, '_numerator', 999)
        self.assertEqual(str(lease[0][0]), original_center)
        self.assertEqual(str(lease[0][1][key]), original_value)
        for obj, name in ((lease, '_live'), (mapping, '_row')):
            with self.assertRaises(AttributeError):
                setattr(obj, name, None)
            with self.assertRaises(AttributeError):
                delattr(obj, name)
        with self.assertRaises(TypeError):
            mapping[key] = F(7)
        items = iter(mapping.items()); next(items)
        sequence = iter(lease); next(sequence)
        d.submit(token, candidates(d.forest, tasks), lambda i,b:(digest(b), len(b)))
        for fn in (lambda:len(lease), lambda:len(mapping), lambda:mapping[key], lambda:next(items), lambda:next(sequence)):
            with self.assertRaises(ValueError):
                fn()
        revoke(lease)
        self.assertIsNone(lease._roots)

    def test_foreign_source_work_task_token_and_ordinal(self):
        for changed in ('source', 'work', 'task', 'token', 'ordinal', 'contract'):
            d = self.driver(); _, lease, _ = prepare(d)
            if changed == 'source':
                d.forest = self.driver().forest
            elif changed == 'work':
                d.work = work()
            elif changed == 'task':
                d.pending = (tuple(reversed(d.pending[0])), d.pending[1])
            elif changed == 'token':
                d.pending = (d.pending[0], object())
            elif changed == 'ordinal':
                d.ordinal += 1
            else:
                d.contract = '0' * 64
            with self.assertRaises(ValueError):
                len(lease)
            revoke(lease)

    def test_abort_and_proposal_revoke(self):
        d = self.driver(); _, lease, _ = prepare(d); d.abort()
        with self.assertRaises(ValueError):
            len(lease)
        d = self.driver(); life = Lifecycle(method('pair_scoped')); life.setup(d.forest, d.work)
        token, records = life.propose(d)
        with self.assertRaises(ValueError):
            len(life.proposer.saved)
        d.submit(token, records, lambda i,b:(digest(b), len(b)))
        life.close()
        with self.assertRaises(ValueError):
            life.close()
        with self.assertRaises(ValueError):
            life.propose(d)
        with self.assertRaises(ValueError):
            life.setup(d.forest, d.work)

    def test_proposer_rejects_other_driver_and_tasks(self):
        d, other = self.driver(), self.driver()
        p = FixedProposer(method('pair_scoped')); p.setup(d.forest, d.work)
        tasks, lease, _ = prepare(other)
        with self.assertRaises(ValueError):
            p.propose(tasks, lease)
        revoke(lease); p.close()
        p = FixedProposer(method('pair_scoped')); p.setup(d.forest, d.work)
        tasks, lease, _ = prepare(d)
        with self.assertRaises(ValueError):
            p.propose(tuple(reversed(tasks)), lease)
        revoke(lease); p.close()

    def test_same_candidate_full_differential(self):
        for name in self.cases:
            for profile in (0, 1):
                results = []
                for arm in ('dense', 'pair_scoped'):
                    d = self.driver(name, arm); life = Lifecycle(method(arm, profile))
                    life.setup(d.forest, d.work)
                    oracle_driver = self.driver(name, 'dense')
                    old = oracle_driver.session
                    matrix = expanded(oracle_driver.forest, oracle_driver.work)
                    checks = 0; receipts = []
                    for i in range(d.session.tile_count):
                        token, records = life.propose(d)
                        tasks = d.pending[0]
                        expected_records = candidates(d.forest, tasks, general=profile == 1)
                        self.assertEqual(records, expected_records)
                        accepted = d.submit(token, records, lambda i,b:(digest(b), len(b)), profile=profile)
                        ot, _ = old.prepare(i)
                        received = old.accept(records); receipts.extend(received)
                        for t, rec, row in zip(ot, records, accepted, strict=True):
                            for role in ('proposed', 'zero'):
                                bound, _ = scalar_bound(oracle_driver.forest, t, rec[role], matrix, oracle_driver.work)
                                self.assertEqual(row.record[role], str(bound)); checks += 1
                    life.close()
                    result = d.finish()
                    self.assertEqual(result, old.finish(receipts))
                    results.append(result)
                    self.differentials.append(dict(case=name, arm=arm, profile=profile, checks=checks,
                        exact_result_sha256=identity(result), work=d.work.counts, lifecycle=life.proposer.stats))
                self.assertEqual(*results)

    def test_full_arms_package_and_cost(self):
        for arm in ('dense', 'pair_scoped'):
            for profile in (0, 1):
                w = work(); w.visit('row_visits', 17)
                root, r = self.package('guarded', arm, profile, w=w)
                self.assertEqual(r['status'], 'COMPLETE', r)
                self.assertEqual(r['result'], r['replay']['result'])
                self.assertEqual(r['replay']['consumer'], method(arm, profile)['replay'])
                self.assertEqual(r['snapshots']['lifecycle']['close_calls'], 1)
                self.assertEqual(r['snapshots']['owner']['live_bytes'], 0)
                for key in r['work']:
                    self.assertEqual(r['initial_work'][key] + sum(p['work'][key] for p in r['phases']), r['work'][key])
                self.assertLessEqual(sum(p['seconds'] for p in r['phases']), r['seconds'])
                self.assertEqual(r['archive']['logical_bytes'], sum(p.stat().st_size for p in root.rglob('*') if p.is_file()))
                with self.assertRaises(ValueError):
                    check_package(root/'package', expected=r['package']['sha256'], invocation='a'*64,
                        expected_method=method('dense' if arm=='pair_scoped' else 'pair_scoped', profile), work=work(), enabled=True)

    def test_rehashed_method_and_obligation_mutations(self):
        root, r = self.package('guarded')
        self.assertEqual(r['status'], 'COMPLETE', r)
        package = root/'package'; manifest = decode((package/'manifest.json').read_bytes())
        mutations = (lambda m:m['tiles'].pop(), lambda m:m['tiles'].reverse(),
                     lambda m:m.__setitem__('invocation', '0'*64),
                     lambda m:m['method']['producer'].__setitem__('profile', 1),
                     lambda m:m['method']['replay'].__setitem__('kind', 'dense'),
                     lambda m:m['method']['production'].__setitem__('kind', 'dense'))
        for mutate in mutations:
            m = copy.deepcopy(manifest); mutate(m); raw = compact(m)
            with self.assertRaises(ValueError):
                check_manifest(package, raw, expected=digest(raw), invocation='a'*64,
                               expected_method=method('pair_scoped'), work=work(), enabled=True)

    def test_actual_reencoded_profile_rejected(self):
        from scoped_source.direct_candidates import encode
        root, r = self.package()
        self.assertEqual(r['status'], 'COMPLETE', r)
        package = root/'package'; m = decode((package/'manifest.json').read_bytes())
        d = self.driver('half'); tasks, _, _ = d.prepare()
        block, raw = encode(d.owner, d.forest, d.contract, tasks, 0, candidates(d.forest, tasks), profile=1)
        block.release()
        (package/m['tiles'][0]['file']).write_bytes(raw)
        m['tiles'][0].update(bytes=len(raw), sha256=digest(raw))
        raw = compact(m)
        with self.assertRaisesRegex(ValueError, 'actual candidate scalar profile'):
            check_manifest(package, raw, expected=digest(raw), invocation='a'*64,
                           expected_method=method('pair_scoped'), work=work(), enabled=True)

    def test_mixed_live_consumer_rejected(self):
        d = self.driver(arm='dense'); life = Lifecycle(method('pair_scoped'))
        life.setup(d.forest, d.work)
        with self.assertRaisesRegex(ValueError, 'actual producer consumer'):
            life.propose(d)
        self.assertIsNone(d.pending)
        life.close()

    def test_exceptions_close_and_missing_do_not_drop_duties(self):
        for fault in ('setup_error', 'propose_error', 'close_error', 'partial', 'stale_roots', 'after_tile_error'):
            root, r = self.package('guarded', fault=fault)
            self.assertEqual(r['status'], 'ERROR', (fault,r))
            self.assertEqual(r['snapshots']['lifecycle']['close_calls'], 1)
            self.assertEqual(r['snapshots']['owner']['live_bytes'], 0)
            self.assertFalse((root/'terminal.json').exists())
        _, r = self.package('guarded', fault='missing')
        self.assertEqual(r['status'], 'COMPLETE', r)
        self.assertEqual(r['result']['status'], 'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(r['result']['required'], 12)

    def test_expired_setup_still_closes(self):
        original = Lifecycle.setup
        def expire(life, f, w):
            original(life, f, w); w.deadline = time.monotonic() - 1
        with patch.object(Lifecycle, 'setup', expire):
            _, r = self.package()
        self.assertEqual(r['status'], 'TIMEOUT', r)
        self.assertEqual(r['snapshots']['lifecycle']['close_calls'], 1)
        self.assertEqual(r['snapshots']['owner']['live_bytes'], 0)

    def test_cleanup_exceptions_preserve_cost_and_primary(self):
        original = Driver.abort
        def fail_abort(d):
            original(d); raise RuntimeError('controlled abort failure')
        with patch.object(Driver, 'abort', fail_abort):
            root, r = self.package(fault='propose_error')
        self.assertEqual(r['status'], 'ERROR', r)
        self.assertIn('propose_error', r['snapshots']['primary_error'])
        self.assertFalse(next(p for p in r['phases'] if p['name']=='production_cleanup')['complete'])
        self.assertFalse((root/'terminal.json').exists())
        with patch('scoped_source.direct_lifecycle_execution.gc.collect', side_effect=RuntimeError('controlled gc')):
            _, r = self.package()
        self.assertEqual(r['status'], 'ERROR')
        self.assertFalse(next(p for p in r['phases'] if p['name']=='production_cleanup')['complete'])
        original_close = FixedProposer.close
        def fail_close(p):
            original_close(p); raise RuntimeError('controlled close too')
        with patch.object(FixedProposer, 'close', fail_close):
            _, r = self.package(fault='setup_error')
        self.assertIn('setup_error', r['snapshots']['primary_error'])
        self.assertIn('close too', r['error'])
        self.assertEqual(r['snapshots']['lifecycle']['close_calls'], 1)

    def test_relocated_isolated_both_arms(self):
        for arm in ('dense', 'pair_scoped'):
            root, r = self.package(arm=arm)
            self.assertEqual(r['status'], 'COMPLETE', r)
            moved = self.path('relocated-'); shutil.copytree(root/'package', moved)
            code = ("import sys;sys.path.insert(0," + repr(str(moved/'code')) + ");"
                    "from scoped_source.direct_lifecycle_package import check_package;"
                    "from scoped_source.direct_method import method;"
                    "from scoped_source.hz_request_store import Work,compact;import time;"
                    "deny=" + repr((str(ROOT), str(root), str(self.cases['half'][0]))) + ";"
                    "\ndef hook(event,args):\n"
                    " if event=='open' and isinstance(args[0],str) and args[0].startswith(deny): raise RuntimeError('original access')\n"
                    "sys.addaudithook(hook)\n"
                    f"r=check_package({str(moved)!r},expected={r['package']['sha256']!r},expected_method=method({arm!r}),invocation={'a'*64!r},work=Work(time.monotonic()+15),enabled=True)\n"
                    "assert not any(n.split('.')[0] in ('torch','numpy','scipy') for n in sys.modules)\n"
                    "print(compact(r['result']).decode())")
            cp = subprocess.run([PYTHON, '-I', '-S', '-B', '-c', code], capture_output=True, timeout=20)
            self.assertEqual(cp.returncode, 0, cp.stderr.decode())
            self.assertEqual(decode(cp.stdout), r['result'])

    def run_supervised(self, case='half', arm='pair_scoped', profile=0, fault='', seconds=10):
        root = self.path('supervised-')
        begin = time.monotonic()
        r = supervise(root, spec(case, arm, profile, fault), enabled=True, seconds=seconds)
        end = time.monotonic()
        checked = audit(root, r)
        self.assertLessEqual(r['seconds'], end-begin)
        if r['status'] == 'COMPLETE':
            inv, _ = read_json(root/'invocation.json', 65536)
            self.assertLess(end, inv['deadline'])
        log = (root/'worker.log').read_text() if (root/'worker.log').exists() else ''
        self.supervised.append(dict(case=case, arm=arm, profile=profile, fault=fault,
            status=r['status'], seconds=r['seconds'], observed_api_seconds=end-begin,
            budget=r['budget'], bytes=r['storage']['bytes'],
            result=None if r['accepted'] is None else r['accepted']['result'],
            work=r['work'], audit=checked, child=r['child'], log=log))
        return root, r

    def test_supervised_complete_both_arms_profiles(self):
        for case in ('half', 'guarded'):
            for profile in (0, 1):
                results = []
                for arm in ('dense', 'pair_scoped'):
                    root, r = self.run_supervised(case, arm, profile)
                    self.assertEqual(r['status'], 'COMPLETE', self.supervised[-1])
                    self.assertLess(r['seconds'], r['budget'])
                    self.assertEqual(r['child']['cleanup_status'], 'LEADER_REAPED_NO_LIVE_GROUP')
                    self.assertEqual(r['child']['remaining_group']['live'], [])
                    self.assertEqual(r['accepted']['method'], method(arm, profile))
                    self.assertLessEqual(sum(r['work'].values()), POLICY['visits'])
                    results.append(r['accepted']['result'])
                self.assertEqual(*results)

    def test_supervised_fault_roster(self):
        for fault in ('startup_error', 'setup_error', 'propose_error', 'close_error', 'output_partial',
                      'wrong_invocation', 'wrong_method', 'after_tile_error', 'partial',
                      'receive_expired', 'terminal_expired'):
            root, r = self.run_supervised(fault=fault, seconds=1.5 if fault in ('receive_expired', 'terminal_expired') else 10)
            self.assertIn(r['status'], ('ERROR', 'TIMEOUT'), self.supervised[-1])
            self.assertIsNone(r['accepted'])
            self.assertGreater(r['storage']['bytes'], 0)

    def test_hard_blocked_stubs_cleanup_and_partial(self):
        for fault in ('setup_block', 'propose_block', 'close_block', 'replay_block', 'output_block', 'late_exit'):
            root, r = self.run_supervised(fault=fault, seconds=1.5)
            self.assertEqual(r['status'], 'TIMEOUT', self.supervised[-1])
            self.assertIsNone(r['accepted'])
            self.assertIsNone(r['work'])
            self.assertEqual(r['child']['cleanup_status'], 'LEADER_REAPED_NO_LIVE_GROUP')
            self.assertGreater((root/'progress.jsonl').stat().st_size, 0)
            if fault == 'late_exit':
                self.assertEqual(read_json(root/'candidate.json', 2**20)[0]['status'], 'COMPLETE')

    def test_late_final_publication_cannot_rescue(self):
        import scoped_source.direct_supervised as module
        original = module.write_json
        def late(path, value, cap):
            if Path(path).name == 'receipt.json':
                inv, _ = read_json(Path(path).parent/'invocation.json', 65536)
                while time.monotonic() < inv['deadline']:
                    time.sleep(min(.01, max(0., inv['deadline']-time.monotonic())))
            return original(path, value, cap)
        with patch.object(module, 'write_json', late):
            root, r = self.run_supervised(seconds=1.5)
        self.assertEqual(r['status'], 'TIMEOUT')
        self.assertIsNone(r['accepted'])
        self.assertEqual(read_json(root/'receipt.json', 2**20)[0]['status'], 'COMPLETE')

    def test_no_process_preserves_timeout(self):
        import scoped_source.direct_supervised as module
        def not_launched(*args, **kw):
            return dict(status='TIMEOUT', cleanup_status='NO_PROCESS', pid=None, returncode=None,
                        seconds=0., execution_seconds=0., cleanup_seconds=0., sampled_peak_rss=0,
                        run_deadline=kw['run_deadline'], cleanup_deadline=kw['cleanup_deadline'])
        with patch.object(module, 'run_owned', not_launched):
            _, r = self.run_supervised()
        self.assertEqual(r['status'], 'TIMEOUT')
        self.assertIsNone(r['accepted'])

    def test_supervised_missing_remains_unknown(self):
        _, r = self.run_supervised('guarded', fault='missing')
        self.assertEqual(r['status'], 'COMPLETE', self.supervised[-1])
        self.assertEqual(r['accepted']['result']['status'], 'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(r['accepted']['result']['required'], 12)

    def test_cleanup_unconfirmed_never_accepts(self):
        import scoped_source.direct_supervised as module
        original = module.run_owned
        def incomplete(*args, **kw):
            r = original(*args, **kw); r['cleanup_status'] = 'CLEANUP_INCOMPLETE'
            r['status'] = 'CLEANUP_INCOMPLETE'; return r
        with patch.object(module, 'run_owned', incomplete):
            _, r = self.run_supervised()
        self.assertEqual(r['status'], 'CLEANUP_INCOMPLETE')
        self.assertIsNone(r['accepted'])

    def test_preconsumed_work_and_audit_anchor(self):
        root = self.path('budget-')
        r = supervise(root, spec('half', 'dense'), enabled=True, seconds=10,
                      initial_work=dict(row_visits=100, coefficient_visits=0, arithmetic=0))
        self.assertEqual(r['status'], 'COMPLETE', (r, (root/'worker.log').read_text()))
        self.assertGreater(r['work']['row_visits'], 100)
        changed = copy.deepcopy(r); changed['seconds'] = 11
        with self.assertRaises(ValueError):
            audit(root, changed)
        changed = copy.deepcopy(r); changed['storage']['bytes'] += 1
        with self.assertRaises(ValueError):
            audit(root, changed)
        with self.assertRaises(ValueError):
            supervise(self.path('over-'), spec('half', 'dense'), enabled=True, seconds=301)

    def test_failure_audit_and_equal_length_corruption(self):
        root, r = self.run_supervised(fault='propose_error')
        for key, value in (('status', 'SUCCESS'), ('seconds', -1), ('seconds', float('nan'))):
            wrong = copy.deepcopy(r); wrong[key] = value
            with self.assertRaises(ValueError):
                audit(root, wrong)
        wrong = copy.deepcopy(r); wrong['child']['seconds'] += 1
        with self.assertRaises(ValueError):
            audit(root, wrong)
        p = root/'candidate.json'; raw = p.read_bytes(); self.assertTrue(raw)
        p.write_bytes(raw.replace(b'propose_error', b'propose_xxxxx', 1))
        with self.assertRaises(ValueError):
            audit(root, r)
        root, r = self.run_supervised()
        p = root/'artifacts/package/forest.bin'; raw = bytearray(p.read_bytes()); raw[-1] ^= 1; p.write_bytes(raw)
        with self.assertRaises(ValueError):
            audit(root, r)

    def test_path_and_storage_member_refusals(self):
        for name in ('/data1/Kane/MOE/../outside-control', '/tmp/not-authorized'):
            with self.assertRaises(ValueError):
                supervise(name, spec('half', 'dense'), enabled=True, seconds=10)
        source_root, source, _ = self.cases['half']
        root = self.path('member-cap-')
        r = execute(root, source_root, source, method('dense'), 'a'*64, ROOT,
                    work=work(), enabled=True, member_limit=1)
        self.assertEqual(r['status'], 'ERROR')
        self.assertIn('member reserve', r['error'])
        self.assertFalse((root/'terminal.json').exists())


if __name__ == '__main__':
    unittest.main()
