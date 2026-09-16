"""Analytic relocation, identity, deadline and full-clock controls only."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.test_general_evidence import fixture, checked
from portable_proof.runtime import original_bytes, digest
from cached_portable.pack import pack
from cached_portable.execution import run, audit, supervise, audit_outer, publish, left, verdict, save_new, read, STAGES

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = []


@contextmanager
def case(*args, **kwargs):
    with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
        root = Path(tmp); source = root/'source'; source.mkdir()
        a = fixture(*args, **kwargs)
        for name, value in a[2].items(): save_new(source/name, value)
        save_new(source/'manifest.json', a[0])
        yield root, source, a


def check(bundle, info, *args):
    return subprocess.run([sys.executable, '-I', '-S', str(bundle/'verify.py'),
                           '--bundle-hash', info['bundle_sha256'], '--statement-hash', info['statement_sha256'],
                           *(args or ('--timeout-seconds', '30'))], cwd=bundle.parent,
                          capture_output=True, text=True, timeout=35)


def repin(bundle, info, update):
    meta = read(bundle/'bundle.json'); update(meta)
    (bundle/'bundle.json').write_bytes(original_bytes(meta))
    return {**info, 'bundle_sha256': digest((bundle/'bundle.json').read_bytes())}


class PortableControls(unittest.TestCase):
    def test_relocation_without_original_sources_or_checkout_imports(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            moved = Path(tmp)/'relocated proof with spaces'
            with case(3, 4, partial=True) as (root, source, a):
                expected = checked(a); info = pack(source, root/'bundle', expected, enabled=True)
                shutil.copytree(root/'bundle', moved)
            # Source/checkpoint/history no longer exists; only portable bytes remain.
            p = check(moved, info)
            self.assertEqual(p.returncode, 0, p.stderr)
            value = json.loads(p.stdout)
            self.assertEqual(value['result'], expected)
            self.assertTrue(value['isolated']); self.assertTrue(value['site_disabled'])
            self.assertFalse(value['solver_imported']); self.assertGreater(value['cache']['hits'], 0)
            self.assertEqual(value['cache']['live_entries'], 0)
            OBSERVATIONS.append({'control': 'relocated_analytic', 'experts': 3, 'classes': 4,
                                 'cache_enabled': True, 'packing': info,
                                 'check_seconds': value['check_seconds'], 'cache': value['cache'],
                                 'result_sha256': digest(original_bytes(expected)), 'no_original_source': True})

    def test_cached_uncached_and_original_all_dimensions(self):
        for e, c, base, partial in ((2, 2, 3, False), (3, 3, 3, True), (4, 4, -2, False)):
            with case(e, c, base, partial) as (root, source, a):
                expected = checked(a); hashes = []
                for enabled in (False, True):
                    bundle = root/str(enabled); info = pack(source, bundle, expected, enabled=enabled)
                    p = check(bundle, info); self.assertEqual(p.returncode, 0, p.stderr)
                    result = json.loads(p.stdout)
                    self.assertEqual(result['result'], expected)
                    self.assertEqual(result['cache']['enabled'], enabled)
                    hashes.append(info['bundle_sha256'])
                self.assertNotEqual(*hashes)

    def test_partial_proof_remains_unknown(self):
        with case(2, 3, partial=False) as (root, source, a):
            row = a[0]['obligations'][1]
            row.update(weighted_status='RANGE_UNAVAILABLE', weighted=None, certificate=None)
            a[0]['supports']['s0_1_p1_lo'].update(status='UNAVAILABLE', certificate=None)
            (source/'manifest.json').write_bytes(original_bytes(a[0]))
            expected = checked(a)
            self.assertEqual(expected['status'], 'UNKNOWN_MISSING_EVIDENCE')
            info = pack(source, root/'bundle', expected, enabled=True)
            p = check(root/'bundle', info)
            self.assertEqual(p.returncode, 0, p.stderr); self.assertEqual(json.loads(p.stdout)['result'], expected)

    def test_missing_obligation_wrong_property_source_and_request_rejected(self):
        for mutation in ('missing', 'property', 'source', 'request'):
            with case() as (root, source, a):
                expected = checked(a); m = a[0]
                if mutation == 'missing': m['obligations'].pop()
                elif mutation == 'property': m['obligations'][0]['property']['constant'] = 10
                elif mutation == 'source': m['supports']['s0_1_p1_lo']['source_sha256'] = 'other'
                else: m['request']['experts'] = 4
                (source/'manifest.json').write_bytes(original_bytes(m))
                info = pack(source, root/'bundle', expected, enabled=True)
                self.assertNotEqual(check(root/'bundle', info).returncode, 0, mutation)

    def test_code_option_inventory_and_statement_bound(self):
        for mutation in ('code', 'option', 'inventory', 'statement', 'extra_code'):
            with case() as (root, source, a):
                b = root/'bundle'; info = pack(source, b, checked(a), enabled=True)
                if mutation == 'code':
                    with (b/'code/exact_matrix_cache/cache.py').open('ab') as f: f.write(b'\n# changed\n')
                elif mutation == 'option':
                    # Original external hash must reject cache-option changes.
                    repin(b, info, lambda m: m['parser'].update(enabled=False))
                elif mutation == 'inventory':
                    info = repin(b, info, lambda m: m['files'].pop('code/exact_matrix_cache/cache.py'))
                elif mutation == 'extra_code': (b/'code/extra.py').write_text('raise RuntimeError()')
                else: info = {**info, 'statement_sha256': '0'*64}
                self.assertNotEqual(check(b, info).returncode, 0, mutation)

    def test_deadline_before_start_and_during_isolated_check(self):
        with case() as (root, source, a):
            b = root/'bundle'; info = pack(source, b, checked(a), enabled=True)
            p = check(b, info, '--deadline-monotonic', str(time.monotonic()-1))
            self.assertEqual(p.returncode, 3); self.assertNotIn('CHECKED_CONDITIONAL', p.stdout)
            p = check(b, info, '--timeout-seconds', '301')
            self.assertNotEqual(p.returncode, 0)
            # Trusted analytic delay after math, before response: no late positive.
            path = b/'code/runtime.py'
            with path.open('a') as f:
                f.write('\n_original=verify\ndef verify(*a, **kw):\n r=_original(*a, **kw)\n import time\n time.sleep(.3)\n kw["tick"]()\n return r\n')
            info = repin(b, info, lambda m: m['files'].update({'code/runtime.py': digest(path.read_bytes())}))
            p = check(b, info, '--timeout-seconds', '.15')
            self.assertEqual(p.returncode, 3); self.assertNotIn('CHECKED_CONDITIONAL', p.stdout)

    def test_isolation_rejects_solver_external_read_and_write(self):
        for command in ('import numpy', f'open({str(ROOT/"AGENTS.md")!r}).read()', 'open("forbidden", "w")'):
            with case() as (root, source, a):
                b = root/'bundle'; info = pack(source, b, checked(a), enabled=True)
                path = b/'code/runtime.py'
                # Deliberate pinned analytic probe, not arbitrary unbound executable injection.
                with path.open('a') as f:
                    f.write('\n_original=verify\ndef verify(*a, **kw):\n '+command+'\n return _original(*a, **kw)\n')
                info = repin(b, info, lambda m: m['files'].update({'code/runtime.py': digest(path.read_bytes())}))
                p = check(b, info)
                self.assertNotEqual(p.returncode, 0, command)
                self.assertFalse((root/'forbidden').exists())

    def test_pack_deadline_and_no_overwrite(self):
        with case() as (root, source, a):
            calls = [0]
            def tick():
                calls[0] += 1
                if calls[0] > 5: raise TimeoutError('control packing deadline')
            with self.assertRaises(TimeoutError): pack(source, root/'failed', checked(a), tick=tick)
            self.assertFalse((root/'failed/bundle.json').exists())
            pack(source, root/'bundle', checked(a))
            with self.assertRaises(FileExistsError): pack(source, root/'bundle', checked(a))

    def test_full_tail_inherits_upstream_cost_and_independent_audit(self):
        for enabled in (False, True):
            with case(3, 3, partial=True) as (root, source, a):
                started = time.monotonic()-270
                result = supervise(source, a[1], root/'outer', started=started, enabled=enabled)
                self.assertEqual(result['status'], 'CHECKED_CONDITIONAL', result)
                self.assertEqual(audit_outer(root/'outer'), result)
                tail = root/'outer/tail'
                t = read(tail/'candidate.json')
                self.assertGreaterEqual(t['upstream_elapsed_at_entry'], 270)
                self.assertLess(t['wall_seconds'], 300)
                self.assertEqual([s['name'] for s in t['stages']], list(STAGES))
                self.assertLess(t['stages'][0]['allowed_seconds'], 28.01)
                self.assertLess(t['stages'][2]['allowed_seconds'], t['stages'][0]['allowed_seconds'])
                self.assertGreater(t['tail_overhead_seconds'], 0)
                self.assertGreater(result['driver_exit_seconds'], read(tail/'admission.json')['observed_submission_seconds'])
                output = read(tail/'check.log')
                self.assertEqual(output['deadline_monotonic'], started+298)
                self.assertEqual(output['result'], checked(a))
                OBSERVATIONS.append({'control': 'analytic_whole_tail', 'cache_enabled': enabled,
                                     'simulated_upstream_seconds': 270, 'outer': result,
                                     'stages': t['stages'], 'tail_overhead_seconds': t['tail_overhead_seconds'],
                                     'packing': read(tail/'packing.json'), 'check_seconds': output['check_seconds']})

    def test_outer_exhaustion_and_late_files_rejected(self):
        with case() as (root, source, a):
            result = supervise(source, a[1], root/'outer', started=time.monotonic()-299, enabled=True)
            self.assertEqual(result['status'], 'TIMEOUT'); self.assertEqual(audit_outer(root/'outer'), result)
            self.assertFalse((root/'outer/tail').exists())
            # A forged late positive is not consulted when outer admission timed out.
            (root/'outer/tail').mkdir()
            save_new(root/'outer/tail/admission.json', {'status': 'CHECKED_CONDITIONAL'})
            self.assertEqual(audit_outer(root/'outer')['status'], 'TIMEOUT')

    def test_exhausted_clock_no_work_and_no_late_result_promotion(self):
        with case() as (root, source, a):
            result = run(source, a[1], root/'tail', started=time.monotonic()-299, enabled=True)
            self.assertEqual(result['status'], 'TIMEOUT'); self.assertEqual(audit(root/'tail'), result)
            self.assertFalse((root/'tail/portable').exists())
        self.assertEqual(left(0, clock=lambda: 270), 28)
        with self.assertRaises(TimeoutError): left(0, clock=lambda: 298)
        stages = [{'name': k, 'state': 'COMPLETED'} for k in STAGES]
        stages[-1]['state'] = 'TIMEOUT'
        self.assertEqual(verdict(stages, None, 299, None), ('TIMEOUT', False))
        stages[-1]['state'] = 'ERROR'
        self.assertEqual(verdict(stages, None, 299, None), ('ERROR', False))
        self.assertEqual(verdict(stages, None, 301, None), ('TIMEOUT', False))

    def test_native_hang_owned_watchdog(self):
        from evidence_cohort.run import wait_owned
        p = subprocess.Popen([sys.executable, '-I', '-S', '-c', 'import time; time.sleep(30)'], start_new_session=True)
        outcome = wait_owned(p, time.monotonic()+.1)
        self.assertTrue(outcome['killed']); self.assertIsNotNone(p.returncode)

    def test_terminal_serialization_is_charged(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            now = [297.]
            def write(p, value):
                save_new(p, value)
                if p.name == 'admission.json': now[0] = 300.01
            t = {'status': 'CHECKED_CONDITIONAL', 'complete_independent_check': True}
            r = publish(Path(tmp), t, 0, clock=lambda: now[0], write=write)
            self.assertEqual(r['status'], 'TIMEOUT'); self.assertFalse(r['complete_independent_check'])
            self.assertTrue((Path(tmp)/'publication_timeout.json').exists())

    def test_accounting_and_binding_mutations_rejected(self):
        with case(2, 2, partial=False) as (root, source, a):
            run(source, a[1], root/'tail', started=time.monotonic()-100, enabled=True)
            audit(root/'tail')
            original_t = read(root/'tail/candidate.json'); original_r = read(root/'tail/admission.json')
            for mutation in ('budget', 'cost', 'stage', 'upstream', 'admission'):
                t = copy.deepcopy(original_t); r = copy.deepcopy(original_r)
                if mutation == 'budget': t['budget_seconds'] = 600
                elif mutation == 'cost': t['tail_overhead_seconds'] = 0
                elif mutation == 'stage': t['stages'][2]['allowed_seconds'] = 300
                elif mutation == 'upstream': t['upstream_elapsed_at_entry'] = 0
                else: r['observed_submission_seconds'] = 0
                (root/'tail/candidate.json').write_bytes(original_bytes(t))
                r['candidate_sha256'] = digest((root/'tail/candidate.json').read_bytes())
                (root/'tail/admission.json').write_bytes(original_bytes(r))
                with self.assertRaises(ValueError, msg=mutation): audit(root/'tail')


if __name__ == '__main__': unittest.main()
