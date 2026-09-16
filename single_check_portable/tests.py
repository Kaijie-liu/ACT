"""V2/reference/V3 differential and single-check trust/budget controls."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import time
import unittest
from unittest.mock import patch

from cached_portable.tests import case, check, repin
from cached_portable.pack import pack as v2_pack
from scripts.test_general_evidence import checked
from portable_proof.runtime import original_bytes, digest
from single_check_portable.pack import pack
from single_check_portable.execution import supervise, audit_outer, read, verdict, STAGES
from single_check_portable.structure import check_result

ROOT = Path(__file__).resolve().parents[1]
OBSERVATIONS = []


class SingleCheckControls(unittest.TestCase):
    def test_pack_never_calls_math_or_reads_old_result(self):
        with case() as (root, source, a):
            (source/'independent.json').write_text('not a result')
            (source/'precheck.json').write_text('not a result')
            with patch('moe_evidence.checker.check_manifest', side_effect=AssertionError('precheck invoked')), \
                 patch('exact_matrix_cache.checker.check_manifest', side_effect=AssertionError('cached precheck invoked')):
                info = pack(source, a[1], root/'bundle', enabled=True)
            meta = read(root/'bundle/bundle.json')
            self.assertNotIn('expected_result', meta)
            self.assertNotIn('independent.json', meta['logical_files'])
            self.assertFalse(info['mathematical_precheck_executed']); self.assertIsNone(info['checked_result'])
            self.assertNotIn('precheck', STAGES)
            with self.assertRaises(TypeError): pack(source, a[1], root/'other', result={'status': 'SAFE'})

    def test_full_exact_differential_v2_v3_cache_on_off(self):
        for e, c, base, partial in ((2, 2, 3, False), (3, 4, 3, True), (4, 3, -2, False)):
            with case(e, c, base, partial) as (root, source, a):
                expected = checked(a)
                old = v2_pack(source, root/'v2', expected, enabled=True)
                p = check(root/'v2', old); self.assertEqual(p.returncode, 0, p.stderr)
                self.assertEqual(json.loads(p.stdout)['result'], expected)
                for enabled in (False, True):
                    b = root/str(enabled); info = pack(source, a[1], b, enabled=enabled)
                    p = check(b, info); self.assertEqual(p.returncode, 0, p.stderr)
                    result = json.loads(p.stdout)
                    self.assertEqual(result['result'], expected)
                    self.assertEqual(result['cache']['enabled'], enabled)
                    self.assertEqual(result['decision'], 'PACK_ONLY_THEN_ISOLATED_FULL_CHECK')
                    check_result(result['result'], read(b/'bundle.json'))

    def test_missing_and_unresolved_stay_unknown(self):
        for what in ('missing', 'unresolved'):
            with case(2, 3, partial=False) as (root, source, a):
                if what == 'missing':
                    a[0]['supports']['s0_1_p1_lo'].update(status='UNAVAILABLE', certificate=None)
                    a[0]['obligations'][1].update(weighted_status='RANGE_UNAVAILABLE', weighted=None, certificate=None)
                else: a[0]['routes']['exact'] = False
                (source/'manifest.json').write_bytes(original_bytes(a[0]))
                expected = checked(a); self.assertTrue(expected['status'].startswith('UNKNOWN'))
                info = pack(source, a[1], root/'b', enabled=True)
                p = check(root/'b', info); self.assertEqual(p.returncode, 0, p.stderr)
                result = json.loads(p.stdout)['result']; self.assertEqual(result, expected)
                check_result(result, read(root/'b/bundle.json'))

    def test_bad_proofs_pack_but_sole_checker_rejects(self):
        for kind in ('missing', 'property', 'source', 'dual', 'plane', 'range', 'order', 'threshold'):
            with case(3, 3, partial=False) as (root, source, a):
                m = a[0]; row = m['obligations'][1]; ref = None
                if kind == 'missing': m['obligations'].pop()
                elif kind == 'property': row['property']['constant'] = 999
                elif kind == 'source': m['supports']['s0_1_p1_lo']['source_sha256'] = 'wrong'
                elif kind == 'dual': ref = row['certificate']; a[2][ref['file']]['claimed_lower_bound'] = '99999'
                elif kind == 'plane': ref = row['weighted']; a[2][ref['file']]['lp']['c'][0] = 999
                elif kind == 'range': row['gate_bounds'] = ['3/4', '1']
                elif kind == 'order': m['contexts']['s0_1']['expert_order'] = [1, 0]
                else: m['positive_threshold'] = 0
                if ref:
                    raw = original_bytes(a[2][ref['file']]); (source/ref['file']).write_bytes(raw); ref['sha256'] = digest(raw)
                (source/'manifest.json').write_bytes(original_bytes(m))
                info = pack(source, a[1], root/'bundle', enabled=True)
                p = check(root/'bundle', info)
                self.assertNotEqual(p.returncode, 0, kind)

    def test_expected_result_injection_and_stale_request_rejected(self):
        with case() as (root, source, a):
            b = root/'bundle'; info = pack(source, a[1], b, enabled=True)
            injected = repin(b, info, lambda m: m.update(expected_result=checked(a)))
            self.assertNotEqual(check(b, injected).returncode, 0)
            req = copy.deepcopy(a[1]); req['epsilon'] = .1
            with self.assertRaises(ValueError): pack(source, req, root/'wrong')
            with self.assertRaises(FileExistsError): pack(source, a[1], b)

    def test_relocated_sole_check_without_source(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            moved = Path(tmp)/'only portable evidence'
            with case(3, 4, partial=True) as (root, source, a):
                expected = checked(a); info = pack(source, a[1], root/'b', enabled=True)
                shutil.copytree(root/'b', moved)
            p = check(moved, info); self.assertEqual(p.returncode, 0, p.stderr)
            result = json.loads(p.stdout)
            self.assertEqual(result['result'], expected)
            self.assertTrue(result['isolated']); self.assertFalse(result['solver_imported'])
            OBSERVATIONS.append({'control': 'relocated_analytic', 'packing': info,
                                 'check_seconds': result['check_seconds'], 'cache': result['cache']})

    def test_one_adapter_invocation_and_immutable_manifest(self):
        with case() as (root, source, a):
            b = root/'b'; info = pack(source, a[1], b, enabled=True)
            # Pin a control-only assertion around the unchanged complete checker adapter.
            p = b/'code/exact_matrix_cache/checker.py'
            with p.open('a') as f:
                f.write('\n_saved=check_manifest\n_calls=0\ndef check_manifest(*a, **kw):\n global _calls\n _calls+=1\n assert _calls==1\n return _saved(*a, **kw)\n')
            info = repin(b, info, lambda m: m['files'].update({'code/exact_matrix_cache/checker.py': digest(p.read_bytes())}))
            response = check(b, info); self.assertEqual(response.returncode, 0, response.stderr)
            self.assertEqual(json.loads(response.stdout)['result'], checked(a))
            self.assertEqual(digest((source/'manifest.json').read_bytes()), read(b/'bundle.json')['source_manifest_sha256'])

    def test_deadline_after_math_before_response_is_not_accepted(self):
        with case() as (root, source, a):
            b = root/'bundle'; info = pack(source, a[1], b, enabled=True)
            self.assertEqual(check(b, info, '--deadline-monotonic', str(time.monotonic()-1)).returncode, 3)
            p = b/'code/runtime.py'
            with p.open('a') as f:
                f.write('\n_saved=verify\ndef verify(*a, **kw):\n r=_saved(*a, **kw)\n import time\n time.sleep(.3)\n kw["tick"]()\n return r\n')
            info = repin(b, info, lambda m: m['files'].update({'code/runtime.py': digest(p.read_bytes())}))
            result = check(b, info, '--timeout-seconds', '.15')
            self.assertEqual(result.returncode, 3); self.assertNotIn('CHECKED_CONDITIONAL', result.stdout)

    def test_outer_shared_clock_no_precheck_file_and_no_late_promotion(self):
        for enabled in (False, True):
            with case() as (root, source, a):
                started = time.monotonic()-270
                result = supervise(source, a[1], root/'outer', started=started, enabled=enabled)
                self.assertEqual(result['status'], 'CHECKED_CONDITIONAL', result)
                self.assertEqual(audit_outer(root/'outer'), result)
                tail = root/'outer/tail'; terminal = read(tail/'candidate.json')
                self.assertFalse((tail/'precheck.json').exists()); self.assertFalse((tail/'precheck.log').exists())
                self.assertEqual([s['name'] for s in terminal['stages']], ['package', 'check'])
                self.assertLess(terminal['stages'][0]['allowed_seconds'], 28)
                self.assertLess(result['observed_seconds'], 300)
                self.assertEqual(read(tail/'check.log')['deadline_monotonic'], started+298)
                OBSERVATIONS.append({'control': 'analytic_inherited_budget', 'cache_enabled': enabled,
                                     'simulated_prior_seconds': 270, 'outer': result, 'stages': terminal['stages']})
        stages = [{'name': k, 'state': 'COMPLETED'} for k in STAGES]
        stages[-1]['state'] = 'TIMEOUT'
        self.assertEqual(verdict(stages, None, 299, None), ('TIMEOUT', False))

    def test_pack_only_or_bad_evidence_cannot_admit_positive(self):
        with case() as (root, source, a):
            result = supervise(source, a[1], root/'expired', started=time.monotonic()-299, enabled=True)
            self.assertEqual(result['status'], 'TIMEOUT'); self.assertEqual(audit_outer(root/'expired'), result)
            a[0]['obligations'].pop(); (source/'manifest.json').write_bytes(original_bytes(a[0]))
            result = supervise(source, a[1], root/'invalid', started=time.monotonic(), enabled=True)
            self.assertEqual(result['status'], 'ERROR'); self.assertFalse(result['complete_independent_check'])
            self.assertEqual(audit_outer(root/'invalid'), result)

    def test_result_structure_tampering_rejected_without_precheck_oracle(self):
        with case() as (root, source, a):
            pack(source, a[1], root/'b'); meta = read(root/'b/bundle.json'); expected = checked(a)
            for kind in ('missing', 'counts', 'bound', 'scope', 'minimum'):
                result = copy.deepcopy(expected)
                if kind == 'missing': result['obligations'].pop()
                elif kind == 'counts': result['positive_obligations'] = 0
                elif kind == 'bound': result['obligations'][0]['lower_bound'] = '-1'
                elif kind == 'scope': result['route_pairs'] = []
                else: result['minimum_lower_bound'] = '9999'
                with self.assertRaises(ValueError, msg=kind): check_result(result, meta)


if __name__ == '__main__': unittest.main()
