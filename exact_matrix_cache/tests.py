"""Poisoning, binding and exact differential controls; no real-model queries."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from act.back_end.solver.check_hz_lp_export import _entries as reference_entries
from act.back_end.solver.sparse_lp_certificate import rows as reference_rows
from moe_evidence.checker import check_manifest as reference_check
from portable_proof.runtime import original_bytes, digest
from scripts.test_general_evidence import fixture
from exact_matrix_cache.cache import MatrixCache
from exact_matrix_cache.checker import check_manifest


def matrix(values=(.25, '1/3')):
    return {'shape': [2, 3], 'data': list(values), 'indices': [0, 2], 'indptr': [0, 1, 2]}


def load_files(files):
    def load(ref):
        obj = files[ref['file']]
        if digest(original_bytes(obj)) != ref['sha256']:
            raise ValueError('logical file binding changed')
        return copy.deepcopy(obj)
    return load


def run(a, mode):
    manifest, request, files = a
    if mode == 'reference': return reference_check(manifest, request, load_files(files))
    return check_manifest(manifest, request, load_files(files), enabled=mode == 'cached')['result']


class CacheTests(unittest.TestCase):
    def test_immutable_parse_and_input_alias_changes(self):
        cache = MatrixCache('request1'); value = matrix()
        shape, entries = cache.entries(value)
        with self.assertRaises(TypeError): entries[0, 0] = Fraction(99)
        rows = list(cache.rows(value, 3))
        with self.assertRaises(TypeError): rows[0][0] = (0, Fraction(99))
        self.assertEqual(entries[0, 0], Fraction(1, 4))
        value['data'][0] = .5
        self.assertEqual(cache.entries(value)[1][0, 0], Fraction(1, 2))
        self.assertEqual(entries[0, 0], Fraction(1, 4))
        value['indices'][0] = 1
        self.assertNotIn((0, 0), cache.entries(value)[1])
        value['shape'][1] = 4
        self.assertEqual(cache.entries(value)[0], (2, 4))
        with self.assertRaises(ValueError): list(cache.rows(value, 3))
        self.assertGreater(cache.stats()['hits'], 0)

    def test_content_collision_rejects_stale_value(self):
        with patch('exact_matrix_cache.cache._digest', return_value='deliberate-collision'):
            cache = MatrixCache('request1'); cache.entries(matrix())
            with self.assertRaises(ValueError): cache.entries(matrix((.5, '1/3')))

    def test_no_cross_request_or_persistent_cache(self):
        first = MatrixCache('r1'); second = MatrixCache('r2')
        first.entries(matrix()); second.entries(matrix())
        self.assertEqual((first.stats()['hits'], second.stats()['hits']), (0, 0))
        first.clear(); self.assertEqual(first.stats()['live_entries'], 0)
        first.entries(matrix()); self.assertEqual(first.stats()['parses'], 2)
        with self.assertRaises(TypeError): MatrixCache('r1', persisted_entries={'poison': True})

    def test_eviction_and_oversize_preserve_exact_values(self):
        cache = MatrixCache('r', limits={'entries': 1, 'payload_bytes': 1024, 'cells': 10})
        for x in (.25, .5, .25):
            a = matrix((x, '1/3'))
            self.assertEqual(cache.entries(a), reference_entries(a))
        self.assertEqual(cache.stats()['evictions'], 2)
        small = MatrixCache('r', limits={'entries': 1, 'payload_bytes': 1, 'cells': 1})
        self.assertEqual(small.entries(matrix()), reference_entries(matrix()))
        self.assertEqual(small.stats()['live_entries'], 0)
        self.assertEqual(small.stats()['oversized'], 1)

    def test_malformed_data_cannot_hit_warm_cache(self):
        for what in ('shape', 'pointer', 'column', 'duplicate', 'bool', 'nan', 'infinity'):
            cache = MatrixCache('r'); cache.entries(matrix()); bad = matrix()
            if what == 'shape': bad['shape'][1] = -1
            elif what == 'pointer': bad['indptr'] = [0, 2, 1]
            elif what == 'column': bad['indices'][1] = 99
            elif what == 'duplicate': bad.update(indices=[0, 0], indptr=[0, 2, 2])
            elif what == 'bool': bad['data'][0] = True
            elif what == 'nan': bad['data'][0] = float('nan')
            else: bad['data'][0] = float('inf')
            with self.subTest(what=what), self.assertRaises((ValueError, TypeError)): cache.entries(bad)

    def test_random_canonical_csr_differential(self):
        rng = random.Random(20260916)
        cache = MatrixCache('control')
        for _ in range(80):
            nr, nc = rng.randrange(5), rng.randrange(5)
            a = {'shape': [nr, nc], 'data': [], 'indices': [], 'indptr': [0]}
            for i in range(nr):
                for j in range(nc):
                    if rng.random() < .5:
                        a['indices'].append(j)
                        a['data'].append(rng.choice([0, -1, .125, -.5, '2/3']))
                a['indptr'].append(len(a['data']))
            for _ in range(2):
                self.assertEqual(cache.entries(a), reference_entries(a))
                self.assertEqual([list(r) for r in cache.rows(a, nc)], list(reference_rows(a, nc)))

    def test_request_results_equal_all_three_modes(self):
        for e, c, base, partial in ((2, 2, 3, False), (3, 3, 3, True), (5, 4, -2, False)):
            for custom in (False, True):
                a = fixture(e, c, base, partial, custom)
                results = [run(copy.deepcopy(a), mode) for mode in ('reference', 'uncached', 'cached')]
                self.assertEqual(results[0], results[1]); self.assertEqual(results[0], results[2])
        a = fixture(); item = a[0]['supports']['s0_1_p1_lo']
        item.update(status='UNAVAILABLE', certificate=None)
        a[0]['obligations'][1].update(weighted_status='RANGE_UNAVAILABLE', weighted=None, certificate=None)
        self.assertEqual(run(a, 'reference'), run(a, 'cached'))
        self.assertEqual(run(a, 'cached')['status'], 'UNKNOWN_MISSING_EVIDENCE')

    def test_result_cache_never_skips_binding_or_dual_mutations(self):
        # Mutations are placed on a later property: its shared matrices have
        # already populated the per-request cache. Transport hashes are updated
        # so rejection cannot rely on a stale file checksum alone.
        for mutation in ('objective', 'dual_sign', 'dual_claim', 'gate', 'frame', 'source',
                         'factor_order', 'property', 'scope', 'expert', 'route', 'missing'):
            a = fixture(3, 4, partial=False); m, r, files = a
            row = m['obligations'][2]; ref = None
            if mutation == 'objective': ref = row['weighted']; files[ref['file']]['lp']['c'][0] = 100
            elif mutation == 'dual_sign': ref = row['certificate']; files[ref['file']]['inequality_dual'][0] = 1
            elif mutation == 'dual_claim': ref = row['certificate']; files[ref['file']]['claimed_lower_bound'] = '999'
            elif mutation == 'gate': row['gate_bounds'] = ['3/4', '1']
            elif mutation == 'frame':
                ref = m['contexts']['s0_2']['joint_source']; files[ref['file']]['frame_id'] = 'other'
            elif mutation == 'source':
                ref = row['weighted']; files[ref['file']]['source']['c'][0] = 99
            elif mutation == 'factor_order': m['contexts']['s0_2']['expert_order'] = [2, 0]
            elif mutation == 'property': row['property']['constant'] = 99
            elif mutation == 'scope': m['supports']['s0_1_p2_lo']['request_id'] = 'other'
            elif mutation == 'expert': m['supports']['s0_1_p2_lo']['pair'] = [1, 2]
            elif mutation == 'route': m['routes']['feasible'].pop()
            else: m['obligations'].pop()
            if ref: ref['sha256'] = digest(original_bytes(files[ref['file']]))
            for mode in ('reference', 'uncached', 'cached'):
                with self.subTest(mutation=mutation, mode=mode), self.assertRaises((ValueError, KeyError)):
                    run(copy.deepcopy(a), mode)

    def test_bad_file_hash_not_bypassed_and_no_global_patch(self):
        import act.back_end.solver.check_hz_lp_export as hz
        import act.back_end.solver.sparse_lp_certificate as sparse
        from moe_evidence import checker
        functions = (hz._entries, sparse.rows, checker.check_manifest)
        a = fixture(); original = run(a, 'reference')
        cache_result = check_manifest(a[0], a[1], load_files(a[2]))
        self.assertEqual(cache_result['result'], original)
        self.assertGreater(cache_result['cache']['hits'], 0)
        self.assertEqual(cache_result['cache']['live_entries'], 0)
        self.assertEqual((hz._entries, sparse.rows, checker.check_manifest), functions)
        a[2][a[0]['obligations'][1]['weighted']['file']]['source']['c'][0] = 100
        with self.assertRaises(ValueError): run(a, 'cached')
        fresh = fixture(); self.assertEqual(run(fresh, 'reference'), original)

    def test_timeout_during_cache_hit_and_aggregate_is_fail_closed(self):
        calls = [0]
        def tick():
            calls[0] += 1
            if calls[0] >= 4: raise TimeoutError('test deadline')
        cache = MatrixCache('r', tick=tick); cache.entries(matrix())
        with self.assertRaises(TimeoutError): cache.entries(matrix())
        calls[0] = 0
        a = fixture()
        with self.assertRaises(TimeoutError): check_manifest(a[0], a[1], load_files(a[2]), tick=tick)

    def test_stdlib_isolated_checker_control(self):
        # Prepare trusted checker code exactly as the existing portable bundle
        # does; add the opt-in parser without editing any frozen module.
        from moe_evidence.bundle import pack
        from scripts.optional_evidence_dev_contract import ROOT, save
        a = fixture(); result = run(a, 'reference')
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            root = Path(tmp); src = root/'source'; src.mkdir()
            for name, obj in a[2].items(): save(src/name, obj)
            save(src/'manifest.json', a[0]); pack(src, root/'portable', result)
            code = root/'portable/code'; dest = code/'exact_matrix_cache'; dest.mkdir()
            for name in ('cache.py', 'checker.py'):
                (dest/name).write_bytes((ROOT/'exact_matrix_cache'/name).read_bytes())
            (dest/'__init__.py').write_text('')
            # Use copied logical files and hash-checked loader, not a model.
            (code/'moe_evidence/storage.py').write_bytes((ROOT/'moe_evidence/storage.py').read_bytes())
            (code/'portable_proof').mkdir(); (code/'portable_proof/__init__.py').write_text('')
            (code/'portable_proof/runtime.py').write_bytes((ROOT/'portable_proof/runtime.py').read_bytes())
            program = ('import sys,json;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
                       'from exact_matrix_cache.checker import check_manifest;from moe_evidence.storage import loader;'
                       'p=Path(sys.argv[2]);m=json.loads((p/"manifest.json").read_text());'
                       'r=check_manifest(m,m["request"],loader(p));'
                       'assert not any(n in sys.modules for n in ("torch","numpy","scipy"));'
                       'print(json.dumps(r["result"],sort_keys=True))')
            p = subprocess.run([sys.executable, '-I', '-S', '-c', program, str(code), str(src)],
                               cwd=root, capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual(json.loads(p.stdout), result)


if __name__ == '__main__': unittest.main()
