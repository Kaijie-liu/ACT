"""Identity/alias/differential gates before any finite cost comparison.

Inherited controls are rebound in PRIVATE function globals to the new parser
and checker. They execute the same mutations/predicates, not the old cache.
"""
import copy
from fractions import Fraction as F
import shutil
import subprocess
import unittest
from unittest.mock import patch

from parsed_source_reuse import tests as prior
from parsed_source_reuse.cache import scope_key
from parsed_source_reuse.check import clone
from readonly_source.cache import SourceParser
from readonly_source.check import check
from readonly_source.view import ReadOnlyFraction, ReadOnlySequence, seal


class ParserControls(prior.ParserControls):
    def parser(self, **kw):
        return SourceParser(self.scope, enabled=kw.pop('enabled', True),
                            tick=kw.pop('tick', lambda: None), **kw)

    def test_container_alias_and_fraction_slots_cannot_poison(self):
        p = self.parser(); value = p.unpack(self.state, scope=self.scope)
        h, ci, bi = value
        operations = [lambda: h.__setitem__('frame_id', 999),
                      lambda: h['Gc'][0].__setitem__(0, F(777)),
                      lambda: h['Auc'].clear(), lambda: ci.append('bad'),
                      lambda: setattr(h['Gc'][0][0], '_numerator', 999),
                      lambda: setattr(h['c'][0], '_denominator', 0),
                      lambda: delattr(h['c'][0], '_numerator'),
                      lambda: setattr(ci, 'payload', ['bad'])]
        for op in operations:
            with self.assertRaises((TypeError, AttributeError)): op()
        self.assertIs(p.unpack(self.state, scope=self.scope), value)
        self.assertEqual(value, prior.unpack(self.state))

    def test_hash_collision_refused(self):
        p = self.parser()
        with patch.dict(SourceParser.unpack.__globals__, digest=lambda _: 'same'):
            p.unpack(self.state, scope=self.scope)
            with self.assertRaises(ValueError): p.unpack(prior.specimen(2), scope=self.scope)

    def test_wrong_scope_and_transplanted_entry(self):
        p = self.parser(); p.unpack(self.state, scope=self.scope)
        other = dict(self.scope, invocation='other')
        with self.assertRaises(ValueError): p.unpack(self.state, scope=other)
        q = SourceParser(other, enabled=True, tick=lambda: None)
        key, value = next(iter(p._SourceParser__items.items()))
        q._SourceParser__items[(scope_key(other), key[1], key[2])] = value
        with self.assertRaises(ValueError): q.unpack(self.state, scope=other)

    def test_detached_slices_concatenation_and_original_parse(self):
        raw = prior.unpack(self.state); frozen, _ = seal(raw, lambda: None)
        # deepcopy(Fraction) returns itself; unpack also borrows input factor IDs.
        # Reparse an independent JSON source, not the already parsed object.
        expected = prior.unpack(copy.deepcopy(self.state))
        raw[0]['c'][0]._numerator = 999; raw[0]['Gc'][0].clear(); raw[1].append('bad')
        self.assertEqual(frozen, expected)
        for detached in (frozen[0]['c'][:], frozen[1] + ['new'], ['new'] + frozen[1]):
            self.assertIs(type(detached), list); detached.clear()
        projection = frozen[0]['c'][:]; projection[0] -= F(1, 13)
        self.assertEqual(frozen, expected)

    def test_fraction_arithmetic_comparison_and_derived_aliases(self):
        for q in (F(0), F(1, 3), F(-2, 7), F(2**200+1, 7**43)):
            x = ReadOnlyFraction(q)
            self.assertEqual(hash(x), hash(q)); self.assertEqual(F(x), q)
            self.assertEqual(str(x), str(q)); self.assertEqual(abs(x), abs(q))
            for y in (F(-3, 11), F(2), 3):
                for op in (lambda a,b:a+b, lambda a,b:a-b, lambda a,b:a*b,
                           lambda a,b:a/b, lambda a,b:a < b, lambda a,b:a == b):
                    self.assertEqual(op(x, y), op(q, y))
                    if q: self.assertEqual(op(y, x), op(y, q))
            self.assertEqual(sum([x, x]), q*2)
            derived = x + F(1)
            if type(derived) is F: derived._numerator = 123
            self.assertEqual(x, q)

    def test_sequence_comparisons_both_directions_and_nested_mapping(self):
        for values in ([], [F(1,3)], [{0:F(-1,7)}, {}]):
            view, _ = seal(values, lambda: None)
            self.assertTrue(view == values); self.assertTrue(values == view)
            self.assertFalse(view != values); self.assertFalse(values != view)
            self.assertEqual([view], [values]); self.assertEqual({'a':view}, {'a':values})
            self.assertEqual(tuple(view), tuple(values))
        self.assertFalse(ReadOnlySequence([1]) == [1,2])

    def test_readonly_no_cache_differential_and_identity(self):
        off = self.parser(enabled=False); on = self.parser()
        a = off.unpack(self.state, scope=self.scope); b = on.unpack(self.state, scope=self.scope)
        self.assertEqual(a, b); self.assertIs(on.unpack(self.state, scope=self.scope), b)
        a[0]['c'][0]._numerator = 444
        self.assertEqual(on.unpack(self.state, scope=self.scope), prior.unpack(self.state))
        self.assertEqual(on.stats()['policy'], 'READONLY_EXACT_SOURCE_VIEW_R1')

    def test_reentrant_lookup_fails_closed(self):
        armed = [False]; p = None
        def tick():
            if armed[0]: p.unpack(self.state, scope=self.scope)
        p = self.parser(tick=tick); armed[0] = True
        with self.assertRaises(ValueError): p.unpack(self.state, scope=self.scope)
        armed[0] = False
        self.assertEqual(p.unpack(self.state, scope=self.scope), prior.unpack(self.state))


class CheckerControls(prior.CheckerControls):
    def test_relocated_fresh_python_S_no_solver_or_producer(self):
        target = self.root/'readonly_relocated'; shutil.copytree(self.root/'ties', target)
        code = '''import sys,time
from pathlib import Path
def forbid(event,args):
    if event=='import' and (args[0].split('.')[0] in ('torch','numpy','scipy','act','highspy') or args[0] in ('residual_proof.build','checked_route_frontier.build','shared_route_residual.propose','source_construction_lab.fixtures')):raise ImportError(args[0])
sys.addaudithook(forbid)
from scoped_proof.io import load
from source_enclosure.format import identity
from readonly_source.check import check
p=Path(sys.argv[1]);doc=load(p/'source.json');b=load(p/'bundle.json');results=[]
for mode in (False,True):results.append(check(doc,b,invocation='ties',expected_source_sha256=identity(doc),deadline=time.monotonic()+30,enabled=mode))
assert results[0]==results[1] and not results[1]['complete_output_positive_proof']
'''
        out = subprocess.run([prior.PYTHON, '-S', '-c', code, str(target)], cwd=prior.ROOT,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)


for name, fn in vars(prior.CheckerControls).items():
    if name.startswith('test_') and name not in vars(CheckerControls):
        replacements = {k:v for k,v in {'check':check, 'SourceParser':SourceParser}.items() if k in fn.__globals__}
        setattr(CheckerControls, name, clone(fn, replacements))


if __name__ == '__main__': unittest.main()
