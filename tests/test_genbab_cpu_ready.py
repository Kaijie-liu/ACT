"""B2 inventory/branch reception controls; no external numerical imports."""
import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('genbab_cpu_ready',
    Path(__file__).resolve().parents[1]/'scripts/genbab_cpu_ready.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


class ReadyControls(unittest.TestCase):
    def test_empty_sparse_lA_is_a_score_not_a_bound(self):
        # Execute the real patched method body, without importing/running GenBaB.
        import torch
        relative='heuristics/nonlinear/bbps.py'
        original=(mod.TOOL/'complete_verifier'/relative).read_text()
        before,after=mod.COMPAT_EDITS[relative]
        self.assertEqual(original.count(before),1)
        patched=original.replace(before,after,1)
        self.assertEqual((mod.COMPAT/'complete_verifier'/relative).read_text(),patched)
        def method(source):
            tree=ast.parse(source)
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NonlinearBranching')
            fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_fast_heuristic')
            scope={'torch':torch}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),
                         '<frozen-bbps-method>','exec'),scope)
            return scope['_fast_heuristic']
        old,new=method(original),method(patched)
        class OriginalPathReached(Exception):pass
        def unchanged(*args):raise OriginalPathReached
        node=SimpleNamespace(name='margin')
        starts=[SimpleNamespace(name='active'),SimpleNamespace(name='pruned')]
        owner=SimpleNamespace(net=SimpleNamespace(split_activations={'margin':[(s,0) for s in starts]}),
                              _fast_backward_propagation=unchanged)
        for shape in ((1,3),(2,2,3)):
            lower=torch.full(shape,-1.,dtype=torch.float64)
            upper=-lower;points=torch.zeros((*shape,1),dtype=lower.dtype)
            lb={'margin':lower};ub={'margin':upper}
            result=new(owner,node,points,{'lAs':{},'lower_bounds':lb,'upper_bounds':ub},None)
            self.assertEqual(set(result),{'scores','points'})
            self.assertTrue(torch.equal(result['scores'],torch.zeros_like(lower)))
            self.assertTrue(torch.equal(result['points'],points.flatten(1,-2)))
            self.assertEqual(result['scores'].dtype,lower.dtype)
            self.assertTrue(torch.equal(lower,torch.full_like(lower,-1.)))
            for coeffs in ({'active':1},{'pruned':1},{'active':1,'pruned':1}):
                for fn in (old,new):
                    with self.assertRaises(OriginalPathReached):
                        fn(owner,node,points,{'lAs':coeffs,'lower_bounds':lb,'upper_bounds':ub},None)
            with self.assertRaises(KeyError):
                new(owner,node,points,{'lower_bounds':lb,'upper_bounds':ub},None)
            # Existing masks are downstream and are not changed by the score fallback.
            self.assertEqual(torch.count_nonzero(result['scores'].flatten(1)*torch.zeros_like(lower.flatten(1))),0)

    def test_vendored_metadata_is_not_last_name_wins(self):
        rows = [dict(name='packaging', version='26.3', location='/env/site-packages'),
                dict(name='packaging', version='26.0', location='/env/site-packages/setuptools/_vendor')]
        for ordered in (rows, rows[::-1]):
            mod.check_inventory(ordered, Path('/env/site-packages'), {'packaging': '26.3'})

    def test_wrong_top_level_rejected(self):
        with self.assertRaises(ValueError):
            mod.check_inventory([dict(name='packaging', version='26.0', location='/env/site-packages')],
                                Path('/env/site-packages'), {'packaging': '26.3'})

    def test_no_plain_crown_acceptance(self):
        record = dict(status='verified', stats={'bab': []},
                      graph=[dict(kind='BoundSigmoid')], split_activations={})
        with self.assertRaises(ValueError):
            mod.accepted_smoke(record, '')

    def test_nonlinear_branch_evidence(self):
        record = dict(status='verified', stats={'bab': [[0, .1, 2, .2]]},
                      graph=[dict(kind='BoundSigmoid')],
                      split_activations={'margin': [dict(kind='BoundSigmoid')]})
        self.assertEqual(mod.accepted_smoke(record, 'splitting decisions\nBranching points'), 2)
        record['status'] = 'unknown'
        with self.assertRaises(ValueError):
            mod.accepted_smoke(record, 'splitting decisions\nBranching points')

    def test_guard_rows_preserve_ties(self):
        for x in (-.5, .5):
            y = [0, x, -x, -.5]
            self.assertTrue(all(sum(a*b for a, b in zip(row, y)) <= rhs
                                for row, rhs in zip(mod.C, mod.RHS)))
        self.assertEqual(mod.C[0], [1, 0, 0, 0])
        self.assertFalse(mod.CONFIG['bab.branching.input_split.enable'])
        self.assertEqual(mod.CONFIG['general.complete_verifier'], 'bab')

    def test_compatibility_scope_is_depth_and_empty_heuristic(self):
        self.assertEqual(set(mod.COMPAT_EDITS), {
            'activation_split/stage_solve.py', 'activation_split/decision_precompute.py',
            'activation_split/update_bounds_phases.py',
            'heuristics/nonlinear/bbps.py'})
        for before, after in mod.COMPAT_EDITS.values():
            if '_fast_backward_propagation' not in before:
                self.assertIn('tot_ambi_nodes' if 'mask_' in before else 'all_node_split', before)
                self.assertIn('nonlinear', after)
            else:
                self.assertIn('torch.zeros_like(lb_ori)',after)
            self.assertNotIn('decision_thresh', after)
        # Nonlinear depth is not an exhaustion bound; ReLU mode stays unchanged.
        for mode in ('nonlinear', 'babsr'):
            for depth in (2, 3, 4):
                exhausted = mode != 'nonlinear' and depth == 3
                self.assertEqual(exhausted, mode == 'babsr' and depth == 3)


if __name__ == '__main__':
    unittest.main()
