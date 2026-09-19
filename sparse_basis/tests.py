"""Synthetic sparse scale and native mapping controls; zero real network queries."""
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from single_check_portable.execution import ROOT, ACT, read, save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity, check
from lp_sandwich.tests import fixture, csr
from exact_basis.tests import case as small_case
from sparse_basis.engine import (POLICY, Budget, Limit, Singular, manifest, propose, eliminate)
from sparse_basis.native import capture, map_capture

OBSERVATIONS = []
ARTIFACT_ROOT = None


def statement(lp):
    s = deepcopy(fixture()['statement'])
    s['lp_sha256'] = identity(lp)
    return s


def hints(lp, basic, anchors, point=None):
    s = statement(lp)
    c = {'lp_sha256': identity(lp), 'statement_sha256': identity(s),
         'x': point if point is not None else [0] * len(lp['c'])}
    h = manifest(lp, s, c, basic, anchors, deadline=time.monotonic()+30)
    return s, c, h


def generate(lp, s, c, h, seconds=30):
    return propose(lp, s, c, h, identity(s), identity(h), deadline=time.monotonic()+seconds)


def stats():
    return dict.fromkeys(('peak_active_nnz', 'peak_live_nnz', 'peak_heap_entries', 'fill_insertions',
                         'heap_pops', 'heap_rebuilds', 'pivots', 'pivot_row_candidates',
                         'singleton_column_pivots', 'row_updates'), 0)


def col(kind, i):
    return {'kind': kind, 'index': i}


def zero(kind, i):
    return {'column': col(kind, i), 'at': 'zero'}


def sparse_rows(rows, n):
    ptr, ix, data = [0], [], []
    for row in rows:
        for j, v in row:
            ix.append(j)
            data.append(v)
        ptr.append(len(ix))
    return {'shape': [len(ptr)-1, n], 'indptr': ptr, 'indices': ix, 'data': data}


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ARTIFACT_ROOT
        ARTIFACT_ROOT = Path(tempfile.mkdtemp(prefix='sparse_basis_controls_', dir=ROOT/'data/moe/results'))
        cls.root = ARTIFACT_ROOT

    def native_case(self, lp, name):
        s = statement(lp)
        r = capture(lp, s, self.root/name, deadline=time.monotonic()+60)
        m = map_capture(lp, s, r, identity(r), deadline=time.monotonic()+60)
        save_new(self.root/name/'mapping.json',m)
        self.assertEqual(m['status'], 'MAPPED_HINT_ONLY', m)
        p = generate(lp, s, m['candidate'], m['hint'], 60)
        save_new(self.root/name/'proposal.json',p)
        self.assertEqual(p['status'], 'CANDIDATE_ONLY', p['error'])
        save_new(self.root/name/'bundle.json',p['bundle'])
        save_new(self.root/name/'check_reference.json',check(p['bundle'],identity(s)))
        return s, r, m, p

    def test_existing_small_control_and_positive_slack(self):
        lp, _, _, _ = small_case()
        s, r, m, p = self.native_case(lp, 'small')
        d = check(p['bundle'], identity(s))
        self.assertEqual(d['primal_status'], 'EXACT_FEASIBLE')
        self.assertEqual(p['row_residuals']['A_nonnegative_required'], ['1'])
        self.assertEqual(p['row_residuals']['E_fixed_zero_required'], ['0'])
        self.assertFalse(p['feasibility_certified'])
        self.assertTrue((self.root/'small/raw_native.json').exists())
        with self.assertRaises(FileExistsError):
            capture(lp, s, self.root/'small', deadline=time.monotonic()+10)
        for field, value in [('version', 'other'), ('readback_after', {}), ('policy', {})]:
            bad = deepcopy(r)
            bad[field] = value
            with self.assertRaises(ValueError):
                map_capture(lp, s, bad, identity(bad), deadline=time.monotonic()+10)
        bad = deepcopy(r)
        bad['row_status'][0] = 'kZero'
        self.assertEqual(map_capture(lp, s, bad, identity(bad), deadline=time.monotonic()+10)['status'], 'UNSUPPORTED_MAPPING')

    def test_actual_redundant_equalities_keep_all_rows(self):
        lp = {'matrix_format': 'csr_v1', 'c': [-1,0], 'offset': 0, 'lower': [0,0], 'upper': [1,1],
              'E': csr([[1,1],[2,2]],2), 'h': [1,2], 'A': csr([],2), 'b': []}
        s, r, m, p = self.native_case(lp, 'redundant')
        self.assertIn('kBasic', r['row_status'])
        self.assertTrue(any(x['kind']=='E_residual' for x in m['hint']['basic_columns']))
        self.assertEqual(len(m['hint']['rows']), 2)
        self.assertEqual(p['row_residuals']['E_fixed_zero_required'], ['0','0'])
        self.assertEqual(check(p['bundle'], identity(s))['primal_status'], 'EXACT_FEASIBLE')
        OBSERVATIONS.append({'case': 'native_redundant_E', 'row_status': r['row_status'],
                             'result': check(p['bundle'], identity(s)), 'stats': p['stats']})

    def test_float_collapsed_distinct_equalities_still_rejected(self):
        # Both RHS values convert to the same binary64. Original rational LP is
        # inconsistent; the capture is only a hint and must not erase that fact.
        lp = {'matrix_format': 'csr_v1', 'c': [-1], 'offset': 0, 'lower': [0], 'upper': [1],
              'E': csr([[1],[1]],1), 'h': ['1/3',str(F(1,3)+F(1,2**60))], 'A': csr([],1), 'b': []}
        s, r, m, p = self.native_case(lp, 'float_collapsed')
        self.assertEqual(r['submitted']['rows'][0]['upper'], r['submitted']['rows'][1]['upper'])
        d = check(p['bundle'], identity(s))
        self.assertEqual(d['primal_status'], 'NOT_EXACTLY_FEASIBLE')
        self.assertIsNone(d['upper_bound'])
        self.assertTrue(any(F(x) for x in p['row_residuals']['E_fixed_zero_required']))
        self.assertFalse(d['network_UNSAFE'])
        OBSERVATIONS.append({'case':'native_float_collapsed_distinct_E', 'primal_status':d['primal_status'],
                             'upper_bound':d['upper_bound'], 'residuals':p['row_residuals']})

    def test_full_checker_rejects_nonzero_basic_equality_residual(self):
        lp = {'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[1],
              'E':csr([[3]],1),'h':[1],'A':csr([],1),'b':[]}
        s,c,h = hints(lp, [col('E_residual',0)], [{'column':col('x',0),'at':'lower'}])
        p = generate(lp,s,c,h)
        self.assertEqual(p['status'],'CANDIDATE_ONLY')
        self.assertEqual(p['row_residuals']['E_fixed_zero_required'],['1'])
        d=check(p['bundle'],identity(s))
        self.assertEqual(d['primal_status'],'NOT_EXACTLY_FEASIBLE')
        self.assertIsNone(d['upper_bound'])

    def test_missing_duplicate_rows_and_illegal_coordinates(self):
        lp,_,_,_=small_case()
        s,c,h=hints(lp,[col('x',0),col('x',1)],[zero('E_residual',0),zero('A_slack',0)])
        for change in ('row_missing','row_duplicate','coordinate_missing','overlap','sign','hash','fixed_residual'):
            bad=deepcopy(h)
            if change=='row_missing':bad['rows'].pop()
            elif change=='row_duplicate':bad['rows'][1]=bad['rows'][0]
            elif change=='coordinate_missing':bad['anchors'].pop()
            elif change=='overlap':bad['basic_columns'][1]=bad['basic_columns'][0]
            elif change=='sign':bad['coordinates']='NEGATIVE_NATIVE_ROW_VARIABLE'
            elif change=='hash':bad['lp_sha256']='0'*64
            else:bad['anchors'][0]['at']='candidate'
            self.assertEqual(generate(lp,s,c,bad)['status'],'ERROR',change)
        bad=deepcopy(c);bad['statement_sha256']='f'*64
        self.assertEqual(generate(lp,s,bad,h)['status'],'ERROR')

    def test_differential_with_old_exact_elimination(self):
        from exact_basis.propose import solve
        from exact_primal.propose import Budget as OldBudget
        rng=random.Random(721)
        for n in range(2,10):
            mat=[[F(rng.randint(-3,3)) for j in range(n)] for i in range(n)]
            for i in range(n):mat[i][i]=1+sum(abs(v) for j,v in enumerate(mat[i]) if i!=j)
            x=[F(i+1,7) for i in range(n)]
            sys=[({j:v for j,v in enumerate(row) if v},sum(v*z for v,z in zip(row,x))) for row in mat]
            ref=solve(deepcopy(sys),OldBudget(time.monotonic()+10),{'peak_elimination_nnz':0,'fill_in_insertions':0})
            st=stats();actual=eliminate(sys,Budget(time.monotonic()+10),st)
            self.assertEqual(actual,ref);self.assertEqual(actual,x)

    def test_fill_in_and_no_truncation(self):
        system=[({0:F(1),1:F(1)},F(2)),({0:F(1),2:F(1)},F(2)),({1:F(1),2:F(1)},F(2))]
        st=stats();out=eliminate(deepcopy(system),Budget(time.monotonic()+5),st)
        self.assertEqual(out,[1,1,1]);self.assertGreater(st['fill_insertions'],0)
        with patch.dict(POLICY,fill_insertions=0):
            with self.assertRaises(Limit):eliminate(deepcopy(system),Budget(time.monotonic()+5),stats())
        with patch.dict(POLICY,live_nnz=2):
            with self.assertRaises(Limit):eliminate(deepcopy(system),Budget(time.monotonic()+5),stats())

    def test_singular_is_unresolved_not_infeasible(self):
        lp={'matrix_format':'csr_v1','c':[0,0],'offset':0,'lower':[0,0],'upper':[1,1],
            'E':csr([[1,1],[2,2]],2),'h':[1,2],'A':csr([],2),'b':[]}
        s,c,h=hints(lp,[col('x',0),col('x',1)],[zero('E_residual',0),zero('E_residual',1)])
        p=generate(lp,s,c,h)
        self.assertEqual(p['status'],'UNRESOLVED_SINGULAR_BASIS')
        self.assertIsNone(p['bundle']);self.assertFalse(p['network_UNSAFE'])

    def test_fixed_box_empty_basis_and_exact_binary_input(self):
        lp={'matrix_format':'csr_v1','c':[0.1],'offset':0,'lower':[0.2],'upper':[0.2],
            'E':csr([],1),'h':[],'A':csr([],1),'b':[]}
        s,c,h=hints(lp,[],[{'column':col('x',0),'at':'upper'}])
        p=generate(lp,s,c,h)
        self.assertEqual(p['status'],'CANDIDATE_ONLY')
        self.assertEqual(F(p['bundle']['primal']['claimed_objective']),F.from_float(.1)*F.from_float(.2))
        self.assertEqual(check(p['bundle'],identity(s))['primal_status'],'EXACT_FEASIBLE')

    def test_expired_limits_and_bit_growth(self):
        lp,_,_,_=small_case()
        s,c,h=hints(lp,[col('x',0),col('x',1)],[zero('E_residual',0),zero('A_slack',0)])
        self.assertEqual(generate(lp,s,c,h,-1)['status'],'TIMEOUT')
        for policy in ({'variables':1},{'operations':1},{'max_bits':1}):
            with patch.dict(POLICY,policy):self.assertEqual(generate(lp,s,c,h)['status'],'LIMIT')
        with self.assertRaises(ValueError):Budget(time.monotonic()+301)
        with patch.dict(POLICY,max_bits=8):
            b=Budget(time.monotonic()+1)
            with self.assertRaises(Limit):b.value(F(1,257))

    def test_growth_limit_during_elimination(self):
        lp={'matrix_format':'csr_v1','c':[0,0],'offset':0,'lower':[0,0],'upper':[1,1],
            'E':csr([[3,1],[1,3]],2),'h':[1,0],'A':csr([],2),'b':[]}
        s,c,h=hints(lp,[col('x',0),col('x',1)],[zero('E_residual',0),zero('E_residual',1)])
        with patch.dict(POLICY,max_bits=3):
            result=generate(lp,s,c,h)
        self.assertEqual(result['status'],'LIMIT')
        self.assertEqual(result['error'],'rational bit budget')
        self.assertGreater(result['stats']['pivots'],0)
        self.assertIsNone(result['bundle'])

    def test_permuted_columns_and_mixed_row_slack(self):
        lp={'matrix_format':'csr_v1','c':[0,-1],'offset':-1,'lower':[0,0],'upper':[1,1],
            'E':csr([[1,1]],2),'h':[1],'A':csr([[1,0],[0,3]],2),'b':[1,1]}
        s,c,h=hints(lp,[col('A_slack',0),col('x',1),col('x',0)],
                    [zero('E_residual',0),zero('A_slack',1)])
        result=generate(lp,s,c,h)
        self.assertEqual(result['bundle']['primal']['x'],['2/3','1/3'])
        self.assertEqual(result['row_residuals']['A_nonnegative_required'],['1/3','0'])
        self.assertEqual(check(result['bundle'],identity(s))['upper_bound'],'-4/3')

    def test_new_native_expired_and_oversize_no_capture(self):
        lp,_,_,_=small_case();s=statement(lp)
        root=self.root/'expired_native'
        with self.assertRaises(TimeoutError):capture(lp,s,root,deadline=time.monotonic()-1)
        self.assertFalse(root.exists())
        with patch.dict(POLICY,variables=1):
            with self.assertRaises(Limit):capture(lp,s,root,deadline=time.monotonic()+10)
        self.assertFalse(root.exists())

    def test_deadline_and_heap_limit_inside_sparse_engine(self):
        sys=[({0:F(1)},F(1))]
        with self.assertRaises(TimeoutError):eliminate(deepcopy(sys),Budget(time.monotonic()-1),stats())
        with patch.dict(POLICY,heap_entries=0):
            with self.assertRaises(Limit):eliminate(deepcopy(sys),Budget(time.monotonic()+10),stats())

    def test_actual_large_native_redundant_equality_basis(self):
        n=4096
        lp={'matrix_format':'csr_v1','c':[-1]*n,'offset':0,'lower':[0]*n,'upper':[1]*n,
            'E':sparse_rows(([(i%n,3 if i<n else 6)] for i in range(2*n)),n),
            'h':[1]*n+[2]*n,'A':csr([],n),'b':[]}
        started=time.monotonic()
        s,r,m,p=self.native_case(lp,'large_native')
        self.assertTrue(any(x['kind']=='E_residual' for x in m['hint']['basic_columns']))
        d=check(p['bundle'],identity(s))
        self.assertEqual(d['primal_status'],'EXACT_FEASIBLE')
        self.assertEqual(F(d['upper_bound']),-F(n,3))
        self.assertEqual(p['stats']['pivots'],2*n)
        self.assertTrue(all(F(x)==0 for x in p['row_residuals']['E_fixed_zero_required']))
        OBSERVATIONS.append({'case':'native_4096_variables_8192_E_rows','variables':n,'rows':2*n,
                             'upper_bound':d['upper_bound'],'stats':p['stats'],'operations':p['operations'],
                             'construction_seconds':p['seconds'],'control_seconds':time.monotonic()-started,
                             'native_seconds':r['native_seconds'],'E_residual_basic_count':sum(x['kind']=='E_residual' for x in m['hint']['basic_columns'])})

    def test_target_size_sparse_triangular_and_movable_full_check(self):
        # Explicit synthetic scale: 9500 variables, 6500 rows, 356015 entries.
        # No real network arrays, point selection, or solver tuning are used.
        n,m,ne,width=9500,6500,2000,55
        rows=[[(j,3) for j in range(i,min(m,i+width))] for i in range(m)]
        lp={'matrix_format':'csr_v1','c':[-1]*m+[0]*(n-m),'offset':0,'lower':[0]*n,'upper':[1]*n,
            'E':sparse_rows(rows[:ne],n),'h':[len(r) for r in rows[:ne]],
            'A':sparse_rows(rows[ne:],n),'b':[len(r) for r in rows[ne:]]}
        started=time.monotonic();before=identity(lp)
        basic=[col('x',i) for i in range(m)]
        anchors=[{'column':col('x',i),'at':'lower'} for i in range(m,n)]
        anchors += [zero('E_residual',i) for i in range(ne)]+[zero('A_slack',i) for i in range(m-ne)]
        s,c,h=hints(lp,basic,anchors)
        p=generate(lp,s,c,h,120)
        self.assertEqual(p['status'],'CANDIDATE_ONLY',p['error'])
        self.assertEqual(identity(lp),before)
        self.assertEqual(p['stats']['pivots'],m)
        self.assertEqual(p['stats']['singleton_column_pivots'],m)
        self.assertEqual(p['stats']['row_updates'],0)
        self.assertEqual(p['stats']['fill_insertions'],0)
        self.assertGreater(p['operations'],200000)
        self.assertLess(p['operations'],POLICY['operations'])
        pack=self.root/'large_pack';pack.mkdir()
        save_new(pack/'bundle.json',p['bundle'])
        shutil.copyfile(ROOT/'lp_sandwich/check.py',pack/'verify.py')
        moved=self.root/'large_moved';shutil.copytree(pack,moved)
        begun=time.monotonic()
        result=subprocess.run([ACT,'-I','-S',str(moved/'verify.py'),str(moved/'bundle.json'),
            '--bundle-sha256',digest((moved/'bundle.json').read_bytes()),'--statement-sha256',identity(s),
            '--timeout-seconds','60'],cwd=moved,capture_output=True,text=True,check=True,timeout=65)
        d=json.loads(result.stdout)
        self.assertEqual(d['primal_status'],'EXACT_FEASIBLE')
        self.assertEqual(F(d['upper_bound']),-F(m,3))
        self.assertFalse(d['solver_or_model_imported']);self.assertFalse(d['network_UNSAFE'])
        OBSERVATIONS.append({'case':'synthetic_target_size_triangular','variables':n,'rows':m,
                             'stored_nnz':sum(len(lp[k]['data']) for k in ('E','A')),
                             'stats':p['stats'],'operations':p['operations'],'construction_seconds':p['seconds'],
                             'control_seconds':time.monotonic()-started,'isolated_check_wall_seconds':time.monotonic()-begun,
                             'upper_bound':d['upper_bound'],'moved':True,'native_calls':0,
                             'bundle_sha256':digest((moved/'bundle.json').read_bytes())})


if __name__=='__main__':unittest.main()
