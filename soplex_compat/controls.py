"""Analytic controls only. Persists attempts; no real LPs, no dependency installs."""
import argparse
import copy
from fractions import Fraction as F
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

from lp_sandwich.check import check, identity, rational
from soplex_compat.io import candidate_bundle, export_lp, parse_point, verify_readback

ROOT = Path(__file__).resolve().parents[1]
PREFIX = Path('/data1/Kane/MOE/envs/soplex-8.0.3')
PROBE = PREFIX / 'bin/exact_io_probe'
RAW = None
NATIVE = []


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as f: json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)


def csr(rows, n):
    data, indices, ptr = [], [], [0]
    for row in rows:
        for j, v in enumerate(row):
            if rational(v): data.append(v); indices.append(j)
        ptr.append(len(data))
    return dict(shape=[len(rows), n], data=data, indices=indices, indptr=ptr)


def lp(c=(1,), lo=(-1,), hi=(1,), a=(), b=(), e=(), h=(), offset=0):
    n=len(c)
    return dict(matrix_format='csr_v1', c=list(c), lower=list(lo), upper=list(hi),
                A=csr(a,n), b=list(b), E=csr(e,n), h=list(h), offset=offset)


def statement(p):
    return dict(schema='LP_OBLIGATION_IDENTITY_V1', request_id='1'*64,
                source_sha256='2'*64, export_sha256='3'*64, pair=[0,1],
                property_index=0, property={'q':[1,-1], 'constant':0},
                lp_sha256=identity(p), acceptance_threshold=0)


def cases():
    tiny = str(F(1,2**40))
    return [
        ('third', lp(e=[[3]], h=[1], offset=-1), '-2/3'),
        ('small_negative_coefficient', lp(a=[['-'+tiny]], b=[str(-F(tiny)/3)], offset=-1), '-2/3'),
        ('binary64_input', lp(e=[[1]], h=[0.1]), str(F.from_float(0.1))),
        ('nonterminating_coefficients', lp(c=[1,2],lo=[0,0],hi=[2,2],e=[['1/7','2/11']],h=['5/13']), '281/91'),
        ('negative_box', lp(c=[1,-1],lo=[-2,-3],hi=[-1,-2],offset='1/7'), '1/7'),
        ('fixed_and_zero_columns', lp(c=[0,1,0],lo=[0,'2/3',-1],hi=[0,'2/3',1],offset='-1/7'), '11/21'),
        ('empty_row', lp(a=[[0]],b=[1]), '-1'),
        ('redundant_equalities', lp(e=[[3],[6]],h=[1,2]), '1/3'),
        ('zero_point', lp(lo=[0],hi=[0]), '0'),
        ('infeasible', lp(e=[[0]],h=[1]), None),
    ]


class Controls(unittest.TestCase):
    def test_native_exact_io_and_relocated_original_check(self):
        for name,p,expected in cases():
            with self.subTest(name=name):
                dest=RAW/name;dest.mkdir()
                save(dest/'original.json',p)
                (dest/'input.lp').write_text(export_lp(p))
                start=time.monotonic()
                command=[str(PROBE),str(dest/'input.lp'),str(dest/'native')]
                result=subprocess.run(command,capture_output=True,text=True,timeout=20,
                                      env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
                native_seconds=time.monotonic()-start
                save(dest/'process.json',dict(command=command,returncode=result.returncode,
                     stdout=result.stdout,stderr=result.stderr,seconds=native_seconds))
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual((dest/'native.settings').read_text().split(),['1','1','2','2','-1','0','0','10'])
                for phase in ('before','after'):
                    verify_readback(p,(dest/f'native.{phase}').read_text())
                s=statement(p)
                b=candidate_bundle(p,s,identity(s),(dest/'native.point').read_text())
                # A relocated stdlib-only checker has no source model or solver dependency.
                moved=dest/'moved';moved.mkdir()
                save(moved/'bundle.json',b)
                shutil.copyfile(ROOT/'lp_sandwich/check.py',moved/'check.py')
                command=[sys.executable,'-I','-S',str(moved/'check.py'),str(moved/'bundle.json'),
                         '--bundle-sha256',sha(moved/'bundle.json'),'--statement-sha256',identity(s),
                         '--timeout-seconds','10']
                start=time.monotonic()
                proc=subprocess.run(command,cwd=moved,capture_output=True,text=True,timeout=15)
                elapsed=time.monotonic()-start
                save(dest/'check_process.json',dict(returncode=proc.returncode,stdout=proc.stdout,
                     stderr=proc.stderr,seconds=elapsed))
                self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
                checked=json.loads(proc.stdout);save(dest/'checked.json',checked)
                self.assertEqual(checked['upper_bound'],expected)
                self.assertFalse(checked['network_SAFE']);self.assertFalse(checked['network_UNSAFE'])
                self.assertTrue(checked['isolated']);self.assertTrue(checked['site_disabled'])
                self.assertFalse(checked['solver_or_model_imported'])
                self.assertIsNone(checked['lower_bound'])
                if expected is None: self.assertEqual(checked['primal_status'],'MISSING')
                else: self.assertEqual(checked['primal_status'],'EXACT_FEASIBLE')
                NATIVE.append(dict(case=name,upper_bound=checked['upper_bound'],
                    native_seconds=native_seconds,check_seconds=elapsed,
                    point_bytes=(dest/'native.point').stat().st_size,
                    bundle_bytes=(moved/'bundle.json').stat().st_size))

    def test_export_preserves_binary_and_fraction(self):
        p=lp(c=[0.1],e=[['1/7']],h=['1/3'],offset='1/17')
        text=export_lp(p)
        for value in (str(F.from_float(.1)),'1/7','1/3'):self.assertIn(value,text)
        self.assertNotIn('1/17',text) # Explicit objective translation only; checker re-adds it.

    def test_invalid_input_rejected_before_solver(self):
        for key,value in [('lower',[2]),('c',[float('nan')]),('upper',[True])]:
            p=lp();p[key]=value
            with self.assertRaises(ValueError):export_lp(p)

    def test_real_dimensions_deliberately_out_of_scope(self):
        with self.assertRaises(ValueError):export_lp(lp(c=[1]*9,lo=[0]*9,hi=[1]*9))

    def test_point_missing_duplicate_wrong_dimension_rejected(self):
        for text in ['SOPLEX_CANDIDATE_V1 1\nPOINT 1\nx0 0\n',
                     'SOPLEX_CANDIDATE_V1 1\nPOINT 2\nx0 0\nx0 1\nEND\n',
                     'SOPLEX_CANDIDATE_V1 1\nPOINT 1\nx5 0\nEND\n',
                     'SOPLEX_CANDIDATE_V1 1\nPOINT 1\nx0 NaN\nEND\n']:
            with self.assertRaises((ValueError,IndexError)):parse_point(text,1)

    def test_native_status_never_acceptance(self):
        p=lp(e=[[3]],h=[1]);s=statement(p)
        b=candidate_bundle(p,s,identity(s),'SOPLEX_CANDIDATE_V1 1\nPOINT 1\nx0 0\nEND\n')
        r=check(b,identity(s));self.assertEqual(r['primal_status'],'NOT_EXACTLY_FEASIBLE')
        self.assertIsNone(r['upper_bound'])

    def test_binding_and_property_mutation_rejected(self):
        p=lp();s=statement(p);expected=identity(s)
        for key,value in [('request_id','4'*64),('pair',[1,2]),('property_index',1),
                           ('property',{'q':[-1,1],'constant':0})]:
            bad=copy.deepcopy(s);bad[key]=value
            with self.assertRaises(ValueError):candidate_bundle(p,bad,expected,'')
        p['offset']=1
        with self.assertRaises(ValueError):candidate_bundle(p,s,expected,'')

    def test_model_readback_changes_rejected(self):
        p=lp(e=[[3]],h=[1])
        good='SOPLEX_RATIONAL_READBACK_V1\nCOLUMNS 1\nx0 1 -1 1\nROWS 1\ne0 1 1 1 x0 3\nEND\n'
        verify_readback(p,good)
        for bad in [good.replace('x0 3','x0 0'),good.replace('x0 1 -1 1','x0 -1 -1 1'),
                    good.replace('e0 1 1','e0 0 1'),good.replace('END\n',''),good+'extra\n']:
            with self.assertRaises((ValueError,StopIteration)):verify_readback(p,bad)

    def test_fractional_objective_exact_and_no_dual_claim(self):
        p=lp(c=['1/7'],offset='-1/11');s=statement(p)
        b=candidate_bundle(p,s,identity(s),'SOPLEX_CANDIDATE_V1 1\nPOINT 1\nx0 1/3\nEND\n')
        r=check(b,identity(s));self.assertEqual(r['upper_bound'],str(F(1,21)-F(1,11)))
        self.assertFalse(r['exact_optimality']);self.assertIsNone(r['lower_bound'])

    def test_partial_candidate_cannot_form_evidence(self):
        p=lp();s=statement(p)
        with self.assertRaises(ValueError):candidate_bundle(p,s,identity(s),'SOPLEX_CANDIDATE_V1 1\nPOINT 1\n')


def main():
    global RAW
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',type=Path,required=True)
    a=parser.parse_args()
    if a.receipt.exists():raise FileExistsError(a.receipt)
    RAW=Path(tempfile.mkdtemp(prefix='soplex_compat_',dir=ROOT/'data/moe/results'))
    captured=io.StringIO();started=time.monotonic()
    r=unittest.TextTestRunner(stream=captured,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    log=captured.getvalue();(RAW/'tests.log').write_text(log)
    files={str(p.relative_to(ROOT)):sha(p) for p in RAW.rglob('*') if p.is_file()}
    sources={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'soplex_compat').glob('*') if p.is_file()}
    sources['lp_sandwich/check.py']=sha(ROOT/'lp_sandwich/check.py')
    receipt=dict(schema='SOPLEX_EXACT_IO_CONTROLS_V1',status='PASS' if r.wasSuccessful() else 'FAIL',
        tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skips=len(r.skipped),
        seconds=time.monotonic()-started,raw_root=str(RAW),source_sha256=sources,
        artifacts=files,probe_sha256=sha(PROBE),soplex_sha256=sha(PREFIX/'bin/soplex'),
        upstream_commit='13e2ab2467e0016d02116802ac4dc7a89560dbc1',native_controls=NATIVE,
        real_queries=0,network_SAFE=False,network_UNSAFE=False)
    save(a.receipt,receipt);print(log);print(json.dumps({k:receipt[k] for k in ['status','tests','failures','errors','raw_root']}))
    return 0 if r.wasSuccessful() else 1


if __name__=='__main__':sys.exit(main())
