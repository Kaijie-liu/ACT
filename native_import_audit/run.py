"""Numbered import-only analysis; all real solver artifacts remain untouched."""
import io
import math
from fractions import Fraction
from pathlib import Path
import tempfile
import time
import unittest
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from sparse_supervised.study import verify
from native_import_audit.inventory import run as inventory, ARCHIVE
from native_import_audit.probe import runtime, model, submit


def sources():
    paths=list(Path(__file__).parent.glob('*.py'))+[ROOT/'sparse_basis/native.py',ROOT/'lp_sandwich/check.py']
    return {str(p.relative_to(ROOT)):digest(p.read_bytes()) for p in sorted(paths)}


def run():
    start=time.monotonic();verify();before=sources();rt=runtime();data=inventory(rt['defaults'])
    out=Path(tempfile.mkdtemp(prefix='native_import_analytic_',dir=ROOT/'data/moe/results'))
    stream=io.StringIO();tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('native_import_audit.tests'))
    cases=[];issues=[]
    if not tests.wasSuccessful() or tests.skipped:issues.append('unit controls failed')
    t=rt['defaults']['small_matrix_value'];v=data['matrix']['min_abs_nonzero']
    specs=[('ordinary',t*10,False),('below',math.nextafter(t,0.),False),
           ('at',t,False),('above',math.nextafter(t,math.inf),False),
           ('negative_below',-t/2,False),('observed_min_A',v,False),('observed_min_E',v,True),
           ('observed_signed_A',data['small_matrix_entries'][0]['submitted_value'],False)]
    for name,value,equality in specs:
        record=submit(name,model(value,equality),out/name);cases.append(record)
        expected_drop=abs(value)<=t
        d=record['comparison']
        if (record['status']!=('HighsStatus.kWarning' if expected_drop else 'HighsStatus.kOk') or
                d['matrix_changes']!=([{'row':0,'column':1,'before':value,'after':None}] if expected_drop else []) or
                not d['other_fields_unchanged'] or not d['row_bounds_unchanged']):
            issues.append('unexpected import/readback: '+name)
        if expected_drop and 'ignored' not in record['log'].lower():
            issues.append('missing native filtering message: '+name)
    # A concrete exact arithmetic semantic counterexample, not a model witness.
    selected=next(c for c in cases if c['case']=='observed_min_A')
    point=[Fraction(0),Fraction(1)]
    def holds(row):
        lhs=sum((Fraction(x)*point[j] for j,x in row['entries']),Fraction(0))
        return lhs<=Fraction(row['upper']),str(lhs)
    original,lhs=holds(selected['intended']['rows'][0])
    imported,after_lhs=holds(selected['readback']['rows'][0])
    if original or not imported:issues.append('analytic semantic witness failed')
    # Deleting a negative coefficient can instead exclude an originally feasible point.
    signed=next(c for c in cases if c['case']=='observed_signed_A')
    point=[Fraction(v)/2,Fraction(1)]
    signed_original,signed_lhs=holds(signed['intended']['rows'][0])
    signed_imported,signed_after=holds(signed['readback']['rows'][0])
    if not signed_original or signed_imported:issues.append('signed semantic witness failed')
    verify()
    # Recheck all sealed artifact bytes, not just file existence.
    for path,h in read(ARCHIVE)['artifact_sha256'].items():
        if digest((ROOT/path).read_bytes())!=h:raise ValueError('sealed artifact mutated')
    if sources()!=before:raise ValueError('source changed during inspection')
    n=1
    while (ROOT/f'docs/native_import_analysis_attempt{n:03}.json').exists():n+=1
    dest=ROOT/f'docs/native_import_analysis_attempt{n:03}.json'
    result={'status':'PASS' if not issues else 'FAIL','issues':issues,'sources':before,'runtime':rt,
            'inventory':data,'analytic_cases':cases,'tests_run':tests.testsRun,'test_log':stream.getvalue(),
            'analytic_semantic_witness':{'point':['0','1'],'original_feasible':original,'imported_feasible':imported,
                                         'original_lhs':lhs,'imported_lhs':after_lhs,'rhs':'0'},
            'signed_semantic_witness':{'point':[str(x) for x in point],'original_feasible':signed_original,
                                      'imported_feasible':signed_imported,'original_lhs':signed_lhs,
                                      'imported_lhs':signed_after,'rhs':'0'},
            'artifact_root':str(out),'artifact_sha256':{str(p.relative_to(out)):digest(p.read_bytes())
                                                      for p in sorted(out.rglob('*')) if p.is_file()},
            'seconds':time.monotonic()-start,'real_native_imports':0,'optimization_calls':0,
            'analytic_native_imports':len(cases),'frozen_interface_or_options_changed':False,
            'logging_override_only':True,'sealed_results_unchanged':True,
            'interpretation':'default tiny-coefficient filtering reproduced analytically; saved real submission has eligible entries',
            'limitation':'real import was not repeated; no real readback or exhaustive attribution of all original warnings'}
    save_new(dest,result)
    print(stream.getvalue());print(dest,result['status']);print([(c['case'],c['status'],c['log']) for c in cases])
    if issues:raise SystemExit(1)


if __name__=='__main__':run()
