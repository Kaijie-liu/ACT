"""Finite frozen enclosure controls and saved-evidence audit; no solver/model."""
import argparse
from pathlib import Path
import shutil
import subprocess
import time
import unittest

from scoped_proof.io import ROOT,load,save,sha

DESIGN='docs/hz_binary64_enclosure_design_20261001.md'
DESIGN_SHA='62d58dc9f18a5a498679c92f905b99910badc7f2e6e89c8aa7f2da49520b4ab5'
FREEZE_HEAD='128c69c4b456ff05b99413b6bc016634d34cde9c'
FILES=(DESIGN,'scoped_source/hz_binary64.py','scoped_source/check_hz_binary64.py',
       'scoped_source/test_hz_binary64.py','scripts/run_hz_binary64.py',
       'source_enclosure/format.py','upstream_source/checker.py','scoped_source/graph.py',
       'router_source/checker.py','scoped_proof/io.py','act/back_end/solver/solver_hz.py',
       'act/back_end/solver/hz_lp_export.py')
TESTS=sorted('test_'+v for v in ('all_eight_normal_patterns','affine_radius_rounding','exact_identity',
    'equality_slack','guard_signs','subnormal','owner_and_binary_offset','zero_and_empty','overflow',
    'layout_mutations','compensation_mutations','external_binding','canonical_schema','deadline',
    'mutation_during_execution','checker_independence','original_capacity'))
NEGATIVE_COUNTS={'test_canonical_schema':5,'test_compensation_mutations':7,'test_external_binding':4,
                 'test_guard_signs':1,'test_layout_mutations':3,'test_owner_and_binary_offset':3,'test_zero_and_empty':1}
# Fixed concrete mutation receipts, inspected in preserved R2 before R3. Bind
# the source, mutated object, external anchor, failure phase AND refusal reason;
# a self-consistent archive must not substitute one rejected input for another.
NEGATIVE_SHA256={
    'test_canonical_schema:0':'831a8c091c27feec6e546ec4aa211bef2d6ad131f4bb6ae7898b0a934c8d2825',
    'test_canonical_schema:1':'9e0500047fc12c612f0cca7d37bf28e2d3038c8aca4ef089e3d55abc4de900f5',
    'test_canonical_schema:2':'e930dc0864d0ea7b81e844ad0e4778f3a8ef460440644e05bee28af66428dac0',
    'test_canonical_schema:3':'86ce67b4c807c941a500907cae4ecb32dcdde6c9a05a39665b26f4189b1b76d0',
    'test_canonical_schema:4':'6f1c10a719090cc40c4b5ae8ca451c02b56c82bf2724a1143a8368b209f63e52',
    'test_compensation_mutations:0':'4064025ff9c7018f1e25cbe0dd33ac1b1f4e97645e74da2bdaecd1524665be79',
    'test_compensation_mutations:1':'07b671b00f10a224c25bd9e022cd75bafef0141eea0ce0e0bd9ab0c7a51df28a',
    'test_compensation_mutations:2':'31e5b781a230a5b518fdb0d36b1ee43effc0e315a1706ff3311b5ab36f55068f',
    'test_compensation_mutations:3':'47609de68b2d11f0714ab63111dd763fb19eb49b0802670d8f7374174e6ba2e4',
    'test_compensation_mutations:4':'7521667d51efd949d7784177e21241f3d9742a9f5c8a39d8b7c9a066e4fad18b',
    'test_compensation_mutations:5':'13c65048cfef587d47f74f230df3d3077a56d19bbe99fb763be3f3f85470ea36',
    'test_compensation_mutations:6':'1c53b3a4702bf1eec6b5a4a8b4d0d67e095090ff28eeb6ee46d9c3328eaafa8b',
    'test_external_binding:0':'3877a717d87725064077606a469ca1cba43870d133f6e22d63eb1b82c7646d43',
    'test_external_binding:1':'2f59a17c59a71253d4558f65fabd0591bdd218ae76b33442e742a09ff5d2d959',
    'test_external_binding:2':'66e57a67d58e5f3bae841b8b67bdeb1071b13b8e7e7854a21a1654bc4c122e6a',
    'test_external_binding:3':'d6ba530602aeceaad8066be734bc233bd71dfceb3c42f242620e34f524da82ae',
    'test_guard_signs:0':'5a97ff5b567f329120c540af3a1a0bd5225eef85196e6c6f469b1813af2678d6',
    'test_layout_mutations:0':'b20960e167bb7eb3ee00238cbd0610640c5061c8652b1da0d29f591220c7b1ac',
    'test_layout_mutations:1':'71923d89446b5bb48c7c33ee4e7103250fe08a17f929b336becedf5c55960a26',
    'test_layout_mutations:2':'faed0ad7e17761b2debcd17e0e422fc5dbd46412993a5f49cd19e370d04c1ecc',
    'test_owner_and_binary_offset:0':'9385009963444aa16863c3bd975e15972c049de7b33db30b2c3c7a427afd085e',
    'test_owner_and_binary_offset:1':'4c6536108fcbf712ca90aebfd3822eb076afc12850b74b316cef85a2cce40c34',
    'test_owner_and_binary_offset:2':'0442c3ba301f870f3927cc26cb8d74fd9001554fb0754462c91600d75c790c23',
    'test_zero_and_empty:0':'f6b040538b2ae160ba6d972bfb0f0fefc8ac51faaf4c7ddbfd27c19c451b7c1a',
}


def check_negative_binding(key,saved):
    from source_enclosure.format import identity
    if key not in NEGATIVE_SHA256 or identity(saved)!=NEGATIVE_SHA256[key]:
        raise ValueError('fixed mutation receipt binding')


def run(root):
    if sha(ROOT/DESIGN)!=DESIGN_SHA:raise ValueError('precommitted design changed')
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=FREEZE_HEAD:raise ValueError('separate finite execution identity required')
    root.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    bindings={n:sha(ROOT/n) for n in FILES}
    for n in FILES:
        target=root/'implementation'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/n,target)
    save(root/'implementation.json',bindings)
    from scoped_source import test_hz_binary64 as tests
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(tests.EnclosureTests)
    names=['scoped_source.test_hz_binary64.EnclosureTests.'+n for n in TESTS]
    if [t.id() for t in suite]!=names:raise ValueError('control inventory')
    with (root/'tests.log').open('x') as out:
        class Result(unittest.TextTestResult):
            outcomes=[]
            def startTest(self,t):self.state='PASS';super().startTest(t)
            def addError(self,t,e):self.state='ERROR';super().addError(t,e)
            def addFailure(self,t,e):self.state='FAIL';super().addFailure(t,e)
            def addSkip(self,t,e):self.state='SKIP';super().addSkip(t,e)
            def addExpectedFailure(self,t,e):self.state='EXPECTED_FAILURE';super().addExpectedFailure(t,e)
            def addUnexpectedSuccess(self,t):self.state='UNEXPECTED_SUCCESS';super().addUnexpectedSuccess(t)
            def stopTest(self,t):self.outcomes.append({'test':t.id(),'status':self.state});super().stopTest(t)
        result=unittest.TextTestRunner(stream=out,verbosity=2,resultclass=Result,failfast=True).run(suite)
    save(root/'observations.json',tests.OBS)
    save(root/'rejections.json',tests.REJECTIONS);save(root/'overflow.json',tests.OVERFLOW)
    ok=result.wasSuccessful() and result.outcomes==[{'test':n,'status':'PASS'} for n in names]
    summary={'status':'PASS' if ok else 'FAIL','tests':result.testsRun,'outcomes':result.outcomes,
        'exceptions':{k:len(getattr(result,k)) for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')},
        'implementation_sha256':sha(root/'implementation.json'),'observations_sha256':sha(root/'observations.json'),
        'rejections_sha256':sha(root/'rejections.json'),'overflow_sha256':sha(root/'overflow.json'),
        'tests_log_sha256':sha(root/'tests.log'),'seconds':time.monotonic()-start,
        'head':head,'design_sha256':DESIGN_SHA,
        'real_requests':0,'native_solves':0,'cuda_calls':0,'performance_claim':False,'hard_supervision':False}
    save(root/'summary.json',summary);return summary


def audit(root):
    import math
    from source_enclosure.format import identity
    from upstream_source.checker import hz
    from scoped_source.check_hz_binary64 import check,check_embedding,parse
    from scoped_source.test_hz_binary64 import patterns,binary_reference,overflow_patterns,safe
    s=load(root/'summary.json');bindings=load(root/'implementation.json',s['implementation_sha256'])
    obs=load(root/'observations.json',s['observations_sha256'])
    rejects=load(root/'rejections.json',s['rejections_sha256']);overflow=load(root/'overflow.json',s['overflow_sha256'])
    names=['scoped_source.test_hz_binary64.EnclosureTests.'+n for n in TESTS]
    if (s['status']!='PASS' or s['tests']!=len(TESTS) or s['outcomes']!=[{'test':n,'status':'PASS'} for n in names]
            or s['exceptions']!={k:0 for k in ('errors','failures','skipped','expectedFailures','unexpectedSuccesses')}
            or sha(root/'tests.log')!=s['tests_log_sha256'] or set(bindings)!=set(FILES)
            or s['head']!=FREEZE_HEAD or s['design_sha256']!=DESIGN_SHA or bindings[DESIGN]!=DESIGN_SHA
            or type(s['seconds']) not in (float,int) or not math.isfinite(s['seconds']) or s['seconds']<0
            or any(type(s[k]) is not int or s[k]!=0 for k in ('real_requests','native_solves','cuda_calls'))
            or s['performance_claim'] is not False or s['hard_supervision'] is not False):
        raise ValueError('control outcomes/scope')
    for n,h in bindings.items():
        if sha(ROOT/n)!=h or sha(root/'implementation'/n)!=h:raise ValueError('implementation binding: '+n)
    expected=patterns();rows={};points=0
    if set(obs)!=set(expected):raise ValueError('all eight normal patterns required')
    for name,(ref,assignments) in expected.items():
        r=obs[name];p=r['proof'];deadline=time.monotonic()+300
        if r['reference']!=ref or r['owner']!='pair/0-1':raise ValueError('fixed reference/owner')
        checked=check(ref,p,expected_reference_sha256=identity(ref),expected_owner=r['owner'],deadline=deadline)
        if checked!=r['checked'] or hz(r['snapshot'])!=parse(p['target'],target=True)[0] or r['snapshot']['exact'] is not False:
            raise ValueError('saved mathematical/actual-HZ differential')
        if len(r['embeddings'])!=len(assignments):raise ValueError('point inventory')
        for saved,(a,b) in zip(r['embeddings'],assignments):
            if saved['source']!={'continuous':a,'binary':b}:raise ValueError('fixed point')
            point=check_embedding(ref,p,saved['source'],saved['target'],expected_reference_sha256=identity(ref),expected_owner=r['owner'],deadline=deadline)
            if point!=saved['checked']:raise ValueError('point check changed')
            points+=1
        cost=r['cost']
        if (set(cost)!={'produce','instantiate','serialize','check_including_points','total'}
                or any(type(v) not in (float,int) or not math.isfinite(v) or v<0 for v in cost.values())
                or r['deadline']!=r['start']+300 or r['end']>=r['deadline'] or cost['total']!=r['end']-r['start']
                or sum(v for k,v in cost.items() if k!='total')>cost['total']):
            raise ValueError('finite cost/deadline inventory')
        rows[name]={'reference_sha256':identity(ref),'proof_sha256':identity(p),
                    'target_sha256':p['target_sha256'],'added_continuous':checked['added_continuous'],
                    'binary_preserved':checked['binary_preserved'],'compensation':checked['compensation'],
                    'inequalities':checked['inequalities'],'cost_seconds':cost}
    if s['seconds']<sum(r['cost']['total'] for r in obs.values()):raise ValueError('aggregate cost undercount')
    if set(rejects)!={name+':'+str(i) for name,n in NEGATIVE_COUNTS.items() for i in range(n)}:
        raise ValueError('negative input inventory')
    for key,saved in rejects.items():
        check_negative_binding(key,saved)
        item=safe(saved,restore=True)
        if item['pattern'] not in expected or type(item['auxiliary']) is not bool or item['owner']!='pair/0-1':
            raise ValueError('negative original case/owner')
        original=binary_reference() if item['auxiliary'] else expected[item['pattern']][0]
        if item['expected_reference_sha256']!=identity(original) or item['phase'] not in ('target_hash','check'):
            raise ValueError('negative external anchor')
        try:
            if item['phase']=='target_hash':identity(item['proof']['target'])
            else:check(item['reference'],item['proof'],expected_reference_sha256=item['expected_reference_sha256'],
                       expected_owner=item['owner'],deadline=time.monotonic()+300)
        except (ValueError,TypeError,KeyError) as exc:
            if (type(exc).__name__,str(exc))!=(item['error_type'],item['error']):raise ValueError('negative reason changed')
        else:raise ValueError('saved negative input accepted')
    # Check the exact supplied source and the reason for the fixed rounding
    # strategy's refusal without invoking that producer again.
    expected_overflow=overflow_patterns()
    reasons=['no finite binary64 coefficient','no finite outward binary64 endpoint']
    if len(overflow)!=2:raise ValueError('overflow inventory')
    for item,ref,reason in zip(overflow,expected_overflow,reasons):
        if item!={'reference':ref,'error':reason,'accepted':False}:raise ValueError('overflow source/reason')
        parse(ref)
    return {'status':'PASS','root':str(root),'summary_sha256':sha(root/'summary.json'),
            'design_sha256':DESIGN_SHA,'freeze_head':FREEZE_HEAD,
            'tests':len(TESTS),'normal_patterns_checked':len(rows),'embeddings_checked':points,'cases':rows,
            'negative_inputs_rechecked':len(rejects),'overflow_refusals_bound':len(overflow),'suite_seconds':s['seconds'],
            'real_requests':0,'native_solves':0,'cuda_calls':0,'performance_claim':False,
            'hard_supervision':False,'all_six_goal_gates_remain_open':True}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['run','audit']);p.add_argument('root',type=Path)
    p.add_argument('--report',type=Path);a=p.parse_args()
    if not a.root.is_absolute() or not a.root.resolve().is_relative_to(ROOT.parent/'baseline_runs'):
        raise ValueError('new project control directory required')
    result=run(a.root) if a.action=='run' else audit(a.root)
    if a.report:save(a.report,result)
    print({'status':result['status'],'tests':result['tests'],'root':str(a.root)})
