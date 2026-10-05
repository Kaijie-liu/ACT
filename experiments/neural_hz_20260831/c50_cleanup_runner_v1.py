"""One audited full pytest run with equivalent private-temp cleanup only.

All original items execute, no cached outcomes. Per-phase records survive a
timeout. The ordinary pytest rewriter and every condition/exception remain.
"""
import ast
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import types
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pytest
from _pytest.assertion import rewrite
from experiments.neural_hz_20260831.c50_assert_protocol_v1 import guard,code_image,code_key,code_tree,unwrap_function
from experiments.neural_hz_20260831.c50_cleanup_compile_v1 import compile_cleanup,exact_canaries
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c50_checked_cleanup_20260911_v1'
OLD_IDS_SHA='143258636d9b6840970829d30810ae55649671a30f0e40a43a63cd7302797c0e'


def module_audit(module,path,sha,items,config):
    guard();source=path.read_bytes()
    if hashlib.sha256(source).hexdigest()!=sha or Path(module.__file__).resolve()!=path:
        raise ValueError('loaded test module differs from complete frozen source')
    tree=ast.parse(source,filename=str(path))
    for n in ast.walk(tree):
        if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id=='locals')
                or (isinstance(n.func,ast.Attribute) and n.func.attr in ('currentframe','_getframe'))):
            raise ValueError('test inspects private frame locals; cleanup equivalence is not established')
        if isinstance(n,ast.Attribute) and n.attr in ('f_locals','tb_frame'):
            raise ValueError('test depends on reflected frame scratch state')
    functions=[];classes=set()
    def visit(namespace):
        for obj in namespace.values():
            fn=unwrap_function(obj)
            if fn is not None and fn.__module__==module.__name__:functions.append(fn)
            elif isinstance(obj,type) and obj.__module__==module.__name__ and id(obj) not in classes:
                classes.add(id(obj));visit(vars(obj))
    visit(vars(module));functions.extend(unwrap_function(item.obj) for item in items)
    functions=list({id(f):f for f in functions}.values());expected={};images=[];stats=None
    for fn in functions:
        if fn is None or Path(fn.__code__.co_filename).resolve()!=path:
            raise ValueError('unknown wrapped selected test/helper function')
        filename=fn.__code__.co_filename
        if filename not in expected:
            code,stats=compile_cleanup(source,filename,config=config,enabled=True)
            expected[filename]={code_key(c):c for c in code_tree(code)}
        wanted=expected[filename].get(code_key(fn.__code__))
        if wanted is None or code_image(fn.__code__)!=code_image(wanted):
            raise ValueError('loaded test/helper is not the independently rebuilt cleanup code')
        images.append(code_image(fn.__code__))
    if stats is None:raise ValueError('no loaded test code checked')
    return dict(file=str(path),functions=len(functions),compile=stats,
        actual_loaded_code_image=hashlib.sha256('\n'.join(sorted(images)).encode()).hexdigest(),
        all_loaded_test_and_helper_code_checked=True)


class Ledger:
    def __init__(self,freeze,emit):
        self.freeze,self.emit=freeze,emit;self.ids=[];self.reports=[];self.audits=[];self.received=[]
        self.paths={(EXP/n).resolve():freeze['source_sha256'][n] for n in freeze['tests']}
        self.old_paths={(EXP/n).resolve() for n in freeze['original_tests']}

    def pytest_collection_modifyitems(self,session,config,items):
        guard()
        if config.getoption('assertmode')!='rewrite' or config.getini('enable_assertion_pass_hook'):
            raise ValueError('unchanged ordinary assertion evaluation is mandatory')
        old=[item.nodeid for item in items if Path(item.path).resolve() in self.old_paths]
        sha=hashlib.sha256('\n'.join(old).encode()).hexdigest()
        if len(old)!=2104 or sha!=OLD_IDS_SHA:raise ValueError('original full ordered2104-test population changed')
        groups={}
        for item in items:
            path=Path(item.path).resolve()
            if path not in self.paths:raise ValueError('unregistered extra test path')
            if any(m.name in ('skip','skipif','xfail') for m in item.iter_markers()):
                raise ValueError('test/assertion coverage cannot be skipped or xfailed')
            groups.setdefault(path,[]).append(item)
        if set(groups)!=set(self.paths):raise ValueError('a frozen test file was not collected')
        for path,group in groups.items():self.audits.append(module_audit(group[0].module,path,self.paths[path],group,config))
        self.ids=[item.nodeid for item in items]
        self.emit(dict(event='complete_loaded_code_and_test_order_checked',original_cases=len(old),total_cases=len(items),
            original_ordered_ids_sha256=sha,files=len(groups),audits=self.audits))

    def pytest_runtest_logreport(self,report):
        row=dict(event='test_phase',nodeid=report.nodeid,when=report.when,outcome=report.outcome,
            duration_s=report.duration,wasxfail=bool(getattr(report,'wasxfail',False)))
        self.reports.append(row);self.emit(row)
        if report.when=='call':self.received.append(report.nodeid)

    def pytest_sessionfinish(self,session,exitstatus):
        good=(int(exitstatus)==0 and self.received==self.ids and self.ids
            and all(r['outcome']=='passed' and not r['wasxfail'] for r in self.reports)
            and len(self.reports)==3*len(self.ids))
        self.emit(dict(event='complete_test_protocol_finished',pytest_exit_code=int(exitstatus),
            all_ordered_tests_and_all_three_phases_passed=bool(good),cases=len(self.ids),phases=len(self.reports)))
        if not good and int(exitstatus)==0:session.exitstatus=1


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered complete cleanup test run')
    guard()
    if not sys.dont_write_bytecode:raise ValueError('test loader may not write old source caches')
    freeze=json.loads((DIRECTORY/'preregistered.json').read_text())
    paths={(EXP/n).resolve():freeze['source_sha256'][n] for n in freeze['tests']}
    started=time.monotonic();record=dict(completed=False,formal_gain=0)
    with (DIRECTORY/'test_events.jsonl').open('x') as stream:
        def emit(row):stream.write(json.dumps(dict(elapsed_s=time.monotonic()-started,**row),sort_keys=True)+'\n');stream.flush()
        normal_compile,normal_cache=rewrite._rewrite_test,rewrite._read_pyc
        def changed_compile(path,config):
            key=Path(path).resolve()
            if key not in paths:return normal_compile(path,config)
            source=path.read_bytes()
            if hashlib.sha256(source).hexdigest()!=paths[key]:raise ValueError('test source changed before compilation')
            code,stats=compile_cleanup(source,str(path),config=config,enabled=True)
            return path.stat(),code
        def checked_cache(source,*args,**kwargs):
            # A standard old rewritten pyc is not an authority for NEW code.
            # Force authenticated compilation only on this exact frozen list.
            if Path(source).resolve() in paths:return None
            return normal_cache(source,*args,**kwargs)
        try:
            record['canaries']=exact_canaries(enabled=True);emit(dict(event='exact_evaluation_and_owner_canaries_passed',cases=17))
            rewrite._rewrite_test=changed_compile;rewrite._read_pyc=checked_cache
            ledger=Ledger(freeze,emit)
            exitcode=pytest.main(['-q','--tb=short','-p','no:cacheprovider',*(str(EXP/n) for n in freeze['tests'])],plugins=[ledger])
            record.update(completed=int(exitcode)==0,pytest_exit_code=int(exitcode),
                cases=len(ledger.ids),phase_reports=len(ledger.reports),module_audits=ledger.audits,
                all_original_ordered_cases_retained=True)
        except Exception as exc:record['failure']=dict(type=type(exc).__name__,reason=str(exc));raise
        finally:
            rewrite._rewrite_test=normal_compile;rewrite._read_pyc=normal_cache
            record.update(wall_s=time.monotonic()-started,compiler_hooks_restored=True)
            _atomic_exclusive_json(DIRECTORY/'test_result.json',record)
    if not record['completed']:raise SystemExit(record.get('pytest_exit_code',1))


if __name__=='__main__':main()
