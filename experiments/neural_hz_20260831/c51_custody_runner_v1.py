"""One complete qualification, with source-first custody and normal pytest."""
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import pytest
from _pytest.assertion import rewrite
from experiments.neural_hz_20260831.c50_assert_protocol_v1 import guard
from experiments.neural_hz_20260831.c50_cleanup_compile_v1 import exact_canaries
from experiments.neural_hz_20260831.c51_source_custody_v1 import custody
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json
EXP=Path(__file__).resolve().parent;DIRECTORY=EXP/'results/c51_source_custody_20260911_v1'
OLD_IDS_SHA='143258636d9b6840970829d30810ae55649671a30f0e40a43a63cd7302797c0e'


class Ledger:
    def __init__(self,freeze,reg,emit,collection_only=False):
        self.freeze,self.reg,self.emit=freeze,reg,emit
        self.ids=[];self.received=[];self.reports=[];self.audits=[];self.aliases=0
        self.collection_only=collection_only
        self.old_paths={(EXP/n).resolve() for n in freeze['original_tests']}

    def pytest_collection_modifyitems(self,session,config,items):
        guard()
        if config.getoption('assertmode')!='rewrite' or config.getini('enable_assertion_pass_hook'):
            raise ValueError('ordinary assertion evaluation is mandatory')
        old=[x.nodeid for x in items if Path(x.path).resolve() in self.old_paths]
        sha=hashlib.sha256('\n'.join(old).encode()).hexdigest()
        if len(old)!=2104 or sha!=OLD_IDS_SHA:raise ValueError('complete original ordered test population changed')
        groups={}
        for item in items:
            path=Path(item.path).resolve()
            if path not in self.reg.selected:raise ValueError('unregistered test item')
            if any(m.name in ('skip','skipif','xfail') for m in item.iter_markers()):raise ValueError('no test may be skipped or xfailed')
            groups.setdefault(path,[]).append(item)
        if set(groups)!=self.reg.selected:raise ValueError('missing complete test file')
        audited=set()
        for path,group in groups.items():
            module=group[0].module;self.audits.append(self.reg.audit_module(module,path));audited.add(path)
        for item in items:
            path=Path(item.path).resolve()
            origin,module=self.reg.definition(item.module,path,item.obj,item.originalname)
            if origin!=path:self.aliases+=1
            if origin not in audited:self.audits.append(self.reg.audit_module(module,origin));audited.add(origin)
        if audited!=set(self.reg.specs):raise ValueError('an imported definition source was not completely audited')
        self.ids=[x.nodeid for x in items]
        self.emit(dict(event='complete_loaded_source_code_and_order_checked',original_cases=len(old),total_cases=len(items),
            original_ordered_ids_sha256=sha,selected_files=len(groups),definition_files=len(audited),
            imported_items=self.aliases,source_compilations=self.reg.compile_count,collection_recompilations=0))

    def pytest_runtest_logreport(self,report):
        row=dict(event='test_phase',nodeid=report.nodeid,when=report.when,outcome=report.outcome,
            duration_s=report.duration,wasxfail=bool(getattr(report,'wasxfail',False)))
        self.reports.append(row);self.emit(row)
        if report.when=='call':self.received.append(report.nodeid)

    def pytest_sessionfinish(self,session,exitstatus):
        if self.collection_only:
            self.emit(dict(event='diagnostic_collection_only_finished',pytest_exit_code=int(exitstatus),tests_executed=0))
            return
        good=(int(exitstatus)==0 and self.received==self.ids and self.ids
            and all(r['outcome']=='passed' and not r['wasxfail'] for r in self.reports)
            and len(self.reports)==3*len(self.ids))
        self.emit(dict(event='complete_test_protocol_finished',pytest_exit_code=int(exitstatus),
            all_ordered_tests_and_all_three_phases_passed=bool(good),cases=len(self.ids),phases=len(self.reports)))
        if not good and int(exitstatus)==0:session.exitstatus=1


def run(freeze,emit,*,collection_only=False):
    guard()
    if not sys.dont_write_bytecode:raise ValueError('old source caches must not be written')
    paths=[(EXP/n).resolve() for n in freeze['tests']]
    reg=custody(paths,{(EXP/n).resolve():sha for n,sha in freeze['source_sha256'].items()},ROOT,enabled=True)
    normal_compile,normal_cache=rewrite._rewrite_test,rewrite._read_pyc
    def prepare(path,config):
        if Path(path).resolve() not in reg.specs:return normal_compile(path,config)
        code=reg.prepare(path,config);return path.stat(),code
    def cache(path,*args,**kwargs):
        if Path(path).resolve() in reg.specs:return None
        return normal_cache(path,*args,**kwargs)
    record=dict(completed=False,formal_gain=0,collection_only=collection_only)
    try:
        record['canaries']=exact_canaries(enabled=True);emit(dict(event='exact_canaries_passed',cases=17))
        rewrite._rewrite_test=prepare;rewrite._read_pyc=cache
        ledger=Ledger(freeze,reg,emit,collection_only)
        options=['--collect-only'] if collection_only else []
        code=pytest.main(['-q','--tb=short','-p','no:cacheprovider',*options,*map(str,paths)],plugins=[ledger])
        record.update(completed=int(code)==0 and not collection_only,pytest_exit_code=int(code),
            cases=len(ledger.ids),phase_reports=len(ledger.reports),module_audits=ledger.audits,
            imported_test_items=ledger.aliases,source_compilations=reg.compile_count,
            collection_recompilations=0,all_original_ordered_cases_retained=bool(ledger.ids))
    finally:
        rewrite._rewrite_test=normal_compile;rewrite._read_pyc=normal_cache
        record['compiler_hooks_restored']=True
    return record


def main():
    if Path(sys.argv[1]).resolve()!=DIRECTORY:raise ValueError('unregistered qualification directory')
    freeze=json.loads((DIRECTORY/'preregistered.json').read_text());started=time.monotonic()
    with (DIRECTORY/'test_events.jsonl').open('x') as f:
        def emit(row):f.write(json.dumps(dict(elapsed_s=time.monotonic()-started,**row),sort_keys=True)+'\n');f.flush()
        result=run(freeze,emit);result['wall_s']=time.monotonic()-started
        _atomic_exclusive_json(DIRECTORY/'test_result.json',result)
    if not result['completed']:raise SystemExit(result['pytest_exit_code'] or 1)


if __name__=='__main__':main()
