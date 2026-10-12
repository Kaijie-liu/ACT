"""Finite frozen component controls; not a production request supervisor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

FILES = ('scoped_source/coefficient_dag.py','scoped_source/coefficient_dag_controls.py',
    'scoped_source/hz_request_store.py','scripts/test_coefficient_dag.py',
    'scripts/run_coefficient_dag_controls.py','scripts/run_tensor_rows_controls.py',
    'act/back_end/moe/coefficient_support.py','act/back_end/moe/test_coefficient_support.py',
    'docs/hz_coefficient_dag_design_20261013.md')
LIMIT=64*2**20


def save(path, value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)


def worker(root):
    from scoped_source.coefficient_dag import Graph, query, evaluate
    from scoped_source.coefficient_dag_controls import small, dense, dense_reference, expanded, REQUEST
    from scoped_source.hz_request_store import Work, compact, digest, identity, require
    root.mkdir(); started=time.monotonic(); deadline=started+300; reports=[]
    for name in ('small','dense'):
        start=time.monotonic();doc,rows=small() if name=='small' else (dense(),None)
        built=time.monotonic();raw=compact(doc);serialized=time.monotonic()
        work=Work(deadline)  # separate GIVEN objects; shared absolute worker deadline
        graph=Graph(raw,expected_graph=digest(raw),expected_request=REQUEST,expected_source=doc['source'],work=work)
        checked=time.monotonic()
        with (root/(name+'.json')).open('xb') as f:f.write(raw)
        written=time.monotonic();results=[]
        for cancel in ((False,) if name=='small' else (False,True)):
            q=query(graph,['1','-1','2','-3'] if name=='small' else ['1/8']*256,
                    '1/8' if name=='small' else '0')
            d=dict(eq=['1/4'],ine=['-1/2','-1/4']) if name=='small' else dict(
                eq=['-1/8' if cancel else str(__import__('fractions').Fraction(r%3+1,16)) for r in range(256)],ine=[])
            begin=time.monotonic();bound=evaluate(graph,q,d,expected_query=identity(q),work=work)
            evaluated=time.monotonic()
            if name=='small':reference=expanded(rows,q['weights'],d);expanded_visits=sum(len(row[2]) for row in rows)
            else:
                *reference,expanded_visits=dense_reference(cancel,work.tick)
            ended=time.monotonic()
            require((bound.lower,bound.constant,dict(bound.residual)) == tuple(reference),'independent expanded differential')
            receipt=dict(query=q,duals=d,query_sha256=identity(q),lower=str(bound.lower),constant=str(bound.constant),
                residual_sha256=identity([[j,str(v)] for j,v in bound.residual]),cancel=cancel,
                local_support_visits=bound.local_coefficient_visits,reference_support_visits=bound.reference_visits,
                expanded_reference_coefficient_visits=expanded_visits,exact_support_seconds=evaluated-begin,
                independent_reference_seconds=ended-evaluated)
            results.append(receipt)
        report=dict(name=name,graph_sha256=graph.identity,request=REQUEST,source=graph.source,
            stats=dict(graph.stats),generation_seconds=built-start,serialization_seconds=serialized-built,
            validation_seconds=checked-serialized,file_write_seconds=written-checked,
            total_seconds=time.monotonic()-start,checker_work=dict(work.counts),results=results)
        save(root/(name+'_receipt.json'),report);reports.append(report);work.tick()
    return dict(status='CHECKED_GIVEN_ROW_EQUIVALENCE',cases=reports,worker_seconds=time.monotonic()-started,
                real_models=0,gpu_calls=0,full_request_certificates=0)


def relocated(root, anchor):
    from scoped_source.coefficient_dag import Graph, evaluate
    from scoped_source.hz_request_store import Work, identity
    start=time.monotonic();work=Work(start+300)
    receipt=json.loads((root/'small_receipt.json').read_bytes())
    graph=Graph((root/'small.json').read_bytes(),expected_graph=anchor,
        expected_request=receipt['request'],expected_source=receipt['source'],work=work)
    r=receipt['results'][0];result=evaluate(graph,r['query'],r['duals'],expected_query=r['query_sha256'],work=work)
    assert str(result.lower)==r['lower']
    assert identity([[j,str(v)] for j,v in result.residual])==r['residual_sha256']
    assert not {'numpy','scipy','torch','highspy'}&set(sys.modules)
    return dict(graph=graph.identity,lower=str(result.lower),seconds=time.monotonic()-start)


def execute(archive):
    from scripts.run_tensor_rows_controls import terminal_bytes
    start=time.monotonic();repo=Path(__file__).resolve().parents[1]
    archive.mkdir(parents=True,exist_ok=False);code=archive/'code';identities={}
    for name in FILES:
        target=code/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,target)
        identities[name]=hashlib.sha256(target.read_bytes()).hexdigest()
    # Native fixture is run from the checkout, with its complete dependency hashes
    # recorded. It is NOT presented as the relocated stdlib check.
    native_dependencies={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted((repo/'act').rglob('*.py'))}
    save(archive/'execution.json',dict(code=identities,native_dependencies=native_dependencies,
        python=sys.executable,per_process_deadline_seconds=300,archive_cap=LIMIT))
    calls=[];status='FAILED'
    def run(name,command,cwd=archive):
        begin=time.monotonic();item=dict(name=name,command=command,hard_cap_seconds=300);calls.append(item)
        try:
            r=subprocess.run(command,cwd=cwd,capture_output=True,text=True,timeout=300)
            item.update(returncode=r.returncode,status='COMPLETE' if r.returncode==0 else 'ERROR')
            (archive/(name+'.stdout')).write_text(r.stdout);(archive/(name+'.stderr')).write_text(r.stderr)
            if r.returncode:raise RuntimeError(name+' failed; prefix retained')
            return r.stdout
        except subprocess.TimeoutExpired as exc:
            item.update(returncode=None,status='TIMEOUT')
            (archive/(name+'.stdout')).write_bytes(exc.stdout or b'');(archive/(name+'.stderr')).write_bytes(exc.stderr or b'')
            raise
        finally:item['parent_observed_seconds']=time.monotonic()-begin
    launcher=[sys.executable,'-B','-I','-S','-c']
    try:
        run('unit',launcher+["import sys,unittest;sys.path.insert(0,sys.argv[1]);unittest.main(module='scripts.test_coefficient_dag',argv=['controls','-v'])",str(code)])
        run('native',[sys.executable,'-B','-m','unittest','act.back_end.moe.test_coefficient_support','-v'],repo)
        raw=run('fixed_shapes',launcher+["import sys,json;sys.path.insert(0,sys.argv[1]);from scripts.run_coefficient_dag_controls import worker;from pathlib import Path;print(json.dumps(worker(Path(sys.argv[2]))))",str(code),str(archive/'objects')])
        results=json.loads(raw)
        dest=archive/'relocated';shutil.copytree(code,dest/'code');(dest/'objects').mkdir()
        for name in ('small.json','small_receipt.json'):shutil.copyfile(archive/'objects'/name,dest/'objects'/name)
        run('relocated',launcher+["import sys,json;sys.path.insert(0,sys.argv[1]);from scripts.run_coefficient_dag_controls import relocated;from pathlib import Path;print(json.dumps(relocated(Path(sys.argv[2]),sys.argv[3])))",str(dest/'code'),str(dest/'objects'),results['cases'][0]['graph_sha256']])
        status='PASS'
    except BaseException as exc:
        save(archive/'failure.json',dict(exception=type(exc).__name__,message=str(exc)));raise
    finally:
        payload=sum(p.stat().st_size for p in archive.rglob('*') if p.is_file())
        summary=dict(status=status,calls=calls,code=identities,payload_bytes_before_terminal=payload,
            archive_cap=LIMIT,observed_total_before_terminal_seconds=time.monotonic()-start,
            real_models=0,gpu_calls=0,full_request_certificates=0)
        summary,raw=terminal_bytes(summary,payload)
        with (archive/'terminal.json').open('xb') as f:f.write(raw)
        if sum(p.stat().st_size for p in archive.rglob('*') if p.is_file())>LIMIT:raise ValueError('archive cap; prefix retained')
    return summary


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);args=parser.parse_args()
    print(json.dumps(execute(args.archive.resolve()),indent=2))
