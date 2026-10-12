"""Post-review recheck of unchanged R1 bytes; never regenerate dense controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def save(path,value):
    with path.open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2,allow_nan=False)


def recheck(parent):
    from scoped_source.coefficient_dag import Graph, check_candidate
    from scoped_source.hz_request_store import Work, identity
    start=time.monotonic();deadline=start+300;reports=[]
    for name in ('small','dense'):
        work=Work(deadline);r=json.loads((parent/'objects'/(name+'_receipt.json')).read_bytes())
        raw=(parent/'objects'/(name+'.json')).read_bytes()
        g=Graph(raw,expected_graph=r['graph_sha256'],expected_request=r['request'],expected_source=r['source'],work=work)
        values=[]
        for item in r['results']:
            c=dict(query=item['query_sha256'],**item['duals'],lower=item['lower'])
            b=check_candidate(g,item['query'],c,expected_query=item['query_sha256'],work=work)
            if str(b.lower)!=item['lower'] or identity([[j,str(v)] for j,v in b.residual])!=item['residual_sha256']:
                raise ValueError('recheck changed original bound/residual')
            values.append(str(b.lower))
        reports.append(dict(name=name,graph=g.identity,bounds=values,work=dict(work.counts)))
    return dict(status='PASS',cases=reports,seconds=time.monotonic()-start,generated_objects=0)


def execute(parent,archive):
    from scripts.run_coefficient_dag_controls import FILES,LIMIT
    from scripts.run_tensor_rows_controls import terminal_bytes
    started=time.monotonic();repo=Path(__file__).resolve().parents[1];archive.mkdir(parents=True,exist_ok=False)
    code=archive/'code';names=FILES+('scripts/audit_coefficient_dag_controls.py','scripts/recheck_coefficient_dag_controls.py')
    fingerprints={}
    for name in names:
        p=code/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(repo/name,p)
        fingerprints[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    parent_inputs={str(p.relative_to(parent)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [parent/'terminal.json',parent/'execution.json',*sorted((parent/'objects').glob('*.json'))]}
    save(archive/'execution.json',dict(code=fingerprints,parent=str(parent),parent_inputs=parent_inputs,
        python=sys.executable,hard_cap_seconds=300,generated_objects=0))
    calls=[];status='FAILED'
    def run(name,command,cwd=archive):
        begin=time.monotonic();item=dict(name=name,hard_cap_seconds=300,command=command);calls.append(item)
        try:
            r=subprocess.run(command,cwd=cwd,capture_output=True,text=True,timeout=300)
            item.update(status='COMPLETE' if r.returncode==0 else 'ERROR',returncode=r.returncode)
            (archive/(name+'.stdout')).write_text(r.stdout);(archive/(name+'.stderr')).write_text(r.stderr)
            if r.returncode:raise RuntimeError(name+' failed; retained')
        except subprocess.TimeoutExpired as exc:
            item.update(status='TIMEOUT',returncode=None)
            (archive/(name+'.stdout')).write_bytes(exc.stdout or b'');(archive/(name+'.stderr')).write_bytes(exc.stderr or b'');raise
        finally:item['parent_observed_seconds']=time.monotonic()-begin
    std=[sys.executable,'-B','-I','-S','-c']
    try:
        run('unit',std+["import sys,unittest;sys.path.insert(0,sys.argv[1]);unittest.main(module='scripts.test_coefficient_dag',argv=['controls','-v'])",str(code)])
        run('native_and_regressions',[sys.executable,'-B','-m','unittest','-v',
            'act.back_end.moe.test_coefficient_support','scripts.test_tensor_rows',
            'scripts.test_hz_store_source_units','scripts.test_hz_request_store_units'],repo)
        run('independent_audit',[sys.executable,'-B','-I','-S',str(code/'scripts/audit_coefficient_dag_controls.py'),str(parent)])
        run('new_checker_old_bytes',std+["import sys,json;sys.path.insert(0,sys.argv[1]);from scripts.recheck_coefficient_dag_controls import recheck;from pathlib import Path;print(json.dumps(recheck(Path(sys.argv[2]))))",str(code),str(parent)])
        moved=archive/'moved';moved.mkdir()
        for name in ('small.json','small_receipt.json'):shutil.copyfile(parent/'objects'/name,moved/name)
        anchor=json.loads((moved/'small_receipt.json').read_bytes())['graph_sha256']
        run('relocated_final_checker',std+["import sys,json;sys.path.insert(0,sys.argv[1]);from scripts.run_coefficient_dag_controls import relocated;from pathlib import Path;print(json.dumps(relocated(Path(sys.argv[2]),sys.argv[3])))",str(code),str(moved),anchor])
        for name,expected in fingerprints.items():
            if hashlib.sha256((repo/name).read_bytes()).hexdigest()!=expected:raise ValueError('code changed during review')
        for name,expected in parent_inputs.items():
            if hashlib.sha256((parent/name).read_bytes()).hexdigest()!=expected:raise ValueError('parent archive changed')
        status='PASS'
    except BaseException as exc:
        save(archive/'failure.json',dict(exception=type(exc).__name__,message=str(exc)));raise
    finally:
        payload=sum(p.stat().st_size for base in (parent,archive) for p in base.rglob('*') if p.is_file())
        summary=dict(status=status,calls=calls,code=fingerprints,parent_inputs=parent_inputs,
            observed_total_before_terminal_seconds=time.monotonic()-started,combined_payload_bytes_before_terminal=payload,
            real_models=0,gpu_calls=0,generated_objects=0,full_request_certificates=0)
        summary,raw=terminal_bytes(summary,payload,LIMIT)
        with (archive/'terminal.json').open('xb') as f:f.write(raw)
        if sum(p.stat().st_size for base in (parent,archive) for p in base.rglob('*') if p.is_file())>LIMIT:
            raise ValueError('combined archive cap; prefix retained')
    return summary


if __name__=='__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('archive',type=Path);args=p.parse_args()
    print(json.dumps(execute(args.parent.resolve(),args.archive.resolve()),indent=2))
