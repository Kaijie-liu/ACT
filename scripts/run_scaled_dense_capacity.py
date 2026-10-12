"""Once-only full V3 dense capacity query; no real-model or GPU admission."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
DESIGN = 'docs/hz_scaled_dense_design_20261013.md'
PROTOCOL = '00210f1470fd187c777ea0d1fffd7813d425cc1bfbad346a09be929d86773785'
SOURCE = '0745027b58f00c608a9db921204cb2feae58273600fee658d4c9ccc3d8c19dbc'
SOURCE_ROOT = Path('/data1/Kane/MOE/runs/hz_tensor_rows_20261013_r1/sources/dense')
ARCHIVE = Path('/data1/Kane/MOE/runs/hz_scaled_dense_20261013_r1')
LIMIT = 128 * 2**20
RSS_LIMIT = 8 * 2**30
AS_LIMIT = 16 * 2**30
PHASES = ('imports', 'source_copy', 'source_build_encode', 'forest_publish',
          'source_check', 'complete_candidate_and_check', 'proof_binding_publish')
from scripts.run_scaled_endpoint_controls import FILES as PREVIOUS_FILES
FILES = tuple(sorted(set(PREVIOUS_FILES + (DESIGN, 'scripts/run_scaled_dense_capacity.py',
    'scripts/audit_scaled_dense_capacity.py', 'scripts/test_scaled_dense_capacity.py'))))


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def inventory(root):
    result = {}
    for p in sorted(Path(root).rglob('*')):
        if p.is_symlink(): raise ValueError('archive symlink')
        if p.is_file():
            h = hashlib.sha256()
            with p.open('rb') as stream:
                for chunk in iter(lambda: stream.read(2**20), b''): h.update(chunk)
            result[str(p.relative_to(root))] = dict(bytes=p.stat().st_size, sha256=h.hexdigest())
    return result


def reserve(root, added):
    """Reserve terminal/log space before large writes or relocation copies."""
    used = sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())
    if used + added + 2*2**20 > LIMIT: raise ValueError('archive capacity before publication')


def rss(pid):
    try:
        for line in Path(f'/proc/{pid}/status').read_text().splitlines():
            if line.startswith('VmRSS:'): return int(line.split()[1]) * 1024
    except FileNotFoundError: pass
    return 0


def supervise(root, name, command, *, cwd, env, budget=300, memory=RSS_LIMIT, receiver=None):
    """Own one process group. Logs are files, not an unbounded capture buffer."""
    root = Path(root); start = time.monotonic(); deadline = start + budget
    result = dict(name=name, hard_cap_seconds=budget, rss_cap_bytes=memory,
                  status='ERROR', returncode=None, sampled_peak_rss_bytes=0)
    proc = None
    try:
        args = command(deadline) if callable(command) else command
        result['argv'] = args
        with (root/(name+'.stdout')).open('xb') as out, (root/(name+'.stderr')).open('xb') as err:
            proc = subprocess.Popen(args, cwd=cwd, env=env, stdout=out, stderr=err, start_new_session=True)
            while True:
                result['sampled_peak_rss_bytes'] = max(result['sampled_peak_rss_bytes'], rss(proc.pid))
                if time.monotonic() >= deadline: result['status'] = 'TIMEOUT'; break
                if result['sampled_peak_rss_bytes'] > memory: result['status'] = 'MEMORY_LIMIT'; break
                code = proc.poll()
                if code is not None:
                    result.update(returncode=code, status='COMPLETE' if code == 0 else 'ERROR'); break
                time.sleep(min(.05, max(0., deadline-time.monotonic())))
            if proc.poll() is None: os.killpg(proc.pid, signal.SIGKILL)
            result['returncode'] = proc.wait()
        result['parent_observed_seconds'] = time.monotonic()-start
        if result['status'] == 'COMPLETE' and receiver is not None:
            result['reception'] = receiver(result)
    except BaseException as exc:
        result.update(status='ERROR', exception=type(exc).__name__, message=str(exc))
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL); proc.wait()
        result['parent_observed_seconds'] = time.monotonic()-start
        if result['parent_observed_seconds'] >= budget: result['status'] = 'TIMEOUT'
        # This is a pre-publication snapshot, never an alternative accepting receipt.
        save(root/(name+'.call.json'), result)
        result['parent_observed_seconds'] = time.monotonic()-start
        if result['parent_observed_seconds'] >= budget: result['status'] = 'TIMEOUT'
    return result


def worker(root, deadline):
    started = time.monotonic(); root = Path(root); root.mkdir(exist_ok=False)
    resource.setrlimit(resource.RLIMIT_AS, (AS_LIMIT, AS_LIMIT))
    phases = {}; w = None; active = 'imports'; phase_start = started; number = 0
    receipt = dict(status='INCOMPLETE', source=SOURCE, real_models=0, gpu_calls=0)

    def checkpoint(stage, **extra):
        nonlocal number
        number += 1
        save(root/f'progress-{number:02d}.json', dict(stage=stage, source=SOURCE,
            elapsed_seconds=time.monotonic()-started, phases=dict(phases),
            work=None if w is None else dict(w.counts), **extra))

    def finished(next_phase):
        nonlocal active, phase_start
        phases[active] = time.monotonic()-phase_start
        checkpoint(active)
        active = next_phase; phase_start = time.monotonic()

    try:
        from scoped_source.hz_request_store import Work, compact, digest, identity, require
        w = Work(deadline)
        import torch
        torch.set_num_threads(1)
        from unittest.mock import patch
        from scoped_source.scaled_references import build
        from scoped_source.check_scaled_references import CheckedScaledForest
        from scoped_source.scaled_endpoints import contract
        from act.back_end.moe.scaled_forest_endpoints import produce
        finished('source_copy')
        bundle = root/'bundle'; bundle.mkdir(); source = bundle/'source'
        # The complete captured declaration is fixed, not regenerated or substituted.
        if SOURCE_ROOT.is_symlink() or any(p.is_symlink() for p in SOURCE_ROOT.rglob('*')):
            raise ValueError('captured source symlink')
        reserve(root.parent, sum(p.stat().st_size for p in SOURCE_ROOT.rglob('*') if p.is_file()))
        shutil.copytree(SOURCE_ROOT, source); w.tick()
        finished('source_build_encode')
        raw, generation = build(source, SOURCE, work=w)
        finished('forest_publish')
        reserve(root.parent, len(raw))
        with (bundle/'forest.json').open('xb') as stream: stream.write(raw)
        w.tick(); finished('source_check')
        f = CheckedScaledForest(raw, source, SOURCE, expected=digest(raw), work=w)
        source_work = dict(w.counts); finished('complete_candidate_and_check')
        c = contract(f); partial = root/'partial'; partial.mkdir()

        def event(i, data):
            reserve(root.parent, len(data))
            with (partial/f'{i:05d}.json').open('xb') as stream: stream.write(data)
            checkpoint('raw_candidate_before_exact_acceptance', tile=i)

        with patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CUDA forbidden')):
            proof, result, native = produce(f, c, work=w, width=8, event=event)
        require(result['required_endpoints'] == 2 and result['checked_endpoints'] == 2
                and result['required'] == 1 and result['missing_endpoints'] == 0
                and len(result['results']) == 1 and result['results'][0]['pair'] == [0,1]
                and native['candidates'] == 4,
                'fixed complete endpoint inventory')
        finished('proof_binding_publish')
        for name, value in (('contract.json', c), ('proof.json', proof)):
            data = compact(value); reserve(root.parent, len(data))
            with (bundle/name).open('xb') as stream: stream.write(data)
        binding = dict(schema='CAPTURED_SCALED_COMPLETE_BINDING_V3', source=SOURCE,
                       forest=digest(raw), contract=identity(c), proof=identity(proof))
        with (bundle/'binding.json').open('xb') as stream: stream.write(compact(binding))
        w.tick(); finished(None)
        receipt.update(status='COMPLETE_DENSE_QUERY', forest=digest(raw), binding=identity(binding),
            result=result, stats=dict(f.receipt['stats']), generation=generation,
            native=native, source_work=source_work, production_work=dict(w.counts),
            environment=dict(python=sys.version, torch=torch.__version__,
                             threads=torch.get_num_threads(), address_space_cap_bytes=AS_LIMIT))
    except BaseException as exc:
        if active is not None: phases[active] = time.monotonic()-phase_start
        status = ('TIMEOUT' if isinstance(exc, TimeoutError) else 'MEMORY_LIMIT' if isinstance(exc, MemoryError)
                  else 'WORK_LIMIT' if 'work capacity' in str(exc) else 'REFUSED')
        receipt.update(status=status, exception=type(exc).__name__, message=str(exc), stopped_phase=active)
        raise
    finally:
        receipt.update(phases=phases, worker_before_receipt_seconds=time.monotonic()-started,
            worker_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            work=None if w is None else dict(w.counts))
        save(root/'receipt.json', receipt)
    return receipt


def received(call, root):
    """Late, partial or absent receipts cannot form an accepted terminal."""
    if call['status'] != 'COMPLETE': return 'INCOMPLETE_QUERY'
    path = Path(root)/'receipt.json'
    if not path.is_file(): return 'INCOMPLETE_QUERY'
    r = json.loads(path.read_bytes())
    if r['status'] != 'COMPLETE_DENSE_QUERY': return 'INCOMPLETE_QUERY'
    if r['worker_before_receipt_seconds'] > call['parent_observed_seconds']: raise ValueError('worker/parent cost')
    if set(r['phases']) != set(PHASES): raise ValueError('complete phase inventory')
    if any(not isinstance(x, (int,float)) or not math.isfinite(x) or x < 0 for x in r['phases'].values()):
        raise ValueError('invalid phase cost')
    if sum(r['phases'].values()) > r['worker_before_receipt_seconds']: raise ValueError('uncounted worker cost')
    counts = [r[k] for k in ('source_work', 'production_work', 'work')]
    for c in counts:
        if set(c) != {'row_visits','coefficient_visits','arithmetic'} or any(type(v) is not int or v < 0 for v in c.values()) or sum(c.values()) > 8000000:
            raise ValueError('unchanged complete Work contract')
    if any(a[k] > b[k] for a,b in zip(counts,counts[1:]) for k in a): raise ValueError('work reset')
    if not 0 < r['worker_peak_rss_bytes'] <= RSS_LIMIT: raise ValueError('peak RSS')
    return 'COMPLETE_QUERY'


def run_controls(root, env):
    call = supervise(root,'supervisor_controls',[sys.executable,'-B','-I','-c',
        'import sys,unittest;sys.path.insert(0,sys.argv[1]);'
        'unittest.main(module="scripts.test_scaled_dense_capacity",argv=["controls","-v"])',
        str(root/'code')],cwd=root,env=env)
    save(root/'supervisor_controls.json',call)
    if call['status']!='COMPLETE': raise ValueError('supervisor controls failed; no query')


def execute(root):
    from scripts.run_tensor_rows_controls import terminal_bytes
    root = Path(root).resolve()
    if root != ARCHIVE: raise ValueError('fixed new execution directory')
    if subprocess.check_output(['git','status','--porcelain'], cwd=REPO): raise ValueError('commit clean code first')
    if hashlib.sha256((REPO/DESIGN).read_bytes()).hexdigest() != PROTOCOL: raise ValueError('frozen protocol')
    available = next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
    disk = shutil.disk_usage(root.parent)
    if available < AS_LIMIT or disk.free < 2**30: raise ValueError('CPU memory/disk admission')
    started = time.monotonic(); root.mkdir(exist_ok=False); calls = []; code = {}; status = 'INCOMPLETE_QUERY'
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
    try:
        for name in FILES:
            p = root/'code'/name; p.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(REPO/name, p)
            code[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        save(root/'execution.json', dict(schema='SCALED_DENSE_CAPACITY_V3', protocol=PROTOCOL, code=code,
            git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
            source=SOURCE, width=8, work_cap=8000000, archive_cap=LIMIT, rss_cap=RSS_LIMIT, as_cap=AS_LIMIT,
            python=sys.executable, available_memory_bytes=available, disk_free_bytes=disk.free,
            real_models=0, gpu='NOT_POLLED_OR_RUN',
            environment={k:env[k] for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','CUDA_VISIBLE_DEVICES')}))
        run_controls(root,env)
        call = supervise(root, 'query', lambda deadline:[sys.executable,'-B','-I','-c',
            'import sys;sys.path.insert(0,sys.argv[1]);from scripts.run_scaled_dense_capacity import worker;worker(sys.argv[2],float(sys.argv[3]))',
            str(root/'code'), str(root/'query'), str(deadline)], cwd=root, env=env,
            receiver=lambda call: received(call,root/'query'))
        calls.append(call)
        status = call.get('reception','INCOMPLETE_QUERY') if call['status']=='COMPLETE' else 'INCOMPLETE_QUERY'
        if status == 'COMPLETE_QUERY':
            r = json.loads((root/'query/receipt.json').read_bytes())
            # Offline relocation preparation is charged separately, never a second solve.
            begin = time.monotonic(); dest = root/'relocated'; dest.mkdir()
            reserve(root, sum(p.stat().st_size for d in (root/'code',root/'query/bundle')
                              for p in d.rglob('*') if p.is_file()))
            shutil.copytree(root/'code', dest/'code'); shutil.copytree(root/'query/bundle', dest/'bundle')
            save(root/'relocation_copy.json', dict(seconds=time.monotonic()-begin))
            call = supervise(root, 'relocated', lambda deadline:[sys.executable,'-B','-I','-S','-c',
                'import sys,json,resource;sys.path.insert(0,sys.argv[1]);'
                'resource.setrlimit(resource.RLIMIT_AS,(16*2**30,16*2**30));'
                'from scoped_source.verify_scaled_endpoints import verify;'
                'r=verify(sys.argv[2],sys.argv[3],deadline=float(sys.argv[4]));'
                'assert not {"torch","numpy","scipy","highspy"}&set(sys.modules);print(json.dumps(r))',
                str(dest/'code'),str(dest/'bundle'),r['binding'],str(deadline)],cwd=dest,env=env,
                receiver=lambda call: json.loads((root/'relocated.stdout').read_bytes()))
            calls.append(call)
            if call['status'] == 'COMPLETE':
                result = call['reception']
                if result['result'] != r['result']: raise ValueError('portable result mismatch')
                status = 'COMPLETE_QUERY_AND_PORTABLE_CHECK'
            else: status = 'COMPLETE_QUERY_OFFLINE_CHECK_INCOMPLETE'
    except BaseException as exc:
        status = 'ERROR'; save(root/'failure.json', dict(exception=type(exc).__name__,message=str(exc)))
    finally:
        files = inventory(root); size = sum(v['bytes'] for v in files.values())
        result = dict(status=status, calls=calls, code=code, files=files, archive_cap=LIMIT,
            payload_before_terminal_bytes=size, outer_before_terminal_seconds=time.monotonic()-started,
            real_models=0, gpu_calls=0)
        result, raw = terminal_bytes(result,size,LIMIT)
        with (root/'terminal.json').open('xb') as stream: stream.write(raw)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('root',type=Path)
    r=execute(p.parse_args().root); print(json.dumps({k:v for k,v in r.items() if k not in ('files','code')},indent=2))
