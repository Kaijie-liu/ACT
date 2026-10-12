"""One finite source-intake archive, not a production request supervisor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

FILES = ('scoped_source/tensor_rows.py', 'scoped_source/tensor_row_controls.py',
         'scoped_source/factored_source.py', 'scoped_source/factored_io.py',
         'source_enclosure/format.py', 'upstream_source/checker.py',
         'scripts/test_tensor_rows.py', 'scripts/run_tensor_rows_controls.py',
         'docs/hz_tensor_rows_design_20261013.md')
LIMIT = 64*2**20


def terminal_bytes(summary, payload, limit=LIMIT):
    """Refuse before publishing PASS, including the terminal's own bytes."""
    summary = dict(summary)
    def encoded():
        return (json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
    raw = encoded()
    if payload+len(raw) > limit:
        summary['status'] = 'ARCHIVE_LIMIT_REFUSED'
        raw = encoded()
    return summary, raw


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def worker(root):
    from scoped_source.tensor_row_controls import fixture, oracle
    start = time.monotonic()
    root.mkdir()
    results = []
    for name, large, chunk in [('small40', False, 40), ('small1m', False, 1048576),
                               ('dense', True, 1048576)]:
        begin = time.monotonic()
        sha = fixture(root/name, large=large, chunk_bytes=chunk)
        generated = time.monotonic()
        checked = oracle(root/name, sha, large=large)
        item = {'name': name, 'source_sha256': sha, 'generation_seconds': generated-begin,
                **checked, 'generation_and_check_seconds': time.monotonic()-begin}
        save(root/(name+'.json'), item)
        results.append(item)
    return {'status': 'CHECKED_CAPTURED_ROWS', 'cases': results,
            'worker_seconds': time.monotonic()-start, 'real_models': 0, 'gpu_calls': 0,
            'safety_certificates': 0}


def execute(archive):
    start = time.monotonic()
    repo = Path(__file__).resolve().parents[1]
    archive.mkdir(parents=True, exist_ok=False)
    code = archive/'code'
    identities = {}
    for name in FILES:
        target = code/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo/name, target)
        identities[name] = hashlib.sha256(target.read_bytes()).hexdigest()
    save(archive/'execution.json', {'code': identities, 'python': sys.executable,
                                   'per_process_deadline_seconds': 300, 'archive_cap': LIMIT})
    calls = []

    def run(name, command):
        begin = time.monotonic()
        item = {'name': name, 'command': command, 'hard_cap_seconds': 300}
        calls.append(item)
        try:
            result = subprocess.run(command, cwd=archive, capture_output=True, text=True, timeout=300)
            item.update(returncode=result.returncode, status='COMPLETE' if result.returncode == 0 else 'ERROR')
            (archive/(name+'.stdout')).write_text(result.stdout)
            (archive/(name+'.stderr')).write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(name+' failed; prefix retained')
            return result.stdout
        except subprocess.TimeoutExpired as exc:
            item.update(status='TIMEOUT', returncode=None)
            (archive/(name+'.stdout')).write_bytes(exc.stdout or b'')
            (archive/(name+'.stderr')).write_bytes(exc.stderr or b'')
            raise
        finally:
            item['parent_observed_seconds'] = time.monotonic()-begin

    launcher = [sys.executable, '-B', '-I', '-S', '-c']
    try:
        run('unit_controls', launcher+[
            "import sys,unittest;sys.path.insert(0,sys.argv[1]);unittest.main(module='scripts.test_tensor_rows',argv=['controls','-v'])",
            str(code)])
        raw = run('sources', launcher+[
            "import sys;sys.path.insert(0,sys.argv[1]);from scripts.run_tensor_rows_controls import worker;from pathlib import Path;import json;print(json.dumps(worker(Path(sys.argv[2]))))",
            str(code), str(archive/'sources')])
        generated = json.loads(raw)
        relocation = archive/'relocated'
        shutil.copytree(code, relocation/'code')
        shutil.copytree(archive/'sources'/'small40', relocation/'source')
        sha = generated['cases'][0]['source_sha256']
        raw = run('relocated', launcher+[
            "import sys,json;sys.path.insert(0,sys.argv[1]);from scoped_source.tensor_row_controls import oracle;r=oracle(sys.argv[2],sys.argv[3]);assert not {'torch','numpy','scipy','highspy'}&set(sys.modules);print(json.dumps(r))",
            str(relocation/'code'), str(relocation/'source'), sha])
        relocated = json.loads(raw)
        if relocated['receipt'] != generated['cases'][0]['receipt']:
            raise ValueError('relocated receipt differs')
        status = 'PASS'
    except BaseException as exc:
        status = 'FAILED'
        save(archive/'failure.json', {'exception': type(exc).__name__, 'message': str(exc)})
        raise
    finally:
        payload = sum(p.stat().st_size for p in archive.rglob('*') if p.is_file())
        summary = {'status': status, 'calls': calls, 'code': identities,
                   'observed_total_before_terminal_seconds': time.monotonic()-start,
                   'payload_bytes_before_terminal': payload, 'archive_cap': LIMIT,
                   'real_models': 0, 'gpu_calls': 0, 'safety_certificates': 0}
        summary, raw = terminal_bytes(summary, payload)
        with (archive/'terminal.json').open('xb') as stream:
            stream.write(raw)
        if sum(p.stat().st_size for p in archive.rglob('*') if p.is_file()) > LIMIT:
            raise ValueError('finite archive cap exceeded; prefix retained')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    print(json.dumps(execute(args.archive.resolve()), indent=2))
