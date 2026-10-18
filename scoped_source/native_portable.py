"""Fresh stdlib portable entry; offline checks never promote online timeouts.

Run with python -I -S -B and externally authenticated bindings. Authenticate this
bootstrap itself before executing a copied package; self-hashes are not trust.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time


def main():
    started = time.monotonic()
    if len(sys.argv) != 4 or sys.argv[1] != '--enabled':
        raise ValueError('explicit portable opt-in, package and external bindings required')
    root = Path(sys.argv[2]); external = Path(sys.argv[3])
    if not external.is_file() or external.is_symlink() or external.stat().st_size > 2**20:
        raise ValueError('bounded external bindings')
    b = json.loads(external.read_bytes())
    keys={'bundle','method','invocation','source','code'}
    if set(b) not in (keys, keys|{'budget'}) or not isinstance(b['code'],dict):
        raise ValueError('closed externally authenticated bindings')
    deadline=started+300; initial=None
    if 'budget' in b:
        budget=b['budget']
        if (type(budget) is not dict or set(budget)!={'initial_work','deadline'}
                or type(budget['deadline']) not in (int,float)
                or not started < budget['deadline'] <= deadline):
            raise ValueError('inherited portable budget only tightens deadline')
        deadline=budget['deadline']; initial=budget['initial_work']
    code = root/'code'
    if (not root.is_absolute() or root.resolve() != root or code.is_symlink()
            or any(p.is_symlink() for p in (root, *root.parents))):
        raise ValueError('canonical portable root')
    expected_dirs = set()
    for name, h in b['code'].items():
        if type(name) is not str or str(Path(name)) != name:
            raise ValueError('canonical code member')
        expected_dirs.update(str(p) for p in Path(name).parents if str(p) != '.')
        p = code/name
        if (Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink()
                or any(q.is_symlink() for q in p.parents)
                or not p.is_file() or p.stat().st_nlink != 1 or p.stat().st_size > 65536
                or hashlib.sha256(p.read_bytes()).hexdigest() != h):
            raise ValueError('portable checker externally bound')
        if time.monotonic() >= deadline: raise TimeoutError('portable total budget')
    # Reject shadow modules and package __init__ files BEFORE any package import.
    actual = set(); inodes = set()
    for directory, dirs, files in os.walk(code, followlinks=False):
        if time.monotonic() >= deadline: raise TimeoutError('portable inventory budget')
        for name in dirs:
            p = Path(directory)/name
            if p.is_symlink() or p.relative_to(code).as_posix() not in expected_dirs:
                raise ValueError('extra/aliased portable directory')
        for name in files:
            p = Path(directory)/name; s = p.stat(follow_symlinks=False)
            key = (s.st_dev,s.st_ino)
            if (not stat.S_ISREG(s.st_mode) or s.st_nlink != 1 or key in inodes
                    or p.relative_to(code).as_posix() not in b['code']):
                raise ValueError('extra/aliased portable code')
            inodes.add(key); actual.add(p.relative_to(code).as_posix())
    if actual != set(b['code']): raise ValueError('complete portable code inventory')
    sys.path.insert(0,str(code))
    from scoped_source.hz_request_store import Work, POLICY, compact, require
    from scoped_source.native_package import check_package
    work=Work(deadline)
    if initial is not None:
        require(type(initial) is dict and set(initial)==set(work.counts)
                and all(type(v) is int and v>=0 for v in initial.values())
                and sum(initial.values())<=POLICY['visits'],'inherited cumulative portable Work')
        work.counts.update(initial)
    try:
        result = check_package(root,expected=b['bundle'],expected_method=b['method'],invocation=b['invocation'],
            expected_source=b['source'],expected_code=b['code'],work=work,enabled=True)
    except BaseException as exc:
        print(compact(dict(status='ERROR',work=dict(work.counts),online_acceptance=False,
            error=dict(kind=type(exc).__name__,message=str(exc)[:300]))).decode())
        raise
    if any(n.split('.')[0] in ('torch','numpy','scipy','act') for n in sys.modules):
        raise ValueError('portable numerical import')
    result.update(online_acceptance=False,scope='OFFLINE_CHECK_OF_DECLARED_SOURCE',seconds=time.monotonic()-started)
    raw = compact(result)
    if len(raw) > 2**20 or time.monotonic() >= deadline: raise TimeoutError('portable final budget')
    print(raw.decode())


if __name__ == '__main__': main()
