"""Explicit cache allowlist only. Default is a plan; no research artifacts removed.

Use the existing ACT Python with -I -S. No dependency imports or recursive rm.
Plans bind individual inodes/stats; apply rechecks them and skips active roots.
This is maintenance, not a background service or an experiment scheduler.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time

BASE = Path('/data1/Kane/MOE')
TARGETS = {
    'cache/pip/http-v2': 'http',
    'baseline_runs/dual_rs_install_20260921/pip_cache/http-v2': 'http',
    'baseline_runs/metamoe_install_20260921/pip_cache/http-v2': 'http',
    'baseline_runs/pip_cache/http-v2': 'http',
    'baseline_runs/robust_experts_gpu_environment_20260922_r1/pip_cache/http-v2': 'http',
    '.pycache': 'pyc',
    'cache/pycache': 'pyc',
    'cache/pycache-rt-er': 'pyc',
    'cache/rt-er-blackwell/pycache': 'pyc',
    'codex-stage2-cache.3ecCiu': 'pyc',
}
AGE = 86400
HTTP = re.compile(r'(?:[0-9a-f]/){5}[0-9a-f]{56}(?:\.body)?\Z')
PYC = re.compile(r'\.cpython-[0-9]+(?:\.opt-[0-9]+)?\.pyc\Z')


def identity(s):
    return [s.st_dev, s.st_ino, s.st_size, s.st_blocks, s.st_mtime_ns, s.st_ctime_ns]


def allowed(entry):
    target, relative = entry['target'], entry['relative']
    if target not in TARGETS or Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('outside fixed allowlist')
    kind = TARGETS[target]
    if kind == 'http' and not HTTP.fullmatch(relative):
        raise ValueError('not a pip HTTP cache record')
    if kind == 'pyc':
        if not PYC.search(relative):
            raise ValueError('not prefixed Python bytecode')
        # Do not remove orphan or sourceless bytecode: it is not proven rebuildable.
        source = Path('/') / PYC.sub('.py', relative)
        if not source.is_file():
            raise ValueError('Python source absent')


def parent_fd(base, relative):
    """Open each path component without following symlinks, including base."""
    path = base / relative
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child
        return fd, path.name
    except BaseException:
        os.close(fd)
        raise


def verify(base, entry, now):
    allowed(entry)
    fd, name = parent_fd(base, entry['target'] + '/' + entry['relative'])
    try:
        s = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if (not stat.S_ISREG(s.st_mode) or s.st_uid != os.getuid() or s.st_nlink != 1
                or s.st_mtime > now - AGE or identity(s) != entry['identity']):
            raise ValueError('changed, recent, shared, or nonregular cache file')
        return fd, name
    except BaseException:
        os.close(fd)
        raise


def process_identity(pid):
    proc = Path('/proc') / str(pid)
    raw = (proc / 'stat').read_text()
    return {'pid': pid, 'start_ticks': raw[raw.rfind(')') + 2:].split()[19],
            'comm': (proc / 'comm').read_text().strip(),
            'cmdline_sha256': hashlib.sha256((proc / 'cmdline').read_bytes()).hexdigest()}


def busy_roots(base, reviewed=()):
    """Read-only current-user process check. Never stop any process.

    This is an admission snapshot, not a filesystem lock; changed files are
    also checked individually. Never use while a concurrent installer runs.
    """
    busy = set()
    needles = {t: str(base / (t.removesuffix('/http-v2'))) for t in TARGETS}
    for proc in Path('/proc').iterdir():
        try:
            if not proc.name.isdigit() or proc.stat().st_uid != os.getuid() or int(proc.name) == os.getpid():
                continue
            values = [os.readlink(proc / 'cwd')]
            for item in (proc / 'fd').iterdir():
                try:
                    values.append(os.readlink(item))
                except FileNotFoundError:
                    pass
            values += (proc / 'maps').read_text().splitlines()
            # Environment variables identify a cache between two individual opens.
            values += (proc / 'environ').read_bytes().decode(errors='replace').split('\0')
            for target, needle in needles.items():
                if any(needle in value for value in values):
                    busy.add(target)
        except (FileNotFoundError, ProcessLookupError):
            pass
        except PermissionError:
            # Manual review is explicit and bound to PID/start time in the plan.
            # It does not claim access to this process's hidden open-file state.
            if int(proc.name) not in reviewed:
                busy.update(TARGETS)
    return sorted(busy)


def plan(base, now, busy):
    entries, skipped = [], Counter()
    for target in TARGETS:
        root = base / target
        if target in busy:
            skipped['active_root'] += 1
            continue
        if not root.exists():
            continue
        if root.resolve() != root:
            skipped['symlink_root'] += 1
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(directory) / d).is_symlink()]
            for name in files:
                path = Path(directory) / name
                entry = {'target': target, 'relative': str(path.relative_to(root)),
                         'identity': identity(path.lstat())}
                try:
                    fd, _ = verify(base, entry, now)
                    os.close(fd)
                    entries.append(entry)
                except (ValueError, OSError) as error:
                    skipped[str(error)] += 1
    return {'schema': 1, 'base': str(base), 'created_unix': now,
            'minimum_age_seconds': AGE, 'busy_roots': busy,
            'entries': sorted(entries, key=lambda e: (e['target'], e['relative'])),
            'skipped': dict(skipped)}


def apply(base, document, receipt, busy):
    if document['schema'] != 1 or document['base'] != str(base):
        raise ValueError('plan identity mismatch')
    entries = document['entries']
    keys = [(e['target'], e['relative']) for e in entries]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate entry')
    # Validate the entire plan before the first unlink.
    for entry in entries:
        if entry['target'] in busy:
            raise ValueError('cache is in use')
        fd, _ = verify(base, entry, time.time())
        os.close(fd)
    for entry in entries:
        fd, name = verify(base, entry, time.time())
        try:
            os.unlink(name, dir_fd=fd)
        finally:
            os.close(fd)
        receipt.write(json.dumps(entry, sort_keys=True) + '\n')
        receipt.flush()
    os.fsync(receipt.fileno())


def summary(entries):
    totals = {}
    for e in entries:
        row = totals.setdefault(e['target'], {'files': 0, 'logical_bytes': 0, 'allocated_bytes': 0})
        row['files'] += 1
        row['logical_bytes'] += e['identity'][2]
        row['allocated_bytes'] += e['identity'][3] * 512
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--plan', type=Path, help='create new dry-run JSON')
    group.add_argument('--apply', type=Path, help='apply exactly this reviewed plan')
    parser.add_argument('--receipt', type=Path, help='fresh local JSONL unlink receipt')
    parser.add_argument('--reviewed-unreadable-pid', type=int, action='append', default=[],
                        help='explicitly reviewed unrelated service, recorded in plan; never implicit')
    args = parser.parse_args()
    output = args.plan or args.receipt
    if output is None or not output.is_absolute() or not output.resolve().is_relative_to(BASE):
        parser.error('fresh output must be inside /data1/Kane/MOE')
    if output.exists() or output.is_symlink():
        parser.error('output exists; never overwrite receipts')
    reviewed = [process_identity(pid) for pid in sorted(set(args.reviewed_unreadable_pid))]
    busy = busy_roots(BASE, args.reviewed_unreadable_pid)
    if args.plan:
        document = plan(BASE, time.time(), busy)
        document['reviewed_unreadable_processes'] = reviewed
        with output.open('x') as stream:
            json.dump(document, stream, sort_keys=True, indent=2)
        print(json.dumps({'busy_roots': busy, 'skipped': document['skipped'],
                          'planned': summary(document['entries'])}, indent=2))
    else:
        document = json.loads(args.apply.read_text())
        if document.get('reviewed_unreadable_processes') != reviewed:
            parser.error('reviewed process identities changed')
        with output.open('x') as stream:
            apply(BASE, document, stream, busy)
        print(json.dumps({'deleted': summary(document['entries']),
                          'receipt': str(output)}, indent=2))


if __name__ == '__main__':
    main()
