"""Append-only JSON artifacts and small, bounded supervisor receipts."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = '/data1/Kane/miniconda3/envs/act-py312/bin/python'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def load(path, digest=None, limit=None):
    p = Path(path)
    if p.is_symlink() or (limit is not None and p.stat().st_size > limit):
        raise ValueError('symlink/oversized receipt')
    raw = p.read_bytes()
    if digest is not None and hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError('file identity mismatch: '+str(p))
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result: raise ValueError('duplicate JSON field')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def save(path, value):
    p = Path(path); partial = p.with_name(p.name+'.partial')
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    with partial.open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    # Atomic, exclusive publication: no historical artifact can be overwritten.
    os.link(partial, p); partial.unlink()
    return {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def tick(deadline):
    if time.monotonic() >= deadline: raise TimeoutError('shared request deadline')


class Events:
    def __init__(self, root, phase, start):
        self.path = Path(root)/(phase+'_events.jsonl'); self.start = start

    def emit(self, event, **fields):
        with self.path.open('a') as f:
            f.write(json.dumps({'event': event, 'elapsed': time.monotonic()-self.start, **fields}, allow_nan=False)+'\n')
            f.flush()

    def call(self, name, fn):
        start = time.monotonic(); self.emit('ENTER', operation=name)
        try:
            result = fn()
        except BaseException as exc:
            self.emit('EXIT_ERROR', operation=name, seconds=time.monotonic()-start, error=repr(exc)); raise
        self.emit('EXIT', operation=name, seconds=time.monotonic()-start)
        return result
