"""Bounded, strict serialization for the opt-in factored H2 controls."""
import hashlib
import json
from pathlib import Path
from source_enclosure.format import compact

MEMBER_LIMIT = 16*2**20
HEADER_LIMIT = 4*2**20
BLOCK_LIMIT = 2**20
TOTAL_LIMIT = 2*2**30


def digest(raw): return hashlib.sha256(raw).hexdigest()


def required_hash(value):
    if type(value) is not str or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('mandatory member digest')
    return value


def read(root, name, limit=MEMBER_LIMIT, expected=None):
    p = Path(name)
    if p.is_absolute() or not p.parts or any(v in ('..', '.', '') for v in p.parts) or str(p) != name:
        raise ValueError('noncanonical member path')
    root = Path(root)
    if root.is_symlink(): raise ValueError('symlink root')
    path = root
    for part in p.parts:
        path = path/part
        if path.is_symlink(): raise ValueError('symlink member')
    if not path.is_file() or path.stat().st_size > limit: raise ValueError('member absent or oversized')
    with path.open('rb') as f: raw = f.read(limit+1)
    if len(raw) > limit or expected is not None and digest(raw) != expected:
        raise ValueError('member size/identity')
    return raw


def decode(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result: raise ValueError('duplicate JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def load(root, name, limit=MEMBER_LIMIT, expected=None):
    return decode(read(root, name, limit, expected))


def referenced(root, ref, limit=MEMBER_LIMIT):
    if type(ref['bytes']) is not int or not 0 < ref['bytes'] <= limit:
        raise ValueError('referenced member length')
    raw=read(root,ref['file'],limit,required_hash(ref['sha256']))
    if len(raw)!=ref['bytes']: raise ValueError('referenced actual length')
    return decode(raw)


def write(root, name, value, limit=MEMBER_LIMIT, binary=False):
    raw = value if binary else compact(value)
    if len(raw) > limit: raise ValueError('output member too large')
    path = Path(root)/name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f: f.write(raw)
    return {'file':name, 'bytes':len(raw), 'sha256':digest(raw)}


def inventory(root, expected, total_limit=TOTAL_LIMIT):
    actual = set(); total = 0
    for p in Path(root).rglob('*'):
        if p.is_symlink(): raise ValueError('symlink inventory')
        if p.is_file():
            actual.add(str(p.relative_to(root))); total += p.stat().st_size
    if actual != set(expected) or total > total_limit:
        raise ValueError('missing/extra members or total size')
    return total
