"""V2 resource primitives; no default capacity, approval, model or solver.

Reservations describe named requested objects, NOT allocator RSS. Their bounds
must be reviewed and externally frozen before a run can be authorized. This
module does not close the full source/forest heap bound by measuring small inputs.
"""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
from types import MappingProxyType

from scoped_source.hz_request_store import POLICY, decode, require, sha


def tick(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('RESOURCE_DEADLINE')


def code(exc):
    # Never format an arbitrary exception, its args, or its traceback.
    if isinstance(exc, TimeoutError): return 'DEADLINE'
    if isinstance(exc, MemoryError): return 'MEMORY'
    if isinstance(exc, OSError): return 'IO'
    return 'REJECTED'


def bounded_read(path, cap, deadline, expected=None):
    """Stable regular file, bounded actual read including one sentinel byte."""
    tick(deadline)
    require(type(cap) is int and 0 < cap <= POLICY['total_bytes'], 'read reservation')
    p = Path(path)
    require(p.is_absolute() and p.resolve() == p
            and not any(q.is_symlink() for q in (p, *p.parents)), 'unaliased absolute path')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
                and 0 <= before.st_size <= cap, 'bounded regular input')
        # Reserve cap+1 in the caller before entry; never trust stat as a read cap.
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(cap + 1)
        tick(deadline)
        after = os.fstat(fd); named = p.stat(follow_symlinks=False)
        fields = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)
        require(len(raw) <= cap and len(raw) == before.st_size
                and fields(before) == fields(after) == fields(named)
                and not any(q.is_symlink() for q in (p, *p.parents)), 'input changed or grew')
        h = hashlib.sha256(raw).hexdigest()
        require(expected is None or h == sha(expected), 'bound input hash')
        tick(deadline)
        return raw, h
    finally:
        os.close(fd)


def lexical(raw, deadline, max_depth):
    """Cost preflight, NOT a replacement JSON/schema parser.

    Decode exactly the encodings accepted by json.loads(bytes), including BOMs.
    The enclosing slot reserves raw, decoded text, decoder and canonical copies
    BEFORE calling this function. This counts lexical occurrences, not unique
    Python objects; optional opaque trust fields are deliberately included.
    """
    require(type(raw) is bytes and type(max_depth) is int and max_depth > 0, 'lexical input')
    text = raw.decode(json.detect_encoding(raw), 'surrogatepass')
    depth = peak = strings = structural = token_chars = 0
    quoted = escaped = False
    for ch in text:
        tick(deadline)
        if quoted:
            token_chars += 1
            if escaped: escaped = False
            elif ch == '\\': escaped = True
            elif ch == '"': quoted = False
        elif ch == '"':
            quoted = True; strings += 1
        elif ch in '[{':
            depth += 1; peak = max(peak, depth); structural += 1
            require(depth <= max_depth, 'pre-parse depth reservation')
        elif ch in ']}':
            depth -= 1; structural += 1
            require(depth >= 0, 'pre-parse nesting')
        elif not ch.isspace():
            token_chars += 1
    require(depth == 0 and not quoted, 'pre-parse incomplete JSON')
    return dict(raw_bytes=len(raw), text_chars=len(text), max_depth=peak,
                strings=strings, structural=structural, token_chars=token_chars)


def tokens(value, deadline, depth_limit, _active=None, _depth=0, *, integer_bits=None):
    """Canonical ASCII JSON chunks, exact builtins only; no whole-string dump.

    A repeated alias is encoded again, while an ancestral cycle is rejected.
    Sorting/ancestry and result objects must already have an enclosing slot.
    """
    tick(deadline)
    require(_depth <= depth_limit, 'encoding depth reservation')
    if _active is None: _active = set()
    kind = type(value)
    if value is None: yield b'null'
    elif kind is bool: yield b'true' if value else b'false'
    elif kind is int:
        # JSON provenance is not a mathematical rational acceptance gate.
        # Source integers inherit the bounded raw grammar/stdlib decoder;
        # internal metadata can request a separate representation ceiling.
        require(integer_bits is None or value.bit_length() <= integer_bits, 'integer representation bound')
        yield str(value).encode('ascii')
    elif kind is float:
        require(math.isfinite(value), 'finite metadata float')
        yield json.dumps(value, allow_nan=False).encode('ascii')
    elif kind is str:
        yield b'"'
        for ch in value:
            tick(deadline); n = ord(ch)
            if ch == '"': yield b'\\"'
            elif ch == '\\': yield b'\\\\'
            elif ch in '\b\f\n\r\t': yield {'\b':b'\\b','\f':b'\\f','\n':b'\\n','\r':b'\\r','\t':b'\\t'}[ch]
            elif 32 <= n <= 126: yield bytes((n,))
            elif n <= 65535: yield ('\\u%04x' % n).encode('ascii')
            else:
                n -= 65536
                yield ('\\u%04x\\u%04x' % (55296+(n >> 10), 56320+(n & 1023))).encode('ascii')
        yield b'"'
    elif kind in (dict, list, tuple):
        require(id(value) not in _active, 'cyclic metadata')
        _active.add(id(value))
        try:
            yield b'{' if kind is dict else b'['
            if kind is dict:
                require(all(type(k) is str for k in value), 'plain string keys')
                items = sorted(value)
            else: items = range(len(value))
            for i, key in enumerate(items):
                tick(deadline)
                if i: yield b','
                if kind is dict:
                    yield from tokens(key, deadline, depth_limit, _active, _depth+1, integer_bits=integer_bits)
                    yield b':'
                yield from tokens(value[key], deadline, depth_limit, _active, _depth+1, integer_bits=integer_bits)
            yield b'}' if kind is dict else b']'
        finally:
            _active.remove(id(value))
    else:
        raise ValueError('non-builtin metadata')


def measure(value, deadline, cap, depth_limit):
    """Complete encoding traversal without materializing the encoded object."""
    n = 0; h = hashlib.sha256()
    for part in tokens(value, deadline, depth_limit):
        n += len(part); require(n <= cap, 'metadata encoded reservation'); h.update(part)
    tick(deadline)
    return dict(bytes=n, sha256=h.hexdigest())


def materialize(value, resources, slot, cap):
    """Bounded adapter for unchanged byte-oriented archive writers.

    The enclosing phase owns returned bytes. This reservation covers transient
    count/sort/fill/freeze overlap, not a transfer of their lifetime to nowhere.
    """
    with resources.hold(slot) as spec:
        require(cap <= spec['encoded'], 'materialization reservation')
        ref = measure(value,resources.deadline,cap,spec['depth'])
        out = bytearray(ref['bytes']); pos = 0
        for part in tokens(value,resources.deadline,spec['depth']):
            require(pos+len(part) <= len(out), 'changed materialization')
            out[pos:pos+len(part)] = part; pos += len(part)
        raw = bytes(out)
        require(pos == len(out) and hashlib.sha256(raw).hexdigest() == ref['sha256'], 'materialization identity')
        return raw


class Resources:
    """Finite typed reservation roster, immutable bindings, shared lease counts.

    The freeze supplies absolute requested-object reservations and a reviewed
    joint caller/controller/actor envelope. Neither artifact cap nor RLIMIT_AS
    is used as a fabricated heap allowance. No allocation-by-allocation journal.
    """
    def __init__(self, plan, binding, role, deadline):
        require(type(plan) is dict and set(plan) == {'schema','capacity','joint_peak',
            'formula','slots'} and plan['schema'] == 'NATIVE_OWNERSHIP_PLAN_V2', 'ownership schema')
        require(type(plan['capacity']) is int and 0 < plan['capacity']
                and type(plan['joint_peak']) is int and 0 < plan['joint_peak'] <= plan['capacity'], 'reviewed joint capacity')
        sha(plan['formula']); sha(binding)
        require(role in ('caller','controller','production','replay','reception'), 'ownership role')
        require(type(plan['slots']) is dict and 1 <= len(plan['slots']) <= POLICY['members'], 'finite slots')
        slots = {}
        for name, v in plan['slots'].items():
            require(type(name) is str and name.isascii() and 1 <= len(name) <= 64
                    and type(v) is dict and set(v) == {'requested','encoded','depth','role'}, 'typed slot')
            require(type(v['requested']) is int and 0 < v['requested'] <= plan['capacity']
                    and type(v['encoded']) is int and 0 < v['encoded'] <= (32*2**20 if name == 'source' else POLICY['header_bytes'])
                    and type(v['depth']) is int and 0 < v['depth'] <= v['encoded']
                    and v['role'] in ('caller','controller','production','replay','reception','all'), 'slot bounds')
            require(v['requested'] >= 2*v['encoded']+1, 'known raw/fill/freeze overlap floor')
            slots[name] = MappingProxyType(dict(v))
        # Conservative simultaneous-role bound for this sequential-actor design.
        # These are reviewed named reservations, not an empirical heap bound.
        def ceiling(actor):
            return sum(v['requested'] for v in slots.values() if v['role'] in (actor,'all'))
        joint = ceiling('caller')+ceiling('controller')+max(ceiling(a) for a in ('production','replay','reception'))
        require(joint <= plan['joint_peak'], 'caller/controller/active-actor joint reservation')
        self.slots = MappingProxyType(slots)
        self.binding, self.role, self.deadline = binding, role, deadline
        self.capacity, self.joint_peak, self.formula = plan['capacity'], plan['joint_peak'], plan['formula']
        self.live = self.peak = self.acquired = self.released = 0
        self._held = {}; self.closed = False

    def verify_plan(self, plan, binding):
        require(type(plan) is dict and set(plan) == {'schema','capacity','joint_peak','formula','slots'}
                and plan['schema'] == 'NATIVE_OWNERSHIP_PLAN_V2'
                and binding == self.binding and plan['capacity'] == self.capacity
                and plan['joint_peak'] == self.joint_peak and plan['formula'] == self.formula
                and plan['slots'] == self.slots, 'same externally bound ownership plan')

    @contextmanager
    def hold(self, name):
        tick(self.deadline)
        require(not self.closed and name in self.slots, 'known live reservation')
        slot = self.slots[name]
        require(slot['role'] in (self.role,'all'), 'reservation actor binding')
        # Nested loans of the SAME slot retain its backing until the last loan.
        # Independent copies must use different slots, never reuse the same name.
        if name not in self._held:
            require(self.live + slot['requested'] <= self.capacity, 'preconstruction owned capacity')
            self.live += slot['requested']; self.acquired += slot['requested']
            self.peak = max(self.peak, self.live); self._held[name] = 0
        self._held[name] += 1
        try: yield slot
        finally:
            self._held[name] -= 1
            if self._held[name] == 0:
                self.live -= slot['requested']; self.released += slot['requested']; del self._held[name]

    def active(self, name):
        require(not self.closed and self._held.get(name, 0) > 0, 'reserve before constructing')
        return self.slots[name]

    def receipt(self):
        return dict(binding=self.binding, role=self.role, formula=self.formula,
            requested_live=self.live, requested_peak=self.peak,
            acquired=self.acquired, released=self.released,
            scope='NAMED_REQUESTED_RESERVATIONS_NOT_RSS_OR_HEAP_PROOF')


def parse(raw, resources, slot, *, canonical=True):
    """Caller retains the slot for the entire decoded object's lifetime."""
    spec = resources.active(slot)
    require(type(raw) is bytes and len(raw) <= spec['encoded'], 'reserved raw metadata')
    stats = lexical(raw, resources.deadline, spec['depth'])
    obj = decode(raw)  # Original duplicate-key and nonfinite checks are retained.
    ref = measure(obj, resources.deadline, spec['encoded'], spec['depth'])
    if canonical:
        require(ref == dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()), 'canonical metadata')
    return obj, ref, stats


def publish(path, build, resources, slot, *, cap):
    """Reserve before calling our internal builder, count, then stream/fsync.

    build is implementation-owned, never an input callback from the envelope.
    Caller-owned mutable structures are not accepted at the external API.
    """
    with resources.hold(slot) as spec:
        require(type(cap) is int and 0 < cap <= spec['encoded'], 'publication old cap')
        obj = build()
        expected = measure(obj, resources.deadline, cap, spec['depth'])
        path = Path(path); tick(resources.deadline)
        require(path.is_absolute() and not path.exists() and not path.is_symlink()
                and not any(p.is_symlink() for p in path.parents), 'new publication')
        fd = os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        h = hashlib.sha256(); n = 0
        try:
            buffer = bytearray()
            def flush():
                offset = 0
                while offset < len(buffer):
                    tick(resources.deadline)
                    count = os.write(fd,memoryview(buffer)[offset:])
                    require(count > 0,'partial publication'); offset += count
                buffer.clear()
            for part in tokens(obj,resources.deadline,spec['depth']):
                require(n+len(part) <= cap, 'publication mutated or exceeded')
                if len(buffer)+len(part) > 65536: flush()
                buffer.extend(part); n += len(part); h.update(part)
            flush()
            os.fsync(fd); tick(resources.deadline)
            require(expected == dict(bytes=n,sha256=h.hexdigest()), 'independent encoding length/hash')
        finally:
            os.close(fd)
        raw, actual = bounded_read(path, cap, resources.deadline, expected['sha256'])
        require(len(raw) == expected['bytes'] and actual == expected['sha256'], 'durable publication identity')
        return expected
