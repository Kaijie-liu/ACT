"""Same canonical bytes and caps; batch small scalar arrays through stdlib JSON.

The only optimization is amortizing Python fragment/escape/generator overhead.
Every value is still visited, admitted and emitted; no cache, source omission,
matrix deduplication or altered proof checking. Batches are bounded in both
element count and conservative escaped size, including very large rationals.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time

from bounded_evidence.stream import (BLOCK_BYTES, STRING_CHARS, MAX_DEPTH,
    MAX_KEYS, MAX_KEY_CHARS, MAX_INT_BITS, clock, fragments as scalar_fragments)

BATCH_ITEMS = 128
BATCH_BYTES = 16384


def scalar_size(value):
    """Conservative canonical byte bound, or None to use original traversal."""
    kind = type(value)
    if value is None: return 4
    if kind is bool: return 5
    if kind is int:
        if value.bit_length() > MAX_INT_BITS: raise ValueError('integer token limit')
        return value.bit_length() + 2  # loose, integer-only safe decimal bound
    if kind is float:
        if not math.isfinite(value): raise ValueError('nonfinite JSON')
        return 32
    if kind is str and len(value) <= STRING_CHARS:
        return 12 * len(value) + 2  # includes non-BMP surrogate-pair escaping
    return None


def fragments(value, tick, active=None, depth=0, metrics=None):
    tick()
    if depth > MAX_DEPTH: raise ValueError('JSON nesting limit')
    if active is None: active = set()
    kind = type(value)
    if kind not in (dict, list, tuple):
        yield from scalar_fragments(value, tick, active, depth)
        return
    marker = id(value)
    if marker in active: raise ValueError('circular JSON evidence')
    active.add(marker)
    try:
        if kind is dict:
            if len(value) > MAX_KEYS or any(type(k) is not str or len(k) > MAX_KEY_CHARS for k in value):
                raise ValueError('bounded string-key dictionary required')
            yield b'{'
            for i, key in enumerate(sorted(value)):
                if i: yield b','
                yield from scalar_fragments(key, tick, active, depth + 1)
                yield b':'
                yield from fragments(value[key], tick, active, depth + 1, metrics)
            yield b'}'
        else:
            yield b'['
            i = 0
            while i < len(value):
                tick()
                if depth + 1 > MAX_DEPTH: raise ValueError('JSON nesting limit')
                batch = []; bound = 2
                while i + len(batch) < len(value) and len(batch) < BATCH_ITEMS:
                    tick()  # same admission/deadline visit for every scalar
                    item = value[i + len(batch)]
                    n = scalar_size(item)
                    if n is None or bound + n + 1 > BATCH_BYTES: break
                    batch.append(item); bound += n + 1
                if i: yield b','
                if batch:
                    encoded = json.dumps(batch, separators=(',', ':'), allow_nan=False, ensure_ascii=True)
                    if len(encoded) > bound: raise AssertionError('batch size bound violated')
                    tick()
                    if metrics is not None:
                        metrics['native_batches'] = metrics.get('native_batches', 0) + 1
                        metrics['batched_scalars'] = metrics.get('batched_scalars', 0) + len(batch)
                        metrics['max_batch_bytes'] = max(metrics.get('max_batch_bytes', 0), len(encoded))
                        metrics['max_batch_items'] = max(metrics.get('max_batch_items', 0), len(batch))
                    yield encoded[1:-1].encode('ascii')
                    i += len(batch)
                else:
                    yield from fragments(value[i], tick, active, depth + 1, metrics)
                    i += 1
            yield b']'
    finally:
        active.remove(marker)


def chunks(value, tick, *, block_bytes=BLOCK_BYTES, metrics=None):
    if type(block_bytes) is not int or not 64 <= block_bytes <= BLOCK_BYTES:
        raise ValueError('bounded block size required')
    buffer = bytearray()
    for piece in fragments(value, tick, metrics=metrics):
        if metrics is not None:
            metrics['max_fragment_bytes'] = max(metrics.get('max_fragment_bytes', 0), len(piece))
        for start in range(0, len(piece), block_bytes):
            part = piece[start:start + block_bytes]
            if len(buffer) + len(part) > block_bytes:
                tick(); yield bytes(buffer); buffer.clear()
            buffer.extend(part)
    if buffer:
        tick(); yield bytes(buffer)


def save(path, value, *, deadline, metrics=None, block_bytes=BLOCK_BYTES):
    """Same atomic publication and cost boundary as bounded_evidence R1."""
    started = time.monotonic(); tick = clock(deadline)
    p = Path(path); partial = p.with_name(p.name + '.partial')
    if os.path.lexists(p): raise FileExistsError(p)
    stats = metrics if metrics is not None else {}
    stats.update(bytes_written=0, blocks=0, max_block_bytes=0, max_fragment_bytes=0,
                 published=False, returned=False, native_batches=0, batched_scalars=0,
                 max_batch_bytes=0, max_batch_items=0)
    digest = hashlib.sha256()
    try:
        with partial.open('xb', buffering=0) as out:
            for block in chunks(value, tick, block_bytes=block_bytes, metrics=stats):
                tick(); view = memoryview(block)
                while view:
                    tick(); count = os.write(out.fileno(), view)
                    if count <= 0: raise OSError('zero-length evidence write')
                    digest.update(view[:count]); stats['bytes_written'] += count; view = view[count:]
                stats['blocks'] += 1
                stats['max_block_bytes'] = max(stats['max_block_bytes'], len(block))
            tick(); out.flush(); os.fsync(out.fileno())
        tick(); os.link(partial, p); stats['published'] = True; partial.unlink()
        directory = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
        tick(); stats['returned'] = True
        return {'sha256': digest.hexdigest(), 'bytes': stats['bytes_written']}
    finally:
        stats['seconds_including_fsync_and_publication'] = time.monotonic() - started
