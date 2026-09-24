"""Bounded *additional serialization* memory, byte-identical to scoped save.

The already constructed object is still resident. Readers, construction and
other identity() calls are NOT made streaming here. Only built-in JSON evidence
types with string keys are admitted; unsupported inputs fail closed. Dictionary
width, nesting and numeric-token limits bound sorting/stack/scalar workspace.
Large string VALUES are escaped incrementally, not passed whole to iterencode.
No source matrix, obligation or check is omitted/deduplicated.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import time

BLOCK_BYTES = 64 * 1024
STRING_CHARS = 2048
MAX_DEPTH = 64
MAX_KEYS = 4096
MAX_KEY_CHARS = 4096
MAX_INT_BITS = 12000


def clock(deadline):
    if (type(deadline) not in (int, float) or not math.isfinite(deadline)
            or deadline - time.monotonic() > 300):
        raise ValueError('finite absolute deadline required')
    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('evidence publication deadline')
    tick()
    return tick


def fragments(value, tick, active=None, depth=0):
    """ASCII JSON fragments; no fragment grows with a large string value/list."""
    tick()
    if depth > MAX_DEPTH:
        raise ValueError('JSON nesting limit')
    if active is None:
        active = set()
    kind = type(value)
    if kind is str:
        yield b'"'
        for start in range(0, len(value), STRING_CHARS):
            tick()
            # ensure_ascii preserves isolated surrogates and non-BMP spelling.
            piece = json.dumps(value[start:start + STRING_CHARS], ensure_ascii=True)[1:-1]
            yield piece.encode('ascii')
        yield b'"'
    elif value is None:
        yield b'null'
    elif kind is bool:
        yield b'true' if value else b'false'
    elif kind is int:
        if value.bit_length() > MAX_INT_BITS:
            raise ValueError('integer token limit; rational evidence uses strings')
        yield str(value).encode('ascii')
    elif kind is float:
        yield json.dumps(value, allow_nan=False).encode('ascii')
    elif kind in (list, tuple, dict):
        marker = id(value)
        if marker in active:
            raise ValueError('circular JSON evidence')
        active.add(marker)
        try:
            if kind is dict:
                if len(value) > MAX_KEYS or any(type(k) is not str or len(k) > MAX_KEY_CHARS for k in value):
                    raise ValueError('bounded string-key dictionary required')
                yield b'{'
                for i, key in enumerate(sorted(value)):
                    if i:
                        yield b','
                    yield from fragments(key, tick, active, depth + 1)
                    yield b':'
                    yield from fragments(value[key], tick, active, depth + 1)
                yield b'}'
            else:
                yield b'['
                for i, item in enumerate(value):
                    if i:
                        yield b','
                    yield from fragments(item, tick, active, depth + 1)
                yield b']'
        finally:
            active.remove(marker)
    else:
        raise TypeError('unsupported evidence JSON type: ' + kind.__name__)


def chunks(value, tick, *, block_bytes=BLOCK_BYTES, metrics=None):
    if type(block_bytes) is not int or not 64 <= block_bytes <= BLOCK_BYTES:
        raise ValueError('bounded block size required')
    buffer = bytearray()
    for piece in fragments(value, tick):
        if metrics is not None:
            metrics['max_fragment_bytes'] = max(metrics.get('max_fragment_bytes', 0), len(piece))
        for start in range(0, len(piece), block_bytes):
            part = piece[start:start + block_bytes]
            if len(buffer) + len(part) > block_bytes:
                tick()
                yield bytes(buffer)
                buffer.clear()
            buffer.extend(part)
    if buffer:
        tick()
        yield bytes(buffer)


def save(path, value, *, deadline, metrics=None, block_bytes=BLOCK_BYTES):
    """fsync then exclusive atomic link. Failure retains partial/unreceipted data.

    A final filename alone is NOT acceptance. Caller must publish and validate
    its unchanged bound receipt and meet the outer deadline. In particular a
    cutoff after link/fsync must never be revived by reading that final file.
    """
    started = time.monotonic()
    tick = clock(deadline)
    p = Path(path)
    partial = p.with_name(p.name + '.partial')
    if os.path.lexists(p):
        raise FileExistsError(p)
    stats = metrics if metrics is not None else {}
    stats.update(bytes_written=0, blocks=0, max_block_bytes=0,
                 max_fragment_bytes=0, published=False, returned=False)
    digest = hashlib.sha256()
    try:
        # Exclusive creation rejects preexisting partials, including symlinks.
        with partial.open('xb', buffering=0) as out:
            for block in chunks(value, tick, block_bytes=block_bytes, metrics=stats):
                tick()
                view = memoryview(block)
                while view:
                    tick()
                    count = os.write(out.fileno(), view)
                    if count <= 0:
                        raise OSError('zero-length evidence write')
                    digest.update(view[:count])
                    stats['bytes_written'] += count
                    view = view[count:]
                stats['blocks'] += 1
                stats['max_block_bytes'] = max(stats['max_block_bytes'], len(block))
            tick()
            out.flush()
            os.fsync(out.fileno())
        tick()
        os.link(partial, p)  # never overwrites an earlier artifact
        stats['published'] = True
        partial.unlink()
        directory = os.open(p.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        tick()
        stats['returned'] = True
        return {'sha256': digest.hexdigest(), 'bytes': stats['bytes_written']}
    finally:
        stats['seconds_including_fsync_and_publication'] = time.monotonic() - started
