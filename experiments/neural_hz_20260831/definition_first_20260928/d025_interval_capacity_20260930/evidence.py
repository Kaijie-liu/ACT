"""Prepaid bounded evidence traversal and streaming, not a numerical kernel.

The caller gives an exclusive temporary path and publishes it by atomic rename
ONLY after success. Failure leaves an incomplete temporary file, never a receipt.
One Meter is shared across all ledgers and all three model serializations.
"""
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

LIMIT, ENTRY_CAP, DEPTH_CAP = 40_000_000, 64_000_000, 128
SCALARS = (int, bool, float, Fraction)
ALLOWED = (*SCALARS, str, bytes, tuple, list, dict, type(None))


class Meter:
    def __init__(self, limit=LIMIT):
        if type(limit) is not int or not 0 <= limit <= LIMIT:
            raise ValueError('invalid evidence limit')
        self.limit, self.used = limit, 0

    def check(self, amount):
        if type(amount) is not int or amount < 0 or amount > self.limit - self.used:
            raise ValueError('evidence budget exhausted before operation')

    def charge(self, amount):
        self.check(amount)
        self.used += amount


def _integer_bound(value):
    bits = value.bit_length()
    if bits > 512:
        raise ValueError('evidence integer exceeds 512 bits')
    return (bits * 30103) // 100000 + 2  # Decimal digits plus possible sign.


def _children(value):
    if type(value) is Fraction:
        yield value.numerator
        yield value.denominator
    elif type(value) is dict:
        for key, item in value.items():
            yield key
            yield item
    elif type(value) in (tuple, list):
        yield from value


def bounded_ledger(roots, meter):
    """Same physical-root fields as the old ledger, without a wide work stack.

    Pay 8 per occurrence before its visit and another 8 before first-object
    inspection. Numeric occurrences count before identity deduplication, as in
    the old ledger. The traversal's own storage is covered by process peaks.
    """
    seen, stack = set(), [iter((roots,))]
    total = entries = 0
    while stack:
        meter.charge(1)  # Iterator advance, including exhaustion.
        try:
            value = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        meter.charge(8)
        kind = type(value)
        if kind not in ALLOWED:
            raise ValueError('unsupported evidence object')
        if kind in SCALARS:
            if entries >= ENTRY_CAP:
                raise ValueError('evidence entry cap exceeded')
            entries += 1
        if id(value) in seen:
            continue
        meter.charge(8)
        seen.add(id(value))
        total += sys.getsizeof(value)
        if kind in (tuple, list, dict, Fraction):
            count = 2 if kind is Fraction else len(value) * (2 if kind is dict else 1)
            meter.check(8 * count)
            if len(stack) >= DEPTH_CAP:
                raise ValueError('evidence nesting cap exceeded')
            stack.append(iter(_children(value)))
    return dict(held_instance_bytes=total, retained_entries=entries,
                unique_objects=len(seen), bytes_in_storage_not_numeric_tensor_entries=True)


def _key_bound(value, meter, depth=0):
    """Bound Python str/repr before allocating normalized dictionary keys."""
    meter.charge(8)
    if depth >= DEPTH_CAP:
        raise ValueError('evidence key nesting cap exceeded')
    kind = type(value)
    if kind is str:
        return 12 * len(value) + 2
    if kind is int:
        return _integer_bound(value)
    if kind is Fraction:
        return 16 + _integer_bound(value.numerator) + _integer_bound(value.denominator)
    if kind in (bool, type(None)):
        return 5
    if kind is float and math.isfinite(value):
        return 32
    if kind is tuple:
        meter.check(8 * len(value))
        return 2 + sum(_key_bound(item, meter, depth + 1) + 2 for item in value)
    raise ValueError('unsupported evidence dictionary key')


def write_evidence(path, roots, meter, byte_digests):
    """Stream ordinary JSON; normalize keys with str as old encoded() did.

    Mapping order is insertion order, not an unpaid full-key sort. Colliding
    normalized keys and cycles reject. No full encoded tree/string is built.
    Encoding temporary sizes are checked before allocation; visits and actual
    emitted bytes are charged before output. No raw-byte rehash is performed.
    """
    digest, active, byte_count = hashlib.sha256(), set(), 0
    with Path(path).open('xb') as stream:
        def emit(token):
            nonlocal byte_count
            meter.charge(len(token))
            data = token.encode('ascii')
            stream.write(data)
            digest.update(data)
            byte_count += len(data)

        def string(value):
            meter.check(12 * len(value) + 2)  # Includes surrogate-pair escapes.
            emit(json.dumps(value, ensure_ascii=True))

        def write(value, depth=0):
            meter.charge(8)
            if depth >= DEPTH_CAP:
                raise ValueError('evidence nesting cap exceeded')
            kind = type(value)
            if kind is str:
                string(value)
            elif kind is int:
                meter.check(_integer_bound(value))
                emit(str(value))
            elif kind is bool or value is None:
                emit('null' if value is None else ('true' if value else 'false'))
            elif kind is float:
                if not math.isfinite(value):
                    raise ValueError('nonfinite evidence float')
                meter.check(32)
                emit(json.dumps(value, allow_nan=False))
            elif kind is Fraction:
                emit('['); write(value.numerator, depth + 1)
                emit(','); write(value.denominator, depth + 1); emit(']')
            elif kind is bytes:
                bound_digest = byte_digests.get(id(value))
                meter.charge(64)
                if (type(bound_digest) is not str or len(bound_digest) != 64
                        or any(c not in '0123456789abcdef' for c in bound_digest)):
                    raise ValueError('verified raw-byte digest missing')
                write(dict(byte_count=len(value), sha256=bound_digest), depth + 1)
            elif kind in (tuple, list, dict):
                if id(value) in active:
                    raise ValueError('cyclic JSON evidence')
                meter.charge(8 * len(value))
                active.add(id(value))
                emit('{' if kind is dict else '[')
                seen_keys = set()
                iterator = value.items() if kind is dict else enumerate(value)
                first = True
                for key, item in iterator:
                    if not first:
                        emit(',')
                    first = False
                    if kind is dict:
                        bound = _key_bound(key, meter)
                        meter.check(bound)
                        normalized = str(key)
                        if normalized in seen_keys:
                            raise ValueError('colliding normalized evidence keys')
                        seen_keys.add(normalized)
                        string(normalized); emit(':')
                    write(item, depth + 1)
                emit('}' if kind is dict else ']')
                active.remove(id(value))
            else:
                raise ValueError('unsupported JSON evidence object')

        write(roots)
        emit('\n')
    return dict(sha256=digest.hexdigest(), bytes=byte_count)
