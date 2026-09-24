"""Bounded request-local reuse of the ORIGINAL exact CSR parser.

Adapted from exact_matrix_cache's content-bound design, without ACT imports.
No validation, state, bound, or verdict is cached. Consumers receive fresh
containers; retained rows and exact rational coefficients are immutable.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from upstream_source.checker import csr as reference_csr

POLICY = 'SOURCE_CONSTRUCTION_CSR_PARSE_V1'
LIMITS = {'entries': 64, 'bytes': 64 * 1024**2, 'cells': 2_000_000}


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class Parsed:
    scope: str
    payload: bytes
    rows: tuple
    cells: int


class MatrixParser:
    def __init__(self, scope, *, enabled, tick, limits=None):
        if type(scope) is not str or not scope or type(enabled) is not bool:
            raise ValueError('explicit request scope and boolean mode required')
        self._scope = scope
        self._enabled = enabled
        self._tick = tick
        self._limits = dict(LIMITS if limits is None else limits)
        if set(self._limits) != set(LIMITS) or any(type(v) is not int or v < 1 for v in self._limits.values()):
            raise ValueError('positive bounded cache limits required')
        self._items = OrderedDict()
        self._bytes = self._cells = 0
        self._counts = dict(lookups=0, hits=0, parses=0, evictions=0, oversized=0,
            peak_entries=0, peak_bytes=0, peak_cells=0)

    def csr(self, matrix, shape=None):
        self._tick()
        # Shape is a CURRENT obligation even on a hit; not part of cached trust.
        if shape is not None and matrix['shape'] != list(shape):
            raise ValueError('current matrix shape mismatch')
        payload = json.dumps(matrix, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        key = (self._scope, POLICY, digest(payload))
        self._counts['lookups'] += 1
        if self._enabled and key in self._items:
            value = self._items[key]
            if value.scope != self._scope or value.payload != payload:
                raise ValueError('parse-cache identity/content collision')
            self._items.move_to_end(key)
            self._counts['hits'] += 1
        else:
            # Parse the same canonical content just bound above, not a mutable alias.
            rows = reference_csr(json.loads(payload), shape)
            frozen = tuple(tuple(row.items()) for row in rows)
            value = Parsed(self._scope, payload, frozen, len(frozen) + sum(map(len, frozen)))
            self._counts['parses'] += 1
            self._tick()
            if self._enabled:
                if len(payload) > self._limits['bytes'] or value.cells > self._limits['cells']:
                    self._counts['oversized'] += 1
                else:
                    while self._items and (len(self._items) >= self._limits['entries'] or
                            self._bytes + len(payload) > self._limits['bytes'] or
                            self._cells + value.cells > self._limits['cells']):
                        _, old = self._items.popitem(last=False)
                        self._bytes -= len(old.payload)
                        self._cells -= old.cells
                        self._counts['evictions'] += 1
                    self._items[key] = value
                    self._bytes += len(payload)
                    self._cells += value.cells
                    self._counts['peak_entries'] = max(self._counts['peak_entries'], len(self._items))
                    self._counts['peak_bytes'] = max(self._counts['peak_bytes'], self._bytes)
                    self._counts['peak_cells'] = max(self._counts['peak_cells'], self._cells)
        result = []
        for row in value.rows:
            self._tick()
            result.append(dict(row))  # no mutable cache row escapes to a producer
        self._tick()
        return result

    def stats(self):
        return {**self._counts, 'enabled': self._enabled, 'scope': self._scope, 'policy': POLICY,
            'limits': dict(self._limits), 'live_entries': len(self._items),
            'live_bytes': self._bytes, 'live_cells': self._cells}

    def clear(self):
        self._items.clear()
        self._bytes = self._cells = 0
