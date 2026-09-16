"""Request-local bounded reuse of immutable, exact canonical CSR parsing.

Only parsing is reused, never a bound, a source validation or a proof verdict.
Every lookup serializes the supplied current content. Hash hits additionally
require identical canonical bytes, so object identity and mutable aliases are
not authority. No persistent/user-supplied cache entry is accepted.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType

from act.back_end.solver.sparse_lp_certificate import rows as reference_rows

POLICY = 'EXACT_CSR_PARSE_V1'
LIMITS = {'entries': 64, 'payload_bytes': 64 * 1024 * 1024, 'cells': 2_000_000}


def _digest(payload):
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class _Parsed:
    scope: str
    payload: bytes
    shape: tuple
    rows: tuple
    entries: object
    cells: int


def parse(payload, scope):
    matrix = json.loads(payload)
    shape = matrix['shape']
    if not isinstance(shape, list) or len(shape) != 2 or any(type(v) is not int or v < 0 for v in shape):
        raise ValueError('canonical integer CSR dimensions required')
    # Reference rows validates pointer/index types, ordering and every value.
    rows = tuple(tuple(row) for row in reference_rows(matrix, shape[1]))
    entries = MappingProxyType({(i, j): v for i, row in enumerate(rows) for j, v in row})
    return _Parsed(scope, payload, tuple(shape), rows, entries, len(rows)+len(entries))


class MatrixCache:
    def __init__(self, scope, *, enabled=True, tick=lambda: None, limits=None):
        if not isinstance(scope, str) or not scope:
            raise ValueError('request identity required')
        self.__scope = scope; self.__enabled = bool(enabled); self.__tick = tick
        self.__limits = dict(LIMITS if limits is None else limits)
        if set(self.__limits) != set(LIMITS) or any(type(v) is not int or v < 1 for v in self.__limits.values()):
            raise ValueError('positive bounded cache limits required')
        self.__items = OrderedDict(); self.__bytes = 0; self.__cells = 0
        self.__counts = dict(lookups=0, hits=0, parses=0, evictions=0, oversized=0,
                             peak_entries=0, peak_payload_bytes=0, peak_cells=0)

    def _get(self, matrix):
        self.__tick()
        payload = json.dumps(matrix, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        key = (self.__scope, POLICY, _digest(payload))
        self.__counts['lookups'] += 1
        if self.__enabled and key in self.__items:
            value = self.__items[key]
            if value.scope != self.__scope or value.payload != payload:
                raise ValueError('cache scope/content collision; refusing stale parse')
            self.__items.move_to_end(key); self.__counts['hits'] += 1
            self.__tick(); return value
        value = parse(payload, self.__scope); self.__counts['parses'] += 1
        self.__tick()
        if self.__enabled:
            if len(payload) > self.__limits['payload_bytes'] or value.cells > self.__limits['cells']:
                self.__counts['oversized'] += 1
            else:
                while self.__items and (len(self.__items) >= self.__limits['entries'] or
                        self.__bytes+len(payload) > self.__limits['payload_bytes'] or
                        self.__cells+value.cells > self.__limits['cells']):
                    _, old = self.__items.popitem(last=False)
                    self.__bytes -= len(old.payload); self.__cells -= old.cells
                    self.__counts['evictions'] += 1
                self.__items[key] = value; self.__bytes += len(payload); self.__cells += value.cells
                self.__counts['peak_entries'] = max(self.__counts['peak_entries'], len(self.__items))
                self.__counts['peak_payload_bytes'] = max(self.__counts['peak_payload_bytes'], self.__bytes)
                self.__counts['peak_cells'] = max(self.__counts['peak_cells'], self.__cells)
        return value

    def entries(self, matrix):
        value = self._get(matrix)
        return value.shape, value.entries

    def rows(self, matrix, n):
        value = self._get(matrix)
        if value.shape[1] != n:
            raise ValueError('cached CSR width does not match current LP')
        for row in value.rows:
            self.__tick()
            yield row

    def stats(self):
        return {**self.__counts, 'enabled': self.__enabled, 'policy': POLICY,
                'limits': self.__limits.copy(), 'live_entries': len(self.__items),
                'live_payload_bytes': self.__bytes, 'live_cells': self.__cells}

    def clear(self):
        self.__items.clear(); self.__bytes = 0; self.__cells = 0
