"""Request-local immutable binary64 rows; no solver, tensor library or cache.

The packed bank is a representation of a fully traversed source, not a claim
that source->HZ is already proved. The independent equations still run later.
"""
from collections.abc import Sequence
from fractions import Fraction as F
import struct
from itertools import chain
from types import MappingProxyType

from scoped_source.hz_request_store import POLICY, compact, decode, require
from scoped_source.tensor_rows import TensorRows, AffineRow
from scoped_source.direct_source_format import number


class Immutable:
    __slots__ = ()
    def __setattr__(self, k, v): raise AttributeError('immutable shared source')
    def __delattr__(self, k): raise AttributeError('immutable shared source')


class PackedRow(Immutable):
    __slots__ = ('key', 'raw', 'width', 'nonzero')

    def __init__(self, key, raw, work):
        require(type(key) is tuple and len(key) == 3 and type(key[0]) is str
                and all(type(i) is int and i >= 0 for i in key[1:]), 'source row key')
        require(type(raw) is bytes and 16 <= len(raw) <= 8 * 4097 and len(raw) % 8 == 0,
                'immutable bounded source row')
        from math import isfinite
        count = 0
        for i, (x,) in enumerate(struct.iter_unpack('<d', raw)):
            work.visit('coefficient_visits'); require(isfinite(x), 'finite packed source')
            if i and x: count += 1
        for k, v in dict(key=key, raw=raw, width=len(raw)//8-1, nonzero=count).items():
            object.__setattr__(self, k, v)

    @property
    def bias(self): return F(struct.unpack_from('<d', self.raw)[0])

    def coefficient(self, i):
        require(type(i) is int and 0 <= i < self.width, 'source coefficient index')
        return F(struct.unpack_from('<d', self.raw, (i+1)*8)[0])


class Snapshot:
    """Single replay, packed values immutable; complete source receipt required.

    Python object overhead is NOT included in packed_bytes. At most one source
    row tuple and one row fill buffer coexist while packing. Replay briefly
    holds both a row list and its tuple of references. Source hashes and
    TensorRows' bounded inventory checks are not omitted.
    """
    __slots__=('work','_plan','_inputs','_request','_rows','packed_bytes','_receipt','_state','_sealed')
    def __setattr__(self,k,v):
        require(not getattr(self,'_sealed',False), 'immutable source bank binding')
        object.__setattr__(self,k,v)
    def __delattr__(self,k): raise AttributeError('immutable source bank binding')

    def __init__(self, root, source, work):
        reader = TensorRows(root, source, deadline=work.deadline)
        require(reader.request['experts'] <= 8, 'unchanged expert limit')
        self.work = work; self._plan = reader.plan; self._inputs = reader.inputs
        self._request = compact(reader.request); self._rows = {}; self.packed_bytes = 0
        for row in reader.rows():
            work.visit('row_visits'); key = (row.graph, row.layer, row.row)
            size = 8 * (len(row.coefficients) + 1)
            require(key not in self._rows and len(self._rows) < POLICY['rows']
                    and self.packed_bytes + size <= POLICY['total_bytes'], 'packed source admission')
            raw = bytearray(size)
            for i, q in enumerate(chain((row.bias,),row.coefficients)):
                work.visit('coefficient_visits'); struct.pack_into('<d', raw, i*8, number(q))
            self._rows[key] = PackedRow(key, bytes(raw), work)
            self.packed_bytes += size
        self._receipt = compact(reader.receipt())
        self._rows = MappingProxyType(self._rows); self._state = 'READY'; self._sealed=True; work.tick()

    @property
    def plan(self): return self._plan
    @property
    def inputs(self): return tuple((F(lo.numerator,lo.denominator),F(hi.numerator,hi.denominator)) for lo,hi in self._inputs)
    @property
    def request(self): return decode(self._request)

    def row(self, key):
        self.work.tick(); require(type(key) is tuple and key in self._rows, 'bound source row')
        return self._rows[key]

    def rows(self):
        require(self._state == 'READY', 'one complete snapshot replay'); object.__setattr__(self,'_state','RUNNING')
        try:
            for key, row in self._rows.items():
                self.work.visit('row_visits')
                values = []
                for i in range(row.width):
                    self.work.visit('coefficient_visits'); values.append(row.coefficient(i))
                yield AffineRow(*key, tuple(values), row.bias)
            self.work.tick(); object.__setattr__(self,'_state','COMPLETE')
        except BaseException:
            object.__setattr__(self,'_state','INCOMPLETE'); raise

    def receipt(self):
        self.work.tick(); require(self._state == 'COMPLETE', 'complete snapshot semantics traversal')
        return decode(self._receipt)


class SourceLocal(Immutable, Sequence):
    """One source row plus explicit compensation tail; no retained dense dict."""
    __slots__ = ('row', 'tail', 'work')

    def __init__(self, row, tail, work):
        require(type(row) is PackedRow and type(tail) is tuple and len(tail) <= 2,
                'closed source local/tail')
        last = row.width-1
        for j, q in tail:
            require(type(j) is int and j > last and type(q) is F and q != 0, 'source private tail')
            last = j
        object.__setattr__(self, 'row', row); object.__setattr__(self, 'tail', tail)
        object.__setattr__(self, 'work', work)

    def __len__(self): return self.row.nonzero + len(self.tail)
    def __iter__(self):
        for i in range(self.row.width):
            self.work.visit('coefficient_visits')
            q = self.row.coefficient(i)
            if q: yield i, q
        yield from self.tail

    def __getitem__(self, i):
        if isinstance(i, slice): return tuple(self)[i]
        if i < 0: i += len(self)
        for j, v in enumerate(self):
            if j == i: return v
        raise IndexError(i)

    def __add__(self, tail): return SourceLocal(self.row, self.tail + tail, self.work)
