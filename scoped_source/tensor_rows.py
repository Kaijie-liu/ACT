"""Opt-in bounded rows from captured H2 bytes; not a network/HZ proof.

No full tensor of rational weights, base64 bank, solver or numerical library.
Every replay is a new reader. Exhaust rows() before asking for receipt().
"""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import re
import struct
import time

from scoped_source.factored_io import BLOCK_LIMIT, HEADER_LIMIT, decode, read, required_hash
from scoped_source.factored_source import SCHEMA, SOURCE_TOTAL_LIMIT, Source
from source_enclosure.format import compact, identity

WIDTH_LIMIT = 4096


@dataclass(frozen=True)
class Chunk:
    file: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    sha256: str
    chunks: tuple


@dataclass(frozen=True)
class Layer:
    graph: str
    index: int
    kind: str
    inputs: int
    outputs: int
    weight: str = ''
    bias: str = ''


@dataclass(frozen=True)
class AffineRow:
    graph: str
    layer: int
    row: int
    coefficients: tuple
    bias: Fraction


def _bounded_inventory(root, expected, tick):
    """Only the two levels allowed by H2; refuse extra entries immediately."""
    tick()
    if root.is_symlink() or not root.is_dir():
        raise ValueError('source root')
    found = set()
    total = 0
    for directory, prefix, allowed_dir in ((root, '', 't'), (root/'t', 't/', None)):
        tick()
        if directory.is_symlink() or not directory.is_dir():
            raise ValueError('source member directory')
        with os.scandir(directory) as entries:
            for entry in entries:
                tick()
                if entry.is_symlink():
                    raise ValueError('symlink inventory')
                if entry.name == allowed_dir and entry.is_dir(follow_symlinks=False):
                    continue
                name = prefix+entry.name
                if name not in expected or not entry.is_file(follow_symlinks=False):
                    raise ValueError('extra source member')
                found.add(name)
                total += entry.stat(follow_symlinks=False).st_size
                if total > SOURCE_TOTAL_LIMIT:
                    raise ValueError('source total bytes')
    if found != expected:
        raise ValueError('missing source members')
    return total


class TensorRows:
    """Complete graph validation, then bounded immutable affine-row traversal.

    The expected identity binds an existing captured declaration, not a live
    checkpoint. Public metadata is copied/frozen. Private state is not a security
    boundary against malicious Python code running in this same interpreter.
    """

    def __init__(self, root, expected, *, deadline):
        now = time.monotonic()
        if not math.isfinite(deadline) or not now < deadline <= now+300:
            raise ValueError('one finite cumulative deadline of at most 300 seconds')
        self._deadline = deadline
        self._root = Path(root)
        self._identity = required_hash(expected)
        self._status = 'READY'
        self._checked = {}
        self._stats = dict(chunk_reads=0, raw_bytes=0, values=0, affine_rows=0,
                           max_chunk_bytes=0, max_row_values=0)
        self._tick()
        raw = read(root, 'manifest.json', HEADER_LIMIT)
        self._manifest_digest = hashlib.sha256(raw).hexdigest()
        m = decode(raw)
        self._tick()
        if (set(m) != {'schema', 'declaration', 'tensors'} or m['schema'] != SCHEMA
                or identity(m) != expected):
            raise ValueError('external chunked source identity')
        doc = m['declaration']
        if (set(doc) not in ({'schema', 'request', 'center', 'state_inventory', 'networks'},
                            {'schema', 'request', 'center', 'state_inventory', 'networks', 'trust'})
                or doc['schema'] != 'SCOPED_DECLARED_TOP2_V1'):
            raise ValueError('source declaration schema')
        r = doc['request']
        if (set(r) != {'experts', 'classes', 'label', 'top_k', 'gate', 'tie_policy',
                       'training', 'center', 'radius', 'margin', 'clip', 'model_state'}
                or type(r['experts']) is not int or not 2 <= r['experts'] <= 64
                or type(r['classes']) is not int or not 2 <= r['classes'] <= 1024
                or type(r['label']) is not int or not 0 <= r['label'] < r['classes']
                or type(r['top_k']) is not int or r['top_k'] != 2
                or r['gate'] != 'SELECTED_SOFTMAX' or r['tie_policy'] != 'ANY_LEGAL_TOPK'
                or r['training'] is not False):
            raise ValueError('request semantics/dimensions')
        self._request = compact(r)
        inv = doc['state_inventory']
        state = {v['name']: v for v in inv}
        if [v['name'] for v in inv] != sorted(state):
            raise ValueError('state inventory duplicate/order')
        h = hashlib.sha256()
        count = 0
        for v in inv:
            self._tick()
            if set(v) != {'name', 'dtype', 'shape', 'sha256'}:
                raise ValueError('state fields')
            Source.tensor_shape(v)
            h.update(v['name'].encode())
            h.update(required_hash(v['sha256']).encode())
            count += math.prod(v['shape'])
        if r['model_state'] != {'sha256': h.hexdigest(), 'tensor_count': len(inv),
                               'parameter_count': count}:
            raise ValueError('model-state inventory identity')
        self._tensors = {}
        files = {'manifest.json'}
        names = [v['name'] for v in m['tensors']]
        if names != sorted(set(state)|{'@center'}):
            raise ValueError('tensor inventory')
        for i, v in enumerate(m['tensors']):
            self._tick()
            if set(v) != {'name', 'dtype', 'shape', 'byte_order', 'chunks'}:
                raise ValueError('tensor descriptor fields')
            Source.tensor_shape(v)
            if v['byte_order'] != 'little':
                raise ValueError('tensor byte order')
            reference = r['center'] if v['name'] == '@center' else state[v['name']]
            if (reference['dtype'] != v['dtype'] or reference['shape'] != v['shape']
                    or v['name'] == '@center' and set(reference) != {'dtype', 'shape', 'sha256'}):
                raise ValueError('tensor descriptor/state dimensions')
            required_hash(reference['sha256'])
            chunks = []
            end = 0
            for j, chunk in enumerate(v['chunks']):
                self._tick()
                if (set(chunk) != {'file', 'bytes', 'sha256', 'offset'}
                        or chunk['file'] != f't/{i:05d}-{j:05d}.bin'
                        or type(chunk['offset']) is not int or chunk['offset'] != end
                        or type(chunk['bytes']) is not int
                        or not 0 < chunk['bytes'] <= BLOCK_LIMIT or chunk['bytes'] % 8):
                    raise ValueError('chunk ordering/coverage/size')
                chunks.append(Chunk(chunk['file'], chunk['bytes'], required_hash(chunk['sha256'])))
                files.add(chunk['file'])
                end += chunk['bytes']
            if end != 8*math.prod(v['shape']):
                raise ValueError('complete tensor byte coverage')
            self._tensors[v['name']] = Tensor(v['name'], tuple(v['shape']),
                                                reference['sha256'], tuple(chunks))
        self._files = frozenset(files)
        self._total_bytes = _bounded_inventory(self._root, self._files, self._tick)
        center = self._tensors['@center']
        if (doc['center'] != {'tensor_ref': '@center'} or len(center.shape) < 2
                or center.shape[0] != 1 or math.prod(center.shape) > WIDTH_LIMIT):
            raise ValueError('bounded input tensor binding')
        self._plan = self._graph_plan(doc['networks'], r, center.shape, set(state))
        values = tuple(self._values('@center'))
        radius = self._fraction(r['radius'])
        if type(r['clip']) is not list or len(r['clip']) != 2:
            raise ValueError('clip bounds')
        lo, hi = map(self._fraction, r['clip'])
        margin = self._fraction(r['margin'])
        if radius < 0 or lo >= hi or margin < 0 or any(not lo <= x <= hi for x in values):
            raise ValueError('domain/margin')
        self._inputs = tuple((max(lo, x-radius), min(hi, x+radius)) for x in values)
        self._tick()

    def _tick(self):
        if time.monotonic() >= self._deadline:
            raise TimeoutError('tensor intake cumulative deadline')

    def _fraction(self, value):
        self._tick()
        if type(value) not in (str, int, float):
            raise ValueError('bounded rational literal')
        if type(value) is str and (len(value) > 4096 or
                re.fullmatch(r'[+-]?[0-9]+(?:/[0-9]+|\.[0-9]+)?', value) is None):
            raise ValueError('bounded rational literal')
        if type(value) is int and value.bit_length() > 8192:
            raise ValueError('bounded rational literal')
        if type(value) is float and not math.isfinite(value):
            raise ValueError('bounded rational literal')
        return Fraction(value)

    def _graph_plan(self, graphs, request, shape, state_names):
        names = ['router']+[f'expert{i}' for i in range(request['experts'])]
        if [g['name'] for g in graphs] != names:
            raise ValueError('full ordered graph inventory')
        used = set()
        plan = []
        for graph in graphs:
            self._tick()
            if set(graph) != {'name', 'layers'} or not graph['layers']:
                raise ValueError('network schema')
            current = shape
            prefix = 'router' if graph['name'] == 'router' else 'experts.'+graph['name'][6:]
            for j, layer in enumerate(graph['layers']):
                self._tick()
                if (type(layer['index']) is not int or layer['index'] != j
                        or layer['training'] is not False):
                    raise ValueError('layer order/mode')
                kind = layer['kind']
                width = math.prod(current[1:])
                if kind == 'Flatten':
                    if (set(layer) != {'index', 'training', 'kind', 'dimensions'}
                            or layer['dimensions'] != [1, -1]):
                        raise ValueError('flatten contract')
                    plan.append(Layer(graph['name'], j, kind, width, width))
                    current = (1, width)
                elif kind == 'ReLU':
                    if (set(layer) != {'index', 'training', 'kind', 'inplace'}
                            or layer['inplace'] is not False):
                        raise ValueError('ReLU contract')
                    plan.append(Layer(graph['name'], j, kind, width, width))
                elif kind == 'Linear':
                    if set(layer) != {'index', 'training', 'kind', 'weight', 'bias', 'weight_name', 'bias_name'}:
                        raise ValueError('affine fields')
                    refs = []
                    for role in ('weight', 'bias'):
                        name = f'{prefix}.{j}.{role}'
                        if (layer[role+'_name'] != name or layer[role] != {'tensor_ref': name}
                                or name not in state_names or name in used):
                            raise ValueError('graph parameter binding')
                        used.add(name)
                        refs.append(name)
                    weight, bias = (self._tensors[n] for n in refs)
                    if (len(current) != 2 or len(weight.shape) != 2
                            or weight.shape[1] != current[1] or bias.shape != (weight.shape[0],)
                            or max(weight.shape) > WIDTH_LIMIT):
                        raise ValueError('bounded affine dimensions')
                    current = (1, weight.shape[0])
                    plan.append(Layer(graph['name'], j, kind, width, current[1], *refs))
                else:
                    raise ValueError('unsupported source operator')
            expected = request['experts'] if graph['name'] == 'router' else request['classes']
            if current != (1, expected):
                raise ValueError('network endpoint shape')
        if used != state_names:
            raise ValueError('unused source tensor')
        return tuple(plan)

    def _values(self, name):
        tensor = self._tensors[name]
        h = hashlib.sha256(b'torch.float64'+compact(list(tensor.shape)))
        count = 0
        for chunk in tensor.chunks:
            self._tick()
            raw = read(self._root, chunk.file, BLOCK_LIMIT, chunk.sha256)
            if len(raw) != chunk.bytes:
                raise ValueError('chunk actual byte count')
            self._stats['chunk_reads'] += 1
            self._stats['raw_bytes'] += len(raw)
            self._stats['max_chunk_bytes'] = max(self._stats['max_chunk_bytes'], len(raw))
            h.update(raw)
            for (value,) in struct.iter_unpack('<d', raw):
                if count % 256 == 0:
                    self._tick()
                if not math.isfinite(value):
                    raise ValueError('nonfinite tensor')
                count += 1
                self._stats['values'] += 1
                yield Fraction.from_float(value)
            del raw
        self._tick()
        if count != math.prod(tensor.shape) or h.hexdigest() != tensor.sha256:
            raise ValueError('source tensor/state identity')
        if name in self._checked:
            raise ValueError('tensor consumed twice')
        self._checked[name] = h.hexdigest()

    @staticmethod
    def _exhaust(values):
        try:
            next(values)
        except StopIteration:
            return
        raise ValueError('extra tensor values')

    def rows(self):
        if self._status != 'READY':
            raise ValueError('row traversal is once only')
        self._status = 'RUNNING'
        try:
            for layer in self._plan:
                self._tick()
                if layer.kind != 'Linear':
                    continue
                weights = self._values(layer.weight)
                biases = self._values(layer.bias)
                for row in range(layer.outputs):
                    self._tick()
                    coefficients = tuple(next(weights) for _ in range(layer.inputs))
                    bias = next(biases)
                    self._stats['max_row_values'] = max(self._stats['max_row_values'], len(coefficients))
                    self._stats['affine_rows'] += 1
                    yield AffineRow(layer.graph, layer.index, row, coefficients, bias)
                    del coefficients
                self._exhaust(weights)
                self._exhaust(biases)
            if set(self._checked) != set(self._tensors):
                raise ValueError('incomplete tensor consumption')
            self._tick()
            read(self._root, 'manifest.json', HEADER_LIMIT, self._manifest_digest)
            if _bounded_inventory(self._root, self._files, self._tick) != self._total_bytes:
                raise ValueError('source inventory changed')
            self._tick()
            self._status = 'COMPLETE'
        except BaseException:
            self._status = 'INCOMPLETE'
            raise

    @property
    def plan(self):
        return self._plan

    @property
    def inputs(self):
        return self._inputs

    @property
    def request(self):
        return decode(self._request)

    @property
    def statistics(self):
        return deepcopy(self._stats)

    def receipt(self):
        self._tick()
        if self._status != 'COMPLETE':
            raise ValueError('all rows and final tensor identities must be exhausted')
        return {'schema': 'CAPTURED_TENSOR_ROWS_RECEIPT_V1',
                'source_sha256': self._identity, 'request_sha256': identity(self.request),
                'tensor_hashes': dict(sorted(self._checked.items())),
                'source_bytes': self._total_bytes, 'statistics': self.statistics,
                'scope': 'captured binary64 parameters only; not HZ lowering or model safety'}
