"""Fixed stdlib fixtures and independent analytic row oracle. No solver."""
from fractions import Fraction as F
import hashlib
import math
from pathlib import Path
import struct
import time

from scoped_source.factored_io import BLOCK_LIMIT, HEADER_LIMIT, load, write
from scoped_source.factored_source import SCHEMA
from scoped_source.tensor_rows import TensorRows
from source_enclosure.format import compact, identity


def recipe(large=False):
    width, hidden = (3072, 256) if large else (3, 4)
    graphs = []
    tensors = {'@center': ((1, 1, width), 0, 'center')}
    for name, dims in [('router', [width, 2]), ('expert0', [width, hidden, 2]),
                       ('expert1', [width, 2])]:
        prefix = 'router' if name == 'router' else 'experts.'+name[6:]
        layers = [{'index': 0, 'kind': 'Flatten', 'training': False, 'dimensions': [1, -1]}]
        for k, (ins, outs) in enumerate(zip(dims, dims[1:])):
            j = len(layers)
            salt = (1 if name == 'router' else 2 if name == 'expert0' else 4)+k
            weight, bias = f'{prefix}.{j}.weight', f'{prefix}.{j}.bias'
            tensors[weight] = ((outs, ins), salt, 'weight')
            tensors[bias] = ((outs,), salt, 'bias')
            layers.append({'index': j, 'kind': 'Linear', 'training': False,
                           'weight_name': weight, 'bias_name': bias,
                           'weight': {'tensor_ref': weight}, 'bias': {'tensor_ref': bias}})
            if k < len(dims)-2:
                layers.append({'index': len(layers), 'kind': 'ReLU', 'training': False, 'inplace': False})
        graphs.append({'name': name, 'layers': layers})
    return graphs, tensors


def _encoded_number(spec, index, large):
    shape, salt, role = spec
    if role == 'center':
        return 0.0
    if role == 'bias':
        return (index-salt)/16
    if not large and index < 3:
        return (0.0, -0.0, float.fromhex('0x0.0000000000001p-1022'))[index]
    row, col = divmod(index, shape[1])
    value = ((row*17+col*7+salt*3) % 31+1)/32
    return -value if (row+col+salt) % 2 else value


def fixture(root, *, large=False, chunk_bytes=BLOCK_LIMIT):
    """Write raw chunks directly; never builds a full tensor/base64 string."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    graphs, tensors = recipe(large)
    records = []
    state = []
    center = None
    for i, (name, spec) in enumerate(sorted(tensors.items())):
        shape, _, _ = spec
        h = hashlib.sha256(b'torch.float64'+compact(list(shape)))
        buf = bytearray()
        chunks = []
        offset = 0
        for index in range(math.prod(shape)):
            buf.extend(struct.pack('<d', _encoded_number(spec, index, large)))
            if len(buf) == chunk_bytes or index == math.prod(shape)-1:
                raw = bytes(buf)
                h.update(raw)
                ref = write(root, f't/{i:05d}-{len(chunks):05d}.bin', raw, BLOCK_LIMIT, binary=True)
                chunks.append({**ref, 'offset': offset})
                offset += len(raw)
                buf.clear()
        ident = {'dtype': 'torch.float64', 'shape': list(shape), 'sha256': h.hexdigest()}
        if name == '@center':
            center = ident
        else:
            state.append({'name': name, **ident})
        records.append({'name': name, 'dtype': 'torch.float64', 'shape': list(shape),
                        'byte_order': 'little', 'chunks': chunks})
    h = hashlib.sha256()
    for v in state:
        h.update(v['name'].encode())
        h.update(v['sha256'].encode())
    request = {'experts': 2, 'classes': 2, 'label': 0, 'top_k': 2,
               'gate': 'SELECTED_SOFTMAX', 'tie_policy': 'ANY_LEGAL_TOPK', 'training': False,
               'center': center, 'radius': '1/255', 'clip': ['-1', '1'], 'margin': '0',
               'model_state': {'sha256': h.hexdigest(), 'tensor_count': len(state),
                               'parameter_count': sum(math.prod(v['shape']) for v in state)}}
    manifest = {'schema': SCHEMA, 'tensors': records,
                'declaration': {'schema': 'SCOPED_DECLARED_TOP2_V1', 'request': request,
                                'center': {'tensor_ref': '@center'}, 'state_inventory': state,
                                'networks': graphs}}
    write(root, 'manifest.json', manifest, HEADER_LIMIT)
    return identity(manifest)


def oracle(root, expected, *, large=False):
    """Check every emitted rational against the fixed formula, not another reader."""
    start = time.monotonic()
    reader = TensorRows(root, expected, deadline=start+300)
    graphs, tensors = recipe(large)
    expected_plan = []
    for graph in graphs:
        for layer in graph['layers']:
            if layer['kind'] == 'Linear':
                shape, salt, _ = tensors[layer['weight_name']]
                expected_plan.extend((graph['name'], layer['index'], i, shape[1], salt)
                                     for i in range(shape[0]))
    n = 0
    for row in reader.rows():
        graph, layer, i, width, salt = expected_plan[n]
        if (row.graph, row.layer, row.row, len(row.coefficients)) != (graph, layer, i, width):
            raise ValueError('oracle row order/shape')
        if row.bias != F(i-salt, 16):
            raise ValueError('oracle bias')
        for col, value in enumerate(row.coefficients):
            flat = i*width+col
            if not large and flat < 3:
                expected_value = F(1, 1 << 1074) if flat == 2 else F(0)
            else:
                sign = -1 if (i+col+salt) % 2 else 1
                expected_value = F(sign*((i*17+col*7+salt*3) % 31+1), 32)
            if value != expected_value:
                raise ValueError('independent analytic coefficient')
        n += 1
    if n != len(expected_plan):
        raise ValueError('oracle complete row roster')
    receipt = reader.receipt()
    params = sum(math.prod(spec[0]) for spec in tensors.values())
    if (receipt['statistics']['values'] != params or receipt['statistics']['raw_bytes'] != 8*params
            or receipt['statistics']['affine_rows'] != n
            or len(receipt['tensor_hashes']) != len(tensors)):
        raise ValueError('oracle complete byte/value/tensor census')
    manifest = load(root, 'manifest.json', HEADER_LIMIT)
    expected_chunks = sum(len(t['chunks']) for t in manifest['tensors'])
    if receipt['statistics']['chunk_reads'] != expected_chunks:
        raise ValueError('oracle chunk read census')
    if large and receipt['source_bytes'] <= 4*2**20:
        raise ValueError('registered source size control')
    return {'receipt': receipt, 'component_seconds': time.monotonic()-start,
            'oracle': 'all coefficients, biases, ordered rows, values and bytes checked'}
