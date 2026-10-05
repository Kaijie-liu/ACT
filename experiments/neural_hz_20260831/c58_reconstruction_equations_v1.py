"""Default-off exact inverse equations; not an HZ or native coefficient adapter."""
from fractions import Fraction
import hashlib
import io
import json
import numpy as np
from experiments.neural_hz_20260831.c54_exact_dyadic_v1 import Pool, canonical, unpack
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import word, fraction

SCHEMA = 'c58_bounded_coefficient_inverse_equations_v1'


def operands(ratio):
    """Both returned operands obey the original scalar window, not their ratio."""
    m, e = word(*ratio)
    if not m or abs(fraction((m, e))) > 1:
        raise ValueError('nonzero unit-bounded exact inverse required')
    q = max(0, -20 - (abs(m).bit_length() - 1 + e))
    return canonical(m, e + q), canonical(1, q)


def seal(packet):
    h = hashlib.sha256()
    for name in ('schema', 'source_binding', 'n_cont', 'n_bin'):
        h.update(json.dumps([name, packet[name]]).encode())
    for name, a in [('frame', packet['frame']), ('pairs', packet['pairs']),
                    *sorted(packet['scalars'].items())]:
        if type(a) is not np.ndarray or a.dtype.hasobject or not a.flags.c_contiguous:
            raise ValueError('numeric-only contiguous inverse packet required')
        h.update(json.dumps([name, a.shape, a.dtype.str]).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def build(roots, weights, *, n_bin, source_binding, pool, enabled=False):
    if not enabled:
        return None
    count = len(weights)
    if (type(roots) is not np.ndarray or roots.shape != (count,)
            or count >= 2**32 or n_bin < 0 or not isinstance(source_binding, str)):
        raise ValueError('complete original inverse frame required')
    pool.charge('c58_complete_frame_build', 16 * count + 1024)
    scalar = Pool(pool)
    pair_ids, ratios, pairs = {}, {}, []
    frame = np.empty(count, np.uint64)
    for i, (root, ratio) in enumerate(zip(roots, weights)):
        root = int(root)
        if not 0 <= root <= i:
            raise ValueError('original retained-root birth order required')
        ratio = word(*ratio)
        if ratio not in ratios:
            pool.charge('c58_distinct_equation_operands', 128)
            numerator, denominator = operands(ratio)
            pair = (scalar.intern(numerator), scalar.intern(denominator))
            if pair not in pair_ids:
                pair_ids[pair] = len(pairs)
                pairs.append(pair)
            ratios[ratio] = pair_ids[pair]
        pair_id = ratios[ratio]
        if pair_id >= 2**32:
            raise MemoryError('inverse pair ID capacity')
        frame[i] = np.uint64(root | (pair_id << 32))
    if len(scalar.values) >= 2**32:
        raise MemoryError('inverse operand ID capacity')
    packet = dict(schema=SCHEMA, source_binding=source_binding, n_cont=count,
                  n_bin=int(n_bin), frame=frame,
                  pairs=np.asarray(pairs, np.uint32).reshape(-1, 2),
                  scalars=scalar.pack())
    pool.charge('c58_packet_seal', 8 * count + 32 * len(pairs) + 1024)
    packet['seal'] = seal(packet)
    return packet


def audit(packet, roots, weights, *, n_bin, source_binding, pool):
    if (set(packet) != {'schema', 'source_binding', 'n_cont', 'n_bin', 'frame', 'pairs', 'scalars', 'seal'}
            or packet['schema'] != SCHEMA or packet['seal'] != seal(packet)
            or packet['source_binding'] != source_binding or packet['n_bin'] != n_bin
            or packet['n_cont'] != len(weights) or roots.shape != (len(weights),)):
        raise ValueError('source-bound complete inverse packet differs')
    frame, pairs = packet['frame'], packet['pairs']
    if (frame.dtype != np.uint64 or frame.shape != (len(weights),)
            or pairs.dtype != np.uint32 or pairs.ndim != 2 or pairs.shape[1] != 2):
        raise ValueError('inverse coordinate or equation layout differs')
    table = unpack(packet['scalars'])
    pool.charge('c58_complete_coordinate_audit', 16 * len(weights) + 64 * len(table) + 1024)
    values, used = [], set()
    if len(set(map(tuple, pairs.tolist()))) != len(pairs):
        raise ValueError('duplicate inverse equation')
    for num_id, den_id in pairs:
        pool.charge('c58_independent_Fraction_equation_proof', 160)
        num_id, den_id = int(num_id), int(den_id)
        if max(num_id, den_id) >= len(table):
            raise ValueError('inverse operand outside pool')
        n, d = table[num_id], table[den_id]
        if d[0] != 1 or not 0 <= d[1] <= 40:
            raise ValueError('positive bounded dyadic denominator required')
        exact = fraction(n) / fraction(d)
        value = word(n[0], n[1] - d[1])
        if not 0 < abs(exact) <= 1 or exact != fraction(value):
            raise ValueError('inverse equation or redundant box proof failed')
        values.append(value)
        used.update((num_id, den_id))
    if used != set(range(len(table))):
        raise ValueError('unreferenced scalar pool payload')
    live_pairs = set()
    for index, packed in enumerate(frame):
        packed = int(packed)
        root, pair_id = packed & (2**32 - 1), packed >> 32
        if (pair_id >= len(values) or root != int(roots[index]) or not 0 <= root <= index
                or values[pair_id] != weights[index]):
            raise ValueError('actual original-coordinate inverse differs')
        # A composed root is retained, not another eliminated coordinate.
        if int(roots[root]) != root or weights[root] != (1, 0):
            raise ValueError('inverse root is not retained')
        live_pairs.add(pair_id)
    if live_pairs != set(range(len(pairs))):
        raise ValueError('unused inverse equation payload')
    return dict(coordinates_checked=len(weights), independent_equations_checked=len(pairs),
                scalar_operands_checked=len(table), all_operands_in_original_window=True,
                all_original_roots_and_ratios_exact=True, all_inverse_boxes_redundant=True,
                maximum_denominator_power=max((table[int(p[1])][1] for p in pairs), default=0))


def encode(packet):
    metadata = {k: packet[k] for k in ('schema', 'source_binding', 'n_cont', 'n_bin', 'seal')}
    out = io.BytesIO()
    np.savez(out, metadata=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), np.uint8),
             frame=packet['frame'], pairs=packet['pairs'], **packet['scalars'])
    return out.getvalue()


def decode(payload):
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != {'metadata', 'frame', 'pairs', 'indptr', 'limbs', 'exponent', 'sign'}:
            raise ValueError('complete numeric inverse archive required')
        result = json.loads(archive['metadata'].tobytes())
        result.update(frame=archive['frame'], pairs=archive['pairs'],
                      scalars={k: archive[k] for k in ('indptr', 'limbs', 'exponent', 'sign')})
    if result['seal'] != seal(result):
        raise ValueError('inverse archive changed')
    return result
