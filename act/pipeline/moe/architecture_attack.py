"""S1 common PGD and lossless endpoint-referenced witness storage.

Attack results are auxiliary, never used as free evidence by either verifier.
The codec changes storage only, not the attack point or numerical replay.
"""
import base64
import struct
import time


def encode_point(point, lower, upper):
    import torch
    if point.dtype != torch.float64 or point.shape != lower.shape or lower.shape != upper.shape:
        raise ValueError('witness shape/dtype')
    if not bool(torch.isfinite(point).all()) or not bool(((lower <= point) & (point <= upper)).all()):
        raise ValueError('finite in-box witness')
    mask = bytearray(); interior = []
    for p, lo, hi in zip(point.flatten().tolist(), lower.flatten().tolist(), upper.flatten().tolist()):
        code = 1 if p == lo else 2 if p == hi else 0
        mask.append(code)
        if code == 0:
            interior.append(p)
    return dict(codec='box-endpoints-f64-v1', shape=list(point.shape),
        mask=base64.b64encode(mask).decode(),
        interior=base64.b64encode(struct.pack('<'+'d'*len(interior), *interior)).decode())


def decode_point(record, lower, upper):
    import torch
    if record.get('codec') != 'box-endpoints-f64-v1' or record['shape'] != list(lower.shape) or lower.shape != upper.shape:
        raise ValueError('witness codec/domain shape')
    mask = base64.b64decode(record['mask'], validate=True)
    raw = base64.b64decode(record['interior'], validate=True)
    if len(mask) != lower.numel() or any(c not in (0, 1, 2) for c in mask) or len(raw) != 8*mask.count(0):
        raise ValueError('complete witness coordinates')
    inside = iter(struct.unpack('<'+'d'*mask.count(0), raw))
    values = [next(inside) if c == 0 else lo if c == 1 else hi
        for c, lo, hi in zip(mask, lower.flatten().tolist(), upper.flatten().tolist())]
    point = torch.tensor(values, dtype=torch.float64).reshape(lower.shape)
    if not bool(torch.isfinite(point).all()) or not bool(((lower <= point) & (point <= upper)).all()):
        raise ValueError('decoded witness outside domain')
    return point


def attack(model, tensors, *, radius, seed, deadline):
    import torch
    from act.back_end.moe.class_separated_top1 import validate_replay
    from act.pipeline.moe.multilayer_study import public_pgd
    if radius not in (2, 8) or type(seed) is not int:
        raise ValueError('S1 attack radius/seed')
    if time.monotonic() >= deadline:
        raise TimeoutError('attack domain deadline')
    x, lo, hi, rows, rhs = (tensors[k] for k in ('center', 'lower', 'upper', 'rows', 'thresholds'))
    start = time.monotonic()
    with torch.no_grad():
        if not bool(torch.isfinite(model(x)).all()):
            raise ValueError('nonfinite full-model clean output')
    if validate_replay(model, x, lo, hi, rows, rhs):
        result = dict(status='UNSAFE_REPLAYED', completed_fixed_search=False,
            stop_reason='clean_center_counterexample', iterations=0, witness=x.tolist(),
            guarantee='FULL_MODEL_REPLAY')
    else:
        result = public_pgd(model, tensors, deadline, seed, steps=20, restarts=2,
            step_size=radius/255/4)
    result.update(seed=seed, steps=20, restarts=2, step_size=radius/255/4,
        max_updates=40, replayed=False, formal_SAFE=False)
    if time.monotonic() >= deadline:
        return dict(status='ATTACK_TIMEOUT', completed_fixed_search=False, iterations=result['iterations'],
            replayed=False, guarantee='NONE', formal_SAFE=False, seed=seed, steps=20, restarts=2,
            step_size=radius/255/4, seconds=time.monotonic()-start)
    if result['status'] == 'UNSAFE_REPLAYED':
        point = torch.tensor(result['witness'], dtype=torch.float64)
        packed = ({'reference': 'bound input center'} if torch.equal(point, x)
            else encode_point(point, lo, hi))
        restored = x if 'reference' in packed else decode_point(packed, lo, hi)
        if not torch.equal(restored, point) or not validate_replay(model, restored, lo, hi, rows, rhs):
            raise ValueError('same-point complete-model replay after serialization')
        result.update(witness=packed, replayed=True)
    else:
        result.pop('witness', None)
    if time.monotonic() >= deadline:
        return dict(status='ATTACK_TIMEOUT', completed_fixed_search=False, replayed=False,
            guarantee='NONE', formal_SAFE=False, seconds=time.monotonic()-start)
    result['seconds'] = time.monotonic()-start
    return result
