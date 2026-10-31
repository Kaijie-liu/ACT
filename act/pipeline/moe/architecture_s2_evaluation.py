"""S2a primary batch measurements and shared physical input convention.

Training and primary evaluation call the frozen model_bounds function. The
uint8 source is lossless; normalization reproduces ToTensor's float32 division.
Secondary verification lifts these SAME represented float32 boxes to float64.
No outward-rounded input-containment or native-execution proof is claimed.
"""
import time
import base64
import io
import numpy as np
import torch
from .route_envelope_ibp import model_bounds, margin_rows


def pack_arrays(arrays):
    """Lossless numeric archive in the existing shared JSON/gzip byte account."""
    if any(not isinstance(v,np.ndarray) or v.dtype.hasobject for v in arrays.values()):
        raise ValueError('numeric arrays only')
    stream=io.BytesIO()
    np.savez_compressed(stream,**arrays)
    return dict(format='S2_NUMERIC_NPZ_V1',data=base64.b64encode(stream.getvalue()).decode('ascii'))


def unpack_arrays(packet):
    if set(packet)!={'format','data'} or packet['format']!='S2_NUMERIC_NPZ_V1':
        raise ValueError('numeric archive format')
    raw=base64.b64decode(packet['data'],validate=True)
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        if len(archive.files)!=len(set(archive.files)):
            raise ValueError('duplicate array name')
        arrays={k:archive[k] for k in archive.files}
    if any(v.dtype.hasobject for v in arrays.values()):raise ValueError('object array')
    return arrays


def check_inputs(arrays, *, count=10000):
    if (set(arrays)!={'pixels','labels'} or arrays['pixels'].shape!=(count,3,32,32)
            or arrays['pixels'].dtype!=np.uint8 or arrays['labels'].shape!=(count,)
            or arrays['labels'].dtype!=np.int64 or
            not ((arrays['labels']>=0)&(arrays['labels']<10)).all()):
        raise ValueError('fixed S2a test inputs')


def load_secondary_input(packet, index):
    if type(index) is not int or not 0<=index<1000:
        raise ValueError('fixed secondary input prefix')
    arrays=unpack_arrays(packet);check_inputs(arrays)
    label=int(arrays['labels'][index])
    return secondary_domain(torch.from_numpy(arrays['pixels'][index:index+1]),label),label


def input_batch(pixels, numerator=2):
    if pixels.dtype != torch.uint8 or pixels.ndim < 2 or numerator != 2:
        raise ValueError('S2a fixed uint8 input and radius')
    center = pixels.to(torch.float32)/255
    epsilon = numerator/255
    return center, (center-epsilon).clamp(0, 1), (center+epsilon).clamp(0, 1)


def secondary_domain(pixels, label, *, classes=10):
    if len(pixels) != 1 or type(label) is not int or not 0 <= label < classes:
        raise ValueError('one original input and true class')
    center, lower, upper = input_batch(pixels)
    rows = margin_rows(torch.tensor([label], dtype=torch.long), classes, dtype=torch.float64)[0]
    return dict(center=center.double(), lower=lower.double(), upper=upper.double(),
                rows=rows, thresholds=torch.zeros(classes-1, dtype=torch.float64))


@torch.no_grad()
def primary_batch(model, pixels, labels):
    """Per-input raw margins, clean state and complete route-candidate counts."""
    began = time.monotonic()
    center, lower, upper = input_batch(pixels)
    model.eval()
    logits = model(center)
    if labels.dtype != torch.long or labels.shape != (len(pixels),):
        raise ValueError('batch labels')
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError('finite class outputs')
    rows = margin_rows(labels, logits.shape[1], dtype=center.dtype)
    lb, ub, envelope = model_bounds(model, lower, upper, rows)
    if not (torch.isfinite(lb).all() and torch.isfinite(ub).all() and (lb <= ub).all()):
        raise ValueError('finite ordered property intervals')
    prediction = logits.argmax(1)
    clean = prediction == labels
    certified = clean & (lb > 0).all(1)
    route_counts = (envelope.candidates.sum(1).to(torch.uint8) if envelope is not None
                    else torch.zeros(len(pixels), dtype=torch.uint8))
    # Every input contributes, including clean mistakes. Timing is batch work,
    # not a claim of independent cold per-input latency.
    arrays = dict(prediction=prediction.cpu().numpy(), label=labels.cpu().numpy(),
        clean=clean.cpu().numpy(), certified=certified.cpu().numpy(),
        lower_margin=lb.cpu().numpy(), upper_margin=ub.cpu().numpy(),
        route_count=route_counts.cpu().numpy())
    if envelope is not None:
        arrays.update(router_lower=envelope.router_lower.cpu().numpy(),
                      router_upper=envelope.router_upper.cpu().numpy(),
                      pair_mask=np.packbits(envelope.candidates.cpu().numpy(), axis=1,
                                            bitorder='little'))
    return arrays, dict(seconds=time.monotonic()-began, count=len(pixels),
        clean=int(clean.sum()), certified=int(certified.sum()),
        mean_margin_width=float((ub-lb).mean()),
        single_route=int((route_counts == 1).sum()) if envelope is not None else None,
        evidence_grade='IBP_NUMERICAL_POLICY', bound_dtype='float32',
        time_semantics='batch normalization/forward/bounds/array capture; load/save/check separately charged')


def check_primary_arrays(arrays, *, family, count, classes=10, experts=8):
    """Readback validation, without rerunning a network or altering a result."""
    required = {'prediction','label','clean','certified','lower_margin','upper_margin','route_count'}
    if family == 'moe':
        required |= {'router_lower','router_upper','pair_mask'}
    elif family != 'dense':
        raise ValueError('S2a family')
    if set(arrays) != required:
        raise ValueError('complete primary array fields')
    for k in ('prediction','label','clean','certified','route_count'):
        if arrays[k].shape != (count,):
            raise ValueError('complete primary array lengths')
    for k in ('prediction','label'):
        if not np.issubdtype(arrays[k].dtype,np.integer) or not ((arrays[k]>=0)&(arrays[k]<classes)).all():
            raise ValueError('class identity')
    if arrays['clean'].dtype != np.bool_ or arrays['certified'].dtype != np.bool_:
        raise ValueError('boolean primary flags')
    lo, hi = arrays['lower_margin'], arrays['upper_margin']
    if (lo.shape != (count,classes-1) or hi.shape != lo.shape or
            lo.dtype != np.float32 or hi.dtype != np.float32 or
            not np.isfinite(lo).all() or not np.isfinite(hi).all() or not (lo<=hi).all()):
        raise ValueError('complete finite property arrays')
    clean = arrays['prediction'] == arrays['label']
    if not np.array_equal(clean,arrays['clean']) or not np.array_equal(clean&(lo>0).all(1),arrays['certified']):
        raise ValueError('positive/clean policy mismatch')
    counts=arrays['route_count']
    if counts.dtype != np.uint8:
        raise ValueError('candidate count dtype')
    if family == 'dense':
        if counts.any():raise ValueError('dense route sentinel')
        return
    rl,ru=arrays['router_lower'],arrays['router_upper']
    if (rl.shape!=(count,experts) or ru.shape!=rl.shape or
            rl.dtype!=np.float32 or ru.dtype!=np.float32 or
            not np.isfinite(rl).all() or not np.isfinite(ru).all() or not (rl<=ru).all()):
        raise ValueError('complete router ranges')
    import itertools
    pairs=list(itertools.combinations(range(experts),2))
    mask=arrays['pair_mask']
    if mask.shape!=(count,(len(pairs)+7)//8) or mask.dtype!=np.uint8:
        raise ValueError('pair mask shape')
    bits=np.unpackbits(mask,axis=1,bitorder='little')
    if bits[:,len(pairs):].any():raise ValueError('noncanonical pair-mask padding')
    bits=bits[:,:len(pairs)].astype(bool)
    expected=[]
    for a,b in pairs:
        rest=[c for c in range(experts) if c not in (a,b)]
        threshold=rl[:,rest].max(1) if rest else np.full(count,-np.inf)
        expected.append((ru[:,a]>=threshold)&(ru[:,b]>=threshold))
    if not np.array_equal(bits,np.stack(expected,1)) or not np.array_equal(bits.sum(1),counts) or not (counts>=1).all():
        raise ValueError('complete tie-legal candidate cover')
