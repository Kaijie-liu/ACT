"""Exact bit-decoded binary32 filters, bounded integer transforms, no rounding."""
import math
import numpy as np
from experiments.neural_hz_20260831.c85_exact_tile_algebra_v1 import H,recover_filter


def transform(weights,*,pool,enabled=False):
    if not enabled:return None
    w=np.asarray(weights)
    if w.dtype!=np.float32 or w.ndim!=4 or w.shape[-2:]!=(3,3):
        raise ValueError('ordinary original binary32 3x3 filters required')
    kernels=int(w.shape[0]*w.shape[1])
    pool.charge('c85_complete_integer_filter_transform_and_inverse',400*kernels+4*int(w.size))
    bits=w.view(np.uint32);exponent=((bits>>23)&255).astype(np.int32)
    fraction=bits&np.uint32((1<<23)-1)
    if np.any(exponent==255) or np.any((exponent==0)&(fraction!=0)):
        raise ValueError('nonfinite/subnormal source outside ordinary certified integer domain')
    nonzero=exponent!=0
    mantissa=(fraction|np.uint32(1<<23)).astype(np.int64)
    mantissa[~nonzero]=0;mantissa[(bits>>31)!=0]*=-1
    powers=exponent-127-23
    minimum=np.where(nonzero,powers,300).min(axis=(-2,-1))
    minimum=np.where(minimum==300,0,minimum)
    shifts=np.where(nonzero,powers-minimum[...,None,None],0)
    # Each transformed row has total absolute integer multiplier at most9.
    #24+shift<=59 leaves all prefix/final signed sums inside int64.
    if np.any(shifts>35) or np.any(shifts<0):
        raise ValueError('complete signed-int64 transform envelope not established')
    aligned=np.left_shift(mantissa,shifts)
    temp=np.einsum('ai,kcij->kcaj',H,aligned,optimize=False)
    numerator=np.einsum('kcaj,bj->kcab',temp,H,optimize=False)
    # The two-step inverse intermediates are bounded by4*aligned input values.
    # The established9-fold envelope also covers each difference used below.
    recovered=recover_filter(numerator)
    if not np.array_equal(recovered,4*aligned):
        raise ValueError('independent original-kernel recovery differs')
    absolute=np.abs(numerator).astype(np.uint64)
    lowbit=absolute & (~absolute+np.uint64(1))
    odd=absolute//np.where(lowbit==0,np.uint64(1),lowbit)
    representable=odd<=np.uint64(1<<53)
    native=np.ldexp(numerator.astype(np.float64),minimum[...,None,None]-2)
    native[numerator==0]=0.
    finite=np.isfinite(native)
    # Native values are published only if every exact numerator has <=53
    # significant bits. No rounded prefix is accepted.
    report=dict(kernels=kernels,original_weights=int(w.size),transformed_coefficients=int(numerator.size),
        transformed_nonzero=int(np.count_nonzero(numerator)),
        maximum_alignment_shift=int(shifts.max(initial=0)),
        exact_binary64_failures=int(np.count_nonzero(~representable)),
        nonfinite_native_values=int(np.count_nonzero(~finite)),
        independent_original_integer_kernel_recovery=True,
        complete_signed_int64_envelope=True,
        all_coefficients_exact_binary64=bool(representable.all() and finite.all()))
    if not report['all_coefficients_exact_binary64']:
        return report,None
    # Per-output-channel/per-transform-position channel-sum coefficient envelope.
    # Missing normalizing pivot, source scales and RHS deliberately NOT assumed.
    mag=np.abs(native);nz=mag!=0
    smallest=np.where(nz,mag,np.inf).min(axis=1)
    largest=mag.max(axis=1)
    active=largest!=0
    _,loexp=np.frexp(smallest);himant,hiexp=np.frexp(largest)
    lower=-19-loexp
    upper=np.where(himant==.5,41,40)-hiexp
    compatible=(~active)|(lower<=upper)
    gauge=np.where(active,np.minimum(np.maximum(0,lower),upper),0).astype(np.int32)
    scaled=np.ldexp(native,gauge[:,None,:,:])
    smag=np.abs(scaled)
    coefficient_window=bool(np.all((smag==0)|((smag>=2.**-20)&(smag<=2.**40))))
    report.update(channel_sum_rows=int(active.size),nonzero_channel_sum_rows=int(active.sum()),
        coefficient_only_gauge_failures=int(np.count_nonzero(~compatible)),
        coefficient_only_native_window_pass=coefficient_window and bool(compatible.all()),
        complete_HZ_row_window_proved=False,normalizing_pivot_and_RHS_checked=False,
        min_nonzero_abs=float(mag[nz].min()) if nz.any() else None,
        max_abs=float(mag.max(initial=0)),max_coefficient_only_gauge=int(gauge.max(initial=0)),
        min_coefficient_only_gauge=int(gauge.min(initial=0)),
        all_native_coefficients_checked=True)
    return report,dict(numerator=numerator,exponent=minimum-2,native=native,gauge=gauge)
