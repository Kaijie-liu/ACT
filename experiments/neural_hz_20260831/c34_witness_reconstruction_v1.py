"""Exact surviving-row/alias extension before unchanged native input recovery."""
from fractions import Fraction as F
import hashlib
import math
import numpy as np
import torch

from act.back_end.solver.solver_hz import _HZMILP
from experiments.neural_hz_20260831.c32_fresh_lineage_v1 import decode
from experiments.neural_hz_20260831.c32_splice_binding_v1 import SplicedState
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays


def recover(new,model,x,input_hz,input_shape,lane,original,*,pool,enabled=False):
    if not enabled:return None
    if type(new) is not SplicedState or type(model) is not _HZMILP:
        raise ValueError('new bound state and actual native model required')
    new.validate();hz=new.hz
    pool.charge('terminal_witness_full_frame_and_map_metadata',8*hz.n_cont+8*len(new.lineage.eq_roots)+512)
    x=np.asarray(x)
    if (x.dtype!=np.float64 or x.shape!=(model.n_var,) or not np.isfinite(x).all()
            or model.cont_eliminations or model.bin_fixes or lane!=0
            or not input_shape or input_shape[0]!=1 or math.prod(input_shape)!=input_hz.n_out
            or input_hz.frame_id!=hz.frame_id or input_hz.n_cont>new.lineage.old_n_cont
            or input_hz.n_bin>new.original_fields['old_n_bin']):
        raise ValueError('witness model, input shape, original frame or forbidden reduction changed')
    for source,n,limit in ((model.cont_source,model.n_cont,hz.n_cont),(model.bin_source,model.n_bin,hz.n_bin)):
        if (type(source) is not np.ndarray or source.shape!=(n,) or source.dtype!=np.int64
                or np.any(source<0) or np.any(source>=limit) or np.any(source[1:]<=source[:-1])):
            raise ValueError('invalid native retained global coordinate map')
    continuous=np.zeros(hz.n_cont,np.float64);continuous[model.cont_source]=x[:model.n_cont]
    parents=0
    for column in new.lineage.columns:
        col=int(column);at=new.lineage.old_n_eq+col-new.lineage.old_n_cont
        _,d,c,inequality,pivot,sign=decode(new.lineage.eq_roots[at])
        row=c if inequality else new.lineage.eq_row(c,pool=pool)
        matrix=hz.Auc if inequality else hz.Ac
        a,b=map(int,matrix.indptr[row:row+2])
        parents+=int(np.searchsorted(matrix.indices[a:b],col))
    aliases=int(np.count_nonzero(new.lineage.eq_roots<0))
    # The existing C27 reader priced only row queries. Prepay ALL additional
    # Fraction arithmetic/box/alias work in this separate witness-diagnostic
    # ledger; never claim it as free or as part of the generator's256M total.
    pool.charge('terminal_all_surviving_parent_fraction_arithmetic',128*parents)
    pool.charge('terminal_all_legacy_alias_fraction_arithmetic',64*aliases)
    extended=new.reconstruct_fraction(continuous,pool=pool)
    if any(extended[i]!=F(float(continuous[i])) for i in range(input_hz.n_cont)):
        raise ValueError('internal-factor reconstruction changed an original input coordinate')
    pool.charge('terminal_original_input_reconstruction',64*(input_hz.Gc.nnz+input_hz.Gb.nnz+input_hz.n_out))
    xi=np.asarray([float(v) for v in extended[:input_hz.n_cont]],np.float64)
    binary=-np.ones(input_hz.n_bin,np.float64)
    selected=model.bin_source<input_hz.n_bin
    binary[model.bin_source[selected]]=2*x[model.n_cont:][selected]-1
    value=input_hz.c.copy()
    if input_hz.n_cont:value+=np.asarray(input_hz.Gc@xi).reshape(-1)
    if input_hz.n_bin:value+=np.asarray(input_hz.Gb@binary).reshape(-1)
    actual=original(model,x,input_hz,input_shape,lane)
    expected=torch.from_numpy(value.reshape(input_shape).copy())[lane].clone()
    if actual is None or not torch.isfinite(actual).all() or not torch.equal(actual,expected):
        raise ValueError('unchanged native input recovery disagrees with exact extension')
    return actual,dict(all_continuous_slots_reconstructed=len(extended),all_unit_pairs_reconstructed=len(new.lineage.columns),
        surviving_parent_terms=parents,legacy_aliases_reconstructed=aliases,
        original_input_latents_unchanged=True,native_input_recovery_bitwise_equal=True,
        exact_extension_from_surviving_rows=True,native_point_and_coordinate_sha256=digest_arrays(x,model.cont_source,model.bin_source),
        diagnostic_work=pool.used,diagnostic_work_parts=dict(pool.parts),
        concrete_network_validation_still_required=True,formal_gain=0)
