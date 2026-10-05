"""Independent ordinary direct convolutions and corrupt-row rejection."""
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform
from experiments.neural_hz_20260831.c88_inline_tile_v1 import construct
from experiments.neural_hz_20260831.c89_quotient_budget_v1 import lower
from experiments.neural_hz_20260831.c90_actual_circuit_proof_v1 import rebase,prove


def fixture(channels):
    outputs=channels+1;base=16*channels+4*outputs
    kernel=((np.arange(outputs*channels*9)%23)-11).reshape(outputs,channels,3,3).astype(np.float32)/32
    _,data=transform(kernel,pool=WorkPool(256_000_000),enabled=True)
    ids=np.arange(16*channels).reshape(channels,4,4)
    # Ordinary shared source coordinates and nonuniform dyadic normalization.
    ids[:,1,0]=ids[:,0,0]
    exps=(np.arange(16*channels)%3).reshape(channels,4,4).astype(np.int32)
    out=np.arange(16*channels,base).reshape(outputs,2,2)
    report,raw=construct(data,ids,exps,out,np.full(out.shape,15,np.int32),base,
        pool=WorkPool(256_000_000),enabled=True)
    native,_=lower(report,raw,np.arange(base),[(1,0)]*base,pool=WorkPool(256_000_000),enabled=True)
    original=[]
    for k,i,j in np.ndindex(out.shape):
        poly={int(out[k,i,j]):2.**15}
        for c,a,b in np.ndindex(channels,3,3):
            col=int(ids[c,i+a,j+b]);v=float(kernel[k,c,a,b])*2.**int(exps[c,i+a,j+b])
            poly[col]=poly.get(col,0.)-v
        original.append(dict(pivot=int(out[k,i,j]),rhs=0.,coefficients=tuple((c,v) for c,v in poly.items() if v)))
    return report,native,original,base


@pytest.mark.parametrize('channels',[2,4,8])
@pytest.mark.parametrize('offset',[0,17])
def test_full_independent_original_equations_and_global_names(channels,offset):
    report,packet,original,base=fixture(channels)
    native=rebase(packet,base,offset,pool=WorkPool(256_000_000))
    got=prove(native,original,[0]*len(original),old_n_cont=base,first_aux=base+offset,
        new_factors=report['new_factors'],pool=WorkPool(256_000_000),enabled=True)
    assert got['all_original_output_equations_proved']==len(original)
    assert got['all_auxiliary_equations_and_redundant_boxes_proved']==report['new_factors']
    assert got['universal_unique_box_extension'] and got['original_source_equivalence']
    assert np.array_equal(packet['columns'][packet['columns']<base],native['columns'][packet['columns']<base])


@pytest.mark.parametrize('change',['coefficient','rhs','pivot'])
def test_modified_actual_rows_are_rejected(change):
    report,packet,original,base=fixture(2)
    if change=='coefficient':packet['native'][-2]*=2
    if change=='rhs':packet['rhs'][-1]=1.
    if change=='pivot':packet['pivots'][0]+=1
    with pytest.raises(ValueError):
        prove(packet,original,[0]*len(original),old_n_cont=base,first_aux=base,
            new_factors=report['new_factors'],pool=WorkPool(256_000_000),enabled=True)


def test_default_off_never_reads_or_spends():
    pool=WorkPool(0)
    assert prove(None,None,None,old_n_cont=0,first_aux=0,new_factors=0,pool=pool) is None
    assert pool.used==0
