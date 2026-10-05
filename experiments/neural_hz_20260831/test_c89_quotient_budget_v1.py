"""Ordinary quotient-composition, exact native inverse and whole-budget tests."""
from fractions import Fraction as F
import gc
import weakref
import numpy as np
import pytest
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c85_exact_filter_transform_v1 import transform
from experiments.neural_hz_20260831.c86_complete_tile_hz_v1 import extend_actual, project_outputs
from experiments.neural_hz_20260831.c88_inline_tile_v1 import construct, actual_rows, NativeUnproved
from experiments.neural_hz_20260831.c89_quotient_budget_v1 import lower, bill, select, compose_row


def native_rows(report, packet):
    rows = []
    for i in range(len(packet['rhs'])):
        a,b=map(int,packet['indptr'][i:i+2])
        rows.append(dict(slot=int(packet['pivots'][i]),rhs=float(packet['rhs'][i]),
            gauge=int(packet['gauges'][i]), coefficients=tuple(zip(map(int,packet['columns'][a:b]),map(float,packet['native'][a:b])))))
    return rows[:report['new_factors']], rows[report['new_factors']:]


@pytest.mark.parametrize('channels,outputs',[(2,3),(5,4),(8,8)])
@pytest.mark.parametrize('ratio',[(1,-1),(3,-3),(-5,-3)])
def test_complete_polynomial_commutes_and_inverse_uses_only_native_rows(channels,outputs,ratio):
    base=channels*16+outputs*4
    kernel=((np.arange(outputs*channels*9)%29)-14).reshape(outputs,channels,3,3).astype(np.float32)/32
    _,data=transform(kernel,pool=WorkPool(256_000_000),enabled=True)
    report,original=construct(data,np.arange(channels*16).reshape(channels,4,4),
        (np.arange(channels*16)%3).reshape(channels,4,4).astype(np.int32),
        np.arange(channels*16,base).reshape(outputs,2,2),np.full((outputs,2,2),15,np.int32),
        base,pool=WorkPool(256_000_000),enabled=True)
    original_aux,original_outputs=actual_rows(report,original)
    expected=project_outputs(original_aux,original_outputs,base)
    roots=np.arange(base);weights=[(1,0)]*base
    for i in range(1,channels*16,2):roots[i]=i-1;weights[i]=ratio
    native,info=lower(report,original,roots,weights,pool=WorkPool(256_000_000),enabled=True)
    aux,out=native_rows(report,native)
    exact_expected=[]
    for poly in expected:
        mapped={}
        for col,value in poly.items():
            anchor=int(roots[col]);m,e=weights[col]
            mapped[anchor]=mapped.get(anchor,F(0))+value*m*F(2)**e
        exact_expected.append({c:v for c,v in mapped.items() if v})
    assert project_outputs(aux,out,base)==exact_expected
    point=[F((i%5)-2,4) for i in range(base)]
    for row,poly in zip(out,exact_expected,strict=True):
        pivot=row['slot'];point[pivot]=-sum((v*point[c] for c,v in poly.items() if c!=pivot),F(0))/poly[pivot]
    extended=extend_actual(aux,point)
    assert all(sum((F(v)*extended[c] for c,v in r['coefficients']),F(0))==0 for r in out)
    assert info['changed_old_coordinate_occurrences']>0
    assert info['old_scalar_factors_restored']==0
    refs=[weakref.ref(a) for a in original.values()]
    del original
    gc.collect()
    assert all(r() is None for r in refs)
    assert 'words' not in native and 'powers' not in native


def test_original_projected_output_is_not_restored():
    roots=np.array([0,0]);weights=[(1,0),(1,-1)]
    with pytest.raises(NativeUnproved,match='already projected'):
        compose_row([0,1],[-1,1],[0,0],1,0,roots,weights,pool=WorkPool(1000))


def test_coalescence_and_word_gate_are_exact():
    roots=np.array([0,0,2]);weights=[(1,0),(3,-2),(1,0)]
    row=compose_row([0,1,2],[-3,4,1],[0,0,1],2,1,roots,weights,pool=WorkPool(1000))
    assert row['columns'].tolist()==[2]
    with pytest.raises(NativeUnproved):
        compose_row([0,2],[(1<<53)+1,1],[-40,1],2,1,roots,weights,pool=WorkPool(1000))


def fake_packet(n,rows,aux):
    return dict(columns=np.zeros(n,np.int32),native=np.ones(n,np.float64),
        indptr=np.zeros(rows+1,np.int32),ab_indptr=np.zeros(rows+1,np.int32),
        rhs=np.zeros(rows,np.float64),pivots=np.zeros(rows,np.int32),gauges=np.zeros(rows,np.int32))


def test_declared_bill_counts_both_pointers_and_all_auxiliary_fields():
    packet=fake_packet(200,30,20)
    b=bill(packet,new_factors=20,direct_nnz=500)
    assert b['declared_new_bytes']==sum(a.nbytes for a in packet.values())+8*(10+8*20+8)
    assert b['declared_old_bytes']==12*500+16*10+8
    assert b['new_emission_work']==64*30+16*(200+30)
    assert not b['full_LIVE_physical_gate_proved']


@pytest.mark.parametrize('existing_aux,existing_work',[(28,593353),(16000,593353),(28,15999999)])
def test_global_caps_never_reset(existing_aux,existing_work):
    records=[]
    for index in range(8):
        b=bill(fake_packet(10000,2100,2000),new_factors=2000,direct_nnz=40000+index)
        records.append(dict(mapped_native_pass=True,bill=b))
    got=select(records,existing_aux=existing_aux,existing_work=existing_work,existing_entries=112,
        pool=WorkPool(256_000_000),enabled=True)
    chosen=got['selected_positions']
    assert got['whole_auxiliary_reserve_used']==existing_aux+sum(records[i]['bill']['new_factors'] for i in chosen)<=16384
    assert got['whole_declared_emission_reserve_used']==existing_work+sum(records[i]['bill']['new_emission_work'] for i in chosen)<=16000000
    assert chosen==sorted(chosen,reverse=True)


def test_default_off_and_precharge():
    pool=WorkPool(0)
    assert lower(None,None,None,None,pool=pool) is None
    assert select([],existing_aux=0,existing_work=0,existing_entries=0,pool=pool) is None
    assert pool.used==0
    with pytest.raises(MemoryError):
        select([dict(mapped_native_pass=False)],existing_aux=0,existing_work=0,existing_entries=0,pool=pool,enabled=True)


def test_aggregate_plan_rejects_nnz_only_win_with_storage_growth():
    b=bill(fake_packet(990,110,100),new_factors=100,direct_nnz=1000)
    assert b['nnz_saving']>0 and b['byte_saving']<0
    got=select([dict(mapped_native_pass=True,bill=b)],existing_aux=28,existing_work=593353,
        existing_entries=112,pool=WorkPool(10000),enabled=True)
    assert got['selected_positions']==[]
