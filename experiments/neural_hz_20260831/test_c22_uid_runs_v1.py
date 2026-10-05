import numpy as np
import pytest

from experiments.neural_hz_20260831.c22_uid_runs_v1 import (
    LIMIT, pack, unpack, validate, build, row_for_uid, uid_for_row)
from experiments.neural_hz_20260831.c10_fused_rows_v1 import WorkPool


@pytest.mark.parametrize('triplet', [(0,0,1),(0,0,LIMIT),(LIMIT-1,LIMIT-1,1),(100,7,400)])
def test_exact_60_bit_round_trip(triplet):
    assert unpack(pack(*triplet))==triplet


@pytest.mark.parametrize('triplet', [(-1,0,1),(0,-1,1),(0,0,0),(LIMIT,0,1),(0,LIMIT,1),
    (LIMIT-1,0,2),(0,LIMIT-1,2),(0.,0,1)])
def test_invalid_packed_domains(triplet):
    with pytest.raises(ValueError): pack(*triplet)


@pytest.mark.parametrize('stream,first,end', [([],10,100),([0,1,10,11,90,12,13,18,19,20],10,90),
    ([0,1,10,11,12,13],10,90),([0,1,90,91],10,90),([10,12,14,16],10,90)])
def test_complete_holes_round_trip_and_nonmain_boundaries(stream,first,end):
    a=np.array(stream,np.int64)
    before=a.tobytes()
    pool=WorkPool(0,0)
    words,report=build(a,first,end,pool=pool)
    assert pool.used==report['standalone_build_work']
    expected={uid:row for row,uid in enumerate(stream) if first<=uid<end}
    validate(words)
    q=WorkPool(0,0)
    for uid in range(end+2): assert row_for_uid(words,uid,pool=q)==expected.get(uid)
    inverse={row:uid for uid,row in expected.items()}
    for row in range(len(stream)+2): assert uid_for_row(words,row,pool=q)==inverse.get(row)
    assert a.tobytes()==before


@pytest.mark.parametrize('words', [[1<<60], [pack(10,0,2),pack(11,3,1)],
    [pack(10,2,2),pack(20,3,1)], [pack(10,0,2),pack(12,2,1)]])
def test_invalid_overlap_and_nonmaximal_words(words):
    with pytest.raises(ValueError): validate(np.array(words,np.uint64))


def test_independent_mapping_detects_validly_encoded_corruption():
    words,_=build(np.array([10,11,14],np.int64),10,90,pool=WorkPool(0,0))
    words[0]=pack(9,0,2)
    validate(words)  # Structural validity is not source binding or a proof.
    assert uid_for_row(words,0,pool=WorkPool(0,0))!=10


@pytest.mark.parametrize('stream', [[10,9],[10,10],[-1],[LIMIT]])
def test_invalid_complete_uid_stream(stream):
    with pytest.raises(ValueError): build(np.array(stream,np.int64),0,90,pool=WorkPool(0,0))


@pytest.mark.parametrize('cap', [0,7,23,24,36])
def test_precharged_scan_pack_array_validation_caps(cap):
    # One row: scan8 + pack16 + array1 + validation12 =37.
    source=np.array([10],np.int64)
    with pytest.raises(MemoryError): build(source,10,90,pool=WorkPool(0,0,max_work=cap))
    assert source.tolist()==[10]


def test_query_cap_precedes_any_word_read():
    words=np.array([pack(10,0,1)],np.uint64)
    with pytest.raises(MemoryError): row_for_uid(words,10,pool=WorkPool(0,0,max_work=15))


@pytest.mark.parametrize('key', [-1,LIMIT,1.5])
def test_query_domain(key):
    with pytest.raises(ValueError): row_for_uid(np.empty(0,np.uint64),key,pool=WorkPool(0,0))
