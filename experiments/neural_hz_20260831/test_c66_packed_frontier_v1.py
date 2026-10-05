"""Ordinary packed metadata, exact recurrence and physical lifetime checks."""
import sys
import weakref
import numpy as np
import pytest
from experiments.neural_hz_20260831.c66_packed_frontier_v1 import ScalarColumns,WordColumns
from experiments.neural_hz_20260831.c66_birth_emission_v1 import lift
from experiments.neural_hz_20260831.c64_compact_boundary_dp_v1 import choose
from experiments.neural_hz_20260831.c57_scalar_consumer_probe_v2 import native_word
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.test_c61_retained_boundary_v1 import data
from experiments.neural_hz_20260831.test_c62_physical_boundary_v2 import complete


@pytest.mark.parametrize('kind',['chain','branch','mixed','halves'])
def test_exact_packed_words_preserve_complete_recurrence(kind):
    parents,ratios,external,_,_,_=data(kind);nc=max(parents)+1
    local={v:native_word(float(r)) for v,r in ratios.items()}
    dense=WordColumns(parents,nc)
    for v,r in local.items():dense[v]=r
    assert dense==local and list(dense)==list(parents)
    assert all(type(x) is int for r in dense.values() for x in r)
    maxima=np.zeros(nc,np.uint64)
    for v,cs in external.items():maxima[v]=max((abs(native_word(float(c))[0]) for c in cs),default=0)
    expected=choose(parents,local,maxima,nc,pool=WorkPool(256_000_000))
    got=choose(parents,dense,maxima,nc,pool=WorkPool(256_000_000))
    assert np.array_equal(expected[0],got[0]) and np.array_equal(expected[1],got[1])
    assert expected[2:]==got[2:]


def test_ordinary_repeated_metadata_is_physically_smaller_than_four_dict_tables():
    count=65536;n=2*count;keys={count+i:i for i in range(count)}
    maps=[WordColumns(keys,n),ScalarColumns(keys,n,np.int64,int),
        ScalarColumns(keys,n,np.int64,int),ScalarColumns(keys,n,np.float64,float)]
    for i,v in enumerate(keys):
        maps[0][v]=(2*i+1,-20);maps[1][v]=i;maps[2][v]=-(1<<63)+v;maps[3][v]=.5
    for m in maps:
        assert m.keys is keys
        with pytest.raises(KeyError):m[0]
    actual=sum(m.values.nbytes+sys.getsizeof(m) for m in maps)
    old_tables=sum(sys.getsizeof(dict.fromkeys(keys)) for _ in maps)
    # Conservative: count ONLY old dict tables, none of their Python values.
    assert actual<old_tables
    assert maps[0][n-1]==(2*count-1,-20) and maps[2][n-1]==-(1<<63)+n-1


def test_only_new_tracker_buffers_die_before_return_with_full_graph_retained():
    _,saved=complete('chain');refs=[]
    def observe(encoder,nodes,root,er,es,lr,ls,tracker):
        refs.append(weakref.ref(tracker))
        refs.extend(weakref.ref(getattr(tracker,name).values) for name in ('local','defining','tags','numerators'))
    result=lift(saved['expression'],saved['keep'],enabled=True,before_fold=observe)
    assert refs and all(ref() is None for ref in refs)
    assert result['construction']['nodes'] and result['fields']['expression'] is saved['expression']
    assert result['fields']['report']['alias_quotient']['work_parts']['c66_packed_frontier_bindings']==1024
