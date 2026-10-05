from types import SimpleNamespace
import pytest
from experiments.neural_hz_20260831 import c41_observed_checkpoint_gate_v1 as gate


@pytest.mark.parametrize('rss_bad,trace_bad',[(False,False),(True,False),(False,True),(True,True)])
def test_both_original_transient_caps_enforced_and_stats_retained(monkeypatch,rss_bad,trace_bad):
    cap=1024**3;entry=3000;hwm=entry+cap+1 if rss_bad else entry+100
    monkeypatch.setattr(gate,'rss_bytes',lambda:entry)
    monkeypatch.setattr(gate.resource,'getrusage',lambda _:SimpleNamespace(ru_maxrss=hwm/1024))
    monkeypatch.setattr(gate.tracemalloc,'is_tracing',lambda:False)
    monkeypatch.setattr(gate.tracemalloc,'start',lambda _:None)
    monkeypatch.setattr(gate.tracemalloc,'stop',lambda:None)
    monkeypatch.setattr(gate.tracemalloc,'get_traced_memory',lambda:(10,cap if trace_bad else 10))
    monkeypatch.setattr(gate.tracemalloc,'get_tracemalloc_memory',lambda:1)
    records=[]
    if rss_bad or trace_bad:
        with pytest.raises(MemoryError):gate.measured(lambda:7,observe=records.append)
    else:assert gate.measured(lambda:7,observe=records.append)[0]==7
    assert len(records)==1 and records[0]['build_returned']
    assert records[0]['measured_transient_gate']==(not rss_bad and not trace_bad)


def test_inner_failure_records_stats_and_stops_tracing():
    records=[]
    def fail():raise ValueError('unaccepted owner')
    with pytest.raises(ValueError,match='unaccepted owner'):gate.measured(fail,observe=records.append)
    assert records and not records[0]['build_returned'] and not gate.tracemalloc.is_tracing()


def test_larger_cap_rejected():
    with pytest.raises(ValueError):gate.measured(lambda:None,observe=lambda _:None,transient_cap=2*1024**3)


def test_reordered_complete_checkpoint_ledger_equals_C40(monkeypatch):
    import numpy as np
    from experiments.neural_hz_20260831.test_c32_live_splice_v1 import execute
    from experiments.neural_hz_20260831.c32_splice_binding_v1 import export
    from experiments.neural_hz_20260831.c32_live_splice_runtime_v1 import numeric_roots
    from experiments.neural_hz_20260831.c40_half_gauge_transaction_v1 import checkpoint_payload,checkpoint_closure
    from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
    _,runtime,hz,_=execute(monkeypatch);roots=numeric_roots(runtime);roots['layer']=dict(vars(roots['layer']))
    saved=dict(schema='c34_reconstructable_final_native_checkpoint_v1',spliced_state_fields=export(runtime['lifted']),
        runtime_numeric_roots=roots,final_hz=hz,hz_cache={78:hz},extra_numeric=np.arange(5))
    pool=WorkPool(256_000_000);candidate=checkpoint_payload(saved,hz,hz,hz,np.array([1,2],np.uint64),np.array([],np.uint64),pool=pool)
    expected=checkpoint_closure(saved,candidate,pool=pool);events=[]
    actual=gate.checkpoint_closure(saved,candidate,pool=pool,observe=events.append)
    assert actual==expected
    assert [v['event'] for v in events]==['complete_before_checkpoint_roots_collected',
        'complete_before_checkpoint_numeric_ledger_passed','complete_after_checkpoint_roots_collected',
        'complete_after_checkpoint_numeric_ledger_passed']
