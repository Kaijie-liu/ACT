import json
from pathlib import Path
import pytest
from experiments.neural_hz_20260831.c33_collision_safe_events_v1 import record


@pytest.mark.parametrize('event',[{'event':'entry'}, {'event':'stage','elapsed_s':1.25},
    {'event':'reject','elapsed_s':None,'failure':{'message':'fixture'}}])
def test_distinct_worker_and_stage_clocks_no_drop_or_mutation(event):
    before=json.dumps(event,sort_keys=True)
    result=record(8.75,event)
    assert result['worker_elapsed_s']==8.75
    assert {k:v for k,v in result.items() if k!='worker_elapsed_s'}==event
    assert json.dumps(event,sort_keys=True)==before
    assert json.loads(json.dumps(result))==result


@pytest.mark.parametrize('clock,event',[(-1,{}),(float('nan'),{}),(0,{'worker_elapsed_s':1}),(0,object())])
def test_invalid_clock_or_new_reserved_collision_fail_closed(clock,event):
    with pytest.raises(ValueError):record(clock,event)


def test_only_worker_event_fix_import_and_exclusive_directory_change():
    exp=Path(__file__).resolve().parent
    old=(exp/'c32_live_splice_worker_v1.py').read_text()
    expected=old.replace("from experiments.neural_hz_20260831.c32_splice_binding_v1 import export",
        "from experiments.neural_hz_20260831.c32_splice_binding_v1 import export\nfrom experiments.neural_hz_20260831.c33_collision_safe_events_v1 import record as event_record")
    expected=expected.replace('c32_live_splice_20260911_v1','c33_live_splice_20260911_v1')
    expected=expected.replace('dict(elapsed_s=time.monotonic()-started,**event)','event_record(time.monotonic()-started,event)')
    assert (exp/'c33_live_splice_worker_v1.py').read_text()==expected
