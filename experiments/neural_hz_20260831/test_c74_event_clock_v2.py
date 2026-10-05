"""Preserve an existing stage duration while adding the worker clock."""
from experiments.neural_hz_20260831.c74_live_worker_v2 import event_record


def test_existing_event_duration_and_metadata_are_preserved():
    event={'event':'c5_stage','elapsed_s':1.25,'sequence_products':17}
    before=dict(event)
    assert event_record(event,5.0)==dict(event,worker_elapsed_s=5.0)
    assert event==before
