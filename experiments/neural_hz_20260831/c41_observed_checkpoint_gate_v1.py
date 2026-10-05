"""Same closed numeric ledger and transient gates with complete failure stats.

No storage schema/owner adapter is changed. Observe the before ledger before
walking after roots, so a rejected owner cannot consume both fingerprint walks.
"""
from dataclasses import asdict
import resource
import time
import tracemalloc
from types import SimpleNamespace
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import rss_bytes
from experiments.neural_hz_20260831.c5_live_roots_v2 import collect
from experiments.neural_hz_20260831.c40_half_gauge_transaction_v1 import _measurement_roots


def measured(build,*,observe,transient_cap=1024**3):
    if transient_cap!=1024**3 or tracemalloc.is_tracing():raise ValueError('unchanged exclusive1GiB tracing required')
    entry=rss_bytes();before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    started=time.monotonic();tracemalloc.start(1);returned=False
    try:
        result=build();returned=True
    finally:
        current,peak=tracemalloc.get_traced_memory();metadata=tracemalloc.get_tracemalloc_memory();tracemalloc.stop()
        hwm=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024;growth=max(0,hwm-entry)
        stats=dict(elapsed_s=time.monotonic()-started,entry_rss_bytes=entry,
            lifetime_hwm_before_bytes=before,lifetime_hwm_after_bytes=hwm,
            resident_growth_upper_bound_bytes=growth,traced_current_bytes=current,
            traced_peak_bytes=peak,tracer_metadata_bytes=metadata,transient_cap_bytes=transient_cap,
            measured_transient_gate=growth<=transient_cap and peak+metadata<=transient_cap,
            build_returned=returned)
        observe(stats)
    if not stats['measured_transient_gate']:raise MemoryError('unchanged construction transient cap exceeded')
    return result,stats


def checkpoint_closure(saved,candidate,*,pool,observe):
    pool.charge('half_checkpoint_closure_schema_metadata',1024+64*(len(saved)+len(candidate)))
    before=collect(SimpleNamespace(),dict(complete_saved_checkpoint=_measurement_roots(saved)))
    observe(dict(event='complete_before_checkpoint_roots_collected',numeric_roots=len(before.numeric),
        unique_objects=before.unique_objects,python_shallow_bytes=before.python_shallow_bytes,
        traced=tracemalloc.get_traced_memory()))
    bm=before.measure();bs=before.python_shallow_bytes;bn=len(before.numeric)
    observe(dict(event='complete_before_checkpoint_numeric_ledger_passed',measurement=asdict(bm)))
    del before
    after=collect(SimpleNamespace(),dict(complete_saved_checkpoint=_measurement_roots(candidate)))
    observe(dict(event='complete_after_checkpoint_roots_collected',numeric_roots=len(after.numeric),
        unique_objects=after.unique_objects,python_shallow_bytes=after.python_shallow_bytes,
        traced=tracemalloc.get_traced_memory()))
    am=after.measure()
    observe(dict(event='complete_after_checkpoint_numeric_ledger_passed',measurement=asdict(am)))
    return dict(scope='complete_serialized_C34_checkpoint_roots_only',
        before=asdict(bm),after=asdict(am),numeric_resident_byte_delta=am.resident_bytes-bm.resident_bytes,
        resident_entry_delta=am.resident_entries-bm.resident_entries,
        python_shallow_before=bs,python_shallow_after=after.python_shallow_bytes,
        numeric_root_count_before=bn,numeric_root_count_after=len(after.numeric),
        strict_checkpoint_numeric_decrease=am.resident_bytes<bm.resident_bytes and am.resident_entries<bm.resident_entries,
        original_caller_roots_missing_from_archive=True,whole_C34_LIVE_gate_proved=False,
        runtime_payment_proved=False,formal_gain=0)
