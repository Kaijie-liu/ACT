"""Local construction and fail-closed return; no production state mutation."""

from pathlib import Path
import resource
import time
import tracemalloc

from experiments.neural_hz_20260831.c5_live_roots_v1 import collect


def rss_bytes():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    raise ValueError("VmRSS unavailable")


def measured_build(build, *, transient_cap=1024**3):
    if tracemalloc.is_tracing():
        raise ValueError("construction tracer already active")
    entry_rss = rss_bytes()
    hwm_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    started = time.monotonic()
    tracemalloc.start(1)
    try:
        result = build()
        current, peak = tracemalloc.get_traced_memory()
        metadata = tracemalloc.get_tracemalloc_memory()
    finally:
        tracemalloc.stop()
    hwm_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    upper_growth = max(0, hwm_after - entry_rss)
    stats = {"elapsed_s": time.monotonic() - started, "entry_rss_bytes": entry_rss,
             "lifetime_hwm_before_bytes": hwm_before, "lifetime_hwm_after_bytes": hwm_after,
             "resident_growth_upper_bound_bytes": upper_growth, "traced_current_bytes": current,
             "traced_peak_bytes": peak, "tracer_metadata_bytes": metadata,
             "transient_cap_bytes": transient_cap,
             "measured_transient_gate": upper_growth <= transient_cap and peak + metadata <= transient_cap}
    if not stats["measured_transient_gate"]:
        raise MemoryError(f"construction transient cap exceeded: {stats}")
    return result, stats


def transaction(tf, extra, build, *, after_build=None, measure=False):
    before = collect(tf, extra)
    if measure:
        result, stats = measured_build(build)
    else:
        result, stats = build(), None
    if after_build is not None:
        after_build(result)
    after = collect(tf, extra)
    if after.fingerprint != before.fingerprint:
        raise ValueError("incoming live roots changed; no result published")
    return result, {"input_roots_unchanged": True, "construction": stats,
                    "registered_numeric_roots": len(before.numeric), "schema_counts": before.schema_counts,
                    "python_shallow_bytes": before.python_shallow_bytes}
