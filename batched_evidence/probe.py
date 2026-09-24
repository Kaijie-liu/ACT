"""Fixed synthetic publication probe with tracing explicitly separated."""
import argparse
from pathlib import Path
import resource
import time
import tracemalloc

from bounded_evidence import probe as previous
from bounded_evidence import stream as first_stream
from batched_evidence import stream
from scoped_proof.io import save


def measure(root, mode, size, traced, deadline):
    start = time.monotonic(); obj = previous.payload(size)
    generation = time.monotonic() - start
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if traced: tracemalloc.start()
    before = time.monotonic(); metrics = {}
    if mode == 'legacy': record = save(root / 'payload.json', obj)
    else:
        writer = first_stream if mode == 'stream' else stream
        record = writer.save(root / 'payload.json', obj, deadline=deadline, metrics=metrics)
    seconds = time.monotonic() - before
    peak = tracemalloc.get_traced_memory()[1] if traced else None
    if traced: tracemalloc.stop()
    save(root / 'probe.json', {'mode': mode, 'synthetic_size': size, 'traced': traced,
        'record': record, 'generation_seconds': generation, 'serialization_seconds': seconds,
        'worker_before_report_seconds': time.monotonic() - start,
        'serialization_traced_peak_bytes': peak, 'rss_highwater_before': rss,
        'rss_highwater_after': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'metrics': metrics, 'real_requests': 0, 'native_solver_calls': 0, 'complete_real_certificates': 0})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('kind', choices=('measure', 'hang', 'write_error'))
    p.add_argument('root', type=Path); p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--mode', choices=('legacy', 'stream', 'batch'), default='batch')
    p.add_argument('--size', type=int, choices=(2**18, 2**21), default=2**18)
    p.add_argument('--traced', action='store_true'); a = p.parse_args()
    if a.kind == 'measure': measure(a.root, a.mode, a.size, a.traced, a.deadline)
    else:
        previous.stream = stream
        previous.fault(a.root, a.kind, a.deadline)
