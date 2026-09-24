"""Small synthetic memory/cutoff probes. No network, checkpoint or solver."""
import argparse
import os
from pathlib import Path
import resource
import time
import tracemalloc
from unittest.mock import patch

from bounded_evidence import stream
from scoped_proof.io import save


def payload(size):
    # Fixed rational CSR-shaped data AND a long scalar (iterencode alone would
    # still allocate the entire escaped scalar). Both sizes fixed before tests.
    n = size // 64
    return {'source': 'synthetic-only', 'matrix': {
        'shape': [n, n], 'data': ['123456789012345678901/987654321'] * n,
        'indices': list(range(n)), 'indptr': list(range(n + 1))},
        'large_string': ('x\n中\\😀\ud800' * ((size // 2) // 6))}


def measure(root, mode, size, deadline):
    start = time.monotonic()
    obj = payload(size)
    construction_seconds = time.monotonic() - start
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    tracemalloc.start()
    before = time.monotonic()
    stats = {}
    if mode == 'stream':
        record = stream.save(root / 'payload.json', obj, deadline=deadline, metrics=stats)
    else:
        record = save(root / 'payload.json', obj)
    elapsed = time.monotonic() - before
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end = time.monotonic()
    report = {'mode': mode, 'synthetic_size': size, 'record': record,
        'generation_seconds': construction_seconds, 'serialization_seconds': elapsed,
        'worker_before_report_seconds': end - start, 'serialization_traced_peak_bytes': peak,
        'rss_highwater_before': rss_before,
        'rss_highwater_after': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'metrics': stats, 'native_solver_calls': 0, 'complete_output_certificates': 0}
    save(root / 'probe.json', report)


def fault(root, name, deadline):
    path = root / 'construction.json'
    original = os.write
    called = False

    def write(fd, block):
        nonlocal called
        if called:
            if name == 'write_error':
                raise OSError('injected disk write error')
            # A genuinely uncooperative operation: only outer watchdog stops it.
            while True:
                time.sleep(.1)
        called = True
        return original(fd, block)

    with patch.object(stream.os, 'write', write):
        stream.save(path, payload(2**20), deadline=deadline)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('kind', choices=('measure', 'hang', 'write_error'))
    p.add_argument('root', type=Path)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--mode', choices=('legacy', 'stream'), default='stream')
    p.add_argument('--size', type=int, choices=(2**18, 2**21), default=2**18)
    args = p.parse_args()
    if args.kind == 'measure':
        measure(args.root, args.mode, args.size, args.deadline)
    else:
        fault(args.root, args.kind, args.deadline)
