"""Explicit standard-library measurement root for ONE approved finite CPU batch.

No project import, request/file/identity lookup or algorithm warm-up precedes t0.
The interpreter's earlier startup is unknown, not zero. This is a provisioned-host
measurement, not a claim of full cold-start cost. No additional supervisor.
"""
import time
PROVISION_ENTRY = time.monotonic()
import hashlib
import json
import os
import resource
import sys


def packed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')


def main():
    # Root provisioning only. Never raise a tighter inherited limit.
    for kind, cap in ((resource.RLIMIT_AS, 8*2**30), (resource.RLIMIT_FSIZE, 128*2**20)):
        soft, hard = resource.getrlimit(kind)
        if any(v != resource.RLIM_INFINITY and v < cap for v in (soft, hard)):
            raise ValueError('inherited limit tighter than frozen root; no increase')
        resource.setrlimit(kind, (cap, cap))
    started = time.monotonic()
    result = None
    try:
        if len(sys.argv) != 3 or sys.argv[1] != '--execute-approved-once':
            raise ValueError('explicit default-off finite execution required')
        expected = sys.argv[2]
        if len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('external frozen identity')
        repo = '/data1/Kane/MOE/ACT'
        sys.path.append(repo)
        from scripts.endpoint_midpoint_launcher import launch_from_retained_start, issuer_identity, read
        from pathlib import Path
        own = read(Path(repo)/'scripts/endpoint_midpoint_launcher.py', 131072, started+60)
        with open('/proc/sys/kernel/random/boot_id', 'rb') as stream:
            boot = stream.read(65)
        with open('/proc/self/stat', 'rb') as stream:
            stat = stream.read(8193)
        if len(boot)>64 or len(stat)>8192:
            raise ValueError('bounded root identity')
        ticks = int(stat[stat.rfind(b') ')+2:].split()[19])
        origin = dict(schema='ENDPOINT_RETAINED_ORIGIN_V1', started=started,
            clock='CLOCK_MONOTONIC', origin='AFTER_HOST_PROVISIONING_BEFORE_REQUEST',
            boot_id=boot.decode('ascii').strip(), time_namespace=os.readlink('/proc/self/ns/time'),
            host_pid=os.getpid(), host_start_ticks=ticks, issuer=issuer_identity(own), freeze=expected)
        result = launch_from_retained_start(expected, origin=origin,
            origin_sha256=hashlib.sha256(packed(origin)).hexdigest(), enabled=True)
    except BaseException as exc:
        result = dict(schema='ENDPOINT_MEASUREMENT_FAILURE_V1', status='INCOMPLETE_OR_FAILED',
            error=type(exc).__name__, message=str(exc)[:400], started=started,
            seconds=time.monotonic()-started, online_batch_acceptance=False)
    # Live memory receipt is the trusted event. Export/archival cannot promote a
    # missing/late receipt. Export overhead is measured separately, not hidden.
    received = time.monotonic()
    export_started=received  # Includes receipt copying, hashing and encoding below.
    if result.get('schema') == 'ENDPOINT_CALLER_RETURN_V2':
        result = dict(result, received_at=received, seconds=received-started,
            online_batch_acceptance=result['online_batch_acceptance'] is True and received<started+2100)
    envelope = dict(schema='ENDPOINT_PROVISIONED_HOST_RETURN_V1', caller=result,
        caller_sha256=hashlib.sha256(packed(result)).hexdigest(),
        provisioning=dict(observed_prefix_seconds=started-PROVISION_ENTRY,
            interpreter_startup_seconds=None, full_cold_start_seconds=None,
            scope='STANDARD_LIBRARY_ROOT_ONLY_NO_REQUEST_WORK', missing_is_not_zero=True),
        host=dict(pid=os.getpid(), python=sys.executable, flags=['-I','-S','-B']),
        request_seconds=received-started, live_receipt=True)
    raw=packed(envelope)
    if len(raw)>65536:
        raise ValueError('root export bounded by original receipt capacity')
    sys.stdout.buffer.write(raw+b'\n'); sys.stdout.buffer.flush()
    export_finished=time.monotonic()
    print(json.dumps(dict(schema='ENDPOINT_ROOT_EXPORT_COST_V1',
        envelope_sha256=hashlib.sha256(raw).hexdigest(),
        export_seconds=export_finished-export_started,
        request_through_export_seconds=export_finished-started,
        within_original_budget=export_finished<started+2100)), flush=True)


if __name__ == '__main__':
    main()
