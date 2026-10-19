"""Frozen receive order only; NOT a CLI or a substitute for a launch authority.

The registered launcher must capture and retain the start anchor BEFORE any
request preparation or spawning this interpreter. Never synthesize it here from
now(), process birth, a file timestamp or an old run. This interface does not
create a launcher, register authority, or grant execution permission.
"""
from pathlib import Path
from scoped_source.endpoint_batch_observer import execute, observe_return

RUN_DIRECTORY='/data1/Kane/MOE/runs/hz_endpoint_midpoint_20261019_r2'
FREEZE_FILE='/data1/Kane/MOE/ACT/docs/hz_endpoint_midpoint_freeze_v4_20261019.json'


def receive_once(expected, *, start_anchor, start_sha256, enabled=False):
    proposal=execute(Path(RUN_DIRECTORY),Path(FREEZE_FILE),expected,
                     start_anchor=start_anchor,start_sha256=start_sha256,enabled=enabled)
    return observe_return(proposal,start_anchor=start_anchor,start_sha256=start_sha256)
