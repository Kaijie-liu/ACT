"""Frozen call order only; NOT a CLI or an already-existing trusted service.

The execution host/clock boundary must be separately admitted before use. A fresh
Python process importing this file does not satisfy that pre-existing-host rule.
No source generation, registry activation, retry, output write or job launch here
unless the externally approved observer is explicitly invoked by that host.
"""
from pathlib import Path
from scoped_source.endpoint_batch_observer import execute, observe_return

RUN_DIRECTORY='/data1/Kane/MOE/runs/hz_endpoint_midpoint_20261019_r1'
FREEZE_FILE='/data1/Kane/MOE/ACT/docs/hz_endpoint_midpoint_freeze_candidate_20261019.json'


def receive_once(expected, *, enabled=False):
    proposal=execute(Path(RUN_DIRECTORY),Path(FREEZE_FILE),expected,enabled=enabled)
    return observe_return(proposal)
