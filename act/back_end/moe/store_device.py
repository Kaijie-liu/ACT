"""Explicit request-owned device runtime; proposals are never certificates.

Admission, hard interruption and driver release belong to an outer supervisor.
This interface does not authorize physical CUDA execution by itself.
"""
import math
import os
import re
import time

import torch

from act.back_end.moe.batched_support_device import (
    ALLOCATOR_LIMIT, _initialize_cuda, _synchronize, _readback,
)
from scoped_source.hz_request_store import compact, decode, digest, identity, require

SCHEMA = 'STORE_DEVICE_CONTEXT_V1'


def capture(device, context, expected, *, store, request, contract, work):
    work.tick()
    if device == 'cpu':
        require(context is None and expected is None, 'CPU cannot carry device context')
        return None
    require(device == 'cuda:0', 'explicit cpu or isolated cuda:0 required')
    fields = {'schema', 'store', 'request', 'contract', 'deadline', 'invocation',
              'gpu_uuid', 'allocator_limit_bytes'}
    require(type(context) is dict and set(context) == fields, 'complete device context')
    raw = compact(context)
    require(len(raw) <= 4096 and type(expected) is str and digest(raw) == expected,
            'device context anchor')
    ctx = decode(raw)  # private capture; never use the caller's mutable alias
    require(ctx['schema'] == SCHEMA and ctx['store'] == store
            and ctx['request'] == request and ctx['contract'] == contract
            and all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v)
                    for v in (store, request, contract)), 'device source/property identity')
    require(type(ctx['deadline']) in (int, float) and math.isfinite(ctx['deadline'])
            and ctx['deadline'] == work.deadline, 'device absolute deadline')
    require(type(ctx['allocator_limit_bytes']) is int
            and ctx['allocator_limit_bytes'] == ALLOCATOR_LIMIT, 'device allocator policy')
    require(type(ctx['invocation']) is str and 0 < len(ctx['invocation']) <= 128,
            'device invocation identity')
    uuid = ctx['gpu_uuid']
    require(type(uuid) is str and re.fullmatch(
        r'GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}', uuid)
        and os.environ.get('CUDA_VISIBLE_DEVICES') == uuid, 'isolated GPU UUID')
    work.tick()
    return ctx


class Runtime:
    def __init__(self, device, context, expected, *, store, request, contract, work):
        self.context = capture(device, context, expected, store=store, request=request,
                               contract=contract, work=work)
        self.device, self.work = device, work
        self.context_sha256 = None if self.context is None else identity(self.context)
        self.costs = dict(initialization_sync_seconds=0., transfer_sync_seconds=0.,
                         optimization_sync_seconds=0., readback_sync_seconds=0.)
        self.transfers = 0
        start = time.monotonic()
        self.hardware = _initialize_cuda(self.context['gpu_uuid']) if self.context else None
        self.sync()
        self.costs['initialization_sync_seconds'] = time.monotonic()-start

    def sync(self):
        self.work.tick()
        _synchronize(self.device)
        self.work.tick()

    def move(self, values):
        self.sync(); start = time.monotonic()
        out = tuple(v.to(device=self.device, non_blocking=False) for v in values)
        self.sync()
        self.costs['transfer_sync_seconds'] += time.monotonic()-start
        self.transfers += 1
        return out

    def read(self, y, z):
        self.sync(); start = time.monotonic()
        ys, zs = _readback(y, z, self.device)
        self.work.tick()
        k = y.shape[1]
        require(len(ys) == k and len(zs) == k, 'complete readback columns')
        for rows, n in ((ys, y.shape[0]), (zs, z.shape[0])):
            require(all(len(row) == n for row in rows), 'complete readback dual dimensions')
            require(all(type(v) is float and math.isfinite(v) for row in rows for v in row),
                    'finite float64 readback')
        self.costs['readback_sync_seconds'] += time.monotonic()-start
        return ys, zs

    def report(self):
        self.sync()
        memory = (dict(peak_allocated=torch.cuda.max_memory_allocated(0),
                       peak_reserved=torch.cuda.max_memory_reserved(0))
                  if self.device == 'cuda:0' else None)
        self.work.tick()
        return dict(context_sha256=self.context_sha256, hardware=self.hardware, memory=memory,
                    completed_device_costs=dict(self.costs), transfer_groups=self.transfers,
                    cost_scope='nested in inclusive API/preparation/proposal costs; do not add twice')
