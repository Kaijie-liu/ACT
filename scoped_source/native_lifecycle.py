"""Default-off native bridge. No numerical imports until admitted setup."""
import struct
import time

from scoped_source.direct_candidates import Driver, PREFIX, RECORD, MAGIC, RATIONAL, MAX_TILE, MAX_TOTAL
from scoped_source.direct_lifecycle import prepare, revoke
from scoped_source.hz_request_store import require, sha
from scoped_source.native_method import validate


def submit_owned_duals(driver, token, dual, publish, *, native):
    """Narrow production-only ownership bridge, original exact consumer unchanged."""
    block = None
    try:
        require(type(driver) is Driver and driver.pending is not None and driver.pending[1] is token,
                'native direct owner tile')
        require(type(dual) is native.DualTile, 'closed best-dual producer')
        tasks = driver.pending[0]; dual.check(driver, tasks, token)
        start = time.monotonic()
        try:
            eq, ine = (len(driver.forest.rows(tasks[0].pair, kind)) for kind in ('eq', 'ine'))
            stride = RECORD.size + 2*RATIONAL + 2*(eq+ine)*8
            size = PREFIX.size + len(tasks)*stride
            require(size <= min(MAX_TILE, MAX_TOTAL-driver.total_bytes), 'native frame preallocation ceiling')
            block = driver.owner.allocate(size)
            block.put(0, PREFIX.pack(MAGIC, bytes.fromhex(driver.forest.request),
                bytes.fromhex(driver.forest.identity), bytes.fromhex(sha(driver.contract)),
                driver.ordinal, len(tasks), eq, ine, 0, 0))
            for col, task in enumerate(tasks):
                driver.work.visit('row_visits'); start_row = PREFIX.size + col*stride
                block.put(start_row, RECORD.pack(bytes.fromhex(task.slot), bytes.fromhex(task.query_id), 1, 3, bytes(6)))
                # Fresh zero-filled block contains both advisory zeros and the full zero role.
                for key, count, base in (('equality', eq, 0), ('inequality', ine, eq)):
                    for i in range(count):
                        driver.work.visit('coefficient_visits')
                        raw = struct.pack('<d', dual.scalar(key, i, col))
                        block.put(start_row+RECORD.size+2*RATIONAL+8*(base+i), raw)
                        # Explicit zero role is stored, not silently omitted or reused.
                        driver.work.visit('coefficient_visits')
                        block.put(start_row+RECORD.size+2*RATIONAL+8*(eq+ine+base+i), bytes(8))
            raw = block.seal(); dual.check(driver, tasks, token)
        finally:
            driver.costs['encode_seconds'] += time.monotonic()-start
        return driver._consume_raw(tasks, raw, publish)
    except BaseException:
        driver.abort(); raise
    finally:
        driver._release(block)


class Lifecycle:
    def __init__(self, config, *, code_root, expected_code, enabled=False, midpoint_control=None):
        require(enabled is True, 'native lifecycle opt-in')
        self.method = validate(config); self.proposer = None
        from scoped_source.native_method import MIDPOINT_SCHEMA
        require(midpoint_control in (None, 'midpoint_block', 'midpoint_error')
                and (midpoint_control is None or self.method['schema'] == MIDPOINT_SCHEMA),
                'closed opt-in midpoint lifecycle control')
        self.midpoint_control = midpoint_control
        self.lease = None; self.closed = False
        self.code_root, self.expected_code, self.native = code_root, expected_code, None

    def setup(self, forest, work):
        require(not self.closed and self.proposer is None, 'one native lifecycle setup')
        from scoped_source.native_loader import load
        self.native = load(self.code_root, self.expected_code, work)
        self.proposer = self.native.Proposer(self.method, enabled=True,
                                             midpoint_control=self.midpoint_control)
        self.proposer.setup(forest, work)

    def submit(self, driver, publish):
        from scoped_source.pair_workspace import PairWorkspace
        require(not self.closed and self.proposer is not None and self.lease is None
                and type(driver) is Driver and type(driver.workspace) is PairWorkspace,
                'bound native pair-scoped lifecycle')
        tasks, self.lease, token = prepare(driver)
        try:
            dual = self.proposer.propose(tasks, self.lease)
            revoke(self.lease); self.lease = None
            result = submit_owned_duals(driver, token, dual, publish, native=self.native)
            self.proposer.release()
            return result
        except BaseException:
            driver.abort(); raise
        finally:
            revoke(self.lease); self.lease = None

    def close(self):
        require(not self.closed, 'one native lifecycle close')
        self.closed = True; revoke(self.lease); self.lease = None
        if self.proposer is not None: self.proposer.close()
