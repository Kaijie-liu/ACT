"""Concurrency and overflow gates for the global HybridZ ID allocator."""

from __future__ import annotations

import subprocess
import sys
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

from act.back_end.solver.solver_hz import (
    hz_fresh_col_ids,
    hz_reserve_fresh_col_ids_above,
)


class HybridZColumnIdAllocatorTests(unittest.TestCase):
    def test_above_floor_is_exact_contiguous_and_burned(self):
        anchor = int(hz_fresh_col_ids(1, device="cpu")[0])
        floor = anchor + 1024
        reserved = hz_reserve_fresh_col_ids_above(
            4,
            lower_bound_exclusive=floor,
            device="cpu",
        )
        self.assertEqual(reserved.dtype, torch.int64)
        self.assertEqual(
            reserved.tolist(),
            list(range(floor + 1, floor + 5)),
        )

        # A caller may fail after reservation.  The interval is nevertheless
        # burned and cannot collide with the next ordinary allocation.
        following = hz_fresh_col_ids(3, device="cpu")
        self.assertEqual(
            following.tolist(),
            list(range(floor + 5, floor + 8)),
        )

    def test_mixed_concurrent_allocations_are_disjoint(self):
        worker_count = 32
        width = 7
        barrier = threading.Barrier(worker_count)
        lock = threading.Lock()
        results = []
        failures = []
        shared_floor = int(hz_fresh_col_ids(1, device="cpu")[0]) + 4096

        def worker(index: int) -> None:
            try:
                barrier.wait(timeout=10.0)
                if index % 2:
                    values = hz_reserve_fresh_col_ids_above(
                        width,
                        lower_bound_exclusive=shared_floor,
                        device="cpu",
                    )
                else:
                    values = hz_fresh_col_ids(width, device="cpu")
                with lock:
                    results.append(tuple(int(x) for x in values.tolist()))
            except BaseException as exc:  # pragma: no cover - diagnostic
                with lock:
                    failures.append(exc)

        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(worker_count)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15.0)
        self.assertFalse(failures)
        self.assertEqual(len(results), worker_count)
        self.assertTrue(
            all(
                values == tuple(range(values[0], values[0] + width))
                for values in results
            )
        )
        flattened = [value for values in results for value in values]
        self.assertEqual(len(flattened), len(set(flattened)))

    def test_overflow_and_malformed_floor_fail_without_allocation(self):
        maximum = int(np.iinfo(np.int64).max)
        with self.assertRaises(OverflowError):
            hz_reserve_fresh_col_ids_above(
                1,
                lower_bound_exclusive=maximum,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            hz_reserve_fresh_col_ids_above(
                1,
                lower_bound_exclusive=maximum + 1,
                device="cpu",
            )
        with self.assertRaises(TypeError):
            hz_reserve_fresh_col_ids_above(
                1,
                lower_bound_exclusive=True,
                device="cpu",
            )
        with self.assertRaises(ValueError):
            hz_reserve_fresh_col_ids_above(
                -1,
                lower_bound_exclusive=0,
                device="cpu",
            )

    def test_last_signed_int64_id_is_representable(self):
        script = """
import torch
from act.back_end.solver import solver_hz
maximum = int(torch.iinfo(torch.int64).max)
solver_hz._NEXT_COL_ID[0] = maximum
value = solver_hz.hz_fresh_col_ids(1, device='cpu')
assert value.tolist() == [maximum]
try:
    solver_hz.hz_fresh_col_ids(1, device='cpu')
except OverflowError:
    pass
else:
    raise AssertionError('exhausted allocator reused an int64 ID')
"""
        subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=20.0,
        )


if __name__ == "__main__":
    unittest.main()
