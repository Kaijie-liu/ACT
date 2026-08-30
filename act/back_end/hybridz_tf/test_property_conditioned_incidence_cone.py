#!/usr/bin/env python3
# ===- test_property_conditioned_incidence_cone.py --------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------====#
"""Independent CSR-incidence gates for the PC-CBDE first-stage selector.

Run from the repository root:

    python -m \
      act.back_end.hybridz_tf.test_property_conditioned_incidence_cone
"""

from __future__ import annotations

import gc
import os
import resource
import threading
import time
import unittest

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.gpu_dual_candidates import (
    property_conditioned_incidence_cone_rows,
)


def _causal_packet_fixture(
    *,
    wrong_reverse_copy: bool = False,
):
    """Build one true stable-column path plus disconnected tag decoys.

    Stable columns are ``[y, x, w, d1, e1, d2, e2]``.  The only admissible
    incidence path is

    ``packet(w) -- source(x,w) -- ADD(y,x) -- property(y)``.

    Two numerically closer ADD layer tags live in disconnected components.
    A generated micro-RLT row directly joins ``w`` and ``y`` but must never
    become an ordinary BFS shortcut.
    """

    y, x, w, d1, e1, d2, e2 = range(7)
    dense = np.zeros((9, 7), dtype=np.float64)

    dense[0, w] = 1.0
    dense[1, x] = 1.0
    dense[1, w] = -1.0

    dense[2, y] = 1.0
    dense[2, x] = -1.0
    dense[3, y] = -1.0
    if wrong_reverse_copy:
        dense[3, w] = 1.0
    else:
        dense[3, x] = 1.0

    dense[4, d1] = 1.0
    dense[4, e1] = -1.0
    dense[5] = -dense[4]
    dense[6, d2] = 1.0
    dense[6, e2] = -1.0
    dense[7] = -dense[6]

    # This generated row would create a false one-edge answer if the selector
    # were allowed to consume its own generated packet family.
    dense[8, y] = 1.0
    dense[8, w] = -1.0

    tags = (
        "property_micro_rlt:generated:packet_w",
        "source:packet_to_x",
        "add_materialize:40:forward",
        "add_materialize:40:reverse",
        "add_materialize:38:forward",
        "add_materialize:38:reverse",
        "add_materialize:39:forward",
        "add_materialize:39:reverse",
        "property_micro_rlt:generated:forbidden_shortcut",
    )
    return (
        sp.csr_matrix(dense),
        tags,
        np.ones(len(tags), dtype=np.bool_),
    )


def _large_add_chain_fixture(
    *,
    layers: int,
    width: int = 2048,
    wrong_copy_irrelevant: bool = False,
):
    """Build one or two wide ADD blocks with a one-coordinate causal chain."""

    if layers not in {1, 2} or width < 2:
        raise ValueError("large ADD chain requires one/two layers and width>=2")
    target = width // 3
    n_rows = layers * 2 * width + 1
    n_columns = (layers + 1) * width
    indptr = np.arange(
        0,
        2 * (n_rows - 1) + 2,
        2,
        dtype=np.int32,
    )
    indptr = np.concatenate(
        (
            indptr,
            np.asarray([2 * (n_rows - 1) + 1], dtype=np.int32),
        )
    )
    indices = np.empty(2 * (n_rows - 1) + 1, dtype=np.int32)
    data = np.empty(indices.size, dtype=np.float64)
    tags: list[str] = []
    expected_rows = []

    for layer in range(layers):
        row_start = layer * 2 * width
        previous_start = layer * width
        next_start = (layer + 1) * width
        for coordinate in range(width):
            row = row_start + coordinate
            offset = int(indptr[row])
            indices[offset:offset + 2] = (
                previous_start + coordinate,
                next_start + coordinate,
            )
            data[offset:offset + 2] = (-1.0, 1.0)
            tags.append(f"add_materialize:{100 + layer}:forward")
        for coordinate in range(width):
            row = row_start + width + coordinate
            offset = int(indptr[row])
            indices[offset:offset + 2] = (
                previous_start + coordinate,
                next_start + coordinate,
            )
            data[offset:offset + 2] = (1.0, -1.0)
            tags.append(f"add_materialize:{100 + layer}:reverse")
        expected_rows.extend(
            (
                row_start + target,
                row_start + width + target,
            )
        )

    if wrong_copy_irrelevant:
        decoy = (target + 1) % width
        wrong_row = width + decoy
        wrong_offset = int(indptr[wrong_row])
        data[wrong_offset] = 2.0

    packet_row = n_rows - 1
    indices[int(indptr[packet_row])] = target
    data[int(indptr[packet_row])] = 1.0
    tags.append("property_micro_rlt:generated:wide_packet")
    return (
        sp.csr_matrix(
            (data, indices, indptr),
            shape=(n_rows, n_columns),
        ),
        tuple(tags),
        np.ones(n_rows, dtype=np.bool_),
        packet_row,
        layers * width + target,
        np.asarray(sorted(expected_rows), dtype=np.int64),
    )


def _select(
    A: sp.csr_matrix,
    tags,
    allowed: np.ndarray,
    *,
    property_columns=(0,),
    packet_rows=(0,),
    **kwargs,
) -> np.ndarray:
    return property_conditioned_incidence_cone_rows(
        A,
        property_columns=property_columns,
        packet_rows=packet_rows,
        row_tags=tags,
        allowed_row_mask=allowed,
        **kwargs,
    )


def _production_shape_fixture():
    """Construct a 106,584 x 54,509 frame with exactly 9.3M CSR nnz.

    Its causal path traverses one 2048-forward + 2048-reverse ADD tag block,
    but only the target coordinate pair may be selected.  Dense sparse
    background exercises full-frame CSC construction while remaining
    disconnected from the causal frontier.
    """

    n_rows = 106_584
    n_columns = 54_509
    target_nnz = 9_300_000
    add_width = 2048
    forward_start = 2
    reverse_start = forward_start + add_width
    shortcut_row = reverse_start + add_width
    special_rows = shortcut_row + 1
    special_nnz = 1 + 2 + 4 * add_width + 2
    background_rows = n_rows - special_rows
    background_nnz = target_nnz - special_nnz
    background_width = background_nnz // background_rows
    wider_rows = background_nnz % background_rows

    counts = np.full(n_rows, background_width, dtype=np.int32)
    counts[:special_rows] = 2
    counts[0] = 1
    counts[special_rows:special_rows + wider_rows] += 1
    indptr = np.empty(n_rows + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, dtype=np.int64, out=indptr[1:])
    if int(indptr[-1]) != target_nnz:
        raise AssertionError("production fixture nnz accounting mismatch")

    indices = np.empty(target_nnz, dtype=np.int32)
    data = np.ones(target_nnz, dtype=np.float64)
    indices[int(indptr[0])] = 2
    data[int(indptr[0])] = 1.0
    indices[int(indptr[1]):int(indptr[2])] = (1, 2)
    data[int(indptr[1]):int(indptr[2])] = (1.0, -1.0)
    for coordinate in range(add_width):
        forward_row = forward_start + coordinate
        reverse_row = reverse_start + coordinate
        if coordinate == 0:
            columns = (0, 1)
            forward_values = (1.0, -1.0)
        else:
            columns = (
                3 + coordinate - 1,
                3 + (add_width - 1) + coordinate - 1,
            )
            forward_values = (-1.0, 1.0)
        forward_offset = int(indptr[forward_row])
        reverse_offset = int(indptr[reverse_row])
        indices[forward_offset:forward_offset + 2] = columns
        data[forward_offset:forward_offset + 2] = forward_values
        indices[reverse_offset:reverse_offset + 2] = columns
        data[reverse_offset:reverse_offset + 2] = (
            -forward_values[0],
            -forward_values[1],
        )
    shortcut_offset = int(indptr[shortcut_row])
    indices[shortcut_offset:shortcut_offset + 2] = (0, 2)
    data[shortcut_offset:shortcut_offset + 2] = (1.0, -1.0)

    background_start = int(indptr[special_rows])
    wide_width = background_width + 1
    wide_stop = background_start + wider_rows * wide_width
    if wider_rows:
        indices[background_start:wide_stop].reshape(
            wider_rows,
            wide_width,
        )[:] = np.arange(5000, 5000 + wide_width, dtype=np.int32)
    remaining_rows = background_rows - wider_rows
    if remaining_rows:
        indices[wide_stop:].reshape(
            remaining_rows,
            background_width,
        )[:] = np.arange(
            5000,
            5000 + background_width,
            dtype=np.int32,
        )

    tags = ["background:ordinary"] * n_rows
    tags[0] = "property_micro_rlt:generated:packet_w"
    tags[1] = "source:packet_to_x"
    tags[forward_start:reverse_start] = [
        "add_materialize:40:forward"
    ] * add_width
    tags[reverse_start:shortcut_row] = [
        "add_materialize:40:reverse"
    ] * add_width
    tags[shortcut_row] = (
        "property_micro_rlt:generated:forbidden_shortcut"
    )
    return (
        sp.csr_matrix(
            (data, indices, indptr),
            shape=(n_rows, n_columns),
        ),
        tuple(tags),
        np.ones(n_rows, dtype=np.bool_),
    )


def _resident_set_bytes() -> int:
    with open("/proc/self/statm", encoding="ascii") as status:
        resident_pages = int(status.readline().split()[1])
    return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))


class PropertyConditionedIncidenceConeTests(unittest.TestCase):
    def test_true_csr_path_excludes_packet_decoys_and_generated_shortcut(self):
        A, tags, allowed = _causal_packet_fixture()

        selected = _select(A, tags, allowed)

        np.testing.assert_array_equal(
            selected,
            np.asarray([1, 2, 3], dtype=np.int64),
        )
        selected_tags = {tags[int(row)] for row in selected}
        self.assertEqual(
            selected_tags,
            {
                "source:packet_to_x",
                "add_materialize:40:forward",
                "add_materialize:40:reverse",
            },
        )
        self.assertFalse(
            any(tag.startswith("property_micro_rlt:") for tag in selected_tags)
        )
        self.assertTrue(
            selected_tags.isdisjoint(
                {
                    "add_materialize:38:forward",
                    "add_materialize:38:reverse",
                    "add_materialize:39:forward",
                    "add_materialize:39:reverse",
                }
            )
        )

    def test_wrong_copy_fails_closed_before_incidence_search(self):
        A, tags, allowed = _causal_packet_fixture(
            wrong_reverse_copy=True,
        )

        selected = _select(A, tags, allowed)

        self.assertEqual(selected.dtype, np.dtype(np.int64))
        self.assertEqual(selected.size, 0)

    def test_row_permutation_preserves_selected_semantic_rows(self):
        A, tags, allowed = _causal_packet_fixture()
        baseline = _select(A, tags, allowed)
        permutation = np.asarray(
            [6, 2, 8, 0, 5, 1, 7, 3, 4],
            dtype=np.int64,
        )
        packet_row = int(np.flatnonzero(permutation == 0)[0])

        permuted = _select(
            A[permutation, :],
            tuple(tags[int(row)] for row in permutation),
            allowed[permutation],
            packet_rows=(packet_row,),
        )

        selected_original_rows = np.sort(permutation[permuted])
        np.testing.assert_array_equal(selected_original_rows, baseline)

    def test_deadline_and_hard_candidate_caps_fail_closed(self):
        A, tags, allowed = _causal_packet_fixture()
        cases = {
            "expired_deadline": {
                "deadline": time.monotonic() - 1.0,
            },
            "row_cap": {"max_rows": 2},
            "selected_nnz_cap": {"max_selected_nnz": 5},
            "depth_cap": {"max_depth": 1},
        }

        for name, kwargs in cases.items():
            with self.subTest(name=name):
                self.assertEqual(
                    _select(A, tags, allowed, **kwargs).size,
                    0,
                )

    def test_malformed_mask_and_partial_add_block_fail_closed(self):
        A, tags, allowed = _causal_packet_fixture()

        self.assertEqual(
            _select(A, tags, allowed.astype(np.int8)).size,
            0,
        )
        incomplete = allowed.copy()
        incomplete[3] = False
        self.assertEqual(
            _select(A, tags, incomplete).size,
            0,
        )

    def test_large_add_block_selects_only_causal_atomic_pair(self):
        (
            A,
            tags,
            allowed,
            packet_row,
            property_column,
            expected,
        ) = _large_add_chain_fixture(layers=1)

        selected = _select(
            A,
            tags,
            allowed,
            property_columns=(property_column,),
            packet_rows=(packet_row,),
        )

        np.testing.assert_array_equal(selected, expected)
        self.assertEqual(selected.size, 2)

        permutation = np.random.default_rng(20260729).permutation(
            A.shape[0]
        )
        permuted_packet = int(
            np.flatnonzero(permutation == packet_row)[0]
        )
        permuted = _select(
            A[permutation, :],
            tuple(tags[int(row)] for row in permutation),
            allowed[permutation],
            property_columns=(property_column,),
            packet_rows=(permuted_packet,),
        )
        np.testing.assert_array_equal(
            np.sort(permutation[permuted]),
            expected,
        )

    def test_two_large_add_blocks_select_four_atomic_rows(self):
        (
            A,
            tags,
            allowed,
            packet_row,
            property_column,
            expected,
        ) = _large_add_chain_fixture(layers=2)

        selected = _select(
            A,
            tags,
            allowed,
            property_columns=(property_column,),
            packet_rows=(packet_row,),
        )

        np.testing.assert_array_equal(selected, expected)
        self.assertEqual(selected.size, 4)

    def test_large_add_full_audit_catches_irrelevant_copy_and_partial(self):
        (
            wrong_A,
            wrong_tags,
            wrong_allowed,
            wrong_packet,
            property_column,
            _expected,
        ) = _large_add_chain_fixture(
            layers=1,
            wrong_copy_irrelevant=True,
        )
        self.assertEqual(
            _select(
                wrong_A,
                wrong_tags,
                wrong_allowed,
                property_columns=(property_column,),
                packet_rows=(wrong_packet,),
            ).size,
            0,
        )

        (
            A,
            tags,
            allowed,
            packet_row,
            property_column,
            _expected,
        ) = _large_add_chain_fixture(layers=1)
        allowed[1] = False
        self.assertEqual(
            _select(
                A,
                tags,
                allowed,
                property_columns=(property_column,),
                packet_rows=(packet_row,),
            ).size,
            0,
        )

    def test_production_shape_frontier_wall_and_rss_gate(self):
        A, tags, allowed = _production_shape_fixture()
        gc.collect()
        rss_before = _resident_set_bytes()
        maxrss_before = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ) * 1024
        sampled_peak = [rss_before]
        stop_sampling = threading.Event()

        def sample_rss() -> None:
            while not stop_sampling.wait(0.002):
                sampled_peak[0] = max(
                    sampled_peak[0],
                    _resident_set_bytes(),
                )

        sampler = threading.Thread(target=sample_rss, daemon=True)
        sampler.start()
        started = time.perf_counter()
        try:
            selected = _select(A, tags, allowed)
        finally:
            elapsed = time.perf_counter() - started
            stop_sampling.set()
            sampler.join()
        sampled_peak[0] = max(sampled_peak[0], _resident_set_bytes())
        maxrss_after = int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ) * 1024
        extra_rss = max(
            sampled_peak[0] - rss_before,
            maxrss_after - maxrss_before,
            0,
        )

        np.testing.assert_array_equal(
            selected,
            np.asarray([1, 2, 2050], dtype=np.int64),
        )
        print(
            "production incidence gate: "
            f"wall={elapsed:.6f}s "
            f"extra_rss={extra_rss / (1024 * 1024):.2f}MiB"
        )
        self.assertLessEqual(elapsed, 0.75)
        self.assertLessEqual(extra_rss, 128 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
