#!/usr/bin/env python3
"""Soundness gates for the live Operator-HZ exact clique closure."""

from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction
import hashlib
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import (
    operator_exact_relu_phase_cliques as clique_module,
)
from act.back_end.hybridz_tf import operator_hz as operator_hz_module
from act.back_end.hybridz_tf import (
    property_phase_conflict_clique as legacy_clique_module,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_cliques import (
    OperatorExactReLUPhaseCliqueError,
    RankedOperatorPhase,
    consume_verified_operator_phase_clique_snapshot,
    run_operator_exact_relu_phase_cliques_candidate,
    verify_and_issue_operator_phase_clique_snapshot,
    verify_operator_exact_relu_phase_cliques_result,
)
from act.back_end.hybridz_tf.operator_exact_relu_phase_literals import (
    derive_operator_exact_relu_property_phase_literals,
)
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuild,
    build_operator_hz,
)
from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
    PersistentPairRecord,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
)
from act.back_end.hybridz_tf.raw_vnnlib_rival_adapter import (
    consume_raw_vnnlib_top1_candidate,
    issue_raw_vnnlib_top1_candidate,
    validate_consumed_raw_vnnlib_rival_batch,
)
from act.back_end.hybridz_tf.test_operator_exact_relu_phase_literals import (
    _DTYPE,
    _canonical_csr,
    _clone_hz,
    _dense,
    _exact_k4_margin_upper,
    _k4_corner_build,
    _layer,
    _relaxed_margin_upper,
    _rivals,
)
from act.back_end.solver.solver_hz import SparseHZono


def _corner_build(
    *,
    bias: float,
    issue_constructive_nonempty_seal: Any = False,
) -> OperatorHZBuild:
    """Four corner ReLUs with controllable pair compatibility."""

    lower = torch.tensor([[-1.0, -1.0]], dtype=_DTYPE)
    upper = torch.tensor([[1.0, 1.0]], dtype=_DTYPE)
    layers = [
        _layer(
            0,
            "INPUT",
            {"shape": (1, 2)},
            width=2,
        ),
        _layer(
            1,
            "INPUT_SPEC",
            {"kind": "BOX", "lb": lower, "ub": upper},
            width=2,
        ),
        _dense(
            2,
            (
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ),
            (bias, bias, bias, bias),
        ),
        _layer(3, "RELU", width=4),
        _dense(
            4,
            (
                (0.0, 0.0, 0.0, 0.0),
                (1.0, 1.0, 1.0, 1.0),
                (0.5, 0.5, 0.5, 0.5),
            ),
            (0.75, 0.0, 0.0),
        ),
        _layer(5, "ASSERT", width=3),
    ]
    predecessors = {
        0: [],
        1: [0],
        2: [1],
        3: [2],
        4: [3],
        5: [4],
    }
    successors = {layer.id: [] for layer in layers}
    for child, parents in predecessors.items():
        for parent in parents:
            successors[parent].append(child)
    network = SimpleNamespace(
        layers=layers,
        preds=predecessors,
        succs=successors,
        by_id={layer.id: layer for layer in layers},
    )
    facts = {}
    for layer in layers:
        width = len(layer.out_vars)
        if layer.kind in {"INPUT", "INPUT_SPEC"}:
            fact_lower = lower.clone()
            fact_upper = upper.clone()
        else:
            fact_lower = torch.full(
                (1, width), -1.0e30, dtype=_DTYPE
            )
            fact_upper = torch.full(
                (1, width), 1.0e30, dtype=_DTYPE
            )
        facts[layer.id] = Fact(
            Bounds(fact_lower, fact_upper), ConSet()
        )
    return build_operator_hz(
        network,
        facts,
        facts,
        exact_budget=4,
        materialize_add=True,
        issue_constructive_nonempty_seal=(
            issue_constructive_nonempty_seal
        ),
    )


def _one_zero_effect_mapping(
    build: OperatorHZBuild,
) -> OperatorHZBuild:
    """Make one mapped output cancel in both rivals but remain nonzero."""

    baseline = (
        derive_operator_exact_relu_property_phase_literals(
            build, _rivals()
        )
    )
    position = baseline.mappings[0].output_continuous_position
    generator = build.hz.Gc.tolil(copy=True)
    generator[:, position] = np.asarray(
        [[0.25], [0.25], [0.25]], dtype=np.float64
    )
    hz = _clone_hz(
        build.hz, Gc=_canonical_csr(generator)
    )
    return replace(build, hz=hz)


def _raw_top1_source() -> str:
    return """
    (set-logic QF_LRA)
    (declare-const X_0 Real)
    (declare-const X_1 Real)
    (declare-const Y_0 Real)
    (declare-const Y_1 Real)
    (declare-const Y_2 Real)
    (assert (>= X_0 -1))
    (assert (<= X_0 1))
    (assert (>= X_1 -1))
    (assert (<= X_1 1))
    (assert (or (<= Y_0 Y_1) (<= Y_0 Y_2)))
    """


def _run(
    build: OperatorHZBuild,
    *,
    rivals=None,
    **kwargs: Any,
):
    ordered_rivals = _rivals() if rivals is None else rivals
    selection = (
        derive_operator_exact_relu_property_phase_literals(
            build, ordered_rivals
        )
    )
    result = run_operator_exact_relu_phase_cliques_candidate(
        build,
        ordered_rivals,
        selection,
        deadline=time.monotonic() + 20.0,
        **kwargs,
    )
    return ordered_rivals, selection, result


def _large_cut_copy_hz(
    nnz: int,
    *,
    n_bin: int = 4,
) -> SparseHZono:
    """Controlled canonical CSR core for allocation-slope tests only."""

    columns = 1000
    if type(nnz) is not int or nnz < columns or nnz % columns:
        raise ValueError("nnz must be a positive multiple of 1000")
    rows = nnz // columns
    data = np.ones(nnz, dtype=np.float64)
    indices = np.tile(
        np.arange(columns, dtype=np.int32), rows
    )
    indptr = np.arange(
        0,
        nnz + 1,
        columns,
        dtype=np.int32,
    )
    return SparseHZono(
        c=np.zeros(1, dtype=np.float64),
        Gc=sp.csr_matrix((1, columns), dtype=np.float64),
        Gb=sp.csr_matrix((1, n_bin), dtype=np.float64),
        Ac=sp.csr_matrix((0, columns), dtype=np.float64),
        Ab=sp.csr_matrix((0, n_bin), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix(
            (data, indices, indptr),
            shape=(rows, columns),
            dtype=np.float64,
            copy=False,
        ),
        Aub=sp.csr_matrix((rows, n_bin), dtype=np.float64),
        ub=np.full(rows, float(columns), dtype=np.float64),
        col_ids=np.arange(columns, dtype=np.int64),
        bcol_ids=np.arange(10, 10 + n_bin, dtype=np.int64),
    )


class OperatorExactReLUPhaseCliqueTests(unittest.TestCase):
    def test_owned_clique_cut_is_legacy_bit_exact_for_one_and_many(
        self,
    ) -> None:
        source = _k4_corner_build().hz
        # A finite subnormal exercises exact preservation of coefficients
        # that must never be silently treated as zero/dust.
        self.assertGreater(source.Auc.data.size, 0)
        source.Auc.data[0] = np.nextafter(0.0, 1.0)
        stable_ids = tuple(int(value) for value in source.bcol_ids)
        rows = (
            (
                PhaseLiteral(stable_ids[1], -1, "1" * 64),
                PhaseLiteral(stable_ids[0], 1, "2" * 64),
            ),
            (
                PhaseLiteral(stable_ids[3], 1, "3" * 64),
                PhaseLiteral(stable_ids[2], -1, "4" * 64),
            ),
        )
        caps = clique_module.OperatorPhaseCliqueCaps(
            **clique_module._HARD_LIMITS
        )

        legacy_one = legacy_clique_module._copy_parent_with_clique_cut(
            source, rows[0]
        )
        legacy_many = legacy_clique_module._copy_parent_with_clique_cut(
            legacy_one, rows[1]
        )
        with patch.object(
            SparseHZono,
            "__post_init__",
            side_effect=AssertionError(
                "owned cut invoked redundant SparseHZono normalization"
            ),
        ), patch.object(
            clique_module.sp,
            "vstack",
            side_effect=AssertionError("owned cut invoked scipy vstack"),
        ), patch.object(
            clique_module.sp,
            "hstack",
            side_effect=AssertionError("owned cut invoked scipy hstack"),
        ):
            owned_one = clique_module._copy_parent_with_clique_cut(
                source,
                rows[0],
                caps=caps,
                deadline=time.monotonic() + 10.0,
            )
            owned_many = clique_module._copy_parent_with_clique_cuts(
                source,
                rows,
                caps=caps,
                deadline=time.monotonic() + 10.0,
            )

        self.assertEqual(
            clique_module.sparse_hz_semantic_digest(owned_one),
            clique_module.sparse_hz_semantic_digest(legacy_one),
        )
        self.assertEqual(
            clique_module.sparse_hz_semantic_digest(owned_many),
            clique_module.sparse_hz_semantic_digest(legacy_many),
        )
        self.assertEqual(owned_one.n_ub, source.n_ub + 1)
        self.assertEqual(owned_many.n_ub, source.n_ub + 2)
        for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
            self.assertFalse(
                np.shares_memory(
                    getattr(source, name), getattr(owned_many, name)
                )
            )
        for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
            for buffer_name in ("data", "indices", "indptr"):
                self.assertFalse(
                    np.shares_memory(
                        getattr(getattr(source, name), buffer_name),
                        getattr(getattr(owned_many, name), buffer_name),
                    )
                )

    def test_owned_clique_cut_copies_each_large_core_buffer_once(
        self,
    ) -> None:
        caps = clique_module.OperatorPhaseCliqueCaps(
            **clique_module._HARD_LIMITS
        )
        literals = (
            PhaseLiteral(11, -1, "a" * 64),
            PhaseLiteral(10, 1, "b" * 64),
        )
        real_allocate = clique_module._allocate_owned_exact_array

        for nnz in (250_000, 500_000, 1_000_000):
            with self.subTest(nnz=nnz):
                source = _large_cut_copy_hz(nnz)
                source_digest = (
                    clique_module.sparse_hz_semantic_digest(source)
                )
                allocations: dict[str, np.ndarray] = {}

                def counted_allocate(shape, *, dtype, stage):
                    self.assertNotIn(stage, allocations)
                    result = real_allocate(
                        shape,
                        dtype=dtype,
                        stage=stage,
                    )
                    allocations[stage] = result
                    return result

                with patch.object(
                    SparseHZono,
                    "__post_init__",
                    side_effect=AssertionError(
                        "redundant SparseHZono normalization"
                    ),
                ), patch.object(
                    clique_module.sp,
                    "vstack",
                    side_effect=AssertionError("unexpected vstack"),
                ), patch.object(
                    clique_module.sp,
                    "hstack",
                    side_effect=AssertionError("unexpected hstack"),
                ), patch.object(
                    clique_module,
                    "_allocate_owned_exact_array",
                    side_effect=counted_allocate,
                ):
                    result = clique_module._copy_parent_with_clique_cut(
                        source,
                        literals,
                        caps=caps,
                        deadline=time.monotonic() + 20.0,
                    )

                # Five dense and six times three CSR buffers.  Every output
                # core buffer has exactly one named allocation; no temporary
                # full-size CSR allocation is possible through this path.
                self.assertEqual(len(allocations), 23)
                result_layout = clique_module._exact_hz_core_layout(
                    result
                )
                expected_core_bytes = sum(
                    int(value.nbytes)
                    for _name, value in result_layout.dense
                ) + sum(
                    int(buffer.nbytes)
                    for (
                        _name,
                        _matrix,
                        _shape,
                        data,
                        indices,
                        indptr,
                    ) in result_layout.sparse
                    for buffer in (data, indices, indptr)
                )
                self.assertEqual(
                    sum(
                        int(value.nbytes)
                        for value in allocations.values()
                    ),
                    expected_core_bytes,
                )
                final_buffers = [
                    value for _name, value in result_layout.dense
                ] + [
                    buffer
                    for (
                        _name,
                        _matrix,
                        _shape,
                        data,
                        indices,
                        indptr,
                    ) in result_layout.sparse
                    for buffer in (data, indices, indptr)
                ]
                for buffer in final_buffers:
                    if int(buffer.nbytes):
                        self.assertTrue(
                            any(
                                np.shares_memory(buffer, allocation)
                                for allocation in allocations.values()
                            )
                        )
                self.assertEqual(
                    clique_module.sparse_hz_semantic_digest(source),
                    source_digest,
                )
                self.assertEqual(result.n_ub, source.n_ub + 1)
                self.assertEqual(
                    int(result.Aub.indptr[-1]),
                    int(source.Aub.indptr[-1]) + len(literals),
                )

    def test_owned_clique_cut_caps_deadline_and_malformed_fail_closed(
        self,
    ) -> None:
        source = _k4_corner_build().hz
        ids = tuple(int(value) for value in source.bcol_ids)
        row = (
            PhaseLiteral(ids[0], 1, "5" * 64),
            PhaseLiteral(ids[1], -1, "6" * 64),
        )
        second_row = (
            PhaseLiteral(ids[2], 1, "7" * 64),
            PhaseLiteral(ids[3], -1, "8" * 64),
        )
        caps = clique_module.OperatorPhaseCliqueCaps(
            **clique_module._HARD_LIMITS
        )
        layout = clique_module._exact_hz_core_layout(source)
        metadata_items = (
            clique_module._conditional_metadata_buffer_items(
                source,
                maximum=(
                    caps.max_parent_buffer_items
                    - layout.buffer_items
                ),
                deadline=time.monotonic() + 10.0,
            )
        )
        parent_rows = layout.n_out + layout.n_eq + layout.n_ub
        parent_buffer_items = layout.buffer_items + metadata_items
        cap_cases = (
            replace(caps, max_parent_rows=parent_rows),
            replace(
                caps,
                max_parent_nonzeros=layout.nonzeros,
            ),
            replace(
                caps,
                max_parent_buffer_items=parent_buffer_items,
            ),
            # One row fits exactly; two rows must still be rejected against
            # the actual final shape before any owned allocation.
            replace(caps, max_parent_rows=parent_rows + 1),
        )
        cut_groups = ((row,), (row,), (row,), (row, second_row))
        for limited_caps, groups in zip(cap_cases, cut_groups):
            with self.subTest(caps=limited_caps, rows=len(groups)), patch.object(
                clique_module,
                "_allocate_owned_exact_array",
                side_effect=AssertionError(
                    "output cap was checked after allocation"
                ),
            ):
                with self.assertRaises(
                    OperatorExactReLUPhaseCliqueError
                ):
                    clique_module._copy_parent_with_clique_cuts(
                        source,
                        groups,
                        caps=limited_caps,
                        deadline=time.monotonic() + 10.0,
                    )

        malformed_groups = (
            (row[:1],),
            (
                (
                    row[0],
                    replace(row[1], phase=0),
                ),
            ),
            ((row[0], row[0]),),
            (row, row),
        )
        for groups in malformed_groups:
            with self.subTest(groups=groups), patch.object(
                clique_module,
                "_allocate_owned_exact_array",
                side_effect=AssertionError(
                    "malformed cut reached allocation"
                ),
            ):
                with self.assertRaises(
                    OperatorExactReLUPhaseCliqueError
                ):
                    clique_module._copy_parent_with_clique_cuts(
                        source,
                        groups,
                        caps=caps,
                        deadline=time.monotonic() + 10.0,
                    )

        malformed_parents = []
        nonfinite = _clone_hz(source)
        nonfinite.Auc.data[0] = np.nan
        malformed_parents.append(nonfinite)
        duplicate_ids = _clone_hz(source)
        duplicate_ids.bcol_ids[1] = duplicate_ids.bcol_ids[0]
        malformed_parents.append(duplicate_ids)
        noncanonical = _large_cut_copy_hz(1000)
        noncanonical.Auc.indices[1] = noncanonical.Auc.indices[0]
        malformed_parents.append(noncanonical)
        zero_binary = _large_cut_copy_hz(1000, n_bin=0)
        malformed_parents.append(zero_binary)
        for parent in malformed_parents:
            parent_row = (
                PhaseLiteral(
                    int(parent.bcol_ids[0])
                    if parent.bcol_ids.size
                    else 10,
                    1,
                    "9" * 64,
                ),
                PhaseLiteral(
                    int(parent.bcol_ids[-1])
                    if parent.bcol_ids.size
                    else 11,
                    -1,
                    "a" * 64,
                ),
            )
            with self.subTest(parent=parent), patch.object(
                clique_module,
                "_allocate_owned_exact_array",
                side_effect=AssertionError(
                    "malformed parent reached allocation"
                ),
            ):
                with self.assertRaises(
                    (OperatorExactReLUPhaseCliqueError, RuntimeError)
                ):
                    clique_module._copy_parent_with_clique_cut(
                        parent,
                        parent_row,
                        caps=caps,
                        deadline=time.monotonic() + 10.0,
                    )

        with patch.object(
            clique_module,
            "_allocate_owned_exact_array",
            side_effect=AssertionError("expired deadline allocated"),
        ):
            with self.assertRaises(OperatorExactReLUPhaseCliqueError):
                clique_module._copy_parent_with_clique_cut(
                    source,
                    row,
                    caps=caps,
                    deadline=time.monotonic() - 1.0,
                )

    def test_private_hz_snapshot_is_single_copy_owned_and_semantic_exact(
        self,
    ) -> None:
        build = _k4_corner_build()
        source = build.hz
        source_digest = clique_module.sparse_hz_semantic_digest(source)
        caps = clique_module.OperatorPhaseCliqueCaps(
            **clique_module._HARD_LIMITS
        )
        # The private assembly must not invoke SparseHZono.__post_init__,
        # whose same-dtype CSR normalization would make a second full copy.
        with patch.object(
            SparseHZono,
            "__post_init__",
            side_effect=AssertionError("redundant SparseHZono copy"),
        ):
            snapshot = clique_module._snapshot_sparse_hz(
                source,
                caps=caps,
                deadline=time.monotonic() + 10.0,
                stage="single_copy_toy",
            )
        self.assertEqual(
            clique_module.sparse_hz_semantic_digest(snapshot),
            source_digest,
        )
        for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
            self.assertFalse(
                np.shares_memory(
                    getattr(source, name), getattr(snapshot, name)
                )
            )
        for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
            original = getattr(source, name)
            copied = getattr(snapshot, name)
            for buffer_name in ("data", "indices", "indptr"):
                self.assertFalse(
                    np.shares_memory(
                        getattr(original, buffer_name),
                        getattr(copied, buffer_name),
                    )
                )

        snapshot_digest = clique_module.sparse_hz_semantic_digest(
            snapshot
        )
        source.Auc.data[0] = np.nextafter(
            source.Auc.data[0], np.inf
        )
        self.assertEqual(
            clique_module.sparse_hz_semantic_digest(snapshot),
            snapshot_digest,
        )
        self.assertNotEqual(
            clique_module.sparse_hz_semantic_digest(source),
            snapshot_digest,
        )

    def test_private_hz_snapshot_copies_each_large_core_buffer_once(
        self,
    ) -> None:
        caps = clique_module.OperatorPhaseCliqueCaps(
            **clique_module._HARD_LIMITS
        )
        real_copy = clique_module._copy_exact_array_with_deadline

        # These controlled CSR sizes cover the relevant allocation slope
        # without invoking a production network.  Allocation accounting is
        # deterministic and more stable than an RSS delta: every source core
        # buffer must enter the deadline-aware copier exactly once, and the
        # final CSR must directly own/share that returned allocation.
        for nnz in (250_000, 500_000, 1_000_000):
            with self.subTest(nnz=nnz):
                columns = 1000
                rows = nnz // columns
                data = np.ones(nnz, dtype=np.float64)
                indices = np.tile(
                    np.arange(columns, dtype=np.int32), rows
                )
                indptr = np.arange(
                    0,
                    nnz + 1,
                    columns,
                    dtype=np.int32,
                )
                source = SparseHZono(
                    c=np.zeros(1, dtype=np.float64),
                    Gc=sp.csr_matrix(
                        (1, columns), dtype=np.float64
                    ),
                    Gb=sp.csr_matrix((1, 0), dtype=np.float64),
                    Ac=sp.csr_matrix(
                        (0, columns), dtype=np.float64
                    ),
                    Ab=sp.csr_matrix((0, 0), dtype=np.float64),
                    b=np.zeros(0, dtype=np.float64),
                    Auc=sp.csr_matrix(
                        (data, indices, indptr),
                        shape=(rows, columns),
                        dtype=np.float64,
                        copy=False,
                    ),
                    Aub=sp.csr_matrix(
                        (rows, 0), dtype=np.float64
                    ),
                    ub=np.full(
                        rows, float(columns), dtype=np.float64
                    ),
                    col_ids=np.arange(
                        columns, dtype=np.int64
                    ),
                    bcol_ids=np.zeros(0, dtype=np.int64),
                )
                legacy_digest = (
                    clique_module.sparse_hz_semantic_digest(source)
                )
                layout = clique_module._exact_hz_core_layout(source)
                expected_bytes = sum(
                    int(value.nbytes) for _name, value in layout.dense
                ) + sum(
                    int(buffer.nbytes)
                    for (
                        _name,
                        _matrix,
                        _shape,
                        data_buffer,
                        index_buffer,
                        indptr_buffer,
                    ) in layout.sparse
                    for buffer in (
                        data_buffer,
                        index_buffer,
                        indptr_buffer,
                    )
                )
                copied: dict[str, np.ndarray] = {}

                def counted_copy(value, *, deadline, stage):
                    self.assertNotIn(stage, copied)
                    result = real_copy(
                        value,
                        deadline=deadline,
                        stage=stage,
                    )
                    copied[stage] = result
                    return result

                prefix = f"single_copy_{nnz}"
                with patch.object(
                    SparseHZono,
                    "__post_init__",
                    side_effect=AssertionError(
                        "redundant SparseHZono normalization"
                    ),
                ), patch.object(
                    clique_module,
                    "_copy_exact_array_with_deadline",
                    side_effect=counted_copy,
                ):
                    snapshot = clique_module._snapshot_sparse_hz(
                        source,
                        caps=caps,
                        deadline=time.monotonic() + 10.0,
                        stage=prefix,
                    )

                self.assertEqual(len(copied), 23)
                self.assertEqual(
                    sum(int(value.nbytes) for value in copied.values()),
                    expected_bytes,
                )
                self.assertEqual(
                    clique_module.sparse_hz_semantic_digest(snapshot),
                    legacy_digest,
                )
                for name in (
                    "c",
                    "b",
                    "ub",
                    "col_ids",
                    "bcol_ids",
                ):
                    self.assertIs(
                        getattr(snapshot, name),
                        copied[f"{prefix}_{name}_copy"],
                    )
                for name in (
                    "Gc",
                    "Gb",
                    "Ac",
                    "Ab",
                    "Auc",
                    "Aub",
                ):
                    matrix = getattr(snapshot, name)
                    for buffer_name in (
                        "data",
                        "indices",
                        "indptr",
                    ):
                        owned = copied[
                            f"{prefix}_{name}_{buffer_name}_copy"
                        ]
                        # SciPy may replace a zero-length data/index view
                        # with another zero-length singleton; it cannot add
                        # a full-core allocation because that buffer is 0 B.
                        if owned.size:
                            self.assertTrue(
                                np.shares_memory(
                                    getattr(matrix, buffer_name),
                                    owned,
                                )
                            )

    def test_private_hz_snapshot_reaudits_malformed_core_and_deadline(
        self,
    ) -> None:
        caps = clique_module.OperatorPhaseCliqueCaps(
            **clique_module._HARD_LIMITS
        )
        mutations = (
            (
                "stale_cached_duplicate_csr",
                lambda hz: hz.Gc.indices.__setitem__(
                    2, hz.Gc.indices[1]
                ),
                "semantic_Gc_malformed",
            ),
            (
                "nonfinite_sparse_value",
                lambda hz: hz.Auc.data.__setitem__(0, np.nan),
                "semantic_Auc_malformed",
            ),
            (
                "duplicate_stable_id",
                lambda hz: hz.col_ids.__setitem__(
                    1, hz.col_ids[0]
                ),
                "semantic_col_ids_malformed",
            ),
        )
        for name, mutate, reason in mutations:
            with self.subTest(name=name):
                source = _clone_hz(_k4_corner_build().hz)
                # Prime SciPy's cached flags before direct-buffer tampering;
                # the private semantic audit must not trust these flags.
                self.assertTrue(source.Gc.has_canonical_format)
                mutate(source)
                with self.assertRaisesRegex(RuntimeError, reason):
                    clique_module._snapshot_sparse_hz(
                        source,
                        caps=caps,
                        deadline=time.monotonic() + 10.0,
                        stage=f"malformed_{name}",
                    )

        source = _k4_corner_build().hz
        with patch.object(
            clique_module,
            "_copy_exact_array_with_deadline",
            side_effect=AssertionError(
                "expired snapshot reached allocation"
            ),
        ):
            with self.assertRaisesRegex(
                OperatorExactReLUPhaseCliqueError,
                "deadline_expired",
            ):
                clique_module._snapshot_sparse_hz(
                    source,
                    caps=caps,
                    deadline=time.monotonic() - 1.0,
                    stage="already_expired",
                )

        real_copy = clique_module._copy_exact_array_with_deadline
        delayed_once = [False]

        def expire_before_first_allocation(
            value, *, deadline, stage
        ):
            if not delayed_once[0]:
                delayed_once[0] = True
                time.sleep(0.02)
            return real_copy(
                value, deadline=deadline, stage=stage
            )

        with patch.object(
            clique_module,
            "_copy_exact_array_with_deadline",
            side_effect=expire_before_first_allocation,
        ):
            with self.assertRaisesRegex(
                OperatorExactReLUPhaseCliqueError,
                "deadline_expired",
            ):
                clique_module._snapshot_sparse_hz(
                    source,
                    caps=caps,
                    deadline=time.monotonic() + 0.01,
                    stage="expires_before_copy",
                )
        self.assertTrue(delayed_once[0])

    def test_candidate_rederives_selection_once_and_binds_digest(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals = _rivals()
        selection = (
            derive_operator_exact_relu_property_phase_literals(
                build, rivals
            )
        )
        trusted_derive = (
            clique_module.derive_operator_exact_relu_property_phase_literals
        )
        with patch.object(
            clique_module,
            "derive_operator_exact_relu_property_phase_literals",
            wraps=trusted_derive,
        ) as derive_spy:
            result = run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
            )
        self.assertEqual(derive_spy.call_count, 1)
        self.assertEqual(len(result.certificates), 6)
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "operator_phase_selection_digest_mismatch",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                replace(selection, selection_digest="0" * 64),
                deadline=time.monotonic() + 20.0,
            )

    def test_candidate_progress_is_exact_terminal_or_partial_only(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals = _rivals()
        selection = derive_operator_exact_relu_property_phase_literals(
            build, rivals
        )
        progress: dict[str, object] = {}
        compact = run_operator_exact_relu_phase_cliques_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 20.0,
            emit_cut_hz=False,
            diagnostic_progress=progress,
        )
        self.assertEqual(
            progress["schema"],
            "act.operator_exact_relu_phase_clique_progress.v1",
        )
        self.assertEqual(progress["status"], "complete")
        self.assertIs(progress["terminal_complete"], True)
        self.assertIs(progress["candidate_cut_hz_emitted"], False)
        self.assertIs(progress["partial_never_authorizes_edge"], True)
        self.assertIs(progress["materializer_reached"], False)
        self.assertEqual(progress["pair_target_count"], 6)
        self.assertEqual(progress["pair_attempted_count"], 6)
        self.assertEqual(progress["pair_completed_count"], 6)
        self.assertEqual(progress["certified_conflict_count"], 6)
        self.assertEqual(len(compact.certificates), 6)

        partial: dict[str, object] = {}
        real_probe = clique_module._PersistentHighsPairLP.probe
        calls = [0]

        def timeout_on_second(oracle, pair, *, deadline):
            calls[0] += 1
            if calls[0] == 2:
                raise TimeoutError("synthetic_second_pair_timeout")
            return real_probe(oracle, pair, deadline=deadline)

        with patch.object(
            clique_module._PersistentHighsPairLP,
            "probe",
            new=timeout_on_second,
        ):
            with self.assertRaisesRegex(
                TimeoutError, "synthetic_second_pair_timeout"
            ):
                run_operator_exact_relu_phase_cliques_candidate(
                    build,
                    rivals,
                    selection,
                    deadline=time.monotonic() + 20.0,
                    emit_cut_hz=False,
                    diagnostic_progress=partial,
                )
        self.assertEqual(partial["status"], "pair_probe")
        self.assertEqual(partial["pair_target_count"], 6)
        self.assertEqual(partial["pair_attempted_count"], 2)
        self.assertEqual(partial["pair_completed_count"], 1)
        self.assertLessEqual(
            partial["certified_conflict_count"],
            partial["pair_completed_count"],
        )
        self.assertEqual(partial["last_pair_index"], 1)
        self.assertIs(partial["terminal_complete"], False)
        self.assertIs(partial["candidate_cut_hz_emitted"], False)
        self.assertIs(partial["materializer_reached"], False)

        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "diagnostic_progress_must_be_empty_builtin_dict_or_none",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
                diagnostic_progress={"prepopulated": True},
            )

    def test_persistent_oracle_closes_exactly_once_on_all_pair_exits(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals = _rivals()
        selection = derive_operator_exact_relu_property_phase_literals(
            build, rivals
        )
        real_close = clique_module._PersistentHighsPairLP.close
        real_probe = clique_module._PersistentHighsPairLP.probe
        close_calls: list[int] = []

        def counted_close(oracle) -> None:
            close_calls.append(id(oracle))
            real_close(oracle)

        with patch.object(
            clique_module._PersistentHighsPairLP,
            "close",
            new=counted_close,
        ):
            result = run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
                emit_cut_hz=False,
            )
        self.assertEqual(len(close_calls), 1)
        self.assertEqual(result.telemetry["model_builds"], 1)
        self.assertEqual(result.telemetry["oracle"]["solve_calls"], 6)

        probe_calls = [0]

        def timeout_on_second(oracle, pair, *, deadline):
            probe_calls[0] += 1
            if probe_calls[0] == 2:
                raise TimeoutError("synthetic_pair_timeout_for_close")
            return real_probe(oracle, pair, deadline=deadline)

        with patch.object(
            clique_module._PersistentHighsPairLP,
            "close",
            new=counted_close,
        ), patch.object(
            clique_module._PersistentHighsPairLP,
            "probe",
            new=timeout_on_second,
        ):
            with self.assertRaisesRegex(
                TimeoutError, "synthetic_pair_timeout_for_close"
            ):
                run_operator_exact_relu_phase_cliques_candidate(
                    build,
                    rivals,
                    selection,
                    deadline=time.monotonic() + 20.0,
                    emit_cut_hz=False,
                )
        self.assertEqual(len(close_calls), 2)

        def fail_first_pair(oracle, pair, *, deadline):
            del oracle, pair, deadline
            raise RuntimeError("synthetic_pair_exception_for_close")

        with patch.object(
            clique_module._PersistentHighsPairLP,
            "close",
            new=counted_close,
        ), patch.object(
            clique_module._PersistentHighsPairLP,
            "probe",
            new=fail_first_pair,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "synthetic_pair_exception_for_close"
            ):
                run_operator_exact_relu_phase_cliques_candidate(
                    build,
                    rivals,
                    selection,
                    deadline=time.monotonic() + 20.0,
                    emit_cut_hz=False,
                )
        self.assertEqual(len(close_calls), 3)

    def test_actual_k4_full_parent_exact_closure_tightens_both_rivals(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(build)

        self.assertEqual(build.hz.Gb.nnz, 0)
        self.assertAlmostEqual(
            _relaxed_margin_upper(
                build.hz, rivals[0].objective
            ),
            0.25,
            places=10,
        )
        self.assertEqual(
            _exact_k4_margin_upper(), Fraction(-1, 4)
        )
        self.assertEqual(
            result.status,
            "focused_rival_clique_cut_candidate",
        )
        self.assertEqual(len(result.ranked_phases), 4)
        self.assertEqual(len(result.pair_records), 6)
        self.assertEqual(len(result.certificates), 6)
        self.assertEqual(len(result.cliques), 1)
        self.assertEqual(len(result.cliques[0].literals), 4)
        self.assertEqual(result.telemetry["model_builds"], 1)
        self.assertEqual(
            result.telemetry["oracle"]["base_solve_calls"], 0
        )
        self.assertEqual(
            result.telemetry["oracle"]["solve_calls"], 6
        )
        self.assertTrue(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
            )
        )
        for rival in rivals:
            self.assertLess(
                _relaxed_margin_upper(
                    result.hz, rival.objective
                ),
                0.0,
            )
        self.assertFalse(result.proof_authority)
        self.assertTrue(
            all(
                clique.proof_authority is False
                for clique in result.cliques
            )
        )

    def test_compact_k4_reconstructs_legacy_cut_and_fails_closed(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals = _rivals()
        selection = derive_operator_exact_relu_property_phase_literals(
            build, rivals
        )
        legacy = run_operator_exact_relu_phase_cliques_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 20.0,
        )
        compact = run_operator_exact_relu_phase_cliques_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 20.0,
            emit_cut_hz=False,
        )
        self.assertEqual(
            compact.status,
            "focused_rival_clique_compact_candidate",
        )
        self.assertIsNone(compact.hz)
        self.assertFalse(
            any(
                type(value) is SparseHZono
                for value in vars(compact).values()
            )
        )
        self.assertEqual(
            compact.telemetry["schema"],
            "act.operator_exact_relu_phase_clique_compact_candidate.v1",
        )
        self.assertEqual(compact.pair_records, legacy.pair_records)
        self.assertEqual(compact.certificates, legacy.certificates)
        self.assertEqual(compact.cliques, legacy.cliques)
        self.assertTrue(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                compact,
                deadline=time.monotonic() + 20.0,
            )
        )
        capability = verify_and_issue_operator_phase_clique_snapshot(
            build,
            rivals,
            selection,
            compact,
            deadline=time.monotonic() + 20.0,
        )
        self.assertIsNotNone(capability)
        snapshot = consume_verified_operator_phase_clique_snapshot(
            capability, deadline=time.monotonic() + 20.0
        )
        self.assertEqual(
            clique_module.sparse_hz_semantic_digest(snapshot.cut_hz),
            clique_module.sparse_hz_semantic_digest(legacy.hz),
        )
        for rival in rivals:
            self.assertAlmostEqual(
                _relaxed_margin_upper(snapshot.cut_hz, rival.objective),
                _relaxed_margin_upper(legacy.hz, rival.objective),
                places=12,
            )

        bad_schema = dict(compact.telemetry)
        bad_schema["schema"] = legacy.telemetry["schema"]
        malformed = (
            replace(
                compact,
                status="focused_rival_clique_cut_candidate",
            ),
            replace(
                compact,
                hz=legacy.hz,
            ),
            replace(compact, telemetry=bad_schema),
            replace(
                compact,
                certificates=(
                    replace(
                        compact.certificates[0],
                        certificate_sha256="0" * 64,
                    ),
                    *compact.certificates[1:],
                ),
            ),
        )
        for candidate in malformed:
            self.assertFalse(
                verify_operator_exact_relu_phase_cliques_result(
                    build,
                    rivals,
                    selection,
                    candidate,
                    deadline=time.monotonic() + 20.0,
                )
            )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "emit_cut_hz_not_builtin_bool",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
                emit_cut_hz=np.bool_(False),
            )

    def test_candidate_nonzeros_telemetry_tamper_fails_exact(self) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(build)
        telemetry = copy.deepcopy(result.telemetry)
        telemetry["oracle"]["candidate_nonzeros"] -= 1
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                replace(result, telemetry=telemetry),
                deadline=time.monotonic() + 20.0,
            )
        )

    def test_raw_top1_batch_to_selection_to_exact_clique(self) -> None:
        live = {
            "kind": "TOP1_ROBUST",
            "C": torch.tensor(
                [
                    [-1.0, 1.0, 0.0],
                    [-1.0, 0.0, 1.0],
                ],
                dtype=torch.float64,
            ),
            "thresholds": torch.zeros(
                (1, 2), dtype=torch.float64
            ),
            "M": 2,
            "y_true": torch.tensor([0], dtype=torch.int64),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "top1.vnnlib"
            path.write_text(
                _raw_top1_source().strip() + "\n",
                encoding="utf-8",
            )
            source_sha = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            candidate = issue_raw_vnnlib_top1_candidate(
                path,
                expected_vnnlib_sha256=source_sha,
                live_assert_params=live,
                deadline=time.monotonic() + 10.0,
            )
            batch = consume_raw_vnnlib_top1_candidate(
                candidate,
                live_assert_params=live,
                deadline=time.monotonic() + 10.0,
            )
            self.assertTrue(
                validate_consumed_raw_vnnlib_rival_batch(batch)
            )
            rivals = batch.rivals

            build = _k4_corner_build()
            _, selection, result = _run(
                build, rivals=rivals
            )
            self.assertEqual(
                tuple(item.rival_id for item in rivals),
                (1, 2),
            )
            self.assertEqual(len(result.certificates), 6)
            self.assertTrue(
                verify_operator_exact_relu_phase_cliques_result(
                    build,
                    rivals,
                    selection,
                    result,
                    deadline=time.monotonic() + 20.0,
                )
            )
            for rival in rivals:
                self.assertLess(
                    _relaxed_margin_upper(
                        result.hz, rival.objective
                    ),
                    0.0,
                )

    def test_missing_edges_emit_only_sound_smaller_cliques(
        self,
    ) -> None:
        build = _corner_build(bias=-1.0)
        rivals, selection, result = _run(build)
        statuses = tuple(
            item.status for item in result.pair_records
        )
        self.assertEqual(
            statuses.count("certified_conflict"), 2
        )
        self.assertEqual(
            statuses.count("feasible_or_unknown"), 4
        )
        self.assertEqual(len(result.cliques), 2)
        self.assertEqual(
            tuple(len(item.literals) for item in result.cliques),
            (2, 2),
        )
        self.assertTrue(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
            )
        )
        self.assertTrue(
            all(
                len(item.edge_certificate_sha256s) == 1
                for item in result.cliques
            )
        )

        no_edges = _corner_build(bias=0.0)
        _, empty_selection, empty_result = _run(no_edges)
        self.assertEqual(
            empty_result.status,
            "no_certified_focused_rival_phase_clique",
        )
        self.assertIsNone(empty_result.hz)
        self.assertFalse(empty_result.cliques)
        self.assertTrue(
            verify_operator_exact_relu_phase_cliques_result(
                no_edges,
                rivals,
                empty_selection,
                empty_result,
                deadline=time.monotonic() + 20.0,
            )
        )

    def test_single_focused_rival_cut_is_global_for_other_rivals(
        self,
    ) -> None:
        build = _k4_corner_build()
        all_rivals = _rivals()
        focused_rivals = (all_rivals[0],)
        selection = (
            derive_operator_exact_relu_property_phase_literals(
                build, focused_rivals
            )
        )
        result = run_operator_exact_relu_phase_cliques_candidate(
            build,
            focused_rivals,
            selection,
            deadline=time.monotonic() + 20.0,
        )
        self.assertEqual(len(result.certificates), 6)
        self.assertTrue(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                focused_rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
            )
        )
        self.assertEqual(
            result.focused_property_digest,
            selection.property_digest,
        )
        self.assertNotEqual(
            selection.property_digest,
            derive_operator_exact_relu_property_phase_literals(
                build, all_rivals
            ).property_digest,
        )
        # The second rival did not participate in phase selection.  The cut
        # still tightens it because all six conflicts were proved on the
        # complete parent feasible set, independently of any objective.
        self.assertLess(
            _relaxed_margin_upper(
                result.hz, all_rivals[1].objective
            ),
            0.0,
        )

    def test_exact_zero_omission_never_enters_a_cut(self) -> None:
        build = _one_zero_effect_mapping(_k4_corner_build())
        rivals, selection, result = _run(build)
        self.assertEqual(len(selection.zero_omissions), 1)
        self.assertEqual(len(result.ranked_phases), 3)
        self.assertEqual(len(result.omitted_zero_bcol_ids), 1)
        omitted = result.omitted_zero_bcol_ids[0]
        self.assertTrue(
            all(
                literal.stable_bcol_id != omitted
                for clique in result.cliques
                for literal in clique.literals
            )
        )
        stable_positions = {
            int(stable_id): index
            for index, stable_id in enumerate(
                build.hz.bcol_ids.tolist()
            )
        }
        cut_row = np.asarray(
            result.hz.Aub.getrow(
                result.hz.n_ub - 1
            ).toarray()
        ).reshape(-1)
        self.assertEqual(
            cut_row[stable_positions[omitted]], 0.0
        )
        self.assertEqual(np.count_nonzero(cut_row), 3)
        self.assertTrue(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
            )
        )

    def test_deterministic_top_k_and_caller_caps(self) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(
            build,
            max_top_literals=3,
            max_total_pairs=3,
        )
        self.assertEqual(len(result.ranked_phases), 3)
        self.assertEqual(len(result.pair_records), 3)
        self.assertEqual(len(result.excluded_selected_bcol_ids), 1)
        self.assertEqual(
            tuple(
                item.stable_bcol_id
                for item in result.ranked_phases
            ),
            tuple(
                sorted(
                    item.stable_bcol_id
                    for item in selection.mappings
                )[:3]
            ),
        )
        self.assertTrue(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
                max_top_literals=3,
                max_total_pairs=3,
            )
        )
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
            )
        )

        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "pair_cap_exceeded",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
                max_total_pairs=5,
            )
        parent_variables = (
            build.hz.n_cont + build.hz.n_bin
        )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "parent_size_cap_exceeded",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
                max_parent_variables=parent_variables - 1,
            )
        matrices = (
            build.hz.Gc,
            build.hz.Gb,
            build.hz.Ac,
            build.hz.Ab,
            build.hz.Auc,
            build.hz.Aub,
        )
        dense_arrays = (
            build.hz.c,
            build.hz.b,
            build.hz.ub,
            build.hz.col_ids,
            build.hz.bcol_ids,
        )
        buffer_items = sum(
            int(matrix.data.size)
            + int(matrix.indices.size)
            + int(matrix.indptr.size)
            for matrix in matrices
        ) + sum(int(value.size) for value in dense_arrays)
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "parent_size_cap_exceeded",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
                max_parent_buffer_items=buffer_items - 1,
            )

    def test_nested_conditional_metadata_is_in_parent_buffer_cap(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals = _rivals()
        selection = (
            derive_operator_exact_relu_property_phase_literals(
                build, rivals
            )
        )
        build.hz._audit_conditional_payload = {
            "nested": (np.zeros(2000, dtype=np.float64),)
        }
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "conditional_metadata_cap_exceeded",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
                max_parent_buffer_items=1000,
            )

        cycle = []
        cycle.append(cycle)
        build.hz._audit_conditional_payload = cycle
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "conditional_metadata_cycle",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
            )

    def test_tiny_parent_coefficient_telemetry_is_diagnostic(
        self,
    ) -> None:
        build = _k4_corner_build()
        hz = build.hz
        tiny_continuous = sp.csr_matrix(
            (
                np.asarray([1.0e-13], dtype=np.float64),
                (
                    np.asarray([0], dtype=np.int32),
                    np.asarray([0], dtype=np.int32),
                ),
            ),
            shape=(1, hz.n_cont),
        )
        tiny_binary = sp.csr_matrix(
            (1, hz.n_bin), dtype=np.float64
        )
        tags = tuple(hz._solver_constraint_row_tags)
        extended = _clone_hz(
            hz,
            Auc=_canonical_csr(
                sp.vstack(
                    [hz.Auc, tiny_continuous], format="csr"
                )
            ),
            Aub=_canonical_csr(
                sp.vstack(
                    [hz.Aub, tiny_binary], format="csr"
                )
            ),
            ub=np.concatenate(
                [hz.ub, np.asarray([1.0], dtype=np.float64)]
            ),
            row_tags=tags + ("audit_tiny_redundant",),
        )
        tiny_build = replace(build, hz=extended)
        rivals, selection, result = _run(tiny_build)
        raw_constraint_nonzeros = int(
            extended.Auc.nnz
            + extended.Aub.nnz
            + extended.Ac.nnz
            + extended.Ab.nnz
        )
        self.assertLess(
            result.telemetry["oracle"]["candidate_nonzeros"],
            raw_constraint_nonzeros,
        )
        self.assertTrue(
            verify_operator_exact_relu_phase_cliques_result(
                tiny_build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
            )
        )

    def test_bounded_maximal_search_recovers_greedy_missed_k4(
        self,
    ) -> None:
        ranked = tuple(
            RankedOperatorPhase(
                rank=index,
                stable_bcol_id=index,
                phase=1,
                score_numerator=1,
                score_denominator=1,
            )
            for index in range(10)
        )
        literals = tuple(
            PhaseLiteral(
                stable_bcol_id=index,
                phase=1,
                binding_digest=hashlib.sha256(
                    f"literal:{index}".encode("ascii")
                ).hexdigest(),
            )
            for index in range(10)
        )
        core = (6, 7, 8, 9)
        core_edges = {
            (left, right)
            for offset, left in enumerate(core)
            for right in core[offset + 1 :]
        }
        outsider_edges = {
            (0, 6),
            (0, 7),
            (1, 6),
            (1, 8),
            (2, 6),
            (2, 9),
            (3, 7),
            (3, 8),
            (4, 7),
            (4, 9),
            (5, 8),
            (5, 9),
        }
        edges = tuple(sorted(core_edges | outsider_edges))
        records = tuple(
            PersistentPairRecord(
                literals=(literals[left], literals[right]),
                status="certified_conflict",
                ray_nonzero_rows=1,
                certificate_sha256=hashlib.sha256(
                    f"edge:{left}:{right}".encode("ascii")
                ).hexdigest(),
                rationalization="test_exact",
            )
            for left, right in edges
        )
        cliques, search_nodes, truncated = (
            clique_module._maximal_weighted_cliques(
                ranked=ranked,
                literals=literals,
                pair_records=records,
                subset_digest="f" * 64,
                max_cliques=16,
                max_search_nodes=100000,
                max_exact_bits=4096,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertGreater(search_nodes, 0)
        self.assertFalse(truncated)
        self.assertEqual(
            tuple(
                item.stable_bcol_id
                for item in cliques[0].literals
            ),
            core,
        )
        self.assertEqual(len(cliques[0].literals), 4)

        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "clique_search_node_cap_exceeded",
        ):
            clique_module._maximal_weighted_cliques(
                ranked=ranked,
                literals=literals,
                pair_records=records,
                subset_digest="f" * 64,
                max_cliques=16,
                max_search_nodes=1,
                max_exact_bits=4096,
                deadline=time.monotonic() + 10.0,
            )

    def test_oversized_candidate_collections_reject_before_walk(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals = _rivals()
        selection = (
            derive_operator_exact_relu_property_phase_literals(
                build,
                rivals,
                timeout_seconds=0.1,
            )
        )
        result = run_operator_exact_relu_phase_cliques_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 10.0,
            selection_timeout_seconds=0.1,
        )
        oversized_ranked = (
            result.ranked_phases[0],
        ) * 2_000_000
        started = time.monotonic()
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                replace(
                    result, ranked_phases=oversized_ranked
                ),
                deadline=started + 0.12,
                selection_timeout_seconds=0.1,
            )
        )
        self.assertLess(time.monotonic() - started, 0.1)

        record = result.pair_records[0]
        oversized_literals = (
            record.literals[0],
        ) * 2_000_000
        started = time.monotonic()
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                replace(
                    result,
                    pair_records=(
                        replace(
                            record,
                            literals=oversized_literals,
                        ),
                        *result.pair_records[1:],
                    ),
                ),
                deadline=started + 0.12,
                selection_timeout_seconds=0.1,
            )
        )
        self.assertLess(time.monotonic() - started, 0.1)

    def test_candidate_cut_hz_is_capped_before_digest(self) -> None:
        build = _k4_corner_build()
        rivals = _rivals()
        selection = (
            derive_operator_exact_relu_property_phase_literals(
                build,
                rivals,
                timeout_seconds=0.1,
            )
        )
        result = run_operator_exact_relu_phase_cliques_candidate(
            build,
            rivals,
            selection,
            deadline=time.monotonic() + 10.0,
            selection_timeout_seconds=0.1,
            max_parent_buffer_items=10000,
        )
        result.hz._audit_conditional_payload = np.zeros(
            5000, dtype=np.float64
        )
        started = time.monotonic()
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                result,
                deadline=started + 0.12,
                selection_timeout_seconds=0.1,
                max_parent_buffer_items=10000,
            )
        )
        self.assertLess(time.monotonic() - started, 0.1)

    def test_certificate_clique_and_hz_tampering_fail(self) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(build)
        certificate = result.certificates[0]
        tampered_certificate = replace(
            certificate,
            contradiction_numerator=(
                certificate.contradiction_numerator - 1
            ),
        )
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                replace(
                    result,
                    certificates=(
                        tampered_certificate,
                        *result.certificates[1:],
                    ),
                ),
                deadline=time.monotonic() + 20.0,
            )
        )
        clique = result.cliques[0]
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                replace(
                    result,
                    cliques=(
                        replace(
                            clique,
                            edge_certificate_sha256s=(
                                clique.edge_certificate_sha256s[
                                    :-1
                                ]
                            ),
                        ),
                    ),
                ),
                deadline=time.monotonic() + 20.0,
            )
        )
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                replace(result, hz=build.hz),
                deadline=time.monotonic() + 20.0,
            )
        )

    def test_verifier_never_invokes_candidate_equality(self) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(build)

        class EvilEquality:
            calls = 0

            def __eq__(self, other):
                type(self).calls += 1
                raise AssertionError("candidate equality invoked")

            def __ne__(self, other):
                type(self).calls += 1
                raise AssertionError("candidate inequality invoked")

            def __hash__(self):
                type(self).calls += 1
                raise AssertionError("candidate hash invoked")

        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                replace(result, cliques=(EvilEquality(),)),
                deadline=time.monotonic() + 20.0,
            )
        )
        self.assertEqual(EvilEquality.calls, 0)
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                replace(
                    result,
                    pair_records=(EvilEquality(),)
                    + result.pair_records[1:],
                ),
                deadline=time.monotonic() + 20.0,
            )
        )
        self.assertEqual(EvilEquality.calls, 0)
        malicious_telemetry = dict(result.telemetry)
        malicious_caps = dict(malicious_telemetry["caps"])
        malicious_caps["max_cliques"] = EvilEquality()
        malicious_telemetry["caps"] = malicious_caps
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                replace(result, telemetry=malicious_telemetry),
                deadline=time.monotonic() + 20.0,
            )
        )
        self.assertEqual(EvilEquality.calls, 0)

    def test_slow_csr_is_rejected_before_shape_or_nnz_hooks(
        self,
    ) -> None:
        class SlowCSR(sp.csr_matrix):
            armed = False
            calls = 0

            @property
            def shape(self):
                if type(self).armed:
                    type(self).calls += 1
                    raise AssertionError("shape hook reached")
                return self._shape

            @shape.setter
            def shape(self, value):
                self._shape = tuple(value)

            @property
            def nnz(self):
                if type(self).armed:
                    type(self).calls += 1
                    raise AssertionError("nnz hook reached")
                return int(self.indptr[-1])

        build = _k4_corner_build()
        rivals, selection, result = _run(build)
        slow_parent = SlowCSR(build.hz.Gc, copy=True)
        SlowCSR.calls = 0
        SlowCSR.armed = True
        build.hz.Gc = slow_parent
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "not_exact_csr",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 10.0,
            )
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertIsNone(
            verify_and_issue_operator_phase_clique_snapshot(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertEqual(SlowCSR.calls, 0)

        SlowCSR.armed = False
        build = _k4_corner_build()
        rivals, selection, result = _run(build)
        slow_result = SlowCSR(result.hz.Gc, copy=True)
        SlowCSR.calls = 0
        SlowCSR.armed = True
        result.hz.Gc = slow_result
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertIsNone(
            verify_and_issue_operator_phase_clique_snapshot(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertEqual(SlowCSR.calls, 0)

    def test_issue_uses_only_completed_parent_and_result_snapshots(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(build)
        original_parent_c = build.hz.c.copy()
        original_result_c = result.hz.c.copy()
        parent_done = threading.Event()
        parent_continue = threading.Event()
        result_done = threading.Event()
        result_continue = threading.Event()
        original_copy = (
            clique_module._copy_exact_array_with_deadline
        )
        seen = set()

        def gated_copy(value, *, deadline, stage):
            copied = original_copy(
                value, deadline=deadline, stage=stage
            )
            if value is build.hz.c and "parent" not in seen:
                seen.add("parent")
                parent_done.set()
                if not parent_continue.wait(5.0):
                    raise AssertionError("parent mutation gate timed out")
            elif value is result.hz.c and "result" not in seen:
                seen.add("result")
                result_done.set()
                if not result_continue.wait(5.0):
                    raise AssertionError("result mutation gate timed out")
            return copied

        issued = []
        failures = []

        def worker():
            try:
                issued.append(
                    verify_and_issue_operator_phase_clique_snapshot(
                        build,
                        rivals,
                        selection,
                        result,
                        deadline=time.monotonic() + 30.0,
                    )
                )
            except BaseException as exc:  # pragma: no cover - diagnostic
                failures.append(exc)

        with patch.object(
            clique_module,
            "_copy_exact_array_with_deadline",
            side_effect=gated_copy,
        ):
            thread = threading.Thread(target=worker)
            thread.start()
            self.assertTrue(parent_done.wait(5.0))
            build.hz.c[0] += 17.0
            parent_continue.set()
            self.assertTrue(result_done.wait(5.0))
            result.hz.c[0] -= 19.0
            result_continue.set()
            thread.join(20.0)
            self.assertFalse(thread.is_alive())
        self.assertFalse(failures)
        self.assertEqual(seen, {"parent", "result"})
        self.assertEqual(len(issued), 1)
        self.assertIsNotNone(issued[0])
        snapshot = consume_verified_operator_phase_clique_snapshot(
            issued[0], deadline=time.monotonic() + 5.0
        )
        np.testing.assert_array_equal(
            snapshot.cut_hz.c, original_result_c
        )
        self.assertFalse(
            np.array_equal(build.hz.c, original_parent_c)
        )
        self.assertFalse(
            np.array_equal(result.hz.c, original_result_c)
        )
        result.hz.c[:] = 123.0
        np.testing.assert_array_equal(
            snapshot.cut_hz.c, original_result_c
        )

    def test_bool_verifier_does_not_touch_snapshot_registry(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(build)
        capability = (
            verify_and_issue_operator_phase_clique_snapshot(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
            )
        )
        self.assertIsNotNone(capability)
        with clique_module._SNAPSHOT_REGISTRY_LOCK:
            before = {
                token: (id(record), id(record.snapshot))
                for token, record in (
                    clique_module._SNAPSHOT_REGISTRY.items()
                )
            }
        self.assertTrue(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
            )
        )
        with clique_module._SNAPSHOT_REGISTRY_LOCK:
            after = {
                token: (id(record), id(record.snapshot))
                for token, record in (
                    clique_module._SNAPSHOT_REGISTRY.items()
                )
            }
        self.assertEqual(before, after)
        consume_verified_operator_phase_clique_snapshot(
            capability, deadline=time.monotonic() + 5.0
        )

    def test_ulp_boundary_is_repaired_then_exactly_replayed(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(build)
        exact_replay = (
            clique_module._exact_original_hz_feasible_candidate
        )
        replay_reasons = []

        def audited_exact_replay(*args, **kwargs):
            replayed = exact_replay(*args, **kwargs)
            replay_reasons.append(replayed[1])
            return replayed

        with patch.object(
            clique_module,
            "_exact_original_hz_feasible_candidate",
            side_effect=audited_exact_replay,
        ):
            capability = (
                verify_and_issue_operator_phase_clique_snapshot(
                    build,
                    rivals,
                    selection,
                    result,
                    deadline=time.monotonic() + 20.0,
                )
            )
        self.assertIsNotNone(capability)
        self.assertTrue(replay_reasons)
        self.assertNotEqual(replay_reasons[0], "exact")
        self.assertEqual(replay_reasons[-1], "exact")
        consume_verified_operator_phase_clique_snapshot(
            capability, deadline=time.monotonic() + 5.0
        )

    def test_constructive_seal_default_off_has_no_digest_or_registry_work(
        self,
    ) -> None:
        with operator_hz_module._CONSTRUCTIVE_NONEMPTY_SEAL_LOCK:
            before = {
                token: id(record)
                for token, record in (
                    operator_hz_module
                    ._CONSTRUCTIVE_NONEMPTY_SEAL_RECORDS.items()
                )
            }
        with patch(
            "act.back_end.hybridz_tf.adaptive_phase_forest."
            "sparse_hz_semantic_digest",
            side_effect=AssertionError(
                "default-off builder reached semantic digest"
            ),
        ):
            build = _corner_build(bias=-1.5)
        self.assertIsNone(build.constructive_nonempty_seal)
        with operator_hz_module._CONSTRUCTIVE_NONEMPTY_SEAL_LOCK:
            after = {
                token: id(record)
                for token, record in (
                    operator_hz_module
                    ._CONSTRUCTIVE_NONEMPTY_SEAL_RECORDS.items()
                )
            }
        self.assertEqual(before, after)
        rivals, selection, result = _run(build)
        with patch.object(
            clique_module,
            "_cut_has_exact_private_nonempty_witness",
            return_value=False,
        ) as exact_witness:
            self.assertIsNone(
                verify_and_issue_operator_phase_clique_snapshot(
                    build,
                    rivals,
                    selection,
                    result,
                    deadline=time.monotonic() + 20.0,
                )
            )
        exact_witness.assert_called_once()
        with self.assertRaisesRegex(
            operator_hz_module.OperatorHZBuildError,
            "must be a bool",
        ):
            _corner_build(
                bias=-1.5,
                issue_constructive_nonempty_seal=1,
            )

    def test_sealed_issue_uses_set_equivalence_without_fraction_witness(
        self,
    ) -> None:
        build = _corner_build(
            bias=-1.5,
            issue_constructive_nonempty_seal=True,
        )
        self.assertGreater(build.hz.constraint_nnz, 8)
        rivals, selection, result = _run(
            build, max_exact_nonzeros=8
        )
        seal = build.constructive_nonempty_seal
        self.assertIsNotNone(seal)
        layout = clique_module._exact_hz_core_layout(build.hz)
        owner_identity = clique_module._exact_hz_core_identity(
            build.hz, layout
        )
        self.assertTrue(
            operator_hz_module
            .validate_operator_hz_constructive_nonempty_seal(
                seal,
                owner_build=build,
                owner_hz=build.hz,
                owner_core_identity=owner_identity,
                private_parent_semantic_digest=(
                    clique_module.sparse_hz_semantic_digest(
                        build.hz
                    )
                ),
            )
        )
        with patch.object(
            clique_module,
            "_cut_has_exact_private_nonempty_witness",
            side_effect=AssertionError(
                "sealed path reached Fraction witness"
            ),
        ):
            capability = (
                verify_and_issue_operator_phase_clique_snapshot(
                    build,
                    rivals,
                    selection,
                    result,
                    deadline=time.monotonic() + 20.0,
                    max_exact_nonzeros=8,
                )
            )
        self.assertIsNotNone(capability)
        consume_verified_operator_phase_clique_snapshot(
            capability, deadline=time.monotonic() + 5.0
        )

    def test_constructive_seal_rejects_clones_and_core_replacement(
        self,
    ) -> None:
        build = _corner_build(
            bias=-1.5,
            issue_constructive_nonempty_seal=True,
        )
        seal = build.constructive_nonempty_seal
        self.assertIsNotNone(seal)
        rivals, selection, result = _run(build)

        clone_hz = _clone_hz(build.hz)
        clone_build = replace(
            build,
            hz=clone_hz,
            constructive_nonempty_seal=seal,
        )
        clone_layout = clique_module._exact_hz_core_layout(
            clone_hz
        )
        self.assertFalse(
            operator_hz_module
            .validate_operator_hz_constructive_nonempty_seal(
                seal,
                owner_build=clone_build,
                owner_hz=clone_hz,
                owner_core_identity=(
                    clique_module._exact_hz_core_identity(
                        clone_hz, clone_layout
                    )
                ),
                private_parent_semantic_digest=(
                    clique_module.sparse_hz_semantic_digest(
                        clone_hz
                    )
                ),
            )
        )
        copied_build = replace(build)
        original_layout = clique_module._exact_hz_core_layout(
            build.hz
        )
        original_identity = (
            clique_module._exact_hz_core_identity(
                build.hz, original_layout
            )
        )
        original_digest = (
            clique_module.sparse_hz_semantic_digest(build.hz)
        )
        self.assertFalse(
            operator_hz_module
            .validate_operator_hz_constructive_nonempty_seal(
                seal,
                owner_build=copied_build,
                owner_hz=build.hz,
                owner_core_identity=original_identity,
                private_parent_semantic_digest=original_digest,
            )
        )
        copied_seal = copy.copy(seal)
        self.assertIsNot(copied_seal, seal)
        self.assertFalse(
            operator_hz_module
            .validate_operator_hz_constructive_nonempty_seal(
                copied_seal,
                owner_build=build,
                owner_hz=build.hz,
                owner_core_identity=original_identity,
                private_parent_semantic_digest=original_digest,
            )
        )
        with patch.object(
            clique_module,
            "_cut_has_exact_private_nonempty_witness",
            side_effect=AssertionError(
                "invalid copied seal reached Fraction fallback"
            ),
        ):
            self.assertIsNone(
                verify_and_issue_operator_phase_clique_snapshot(
                    copied_build,
                    rivals,
                    selection,
                    result,
                    deadline=time.monotonic() + 20.0,
                )
            )
        forged = object.__new__(type(seal))
        for name, value in (
            ("_token", seal.token),
            ("_semantic_digest", seal.semantic_digest),
            ("_process_id", seal.process_id),
            ("_reason", seal.reason),
        ):
            object.__setattr__(forged, name, value)
        self.assertFalse(
            operator_hz_module
            .validate_operator_hz_constructive_nonempty_seal(
                forged,
                owner_build=build,
                owner_hz=build.hz,
                owner_core_identity=original_identity,
                private_parent_semantic_digest=original_digest,
            )
        )

        build.hz.Auc = build.hz.Auc.copy()
        replaced_layout = clique_module._exact_hz_core_layout(
            build.hz
        )
        self.assertFalse(
            operator_hz_module
            .validate_operator_hz_constructive_nonempty_seal(
                seal,
                owner_build=build,
                owner_hz=build.hz,
                owner_core_identity=(
                    clique_module._exact_hz_core_identity(
                        build.hz, replaced_layout
                    )
                ),
                private_parent_semantic_digest=(
                    clique_module.sparse_hz_semantic_digest(
                        build.hz
                    )
                ),
            )
        )

        array_build = _corner_build(
            bias=-1.5,
            issue_constructive_nonempty_seal=True,
        )
        array_build.hz.c = array_build.hz.c.copy()
        array_layout = clique_module._exact_hz_core_layout(
            array_build.hz
        )
        self.assertFalse(
            operator_hz_module
            .validate_operator_hz_constructive_nonempty_seal(
                array_build.constructive_nonempty_seal,
                owner_build=array_build,
                owner_hz=array_build.hz,
                owner_core_identity=(
                    clique_module._exact_hz_core_identity(
                        array_build.hz, array_layout
                    )
                ),
                private_parent_semantic_digest=(
                    clique_module.sparse_hz_semantic_digest(
                        array_build.hz
                    )
                ),
            )
        )

        mutated_build = _corner_build(
            bias=-1.5,
            issue_constructive_nonempty_seal=True,
        )
        mutated_layout = clique_module._exact_hz_core_layout(
            mutated_build.hz
        )
        mutated_identity = (
            clique_module._exact_hz_core_identity(
                mutated_build.hz, mutated_layout
            )
        )
        mutated_build.hz.c[0] = np.nextafter(
            mutated_build.hz.c[0], np.inf
        )
        self.assertFalse(
            operator_hz_module
            .validate_operator_hz_constructive_nonempty_seal(
                mutated_build.constructive_nonempty_seal,
                owner_build=mutated_build,
                owner_hz=mutated_build.hz,
                owner_core_identity=mutated_identity,
                private_parent_semantic_digest=(
                    clique_module.sparse_hz_semantic_digest(
                        mutated_build.hz
                    )
                ),
            )
        )

    def test_constructive_seal_rejects_concurrent_mixed_core_copy(
        self,
    ) -> None:
        build = _corner_build(
            bias=-1.5,
            issue_constructive_nonempty_seal=True,
        )
        rivals, selection, result = _run(build)
        original_copy = (
            clique_module._copy_exact_array_with_deadline
        )
        mutated = [False]

        def mix_after_center(value, *, deadline, stage):
            copied = original_copy(
                value, deadline=deadline, stage=stage
            )
            if value is build.hz.c and not mutated[0]:
                mutated[0] = True
                build.hz.ub[0] = np.nextafter(
                    build.hz.ub[0], np.inf
                )
            return copied

        with patch.object(
            clique_module,
            "_copy_exact_array_with_deadline",
            side_effect=mix_after_center,
        ):
            capability = (
                verify_and_issue_operator_phase_clique_snapshot(
                    build,
                    rivals,
                    selection,
                    result,
                    deadline=time.monotonic() + 20.0,
                )
            )
        self.assertTrue(mutated[0])
        self.assertIsNone(capability)

    def test_constructive_seal_is_process_local(
        self,
    ) -> None:
        build = _corner_build(
            bias=-1.5,
            issue_constructive_nonempty_seal=True,
        )
        seal = build.constructive_nonempty_seal
        layout = clique_module._exact_hz_core_layout(build.hz)
        identity = clique_module._exact_hz_core_identity(
            build.hz, layout
        )
        digest = clique_module.sparse_hz_semantic_digest(
            build.hz
        )
        with patch.object(
            operator_hz_module.os,
            "getpid",
            return_value=seal.process_id + 1,
        ):
            self.assertFalse(
                operator_hz_module
                .validate_operator_hz_constructive_nonempty_seal(
                    seal,
                    owner_build=build,
                    owner_hz=build.hz,
                    owner_core_identity=identity,
                    private_parent_semantic_digest=digest,
                )
            )

    def test_outward_binary_rhs_cannot_prove_original_hz_nonempty(
        self,
    ) -> None:
        hz = SparseHZono(
            c=np.zeros(1, dtype=np.float64),
            Gc=sp.csr_matrix((1, 0), dtype=np.float64),
            Gb=sp.csr_matrix((1, 1), dtype=np.float64),
            Ac=sp.csr_matrix((0, 0), dtype=np.float64),
            Ab=sp.csr_matrix((0, 1), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            Auc=sp.csr_matrix((2, 0), dtype=np.float64),
            Aub=sp.csr_matrix(
                np.array(((1.0,), (-1.0,)), dtype=np.float64)
            ),
            ub=np.array(
                (np.nextafter(1.0, 0.0), 0.0),
                dtype=np.float64,
            ),
            col_ids=np.zeros(0, dtype=np.int64),
            bcol_ids=np.array((0,), dtype=np.int64),
        )
        caps = clique_module.OperatorPhaseCliqueCaps(
            max_parent_variables=100,
            max_parent_rows=100,
            max_parent_nonzeros=100,
            max_parent_buffer_items=1000,
            max_top_literals=16,
            max_total_pairs=120,
            max_cliques=16,
            max_clique_search_nodes=1000,
            max_source_terms=100,
            max_multiplier_bits=256,
            max_exact_bits=4096,
            max_exact_nonzeros=100,
        )
        deadline = time.monotonic() + 5.0
        for binary_z in (0.0, 1.0):
            exact, _reason = (
                clique_module._exact_original_hz_feasible_candidate(
                    hz,
                    np.array((binary_z,), dtype=np.float64),
                    caps=caps,
                    deadline=deadline,
                )
            )
            self.assertFalse(exact)
        self.assertFalse(
            clique_module._generated_exact_private_witness(
                hz,
                caps=caps,
                deadline=deadline,
            )
        )

    def test_pair_record_verify_then_mutate_cannot_create_edges(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(build)
        original_records = result.pair_records
        mutable_records = tuple(
            replace(
                record,
                status="feasible_or_unknown",
                certificate_sha256=None,
                rationalization=None,
            )
            for record in original_records
        )
        malicious_telemetry = dict(result.telemetry)
        malicious_telemetry["exact_certificate_count"] = 0
        candidate = replace(
            result,
            pair_records=mutable_records,
            certificates=(),
            telemetry=malicious_telemetry,
        )
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                candidate,
                deadline=time.monotonic() + 20.0,
            )
        )
        original_clique_search = (
            clique_module._maximal_weighted_cliques
        )

        def mutate_then_search(*args, **kwargs):
            for mutable, original in zip(
                mutable_records, original_records
            ):
                object.__setattr__(
                    mutable, "status", "certified_conflict"
                )
                object.__setattr__(
                    mutable,
                    "certificate_sha256",
                    original.certificate_sha256,
                )
                object.__setattr__(
                    mutable,
                    "rationalization",
                    original.rationalization,
                )
            return original_clique_search(*args, **kwargs)

        with patch.object(
            clique_module,
            "_maximal_weighted_cliques",
            side_effect=mutate_then_search,
        ):
            capability = (
                verify_and_issue_operator_phase_clique_snapshot(
                    build,
                    rivals,
                    selection,
                    candidate,
                    deadline=time.monotonic() + 20.0,
                )
            )
        self.assertIsNone(capability)

    def test_capability_identity_ttl_one_use_and_readonly_buffers(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(build)
        capability = (
            verify_and_issue_operator_phase_clique_snapshot(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
            )
        )
        self.assertIsNotNone(capability)
        forged = replace(capability)
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "capability_invalid",
        ):
            consume_verified_operator_phase_clique_snapshot(
                forged, deadline=time.monotonic() + 5.0
            )
        snapshot = consume_verified_operator_phase_clique_snapshot(
            capability, deadline=time.monotonic() + 5.0
        )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "capability_invalid",
        ):
            consume_verified_operator_phase_clique_snapshot(
                capability, deadline=time.monotonic() + 5.0
            )
        with clique_module._SNAPSHOT_REGISTRY_LOCK:
            self.assertNotIn(
                capability.token,
                clique_module._SNAPSHOT_REGISTRY,
            )

        dense_names = ("c", "b", "ub", "col_ids", "bcol_ids")
        for name in dense_names:
            value = getattr(snapshot.cut_hz, name)
            self.assertIs(type(value), np.ndarray)
            self.assertFalse(value.flags.writeable)
            self.assertFalse(
                np.shares_memory(value, getattr(result.hz, name))
            )
        for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
            value = getattr(snapshot.cut_hz, name)
            live = getattr(result.hz, name)
            self.assertIs(type(value), sp.csr_matrix)
            for buffer_name in ("data", "indices", "indptr"):
                buffer = getattr(value, buffer_name)
                self.assertFalse(buffer.flags.writeable)
                self.assertFalse(
                    np.shares_memory(
                        buffer, getattr(live, buffer_name)
                    )
                )
        provenance = (
            (
                snapshot.continuous_layer_ids,
                build.hz._solver_continuous_column_layer_ids,
            ),
            (snapshot.full_col_ids, build.hz.full_col_ids),
            (
                snapshot.input_center,
                build.hz.operator_input_center,
            ),
            (
                snapshot.input_radius,
                build.hz.operator_input_radius,
            ),
            (snapshot.build_input_col_ids, build.input_col_ids),
        )
        for private, live in provenance:
            self.assertIs(type(private), np.ndarray)
            self.assertFalse(private.flags.writeable)
            self.assertFalse(np.shares_memory(private, live))
        self.assertEqual(
            snapshot.cut_hz.n_ub,
            snapshot.original_parent_n_ub
            + len(snapshot.verified_cliques),
        )
        self.assertEqual(
            len(snapshot.parent_row_tags),
            snapshot.cut_hz.n_eq
            + snapshot.original_parent_n_ub,
        )
        self.assertTrue(
            all(
                type(key) is str and type(value) is bool
                for key, value in snapshot.materializer_source_modes
            )
        )

        expiring = (
            verify_and_issue_operator_phase_clique_snapshot(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
            )
        )
        self.assertIsNotNone(expiring)
        with patch.object(
            clique_module.time,
            "monotonic",
            return_value=expiring.expires_monotonic,
        ):
            with self.assertRaisesRegex(
                OperatorExactReLUPhaseCliqueError,
                "capability_invalid",
            ):
                consume_verified_operator_phase_clique_snapshot(
                    expiring,
                    deadline=expiring.expires_monotonic + 1.0,
                )

    def test_capability_cannot_outlive_registry_lock_wait(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals, selection, result = _run(build)
        capability = (
            verify_and_issue_operator_phase_clique_snapshot(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() + 20.0,
                capability_ttl_seconds=0.15,
            )
        )
        self.assertIsNotNone(capability)
        started = threading.Event()
        consumed = []
        failures = []

        def consumer():
            started.set()
            try:
                consumed.append(
                    consume_verified_operator_phase_clique_snapshot(
                        capability,
                        deadline=time.monotonic() + 5.0,
                    )
                )
            except OperatorExactReLUPhaseCliqueError as exc:
                failures.append(exc)

        with clique_module._SNAPSHOT_REGISTRY_LOCK:
            thread = threading.Thread(target=consumer)
            thread.start()
            self.assertTrue(started.wait(2.0))
            time.sleep(0.25)
        thread.join(5.0)
        self.assertFalse(thread.is_alive())
        self.assertFalse(consumed)
        self.assertEqual(len(failures), 1)
        with clique_module._SNAPSHOT_REGISTRY_LOCK:
            self.assertNotIn(
                capability.token,
                clique_module._SNAPSHOT_REGISTRY,
            )

    def test_deadline_and_invalid_caps_fail_without_solver_work(
        self,
    ) -> None:
        build = _k4_corner_build()
        rivals = _rivals()
        selection = (
            derive_operator_exact_relu_property_phase_literals(
                build, rivals
            )
        )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "deadline_expired",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() - 1.0,
            )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "cannot_cover_selection_audit",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 0.01,
                selection_timeout_seconds=1.0,
            )
        with self.assertRaisesRegex(
            OperatorExactReLUPhaseCliqueError,
            "cap_out_of_range",
        ):
            run_operator_exact_relu_phase_cliques_candidate(
                build,
                rivals,
                selection,
                deadline=time.monotonic() + 20.0,
                max_cliques=0,
            )
        _, _, result = _run(build)
        self.assertFalse(
            verify_operator_exact_relu_phase_cliques_result(
                build,
                rivals,
                selection,
                result,
                deadline=time.monotonic() - 1.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
