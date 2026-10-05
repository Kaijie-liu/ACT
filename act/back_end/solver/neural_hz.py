"""Exact, opt-in Hybrid-Zonotope simplifications for neural verification.

This module is deliberately separate from the historical HZ implementation.
It contains representation transforms whose acceptance criterion is structural
compression, not a looser abstract domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class ContinuousElimination:
    """One exact equality-pivot reconstruction in original latent IDs."""

    source: int
    pivot: float
    rhs: float
    cont_sources: np.ndarray
    cont_coefficients: np.ndarray
    bin_sources: np.ndarray
    bin_coefficients: np.ndarray


@dataclass(frozen=True)
class NeuralHZProjection:
    """Result of fill-reducing inactive continuous-factor projection."""

    value_matrix: sp.csr_matrix
    constraint_matrix: sp.csr_matrix
    row_lower: np.ndarray
    row_upper: np.ndarray
    cont_sources: np.ndarray
    eliminations: tuple[ContinuousElimination, ...]
    nnz_before: int
    nnz_after: int

    @property
    def removed_cont(self) -> int:
        return len(self.eliminations)


@dataclass(frozen=True)
class BinaryFix:
    """One predicate-implied binary assignment in original latent IDs."""

    source: int
    value: int


@dataclass(frozen=True)
class NeuralHZPhaseReduction:
    """Exact result of predicate-implied binary phase fixing."""

    value_center: np.ndarray
    value_matrix: sp.csr_matrix
    constraint_matrix: sp.csr_matrix
    row_lower: np.ndarray
    row_upper: np.ndarray
    bin_sources: np.ndarray
    fixes: tuple[BinaryFix, ...]
    proven_empty: bool = False


def _outward_row_interval(matrix: sp.csr_matrix, variable_lower, variable_upper):
    """Return a floating-point enclosure of every sparse affine row."""
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError("Neural-HZ predicates contain non-finite coefficients")
    variable_lower = np.asarray(variable_lower, dtype=np.float64).reshape(-1)
    variable_upper = np.asarray(variable_upper, dtype=np.float64).reshape(-1)
    positive = matrix.maximum(0.0)
    negative = matrix.minimum(0.0)
    lower = np.asarray(
        positive @ variable_lower + negative @ variable_upper,
        dtype=np.float64,
    ).reshape(-1)
    upper = np.asarray(
        positive @ variable_upper + negative @ variable_lower,
        dtype=np.float64,
    ).reshape(-1)
    # A single nextafter is not enough for a long sparse dot product. Bound the
    # standard sequential-summation error by gamma_k * sum(abs(terms)), then
    # round once more outwards. The phase check can therefore miss a fixing,
    # but cannot manufacture one from ordinary floating-point accumulation.
    max_abs_bound = np.maximum(np.abs(variable_lower), np.abs(variable_upper))
    absolute_sum = np.asarray(
        abs(matrix) @ max_abs_bound,
        dtype=np.float64,
    ).reshape(-1)
    terms = np.diff(matrix.indptr).astype(np.float64, copy=False)
    eps = np.finfo(np.float64).eps
    denominator = np.maximum(1.0 - terms * eps, eps)
    rounding_pad = (terms * eps / denominator) * absolute_sum
    return (
        np.nextafter(lower - rounding_pad, -np.inf),
        np.nextafter(upper + rounding_pad, np.inf),
    )


def fix_predicate_implied_binary_phases(
    value_center,
    value_matrix,
    constraint_matrix,
    row_lower,
    row_upper,
    cont_sources,
    bin_sources,
    *,
    max_fixes: int = 128,
) -> NeuralHZPhaseReduction:
    """Fix binary HZ phases ruled out by existing predicate rows.

    Other variables are interval-relaxed only for the implication check. Thus
    a phase is fixed solely when even this superset cannot satisfy at least one
    predicate row. This is sound but intentionally incomplete and introduces
    no branching, backward pass, or convex replacement of an unfixed binary.
    """
    center = np.asarray(value_center, dtype=np.float64).reshape(-1).copy()
    value = sp.csr_matrix(value_matrix, dtype=np.float64)
    constraints = sp.csr_matrix(constraint_matrix, dtype=np.float64)
    lower = np.asarray(row_lower, dtype=np.float64).reshape(-1).copy()
    upper = np.asarray(row_upper, dtype=np.float64).reshape(-1).copy()
    cont_sources = np.asarray(cont_sources, dtype=np.int64).reshape(-1)
    current_bin = np.asarray(bin_sources, dtype=np.int64).reshape(-1).copy()
    n_cont = int(cont_sources.size)
    if constraints.shape[0] != lower.size or lower.shape != upper.shape:
        raise ValueError("Neural-HZ phase predicate bounds do not match matrix")
    if value.shape[1] != constraints.shape[1]:
        raise ValueError("Neural-HZ phase value/predicate widths do not match")
    if value.shape[1] != n_cont + current_bin.size:
        raise ValueError("Neural-HZ phase source maps do not match matrix width")
    if max_fixes < 0:
        raise ValueError("max_fixes must be non-negative")

    fixes: list[BinaryFix] = []
    proven_empty = False
    while len(fixes) < int(max_fixes) and current_bin.size:
        variable_lower = np.concatenate(
            [-np.ones(n_cont, dtype=np.float64), np.zeros(current_bin.size)]
        )
        variable_upper = np.ones(n_cont + current_bin.size, dtype=np.float64)
        row_min, row_max = _outward_row_interval(
            constraints, variable_lower, variable_upper
        )
        binary_csc = constraints[:, n_cont:].tocsc()
        selected = None
        for local_binary in range(current_bin.size):
            start = binary_csc.indptr[local_binary]
            stop = binary_csc.indptr[local_binary + 1]
            rows = binary_csc.indices[start:stop]
            coefficients = binary_csc.data[start:stop]
            if rows.size == 0:
                continue
            minimum_contribution = np.minimum(coefficients, 0.0)
            maximum_contribution = np.maximum(coefficients, 0.0)
            possible = []
            for binary_value in (0.0, 1.0):
                candidate_min = np.nextafter(
                    row_min[rows]
                    - minimum_contribution
                    + coefficients * binary_value,
                    -np.inf,
                )
                candidate_max = np.nextafter(
                    row_max[rows]
                    - maximum_contribution
                    + coefficients * binary_value,
                    np.inf,
                )
                intersects = (
                    (candidate_max >= lower[rows])
                    & (candidate_min <= upper[rows])
                )
                possible.append(bool(np.all(intersects)))
            if not possible[0] and not possible[1]:
                proven_empty = True
                selected = None
                break
            if possible[0] != possible[1]:
                selected = (local_binary, int(possible[1]))
                break
        if proven_empty or selected is None:
            break

        local_binary, binary_value = selected
        column = n_cont + int(local_binary)
        source = int(current_bin[local_binary])
        value_column = value[:, column]
        if binary_value:
            center += np.asarray(value_column.toarray()).reshape(-1)
            shift = np.asarray(constraints[:, column].toarray()).reshape(-1)
            lower -= shift
            upper -= shift
        keep_columns = _without_index(constraints.shape[1], column)
        value = value[:, keep_columns].tocsr()
        constraints = constraints[:, keep_columns].tocsr()
        current_bin = current_bin[_without_index(current_bin.size, local_binary)]
        fixes.append(BinaryFix(source=source, value=binary_value))

    if proven_empty:
        empty_row = sp.csr_matrix((1, constraints.shape[1]), dtype=np.float64)
        constraints = sp.vstack([constraints, empty_row], format="csr")
        lower = np.concatenate([lower, [1.0]])
        upper = np.concatenate([upper, [0.0]])
    return NeuralHZPhaseReduction(
        value_center=center,
        value_matrix=value,
        constraint_matrix=constraints,
        row_lower=lower,
        row_upper=upper,
        bin_sources=current_bin,
        fixes=tuple(fixes),
        proven_empty=proven_empty,
    )


def _without_index(length: int, index: int) -> np.ndarray:
    keep = np.ones(int(length), dtype=bool)
    keep[int(index)] = False
    return keep


def _pivot_record(
    row: sp.csr_matrix,
    pivot_col: int,
    pivot: float,
    rhs: float,
    cont_sources: np.ndarray,
    bin_sources: np.ndarray,
) -> ContinuousElimination:
    cont_ids: list[int] = []
    cont_values: list[float] = []
    bin_ids: list[int] = []
    bin_values: list[float] = []
    n_cont = int(cont_sources.size)
    for column, coefficient in zip(row.indices, row.data):
        column = int(column)
        if column == int(pivot_col):
            continue
        if column < n_cont:
            cont_ids.append(int(cont_sources[column]))
            cont_values.append(float(coefficient))
        else:
            bin_ids.append(int(bin_sources[column - n_cont]))
            bin_values.append(float(coefficient))
    return ContinuousElimination(
        source=int(cont_sources[pivot_col]),
        pivot=float(pivot),
        rhs=float(rhs),
        cont_sources=np.asarray(cont_ids, dtype=np.int64),
        cont_coefficients=np.asarray(cont_values, dtype=np.float64),
        bin_sources=np.asarray(bin_ids, dtype=np.int64),
        bin_coefficients=np.asarray(bin_values, dtype=np.float64),
    )


def project_inactive_equality_factors(
    value_matrix,
    constraint_matrix,
    row_lower,
    row_upper,
    cont_sources,
    bin_sources,
    *,
    max_eliminations: int = 64,
    max_substitution_support: int = 8,
    max_substitution_cells: int = 64,
) -> NeuralHZProjection:
    """Exactly eliminate fill-reducing, output-inactive continuous factors.

    For an equality ``a*x + q(u) = b`` where ``x`` does not occur in the
    represented network output and this is its only defining equality, replace
    ``x`` by ``(b-q(u))/a`` in every other predicate. The latent box
    ``-1 <= x <= 1`` becomes the exactly equivalent two-sided predicate
    ``b-|a| <= q(u) <= b+|a|``.

    A candidate is accepted only when the resulting constraint matrix has
    strictly fewer stored nonzeros. Wide substitutions are not attempted, so
    neural-network sparsity cannot be destroyed speculatively. Binary columns
    are never pivoted or removed.
    """
    value = sp.csr_matrix(value_matrix, dtype=np.float64)
    constraints = sp.csr_matrix(constraint_matrix, dtype=np.float64)
    lower = np.asarray(row_lower, dtype=np.float64).reshape(-1).copy()
    upper = np.asarray(row_upper, dtype=np.float64).reshape(-1).copy()
    current_cont = np.asarray(cont_sources, dtype=np.int64).reshape(-1).copy()
    bin_sources = np.asarray(bin_sources, dtype=np.int64).reshape(-1)
    if constraints.shape[0] != lower.size or lower.shape != upper.shape:
        raise ValueError("Neural-HZ predicate row bounds do not match the matrix")
    if value.shape[1] != constraints.shape[1]:
        raise ValueError("Neural-HZ value/predicate latent widths do not match")
    if value.shape[1] != current_cont.size + bin_sources.size:
        raise ValueError("Neural-HZ latent source maps do not match matrix width")
    if max_eliminations < 0:
        raise ValueError("max_eliminations must be non-negative")

    constraints.sum_duplicates()
    constraints.eliminate_zeros()
    value.sum_duplicates()
    value.eliminate_zeros()
    nnz_before = int(constraints.nnz)
    eliminations: list[ContinuousElimination] = []

    # Neural exact-ReLU graphs commonly leave a whole input-factor block where
    # every factor occurs in one equality and nowhere else. Project this leaf
    # block in one sparse transaction; doing the same algebra one column at a
    # time repeatedly rebuilds the full predicate matrix and dominates small
    # verification instances.
    if max_eliminations and current_cont.size:
        equality_rows = (
            np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
        )
        output_used = (
            np.asarray(value[:, : current_cont.size].getnnz(axis=0))
            .reshape(-1)
            .astype(bool)
        )
        csc = constraints[:, : current_cont.size].tocsc()
        selected_columns: list[int] = []
        selected_rows: list[int] = []
        occupied_rows: set[int] = set()
        for column in np.flatnonzero(~output_used):
            start, stop = csc.indptr[column], csc.indptr[column + 1]
            rows = csc.indices[start:stop]
            if rows.size != 1 or not equality_rows[rows[0]]:
                continue
            row = int(rows[0])
            if row in occupied_rows:
                continue
            selected_columns.append(int(column))
            selected_rows.append(row)
            occupied_rows.add(row)
            if len(selected_columns) >= int(max_eliminations):
                break
        if selected_columns:
            batch_records = []
            pivots = []
            right_sides = []
            for column, row in zip(selected_columns, selected_rows):
                pivot_row = constraints.getrow(row)
                pivot = float(pivot_row[0, column])
                rhs = float(lower[row])
                if pivot == 0.0 or not np.isfinite(pivot) or not np.isfinite(rhs):
                    break
                pivots.append(pivot)
                right_sides.append(rhs)
                batch_records.append(
                    _pivot_record(
                        pivot_row,
                        column,
                        pivot,
                        rhs,
                        current_cont,
                        bin_sources,
                    )
                )
            else:
                q_rows = constraints[np.asarray(selected_rows, dtype=np.int64)].tolil()
                for local_row, column in enumerate(selected_columns):
                    q_rows[local_row, column] = 0.0
                q_rows = q_rows.tocsr()
                q_rows.eliminate_zeros()
                keep_rows = np.ones(constraints.shape[0], dtype=bool)
                keep_rows[np.asarray(selected_rows, dtype=np.int64)] = False
                keep_columns = np.ones(constraints.shape[1], dtype=bool)
                keep_columns[np.asarray(selected_columns, dtype=np.int64)] = False
                candidate_constraints = sp.vstack(
                    [constraints[keep_rows], q_rows], format="csr"
                )[:, keep_columns].tocsr()
                candidate_constraints.sum_duplicates()
                candidate_constraints.eliminate_zeros()
                if int(candidate_constraints.nnz) < int(constraints.nnz):
                    constraints = candidate_constraints
                    lower = np.concatenate(
                        [
                            lower[keep_rows],
                            np.asarray(right_sides) - np.abs(pivots),
                        ]
                    )
                    upper = np.concatenate(
                        [
                            upper[keep_rows],
                            np.asarray(right_sides) + np.abs(pivots),
                        ]
                    )
                    value = value[:, keep_columns].tocsr()
                    keep_cont = np.ones(current_cont.size, dtype=bool)
                    keep_cont[np.asarray(selected_columns, dtype=np.int64)] = False
                    current_cont = current_cont[keep_cont]
                    eliminations.extend(batch_records)

    while len(eliminations) < int(max_eliminations) and current_cont.size:
        equality_rows = (
            np.isfinite(lower)
            & np.isfinite(upper)
            & (lower == upper)
        )
        output_used = (
            np.asarray(value[:, : current_cont.size].getnnz(axis=0))
            .reshape(-1)
            .astype(bool)
        )
        csc = constraints[:, : current_cont.size].tocsc()
        candidates: list[tuple[int, int, int, int]] = []
        for column in np.flatnonzero(~output_used):
            start, stop = csc.indptr[column], csc.indptr[column + 1]
            rows = csc.indices[start:stop]
            defining = rows[equality_rows[rows]]
            if defining.size != 1:
                continue
            pivot_row = int(defining[0])
            row_support = int(constraints.indptr[pivot_row + 1] - constraints.indptr[pivot_row] - 1)
            substitutions = int(rows.size - 1)
            if substitutions and (
                row_support > int(max_substitution_support)
                or substitutions * row_support > int(max_substitution_cells)
            ):
                continue
            candidates.append((substitutions * row_support, row_support, int(column), pivot_row))
        if not candidates:
            break

        accepted = False
        for _, _, pivot_col, pivot_row in sorted(candidates):
            pivot_constraint = constraints.getrow(pivot_row)
            pivot = float(pivot_constraint[0, pivot_col])
            rhs = float(lower[pivot_row])
            if pivot == 0.0 or not np.isfinite(pivot) or not np.isfinite(rhs):
                continue
            q = pivot_constraint.tolil(copy=True)
            q[0, pivot_col] = 0.0
            q = q.tocsr()
            q.eliminate_zeros()

            ratios = constraints[:, pivot_col].tocsr() * (1.0 / pivot)
            substituted = (constraints - ratios @ q).tocsr()
            substituted.sum_duplicates()
            substituted.eliminate_zeros()
            shifts = np.asarray(ratios.toarray(), dtype=np.float64).reshape(-1) * rhs
            shifted_lower = lower - shifts
            shifted_upper = upper - shifts

            keep_rows = _without_index(constraints.shape[0], pivot_row)
            augmented = sp.vstack([substituted[keep_rows], q], format="csr")
            candidate_lower = np.concatenate(
                [shifted_lower[keep_rows], [rhs - abs(pivot)]]
            )
            candidate_upper = np.concatenate(
                [shifted_upper[keep_rows], [rhs + abs(pivot)]]
            )
            keep_columns = _without_index(constraints.shape[1], pivot_col)
            candidate_constraints = augmented[:, keep_columns].tocsr()
            candidate_constraints.sum_duplicates()
            candidate_constraints.eliminate_zeros()
            if not np.all(np.isfinite(candidate_constraints.data)):
                continue
            if int(candidate_constraints.nnz) >= int(constraints.nnz):
                continue

            eliminations.append(
                _pivot_record(
                    pivot_constraint,
                    pivot_col,
                    pivot,
                    rhs,
                    current_cont,
                    bin_sources,
                )
            )
            constraints = candidate_constraints
            lower = candidate_lower
            upper = candidate_upper
            value = value[:, keep_columns].tocsr()
            current_cont = current_cont[_without_index(current_cont.size, pivot_col)]
            accepted = True
            break
        if not accepted:
            break

    return NeuralHZProjection(
        value_matrix=value,
        constraint_matrix=constraints,
        row_lower=lower,
        row_upper=upper,
        cont_sources=current_cont,
        eliminations=tuple(eliminations),
        nnz_before=nnz_before,
        nnz_after=int(constraints.nnz),
    )


def reconstruct_continuous_factors(
    final_cont_values,
    final_bin_values,
    cont_sources,
    bin_sources,
    eliminations: tuple[ContinuousElimination, ...],
) -> dict[int, float]:
    """Reconstruct one original continuous assignment, in reverse pivots."""
    cont_values = {
        int(source): float(value)
        for source, value in zip(cont_sources, final_cont_values)
    }
    binary_values = {
        int(source): float(value)
        for source, value in zip(bin_sources, final_bin_values)
    }
    for elimination in reversed(eliminations):
        q_value = sum(
            float(coefficient) * cont_values[int(source)]
            for source, coefficient in zip(
                elimination.cont_sources, elimination.cont_coefficients
            )
        )
        q_value += sum(
            float(coefficient) * binary_values[int(source)]
            for source, coefficient in zip(
                elimination.bin_sources, elimination.bin_coefficients
            )
        )
        cont_values[int(elimination.source)] = (
            float(elimination.rhs) - q_value
        ) / float(elimination.pivot)
    return cont_values
