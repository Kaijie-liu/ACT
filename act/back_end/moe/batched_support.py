"""Opt-in multi-objective support proposals over one guarded SparseHZono.

Shares A/E across objective columns; it neither merges private factors nor drops
guards. Only CPU synthetic controls are admitted in V1. No native solver, GPU
execution, production dispatch replacement or complete MoE SAFE status.
"""
import copy
from fractions import Fraction
import math
import time

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.solver.hz_lp_export import export
from act.back_end.moe.check_batched_support import SCHEMA, check_batch, validated_records
from scoped_source.rowwise_bound import clock, identity, rational
from scoped_source.rowwise_native import _evaluate


def prepare_batch(hz, queries, *, context, deadline):
    """Keep the HZ factor layout; project each requested property exactly."""
    tick = clock(deadline)
    if (not 1 <= hz.n_cont + hz.n_bin <= 128 or not 1 <= hz.n_out <= 128
            or hz.n_eq + hz.n_ineq > 256 or not 1 <= len(queries) <= 8):
        raise ValueError("finite control capacity exceeded")
    # Do not let the existing exporter silently sum noncanonical duplicates.
    for key in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        matrix = getattr(hz, key)
        if not sp.isspmatrix_csr(matrix):
            raise ValueError("canonical CSR input required")
        matrix.check_format(full_check=True)
        if not np.isfinite(matrix.data).all():
            raise ValueError("nonfinite HZ coefficient")
        for row in range(matrix.shape[0]):
            tick()
            ids = matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]
            if np.any(ids[1:] <= ids[:-1]):
                raise ValueError("duplicate/unsorted HZ coefficient")
    if any(not np.isfinite(getattr(hz, k)).all() for k in ("c", "b", "ub")):
        raise ValueError("nonfinite HZ vector")
    seed = export(hz, [0] * hz.n_out, sparse=True)
    source = seed["source"]
    base = {key: value for key, value in seed["lp"].items() if key not in ("c", "offset")}
    roster = []
    for query in queries:
        tick()
        if len(query["q"]) != hz.n_out or query["side"] not in ("min", "max"):
            raise ValueError("objective dimension/side mismatch")
        q = [rational(value) for value in query["q"]]
        offset = rational(query["offset"])
        sign = 1 if query["side"] == "min" else -1
        coefficients = [Fraction(0)] * (hz.n_cont + hz.n_bin)
        for name, shift in (("Gc", 0), ("Gb", hz.n_cont)):
            matrix = source[name]
            for row, weight in enumerate(q):
                tick()
                for p in range(matrix["indptr"][row], matrix["indptr"][row + 1]):
                    coefficients[shift + matrix["indices"][p]] += sign * weight * rational(matrix["data"][p])
        constant = sign * (offset + sum((weight * rational(value)
                                        for weight, value in zip(q, source["c"])), Fraction(0)))
        roster.append({"id": query["id"], "q": list(map(str, q)), "offset": str(offset),
                       "side": query["side"], "c": list(map(str, coefficients)), "constant": str(constant)})
    batch = {"schema": SCHEMA, "context": copy.deepcopy(context), "source": source,
             "source_sha256": seed["source_sha256"], "base": base, "queries": roster,
             "relaxation": seed["relaxation"], "n_relaxed_binaries": seed["n_relaxed_binaries"]}
    validated_records(batch, expected_batch_sha256=identity(batch), deadline=deadline)
    tick()
    return batch


def _candidate_columns(base, objectives, constants, *, deadline):
    """Floating proposals only. One CPU sparse matrix times many dense columns."""
    tick = clock(deadline)
    if torch.get_num_threads() != 1:
        raise ValueError("frozen control requires one torch CPU thread")

    def tensor(values):
        result = torch.tensor(values, dtype=torch.float64, device="cpu")
        if not bool(torch.isfinite(result).all()):
            raise ValueError("nonfinite tensor conversion")
        tick()
        return result

    def matrix(name):
        value = base[name]
        row_ids = [row for row in range(value["shape"][0])
                   for _ in range(value["indptr"][row], value["indptr"][row + 1])]
        indices = torch.tensor([row_ids, value["indices"]], dtype=torch.int64, device="cpu")
        values = tensor([float(rational(v)) for v in value["data"]])
        return torch.sparse_coo_tensor(indices, values, tuple(value["shape"]),
                                       dtype=torch.float64, device="cpu").coalesce()

    a, e = matrix("A"), matrix("E")
    at, et = a.transpose(0, 1).coalesce(), e.transpose(0, 1).coalesce()
    c = tensor(objectives).T.contiguous()
    d = tensor(constants)
    b, h = tensor(base["b"])[:, None], tensor(base["h"])[:, None]
    low, high = tensor(base["lower"])[:, None], tensor(base["upper"])[:, None]
    k = c.shape[1]
    y = torch.zeros((a.shape[0], k), dtype=torch.float64, device="cpu")
    t = torch.zeros((e.shape[0], k), dtype=torch.float64, device="cpu")
    best_y, best_t = y.clone(), t.clone()
    best = torch.full((k,), -math.inf, dtype=torch.float64, device="cpu")
    # Evaluate zero, followed by every one of the 128 registered updates.
    with torch.no_grad():
        for iteration in range(129):
            tick()
            r = c - torch.sparse.mm(at, y) - torch.sparse.mm(et, t)
            value = d + (b * y).sum(0) + (h * t).sum(0) + torch.minimum(r * low, r * high).sum(0)
            if not bool(torch.isfinite(value).all() and torch.isfinite(r).all()
                        and torch.isfinite(y).all() and torch.isfinite(t).all()):
                raise ValueError("nonfinite candidate iteration")
            improved = value > best
            best = torch.where(improved, value, best)
            best_y[:, improved], best_t[:, improved] = y[:, improved], t[:, improved]
            if iteration == 128:
                break
            box_argmin = torch.where(r > 0, low, torch.where(r < 0, high, (low + high) / 2))
            step = 0.125 / math.sqrt(iteration + 1)
            y = torch.minimum(y + step * (b - torch.sparse.mm(a, box_argmin)), torch.zeros_like(y, device="cpu"))
            t = t + step * (h - torch.sparse.mm(e, box_argmin))
    tick()
    return best_y.T.tolist(), best_t.T.tolist()


def propose_batch(batch, *, expected_batch_sha256, deadline, device="cpu"):
    """Return untrusted candidate records; an exception produces no acceptance."""
    tick = clock(deadline)
    if device != "cpu":
        raise ValueError("CUDA/non-CPU execution not admitted by frozen control protocol")
    records = validated_records(batch, expected_batch_sha256=expected_batch_sha256, deadline=deadline)
    objectives = [[float(rational(v)) for v in lp["c"]] for _, lp in records]
    constants = [float(rational(lp["offset"])) for _, lp in records]
    ys, ts = _candidate_columns(batch["base"], objectives, constants, deadline=deadline)
    if len(ys) != len(records) or len(ts) != len(records):
        raise ValueError("partial candidate column roster")
    entries = []
    for (query, lp), y, t in zip(records, ys, ts):
        tick()
        candidate = {"lp_sha256": identity(lp), "inequality_dual": y, "equality_dual": t}
        proposed = _evaluate(lp, candidate, tick)
        zero = {"lp_sha256": identity(lp), "inequality_dual": [0] * len(lp["b"]),
                "equality_dual": [0] * len(lp["h"])}
        zero_value = _evaluate(lp, zero, tick)
        if zero_value > proposed:
            candidate, proposed = zero, zero_value
        candidate["claimed_lower_bound"] = str(proposed)
        entries.append({"id": query["id"], "certificate": candidate,
                        "zero_candidate_lower_bound": str(zero_value)})
    if identity(batch) != expected_batch_sha256:
        raise ValueError("support batch changed during proposal")
    tick()
    return {"batch_sha256": expected_batch_sha256, "entries": entries,
            "algorithm": "projected_dual_subgradient_multiobjective_v1",
            "iterations": 128, "dtype": "float64", "device": "cpu"}


def support_batch(hz, queries, *, context, deadline, device="cpu"):
    """Cooperatively budgeted opt-in API; not a whole-request supervisor."""
    start = time.monotonic()
    tick = clock(deadline)
    if device != "cpu":
        raise ValueError("only CPU controls admitted; no CUDA initialization")
    batch = prepare_batch(hz, queries, context=context, deadline=deadline)
    prepared = time.monotonic()
    anchored_hash = identity(batch)
    candidates = propose_batch(batch, expected_batch_sha256=anchored_hash, deadline=deadline, device=device)
    proposed = time.monotonic()
    accepted = check_batch(batch, candidates, expected_batch_sha256=anchored_hash, deadline=deadline)
    checked = time.monotonic()
    accepted["cost_seconds"] = {"preparation": prepared - start, "proposal": proposed - prepared,
                                "checking": checked - proposed, "total": checked - start}
    tick()
    return {"batch": batch, "candidates": candidates, "accepted": accepted}
