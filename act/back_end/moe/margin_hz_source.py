"""Opt-in native HZ source/candidate adapter for margin-conditioned support.

No production dispatch, new propagation rule, network proof or SAFE transition.
Existing joint expert composition and untrusted CPU candidate search are reused.
The new interface retains router outputs and checks original-coefficient views.
"""
from dataclasses import dataclass
from fractions import Fraction as F

import numpy as np
import scipy.sparse as sp

from act.back_end.moe.weighted_top2 import shared_input_pair_hz, _remap_columns
from act.back_end.moe.gate_margin_rows import metadata, snapshot
from act.back_end.solver.solver_hz import SparseHZono
from scoped_source.hz_request_store import decode, require
from scoped_source.check_gate_margin_rows import (
    BINDING, GATE_SEMANTICS, MARGIN_PROFILE, capacity, freeze, validate_binding,
)
from scoped_source import check_margin_hz_source as checker


@dataclass(frozen=True)
class MarginHZSource:
    pair_hz: object
    output_hz: object
    payload: bytes
    sha256: str


def bind_source(*, entry, expert_a, expert_b, router, pair, scope,
                router_eq_rows, router_le_rows, work):
    """Keep P unchanged and append router scores in the original shared frame.

    Caller supplies request/model/input/guard identity and retained router-row
    maps. They do not prove the upstream network or tie-legal route exclusions.
    Independent inspection verifies the given-source composition itself.
    """
    work.tick()
    require(type(pair) is tuple and len(pair) == 2
            and all(type(i) is int for i in pair) and 0 <= pair[0] < pair[1] < router.n_out,
            "canonical native pair")
    original = dict(entry=entry, a=expert_a, b=expert_b, router=router)
    # Reject malformed CSR before native prefix subtraction or column remapping.
    captured = {k: snapshot(v, work, profile=MARGIN_PROFILE) for k, v in original.items()}
    factors, rows_cap, outputs = capacity(MARGIN_PROFILE)
    require(expert_a.n_cont+expert_b.n_cont-entry.n_cont
            +expert_a.n_bin+expert_b.n_bin-entry.n_bin <= factors
            and expert_a.n_out+expert_b.n_out+2 <= outputs
            and expert_a.n_eq+expert_a.n_ineq+expert_b.n_eq+expert_b.n_ineq
            -entry.n_eq-entry.n_ineq <= rows_cap, "joint capacity before construction")
    base = shared_input_pair_hz(entry, expert_a, expert_b)
    live = dict(**original, joint=base.output_hz, input=base.input_hz)
    captured.update({k: snapshot(live[k], work, profile=MARGIN_PROFILE) for k in ("joint", "input")})
    value = dict(schema=BINDING, gate_semantics=GATE_SEMANTICS,
                 context=dict(request=scope["request"], domain=scope["domain"], pair=list(pair)),
                 sources=captured,
                 metadata=metadata(base), router_eq_rows=list(router_eq_rows),
                 router_le_rows=list(router_le_rows))
    # Validate ALL router constraints and shared/private maps before appending.
    raw_base, base_id = freeze(value, work, profile=MARGIN_PROFILE)
    owned_base = decode(raw_base)
    validate_binding(owned_base, expected_hash=base_id, work=work, profile=MARGIN_PROFILE)
    h = base.output_hz
    rows = list(pair)
    augmented = SparseHZono(
        c=np.concatenate((h.c, router.c[rows])),
        Gc=sp.vstack((h.Gc, _remap_columns(router.Gc[rows], np.arange(router.n_cont), h.n_cont)),
                     format="csr"),
        Gb=sp.vstack((h.Gb, _remap_columns(router.Gb[rows], np.arange(router.n_bin), h.n_bin)),
                     format="csr"),
        Ac=h.Ac.copy(), Ab=h.Ab.copy(), b=h.b.copy(),
        Auc=h.Auc.copy(), Aub=h.Aub.copy(), ub=h.ub.copy(),
        frame_id=h.frame_id, exact=h.exact and router.exact,
    )
    bundle = dict(schema=checker.SCHEMA, profile=MARGIN_PROFILE, scope=scope,
                  base=owned_base, augmented=snapshot(augmented, work, profile=MARGIN_PROFILE),
                  router_rows=[h.n_out, h.n_out+1],
                  relaxation=dict(kind="BINARY_MINUS_PLUS_ONE_TO_CONTINUOUS_BOX", count=h.n_bin))
    raw, anchor = freeze(bundle, work, profile=MARGIN_PROFILE)
    checker.inspect(raw, expected_source=anchor, work=work)
    for key, hz in live.items():
        require(snapshot(hz, work, profile=MARGIN_PROFILE) == owned_base["sources"][key],
                "native source changed during capture")
    require(snapshot(augmented, work, profile=MARGIN_PROFILE) == bundle["augmented"],
            "augmented source changed during capture")
    work.tick()
    return MarginHZSource(base, augmented, raw, anchor)


def _csr(hz, continuous, binary, work):
    values, ids, ptr = [], [], [0]
    for cr, br in zip(hz[continuous], hz[binary]):
        work.visit("row_visits")
        for row, shift in ((cr, 0), (br, hz["nc"])):
            for j, value in row.items():
                work.visit("coefficient_visits")
                values.append(str(value)); ids.append(shift+j)
        ptr.append(len(values))
    return dict(shape=[len(ptr)-1, hz["nc"]+hz["nb"]], data=values, indices=ids, indptr=ptr)


def _claimed_lower(lp, y, z, work):
    """Producer arithmetic is metered too; receiver uses original HZ, not this LP."""
    residual = [checker.parameter(v, work) for v in lp["c"]]
    value = checker.parameter(lp["offset"], work)
    for matrix_name, rhs_name, duals in (("A", "b", y), ("E", "h", z)):
        matrix = lp[matrix_name]
        require(len(duals) == len(lp[rhs_name]), "producer dual coverage")
        for i, dual in enumerate(duals):
            work.visit("row_visits")
            dual = checker.parameter(dual, work)
            require(matrix_name == "E" or dual <= 0, "producer dual sign")
            rhs = checker.parameter(lp[rhs_name][i], work)
            value = work.exact(value + work.exact(dual*rhs))
            for p in range(matrix["indptr"][i], matrix["indptr"][i+1]):
                work.visit("coefficient_visits")
                j = matrix["indices"][p]
                a = checker.parameter(matrix["data"][p], work)
                residual[j] = work.exact(residual[j] - work.exact(dual*a))
    for r in residual:
        value = work.exact(value-abs(r))
    # Saved claim uses the original stored-bit contract, not just work_bits.
    return checker.parameter(value, work)


def propose_supports(bound, property_bytes, *, expected_property, goals, work):
    """Existing floating CPU proposals; returns only independently checked supports.

    One caller-owned Work/deadline spans source validation, proposal and exact
    reception. It does not install a hard request supervisor or aggregate margin
    cells. The legacy optimizer itself has no authority to accept a bound.
    """
    from act.back_end.moe.batched_support import _candidate_columns

    work.tick()
    src = checker.inspect_property(bound.payload, property_bytes, expected_source=bound.sha256,
                                   expected_property=expected_property, work=work)
    require(type(goals) is tuple and 1 <= len(goals) <= 8, "caller-owned finite goals")
    require(snapshot(bound.output_hz, work, profile=MARGIN_PROFILE) == src["obj"]["augmented"],
            "live joint source changed before proposal")
    hz = src["joint"]
    n = hz["nc"]+hz["nb"]
    base = dict(matrix_format="csr_v1", A=_csr(hz, "Auc", "Aub", work),
                E=_csr(hz, "Ac", "Ab", work), b=list(map(str, hz["ub"])),
                h=list(map(str, hz["b"])), lower=[-1]*n, upper=[1]*n)
    objectives, constants, exact = [], [], []
    for goal in goals:
        require(type(goal) is tuple and len(goal) == 2, "goal shape")
        center, coeff = checker.objective(src, *goal, work)
        work.visit("coefficient_visits", n)
        row = [coeff.get(j, F(0)) for j in range(n)]
        exact.append((center, row))
        objectives.append([float(v) for v in row]); constants.append(float(center))
    # No unbounded fallback/search or altered 128-step strategy.
    ys, zs = _candidate_columns(base, objectives, constants, deadline=work.deadline)
    require(len(ys) == len(zs) == len(goals), "complete candidate columns")
    candidates = []
    for goal, (center, row), y, z in zip(goals, exact, ys, zs):
        work.tick()
        lp = dict(base, c=list(map(str, row)), offset=str(center))
        lower = _claimed_lower(lp, y, z, work)
        zero_y, zero_z = [0]*len(y), [0]*len(z)
        zero_lower = _claimed_lower(lp, zero_y, zero_z, work)
        if zero_lower > lower:
            y, z, lower = zero_y, zero_z, zero_lower
        candidates.append(dict(t=str(checker.parameter(goal[0], work)),
                               nu=str(checker.parameter(goal[1], work)),
                               y=[str(work.exact(F(v))) for v in y],
                               z=[str(work.exact(F(v))) for v in z], lower=str(lower)))
    receipt = checker.check_supports(bound.payload, property_bytes, candidates,
                                    expected_source=bound.sha256, expected_property=expected_property,
                                    expected_goals=goals, work=work)
    require(snapshot(bound.output_hz, work, profile=MARGIN_PROFILE) == src["obj"]["augmented"],
            "live joint source changed during proposal")
    work.tick()
    return dict(candidates=candidates, receipt=receipt,
                algorithm="projected_dual_subgradient_multiobjective_v1", iterations=128,
                device="cpu", hard_budget_supervision=False)
