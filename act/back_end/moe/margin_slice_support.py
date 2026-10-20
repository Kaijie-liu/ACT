"""Default-off exact slab overlay, using the existing CPU candidate kernel.

No new optimizer/supervisor or production dispatch. All rational slab data and
original coefficients are rechecked by a standard-library receiver.
"""
from fractions import Fraction as F

from act.back_end.moe.margin_hz_source import _csr, _claimed_lower
from act.back_end.moe.gate_margin_rows import snapshot
from scoped_source.hz_request_store import require
from scoped_source.check_gate_margin_rows import MARGIN_PROFILE
from scoped_source.check_margin_hz_source import parameter
from scoped_source import check_margin_slices as checker


def propose(bound, property_bytes, *, expected_property, cuts, index, goals, work,
            provider="subgradient_128", query_seconds=None, observe=None):
    require(provider in ("subgradient_128", "highs")
            and (provider == "highs" or (query_seconds is None and observe is None)),
            "explicit fixed slice provider")
    parsed, plan, identity = checker.prepare(
        bound.payload, property_bytes, expected_source=bound.sha256,
        expected_property=expected_property, cuts=cuts, work=work)
    require(type(index) is int and 0 <= index < len(plan["cells"]), "slice index")
    require(type(goals) is tuple and 1 <= len(goals) <= 8, "finite slice goal roster")
    goals = tuple(checker.goal(g, work) for g in goals)
    require(len(set(goals)) == len(goals), "unique slice goals")
    require(snapshot(bound.output_hz, work, profile=MARGIN_PROFILE) == parsed["obj"]["augmented"],
            "live slice source changed before proposal")
    hz, cell = parsed["joint"], plan["cells"][index]
    n = hz["nc"]+hz["nb"]
    a, e = _csr(hz, "Auc", "Aub", work), _csr(hz, "Ac", "Ab", work)
    for sign in (1, -1):
        work.visit("row_visits")
        for j, v in sorted(parsed["rows"]["m"][1].items()):
            work.visit("coefficient_visits")
            a["indices"].append(j); a["data"].append(checker.stored(work.exact(sign*v)))
        a["indptr"].append(len(a["data"]))
    a["shape"][0] += 2
    base = dict(matrix_format="csr_v1", A=a, E=e,
                b=list(map(str, hz["ub"]))+cell["slab_rhs"], h=list(map(str, hz["b"])),
                lower=[-1]*n, upper=[1]*n)
    exact, objectives, constants = [], [], []
    for spec in goals:
        center, sparse = checker.objective(parsed, spec, work)
        work.visit("coefficient_visits", n)
        row = [sparse.get(j, F(0)) for j in range(n)]
        exact.append((center, row))
        if provider == "subgradient_128":
            objectives.append([float(v) for v in row]); constants.append(float(center))
    # Only the proposal receives rounded copies; exact RHS/CSR remain authoritative.
    if provider == "highs":
        from act.back_end.moe.margin_lp_candidates import columns
        ys, zs, native_report = columns(base, exact, work=work, query_seconds=query_seconds, observe=observe)
    else:
        from act.back_end.moe.batched_support import _candidate_columns
        native = dict(base, b=[float(F(v)) for v in base["b"]], h=[float(F(v)) for v in base["h"]])
        ys, zs = _candidate_columns(native, objectives, constants, deadline=work.deadline)
    require(len(ys) == len(zs) == len(goals), "complete slice candidate columns")
    supports, choices = [], []
    for spec, (center, row), y, z in zip(goals, exact, ys, zs):
        lp = dict(base, c=list(map(str, row)), offset=str(center))
        lower = _claimed_lower(lp, y, z, work)
        zero_y, zero_z = [0]*len(y), [0]*len(z)
        zero_lower = _claimed_lower(lp, zero_y, zero_z, work)
        if zero_lower > lower:
            y, z, lower = zero_y, zero_z, zero_lower
            choices.append("CHECKED_ZERO_DUAL_STRONGER")
        else:
            choices.append("NATIVE_DUAL")
        supports.append(dict(goal=[spec[0], checker.stored(spec[1])],
                             y=[str(parameter(v, work)) for v in y[:-2]],
                             z=[str(parameter(v, work)) for v in z],
                             slice_duals=[str(parameter(v, work)) for v in y[-2:]],
                             lower=checker.stored(lower)))
    record = dict(schema=checker.CERT, source=bound.sha256, property=expected_property,
                  partition=identity, cell=cell, supports=supports, grade=checker.GRADE)
    receipt = checker.check(bound.payload, property_bytes, record, expected_source=bound.sha256,
                            expected_property=expected_property, cuts=cuts, expected_index=index,
                            expected_goals=goals, work=work)
    require(snapshot(bound.output_hz, work, profile=MARGIN_PROFILE) == parsed["obj"]["augmented"],
            "live slice source changed during proposal")
    work.tick()
    result = dict(record=record, receipt=receipt, native_candidate_calls=1, objective_slots=len(goals),
                iterations=128, device="cpu", hard_budget_supervision=False)
    if provider == "highs":
        result.update(native_candidate_calls=len(goals), iterations=None,
                      native_report=native_report, selected_candidates=choices)
    return result
