"""Exact-rational lower-bound checking for finite-box linear programs.

LP semantics: min c*x + offset, A*x <= b, E*x = h, lower <= x <= upper.
JSON floats denote their exact binary rational values, not decimal intentions.
This checks the supplied LP only: it does not certify network lowering, MILP
search, deployment arithmetic, or feasibility. No solver is used by check().
"""
from fractions import Fraction
import hashlib
import json
import math


def rational(value):
    if type(value) is int:
        return Fraction(value)
    if type(value) is float and math.isfinite(value):
        return Fraction.from_float(value)
    if type(value) is str:
        return Fraction(value)
    raise ValueError("finite int/float or rational string required")


def identity(lp):
    return hashlib.sha256(json.dumps(lp, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def check(lp, certificate):
    """Check signed dual multipliers, exact residual correction and claim."""
    if certificate["lp_sha256"] != identity(lp):
        raise ValueError("LP identity mismatch")
    c, lower, upper = ([rational(v) for v in lp[k]] for k in ("c", "lower", "upper"))
    n = len(c)
    if not n or len(lower) != n or len(upper) != n or any(a > b for a, b in zip(lower, upper)):
        raise ValueError("invalid finite box")
    residual = c[:]
    bound = rational(lp.get("offset", 0))
    for matrix_key, rhs_key, dual_key, inequality in (
        ("A", "b", "inequality_dual", True), ("E", "h", "equality_dual", False)
    ):
        matrix, rhs = lp.get(matrix_key, []), lp.get(rhs_key, [])
        dual = [rational(v) for v in certificate[dual_key]]
        if len(matrix) != len(rhs) or len(matrix) != len(dual):
            raise ValueError("row/dual shape mismatch")
        for row, right, multiplier in zip(matrix, rhs, dual):
            if len(row) != n or (inequality and multiplier > 0):
                raise ValueError("invalid row width or dual sign")
            bound += multiplier * rational(right)
            for j, value in enumerate(row):
                residual[j] -= multiplier * rational(value)
    # No approximate stationarity is assumed: the residual is minimized
    # exactly on the finite box. For y<=0, y*A*x >= y*b on feasible inputs.
    bound += sum((r * (lo if r >= 0 else hi) for r, lo, hi in zip(residual, lower, upper)), Fraction(0))
    claimed = rational(certificate["claimed_lower_bound"])
    if claimed > bound:
        raise ValueError("claimed lower bound exceeds checked bound")
    return {"status": "CHECKED", "lp_sha256": identity(lp),
            "checked_lower_bound": str(bound), "claimed_lower_bound": str(claimed),
            "residual": [str(r) for r in residual],
            "scope": "Exact rational supplied finite-box LP only; no feasibility or network/MILP proof."}


def propose(lp, *, time_limit=None):
    """Untrusted SciPy proposal, accepted only after the independent checker."""
    from scipy.optimize import linprog
    # A floating proposal may approximate exact rational objective sums.
    # check() still evaluates the original coefficients, including residuals.
    vector = lambda values: [float(rational(v)) for v in values]
    matrix = lambda values: [vector(row) for row in values] or None
    result = linprog(vector(lp["c"]), A_ub=matrix(lp.get("A", [])),
                     b_ub=vector(lp.get("b", [])) or None,
                     A_eq=matrix(lp.get("E", [])), b_eq=vector(lp.get("h", [])) or None,
                     bounds=list(zip(vector(lp["lower"]), vector(lp["upper"]))), method="highs",
                     options={} if time_limit is None else {"time_limit": float(time_limit)})
    if not result.success:
        raise ValueError("proposal solver did not complete")
    candidate = {"lp_sha256": identity(lp),
                 "inequality_dual": [min(0., float(v)) for v in result.ineqlin.marginals],
                 "equality_dual": [float(v) for v in result.eqlin.marginals],
                 "claimed_lower_bound": "0"}
    # Obtain a definitely admissible claim without trusting the primal optimum:
    # first use the box bound plus signed dual terms and residual box radius.
    # An internal exact calculation is rechecked by check(), not by the solver.
    residual = [rational(v) for v in lp["c"]]
    bound = rational(lp.get("offset", 0))
    for M, b, key in ((lp.get("A", []), lp.get("b", []), "inequality_dual"),
                      (lp.get("E", []), lp.get("h", []), "equality_dual")):
        for row, rhs, d in zip(M, b, candidate[key]):
            bound += rational(d) * rational(rhs)
            residual = [r - rational(d) * rational(a) for r, a in zip(residual, row)]
    bound += sum((r * rational(lo if r >= 0 else hi)
                  for r, lo, hi in zip(residual, lp["lower"], lp["upper"])), Fraction(0))
    candidate["claimed_lower_bound"] = str(bound)
    check(lp, candidate)
    return candidate
