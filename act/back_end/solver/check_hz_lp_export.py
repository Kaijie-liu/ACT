"""Independent scalar reconstruction of serialized HZ-to-LP lowering.

Does not import the exporter, SparseHZono, SciPy, or a solver. It establishes
equivalence to the supplied HZ's continuous relaxation, NOT network-to-HZ
soundness or correctness of the original floating-point propagation.
"""
from fractions import Fraction

from act.back_end.solver.lp_certificate import check, identity, rational


def _entries(value):
    nr, nc = value["shape"]
    data, indices, ptr = value["data"], value["indices"], value["indptr"]
    if nr < 0 or nc < 0 or len(ptr) != nr + 1 or ptr[0] != 0 or ptr[-1] != len(data) or len(indices) != len(data):
        raise ValueError("invalid CSR shape")
    result = {}
    for i in range(nr):
        if not 0 <= ptr[i] <= ptr[i + 1] <= len(data):
            raise ValueError("invalid CSR row pointer")
        previous = -1
        for k in range(ptr[i], ptr[i + 1]):
            j = indices[k]
            if not previous < j < nc:
                raise ValueError("noncanonical CSR columns")
            previous = j
            result[i, j] = rational(data[k])
    return (nr, nc), result


def check_export(record, certificate=None, *, expected_source_sha256):
    source, lp = record["source"], record["lp"]
    if identity(source) != expected_source_sha256 or record["source_sha256"] != expected_source_sha256:
        raise ValueError("HZ source identity mismatch")
    if record["relaxation"] != "BINARY_MINUS_PLUS_ONE_TO_CONTINUOUS_BOX":
        raise ValueError("missing binary relaxation declaration")
    matrices = {key: _entries(source[key]) for key in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")}
    outputs, nc = matrices["Gc"][0]
    outputs_b, nb = matrices["Gb"][0]
    if outputs_b != outputs or len(source["c"]) != outputs or len(record["q"]) != outputs:
        raise ValueError("HZ output dimension mismatch")
    n = nc + nb
    if record["n_relaxed_binaries"] != nb or lp["lower"] != [-1] * n or lp["upper"] != [1] * n:
        raise ValueError("factor box or binary count changed")
    q = [rational(v) for v in record["q"]]
    objective = [Fraction(0)] * n
    for key, shift in (("Gc", 0), ("Gb", nc)):
        for (i, j), value in matrices[key][1].items():
            objective[shift + j] += q[i] * value
    constant = rational(record["offset"]) + sum((w * rational(c) for w, c in zip(q, source["c"])), Fraction(0))
    if [rational(v) for v in lp["c"]] != objective or rational(lp["offset"]) != constant:
        raise ValueError("objective or output center changed")
    for left, right, rhs, target, target_rhs in (("Ac", "Ab", "b", "E", "h"),
                                                 ("Auc", "Aub", "ub", "A", "b")):
        (nr, width), c_entries = matrices[left]
        b_shape, b_entries = matrices[right]
        if width != nc or b_shape != (nr, nb) or len(source[rhs]) != nr or len(lp[target]) != nr:
            raise ValueError("constraint shape or row omission")
        if [rational(v) for v in source[rhs]] != [rational(v) for v in lp[target_rhs]]:
            raise ValueError("constraint RHS changed")
        for i, row in enumerate(lp[target]):
            if len(row) != n:
                raise ValueError("constraint width changed")
            for j, value in enumerate(row):
                expected = c_entries.get((i, j), 0) if j < nc else b_entries.get((i, j - nc), 0)
                if rational(value) != expected:
                    raise ValueError("constraint coefficient/sign changed")
    bound = check(lp, certificate) if certificate is not None else None
    return {"status": "CHECKED", "source_sha256": expected_source_sha256,
            "lp_sha256": identity(lp), "n_factors": n, "n_relaxed_binaries": nb,
            "bound": bound,
            "scope": "Given HZ to continuous LP relaxation and optional rational bound only; not network propagation or MILP proof."}
