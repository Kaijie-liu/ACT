"""Export a SparseHZono support obligation with binaries explicitly relaxed."""
from fractions import Fraction

import scipy.sparse as sp

from act.back_end.solver.lp_certificate import identity, rational


def csr(value):
    matrix = value.copy().tocsr()
    matrix.sum_duplicates()
    matrix.sort_indices()
    return {"shape": list(matrix.shape), "data": matrix.data.tolist(),
            "indices": matrix.indices.tolist(), "indptr": matrix.indptr.tolist()}


def snapshot(hz):
    return {"c": hz.c.tolist(), "b": hz.b.tolist(), "ub": hz.ub.tolist(),
            "frame_id": hz.frame_id, "exact": hz.exact,
            **{key: csr(getattr(hz, key)) for key in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub")}}


def export(hz, q, offset=0, *, sparse=False):
    if len(q) != hz.n_out:
        raise ValueError("support objective width mismatch")
    source = snapshot(hz)
    coefficients = [Fraction(0)] * (hz.n_cont + hz.n_bin)
    for matrix, shift in ((hz.Gc, 0), (hz.Gb, hz.n_cont)):
        matrix = matrix.tocsr()
        for i, weight in enumerate(q):
            for k in range(matrix.indptr[i], matrix.indptr[i + 1]):
                coefficients[shift + int(matrix.indices[k])] += rational(weight) * rational(float(matrix.data[k]))
    constant = rational(offset) + sum((rational(w) * rational(float(c)) for w, c in zip(q, hz.c)), Fraction(0))
    n = len(coefficients)
    convert = csr if sparse else lambda m: m.toarray().tolist()
    lp = {"c": [str(v) for v in coefficients], "offset": str(constant),
          "lower": [-1] * n, "upper": [1] * n,
          "A": convert(sp.hstack([hz.Auc, hz.Aub], format="csr")), "b": hz.ub.tolist(),
          "E": convert(sp.hstack([hz.Ac, hz.Ab], format="csr")), "h": hz.b.tolist()}
    if sparse: lp['matrix_format'] = 'csr_v1'
    return {"source": source, "source_sha256": identity(source), "q": list(q), "offset": offset,
            "relaxation": "BINARY_MINUS_PLUS_ONE_TO_CONTINUOUS_BOX",
            "n_relaxed_binaries": hz.n_bin, "lp": lp}
