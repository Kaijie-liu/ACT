"""Window-preserving exact product certificate with either-operand dyadic proof."""

from fractions import Fraction
import numpy as np


def window_products(values, ratios, *, pool):
    """Strict owned float64 vectors; external cached mantissa claims are not accepted.

    Caller separately reserves two gathers per hit. Common12*n+4 accounts for
    two abs maps, four min/max reductions and four scalar comparisons, right
    frexp/equality/all, multiplication and output abs/lower-window check.
    Non-all-right-dyadic rows pay4*n for left frexp/equality, OR and flag walk.
    General products pay64 each BEFORE exact rational arithmetic.
    """
    if (type(values) is not np.ndarray or type(ratios) is not np.ndarray
            or values.dtype != np.dtype(np.float64) or ratios.dtype != np.dtype(np.float64)
            or values.ndim != 1 or ratios.shape != values.shape):
        raise ValueError('exact certifier requires matching owned float64 vectors')
    n = len(values)
    if not n:
        return np.empty(0, bool), np.empty(0, np.float64), {'hits': 0, 'general': 0, 'right_batch_fast': True}
    pool.charge('product_common', 12*n + 4)
    av, ar = np.abs(values), np.abs(ratios)
    # Min/max propagate NaN; the positive comparisons reject NaN, infinity,
    # zero, subnormal and out-of-domain input without separate full passes.
    if not (2.**-20 <= float(av.min()) and float(av.max()) <= 2.**40
            and 2.**-60 <= float(ar.min()) and float(ar.max()) <= 1.):
        raise ValueError('product operand outside unchanged normal window')
    right_dyadic = np.frexp(ar)[0] == .5
    fast = bool(np.all(right_dyadic))
    products = values * ratios
    # |ratio|<=1 proves the upper output window. Smallest possible nonzero
    # result is2^-80, so neither normal product nor dyadic scaling underflows.
    good = np.abs(products) >= 2.**-20
    general = 0
    if not fast:
        pool.charge('product_left_classification', 4*n)
        either_dyadic = right_dyadic | (np.frexp(av)[0] == .5)
        for i in range(n):
            if not either_dyadic[i]:
                pool.charge('product_general_exact', 64)
                exact = Fraction(float(products[i])) == Fraction(float(values[i])) * Fraction(float(ratios[i]))
                good[i] = bool(good[i]) and exact
                general += 1
    return good, products, {'hits': n, 'general': general, 'right_batch_fast': fast}
