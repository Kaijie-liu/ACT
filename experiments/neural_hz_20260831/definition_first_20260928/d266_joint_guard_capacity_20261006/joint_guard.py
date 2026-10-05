"""Default-off joint guard/residual capacity on one certified original H.

Numeric scalars do NOT authenticate a source, model, phase or bound.  The
caller must supply same-source witnesses for every premise in CONTRACT.md.
This is neither a domain implementation nor a source construction bypass.
All arithmetic uses the existing charged, 512-bit checked Fraction kernel.
"""
from fractions import Fraction

from experiments.neural_hz_20260831.definition_first_20260928.d265_joint_source_20261006 import jp_certificate as jp


Rejected = jp.Rejected
_ZERO = Fraction(0)


def certify(a, b, tau, r_bounds, joint_upper, independent_upper, e_upper,
            child_upper, *, budget, enabled=False):
    """Strengthen the old physical row; no solver, new factors or source scan.

    Arguments follow the old certificate, with a same-source e upper bound.
    Exact output pairs retain the original six-coordinate ordering.  The
    sufficient redundancy witness is recomputed for the strengthened rhs;
    failure of that test is still only `not_excluded`, never a strict gain.
    """
    if type(enabled) is not bool:
        raise Rejected('joint guard enabled must be bool')
    if not enabled:
        return None
    old = jp.certify(a, b, tau, r_bounds, joint_upper, independent_upper,
                     child_upper, budget=budget, enabled=True)
    m = jp._Exact(budget)
    a1, a2 = (m.scalar(v) for v in a)
    b1, b2 = (m.scalar(v) for v in b)
    lr, ur = (m.scalar(v) for v in r_bounds)
    t, ue = m.scalar(tau), m.scalar(e_upper)
    mu = m.half(m.add(lr, ur))
    c = m.add(t, mu)
    lk10 = m.sub(m.minimum(_ZERO, m.add(a1, b1)), m.maximum(_ZERO, a2))
    uk01 = m.add(m.neg(m.minimum(_ZERO, a1)), m.maximum(_ZERO, m.add(a2, b2)))
    eta10 = m.maximum(m.sub(m.neg(m.add(lr, lk10)), t), _ZERO)
    eta01 = m.maximum(m.sub(m.add(ur, uk01), t), _ZERO)
    a10 = m.sub(m.neg(lk10), c)
    a01 = m.sub(m.add(uk01, mu), t)
    be = m.twice(m.maximum(ue, _ZERO))
    # Three already checked, small exact records are decoded after charging.
    budget.charge(24, entries=12)
    j = m.checked(Fraction(*old['J']))
    old_rhs = m.checked(Fraction(*old['row_rhs']))
    support = m.checked(Fraction(*old['support_witness']['upper']))
    old_payment = m.add(m.add(j, eta10), eta01)
    guard_joint = m.maximum(j, m.add(a10, be), m.add(a01, be))
    gstar = m.minimum(old_payment, guard_joint)
    reduction = m.sub(old_payment, gstar)
    rhs = m.sub(old_rhs, reduction)
    redundant = m.le(support, rhs)
    margin = m.sub(rhs, support)
    # Account for the shallow copied records and added scalar metadata.
    budget.charge(64, entries=64)
    witness = dict(old['support_witness'])
    witness['rhs_minus_support'] = m.pair(margin)
    result = dict(old)
    result.update(row_rhs=m.pair(rhs), old_row_rhs=m.pair(old_rhs),
                  old_guard_payment=m.pair(old_payment),
                  joint_guard_payment=m.pair(guard_joint),
                  Gstar=m.pair(gstar), rhs_reduction=m.pair(reduction),
                  redundant=redundant,
                  status='excluded' if redundant else 'not_excluded',
                  support_witness=witness,
                  source_authenticated=False, new_domain_qualified=False)
    return result
