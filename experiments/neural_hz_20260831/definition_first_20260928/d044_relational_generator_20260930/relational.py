"""Default-off, exact-rational D042 relational-generator arithmetic.

This is a conditional mathematical component, NOT a verifier, new abstract
domain, model importer, or certificate authority.  The caller must certify the
original shared HZ frame, every supplied bound, each original ReLU relation,
and the binding of an anchor to its original output's original bit.  Tokens
only prevent accidental mixing: their labels and Python identities prove no
network fact.  ``observation`` explicitly accepts such already-certified
premises; it does not establish them.  Original continuous factors, all gate
bits (including both legal zero choices), predicates and decoder stay in the
caller's HZ and are neither replaced nor removed here.

A Pair retains a signed difference AND its right companion, conditioned on
one unchanged original anchor.  Affine transfer uses the fixed left basis
q=d+p, merges identical formal readouts before interval arithmetic, and never
chooses a decomposition from an LP point or result.  ReLU transfer requires
two already-existing original output symbols; it creates no gate or phase
subnetwork.  A formal identity is useful only after the caller binds its
symbols to the same actual source.  Equal labels or overlapping intervals are
not identity certificates.

Only exact Fraction coefficients/endpoints are supported.  Every input and
arithmetic result is limited to 512 bits.  Each affine call admits at most
65,536 observation occurrences and 65,536 total input sparse-term occurrences;
paired transfer applies at most two such bounded combinations.  Scalar
arithmetic/comparison work is O(input occurrences + sparse support), with a
conservative 100 times MAX_SUPPORT bound per transfer; hash/container cost is
additional.  Two output-row copies and immutable carriers are counted as
ordinary storage, not free proof objects.  This is a finite mathematics
qualification contract, NOT native/GPU/full-physical/resource qualification.
No execution budget, solver, imports of model libraries, or numerical search
is introduced.  All public operation functions require explicit opt-in.
"""

from dataclasses import dataclass
from fractions import Fraction


MAX_BITS = 512
MAX_SUPPORT = 65_536
ZERO = Fraction(0)
ONE = Fraction(1)


class KernelError(ValueError):
    """Malformed, inconsistent, or unsupported mathematical premises."""


def _on(enabled):
    if type(enabled) is not bool:
        raise KernelError("enabled must be a bool")
    return enabled


def _f(value):
    if type(value) is not Fraction:
        raise KernelError("exact Fraction values required")
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise KernelError("rational bit limit exceeded")
    return value


def _add(a, b):
    return _f(a + b)


def _neg(a):
    return _f(-a)


def _sub(a, b):
    return _add(a, _neg(b))


def _mul(a, b):
    return _f(a * b)


def _interval(value):
    if type(value) is not tuple or len(value) != 2:
        raise KernelError("an interval is a two-Fraction tuple")
    lo, hi = _f(value[0]), _f(value[1])
    if lo > hi:
        raise KernelError("reversed or contradictory interval")
    return lo, hi


def _states(value):
    if type(value) is not tuple or len(value) != 2:
        raise KernelError("exactly two original-anchor state intervals required")
    return _interval(value[0]), _interval(value[1])


def _label(value):
    if type(value) is not str or not 0 < len(value) <= 256:
        raise KernelError("a bounded nonempty diagnostic label is required")
    return value


@dataclass(frozen=True, eq=False)
class _Frame:
    label: str


@dataclass(frozen=True, eq=False)
class _Symbol:
    frame: _Frame
    label: str


@dataclass(frozen=True, eq=False)
class _Anchor:
    original_output: _Symbol
    label: str

    @property
    def frame(self):
        return self.original_output.frame


@dataclass(frozen=True)
class _Observation:
    anchor: _Anchor
    terms: tuple
    bias: Fraction
    bounds: tuple
    rule: str

    @property
    def frame(self):
        return self.anchor.frame


@dataclass(frozen=True)
class _Pair:
    difference: _Observation
    companion: _Observation

    @property
    def anchor(self):
        return self.difference.anchor


def _frame(value):
    if type(value) is not _Frame:
        raise KernelError("an explicit formal frame token is required")
    _label(value.label)
    return value


def _symbol(value, frame=None):
    if type(value) is not _Symbol:
        raise KernelError("an existing original-readout symbol is required")
    _frame(value.frame)
    _label(value.label)
    if frame is not None and value.frame is not frame:
        raise KernelError("different shared frames")
    return value


def _anchor(value):
    if type(value) is not _Anchor:
        raise KernelError("an original-bit anchor declaration is required")
    _symbol(value.original_output)
    _label(value.label)
    return value


def make_frame(label, *, enabled=False):
    if not _on(enabled):
        return None
    return _Frame(_label(label))


def make_symbol(frame, label, *, enabled=False):
    """Declare an existing formal readout; this does not create a network node."""
    if not _on(enabled):
        return None
    return _Symbol(_frame(frame), _label(label))


def make_anchor(original_output, label, *, enabled=False):
    """Caller declares the existing output's original ReLU bit, not a new bit."""
    if not _on(enabled):
        return None
    return _Anchor(_symbol(original_output), _label(label))


def _normal_terms(frame, terms):
    if type(terms) is not tuple or len(terms) > MAX_SUPPORT:
        raise KernelError("bounded immutable sparse terms required")
    merged = {}
    for item in terms:
        if type(item) is not tuple or len(item) != 2:
            raise KernelError("each sparse term is (symbol, Fraction)")
        symbol, coefficient = _symbol(item[0], frame), _f(item[1])
        merged[symbol] = _add(merged.get(symbol, ZERO), coefficient)
    return tuple((symbol, coefficient) for symbol, coefficient in merged.items()
                 if coefficient != ZERO)


def _create(anchor, terms, bias, bounds, rule):
    anchor, bias = _anchor(anchor), _f(bias)
    terms = _normal_terms(anchor.frame, terms)
    bounds = _states(bounds)
    if not terms:
        if any(not lo <= bias <= hi for lo, hi in bounds):
            raise KernelError("constant identity contradicts supplied endpoints")
        bounds = ((bias, bias), (bias, bias))
    return _Observation(anchor, terms, bias, bounds, rule)


def _obs(value, anchor=None):
    if type(value) is not _Observation:
        raise KernelError("a certified-premise observation is required")
    _anchor(value.anchor)
    if anchor is not None and value.anchor is not anchor:
        raise KernelError("different original-bit anchors")
    _f(value.bias)
    _states(value.bounds)
    if type(value.terms) is not tuple or len(value.terms) > MAX_SUPPORT:
        raise KernelError("invalid sparse observation")
    seen = set()
    for item in value.terms:
        if type(item) is not tuple or len(item) != 2:
            raise KernelError("invalid observation term")
        symbol, coefficient = _symbol(item[0], value.frame), _f(item[1])
        if symbol in seen or coefficient == ZERO:
            raise KernelError("observation terms must already be normalized")
        seen.add(symbol)
    return value


def observation(anchor, terms, bias, state_bounds, *, enabled=False):
    """Accept already-certified bounds, not a string claiming to prove them."""
    if not _on(enabled):
        return None
    return _create(anchor, terms, bias, state_bounds, "caller-certified premise")


def pair(difference, companion, *, enabled=False):
    """The companion is mandatory: it denotes p when difference denotes q-p."""
    if not _on(enabled):
        return None
    difference = _obs(difference)
    return _Pair(difference, _obs(companion, difference.anchor))


def _pair(value, anchor=None):
    if type(value) is not _Pair:
        raise KernelError("a difference-plus-companion pair is required")
    _obs(value.difference, anchor)
    _obs(value.companion, value.anchor)
    return value


def seed_pair(anchor, q, p, g_bounds, f_bounds, difference_bounds, *, enabled=False):
    """Given certified original q=R(g), p=R(f), and L<=g-f<=U, seed D042.

    The original alpha MUST be declared as q's own original bit.  Ranges and
    source/gate bindings are caller-certified premises, not inferred from IDs.
    Contradictory endpoints fail closed; no infeasibility conclusion or bit
    deletion is inferred, including for an unrepresentable empty phase.
    """
    if not _on(enabled):
        return None
    anchor = _anchor(anchor)
    q, p = _symbol(q, anchor.frame), _symbol(p, anchor.frame)
    if anchor.original_output is not q:
        raise KernelError("seed anchor is not q's declared original ReLU bit")
    lg, ug = _interval(g_bounds)
    lf, uf = _interval(f_bounds)
    lower, upper = _interval(difference_bounds)
    if lower > _sub(ug, lf) or upper < _sub(lg, uf):
        raise KernelError("source ranges and difference interval contradict")
    p0 = (max(ZERO, lf), max(ZERO, min(uf, _neg(lower))))
    p1 = (max(ZERO, max(lf, _neg(upper))), max(ZERO, uf))
    d0 = (_neg(p0[1]), _neg(p0[0]))
    d1 = (max(_sub(max(ZERO, lg), p1[1]), min(ZERO, lower)),
          min(_sub(ug, p1[0]), max(ZERO, upper)))
    companion = _create(anchor, ((p, ONE),), ZERO, (p0, p1), "D042 seed companion")
    difference = _create(anchor, ((q, ONE), (p, -ONE)), ZERO, (d0, d1),
                         "D042 seed difference")
    return _Pair(difference, companion)


def _combine(anchor, terms, bias):
    anchor, bias = _anchor(anchor), _f(bias)
    if type(terms) is not tuple or len(terms) > MAX_SUPPORT:
        raise KernelError("bounded immutable affine terms required")
    support = 0
    for item in terms:
        if type(item) is not tuple or len(item) != 2 or type(item[1]) is not _Observation:
            raise KernelError("affine terms are (Fraction, observation)")
        if type(item[1].terms) is not tuple:
            raise KernelError("invalid immutable observation support")
        support += len(item[1].terms)
        if support > MAX_SUPPORT:
            raise KernelError("total affine input support limit exceeded")
    groups = {}
    for coefficient, source in terms:
        coefficient, source = _f(coefficient), _obs(source, anchor)
        key = (source.bias, frozenset(source.terms))
        if key in groups:
            old_coefficient, first, old_bounds = groups[key]
            common = tuple((max(a[0], b[0]), min(a[1], b[1]))
                           for a, b in zip(old_bounds, source.bounds))
            common = _states(common)
            groups[key] = (_add(old_coefficient, coefficient), first, common)
        else:
            groups[key] = (coefficient, source, source.bounds)
    result_terms = []
    result_bias = bias
    result_bounds = [[bias, bias], [bias, bias]]
    for coefficient, source, bounds in groups.values():
        if coefficient == ZERO:
            continue
        result_bias = _add(result_bias, _mul(coefficient, source.bias))
        for symbol, amount in source.terms:
            result_terms.append((symbol, _mul(coefficient, amount)))
        for state in (0, 1):
            lo, hi = bounds[state] if coefficient > ZERO else bounds[state][::-1]
            result_bounds[state][0] = _add(result_bounds[state][0], _mul(coefficient, lo))
            result_bounds[state][1] = _add(result_bounds[state][1], _mul(coefficient, hi))
    return _create(anchor, tuple(result_terms), result_bias,
                   tuple(tuple(value) for value in result_bounds), "fixed signed affine")


def affine(anchor, terms, bias=ZERO, *, enabled=False):
    """Combine same-anchor observations; identical formal readouts merge first."""
    if not _on(enabled):
        return None
    return _combine(anchor, terms, bias)


def paired_affine(A, B, pairs, left_bias, right_bias, *, left_extras=(),
                  right_extras=(), enabled=False):
    """Transfer two actual mixed consumers in the fixed left basis q=d+p.

    A[i]=(a_q,a_p), B[i]=(b_q,b_p), so h=sum(a_q*q+a_p*p)+left
    and w=sum(b_q*q+b_p*p)+right.  The result is (h-w,w).  In particular
    h=A*q+a,w=B*p+b is the a_p=b_q=0 special case.  Extras are ordinary
    (Fraction, observation) tuples with the SAME anchor and shared frame.
    """
    if not _on(enabled):
        return None
    if any(type(value) is not tuple for value in (A, B, pairs, left_extras, right_extras)):
        raise KernelError("immutable paired coefficient/observation tuples required")
    if not 0 < len(pairs) <= MAX_SUPPORT or len(A) != len(pairs) or len(B) != len(pairs):
        raise KernelError("paired coefficient shapes do not match")
    if 2 * len(pairs) + len(left_extras) + len(right_extras) > MAX_SUPPORT:
        raise KernelError("total paired affine term limit exceeded")
    anchor = _pair(pairs[0]).anchor
    left_bias, right_bias = _f(left_bias), _f(right_bias)
    difference_terms, companion_terms = [], []
    support = 0
    for left, right, source in zip(A, B, pairs):
        if type(left) is not tuple or type(right) is not tuple or len(left) != 2 or len(right) != 2:
            raise KernelError("each consumer coefficient pair has two entries")
        source = _pair(source, anchor)
        support += len(source.difference.terms) + len(source.companion.terms)
        if support > MAX_SUPPORT:
            raise KernelError("paired input support limit exceeded")
        aq, ap, bq, bp = _f(left[0]), _f(left[1]), _f(right[0]), _f(right[1])
        left_sum, right_sum = _add(aq, ap), _add(bq, bp)
        difference_terms.extend(((_sub(aq, bq), source.difference),
                                 (_sub(left_sum, right_sum), source.companion)))
        companion_terms.extend(((bq, source.difference), (right_sum, source.companion)))
    difference_terms.extend(left_extras)
    for item in right_extras:
        if type(item) is not tuple or len(item) != 2:
            raise KernelError("invalid right affine extra")
        difference_terms.append((_neg(_f(item[0])), item[1]))
    companion_terms.extend(right_extras)
    difference = _combine(anchor, tuple(difference_terms), _sub(left_bias, right_bias))
    companion = _combine(anchor, tuple(companion_terms), right_bias)
    return _Pair(difference, companion)


def relu_pair(pre_pair, left_output, right_output, *, enabled=False):
    """Caller binds existing original r=R(h),t=R(w); retain the old anchor.

    No new node/bit is introduced, and the existing child gate bits are not
    identified, fixed, replaced, or used as substitute anchors at zero.
    """
    if not _on(enabled):
        return None
    source = _pair(pre_pair)
    left_output = _symbol(left_output, source.anchor.frame)
    right_output = _symbol(right_output, source.anchor.frame)
    difference_bounds = tuple((min(ZERO, lo), max(ZERO, hi))
                              for lo, hi in source.difference.bounds)
    companion_bounds = tuple((max(ZERO, lo), max(ZERO, hi))
                             for lo, hi in source.companion.bounds)
    difference = _create(source.anchor, ((left_output, ONE), (right_output, -ONE)),
                         ZERO, difference_bounds, "ReLU monotone-Lipschitz difference")
    companion = _create(source.anchor, ((right_output, ONE),), ZERO,
                        companion_bounds, "ReLU companion")
    return _Pair(difference, companion)


def compile_rows(source, *, enabled=False):
    """Return two <= rows: (symbol_terms, original_anchor, alpha_coef, rhs).

    Terms are formal readouts, not production HZ columns.  Binding to original
    columns/bit encoding and accounting for their expansion is the caller's job.
    """
    if not _on(enabled):
        return None
    source = _obs(source)
    (l0, u0), (l1, u1) = source.bounds
    return ((source.terms, source.anchor, _neg(_sub(u1, u0)), _sub(u0, source.bias)),
            (tuple((symbol, _neg(value)) for symbol, value in source.terms),
             source.anchor, _sub(l1, l0), _sub(source.bias, l0)))
