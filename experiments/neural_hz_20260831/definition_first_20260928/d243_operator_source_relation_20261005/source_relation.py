"""Owned mathematical source graph and same-source, default-off X relation.

No model binding, native installer, GPU certificate, solver or phase pool is
provided.  Finite physical coordinates are affine coordinates of an HZ; their
boxes never replace the retained predicates.  Conv stores its complete kernel
and geometry, not a spatial CSR copy.  Bounds, row access, reconstruction and
materialization nevertheless charge every consumed connection.

Only D228's Budget, exact arithmetic and Row are reused.  Logical cumulative
work/entry counters are not a complete Python/native/device memory account.
Resource failures and unfinished-builder failures are sticky.  Ordinary
read-only/attachment rejection retains the old H; spent charges are not refunded.
Records are trusted in-memory results of Source/seal/attach; private mutation,
forged records and untrusted deserialization are not supported entry points.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from functools import wraps

from experiments.neural_hz_20260831.definition_first_20260928.d228_joint_bank_component_20261005 import bank as _exact


Budget, Limits, Rejected, Row = _exact.Budget, _exact.Limits, _exact.Rejected, _exact.Row
ZERO, ONE, TWO = Fraction(0), Fraction(1), Fraction(2)
HALF, NEG = Fraction(1, 2), Fraction(-1)
_OWNED = object()


def _tuple(value, length=None, name="value"):
    if type(value) is not tuple or (length is not None and len(value) != length):
        raise Rejected(name + " must be a tuple of the declared length")
    return value


def _op(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        meter = self.budget._branch()
        try:
            return method(self, *args, _meter=meter, **kwargs)
        except Rejected as exc:
            if type(self) is Source and not self._sealed:
                self.budget._reject(str(exc))
            raise
        except MemoryError:
            if type(self) is Source and not self._sealed:
                self.budget._reject("allocation failed during unfinished source construction")
            raise
        except (TypeError, ValueError, IndexError, KeyError, AttributeError, OverflowError) as exc:
            if type(self) is Source and not self._sealed:
                self.budget._reject("invalid operation: " + str(exc))
            raise Rejected("invalid operation: " + str(exc)) from exc
    return call


def _min(a, b, m):
    return a if _exact._cmp(a, b, m) <= 0 else b


def _max(a, b, m):
    return a if _exact._cmp(a, b, m) >= 0 else b


def _abs(a, m):
    return _exact._neg(a, m) if a.numerator < 0 else _exact._number(a, m)


def _row(terms, rhs, width, m):
    terms = tuple(terms)
    n = len(terms)
    m.charge(work=n * (n.bit_length() + 1) + 1, entries=2 * n + 1)
    return _exact._row(terms, rhs, width, m)


@dataclass(frozen=True)
class Affine:
    constant: Fraction = ZERO
    terms: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "constant", _exact._hard_number(self.constant))
        object.__setattr__(self, "terms", Row(self.terms, ZERO).terms)


def _form(constant, terms, width, m):
    c = _exact._number(constant, m)
    r = _row(terms, ZERO, width, m)
    m.charge(work=len(r.terms) + 1, entries=2 * len(r.terms) + 3)
    return Affine(c, r.terms)


def _var(col, width, m):
    return _form(ZERO, ((col, ONE),), width, m)


def _linear(pieces, width, m, constant=ZERO):
    c, terms = _exact._number(constant, m), []
    for a, f in pieces:
        if type(f) is not Affine:
            raise Rejected("expected an exact affine expression")
        a = _exact._number(a, m)
        c = _exact._add(c, _exact._mul(a, f.constant, m), m)
        m.charge(work=len(f.terms) + 1, entries=2 * len(f.terms) + 1)
        for col, value in f.terms:
            terms.append((col, _exact._mul(a, value, m)))
    return _form(c, terms, width, m)


def _le(form, rhs, width, m):
    return _row(form.terms, _exact._sub(_exact._number(rhs, m), form.constant, m), width, m)


def _at(form, values, m):
    result = _exact._number(form.constant, m)
    for col, a in form.terms:
        m.charge(work=1)
        result = _exact._add(result, _exact._mul(a, values[col], m), m)
    return result


def _bound_terms(constant, terms, bounds, m):
    lo = hi = _exact._number(constant, m)
    for col, a in terms:
        m.charge(work=2)
        lower, upper = bounds[col]
        if a.numerator < 0:
            lower, upper = upper, lower
        lo = _exact._add(lo, _exact._mul(a, lower, m), m)
        hi = _exact._add(hi, _exact._mul(a, upper, m), m)
    m.charge(work=1, entries=2)
    return lo, hi


def _coef(form, col, m):
    result = ZERO
    for index, a in form.terms:
        m.charge(work=1)
        if index == col:
            result = a
    return result


@dataclass(frozen=True)
class Scalar:
    column: int
    frame: object = field(repr=False, compare=False)
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("Scalar is an owned source handle")


@dataclass(frozen=True)
class Tensor:
    shape: tuple
    values: tuple
    columns: tuple
    frame: object = field(repr=False, compare=False)
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("Tensor is an owned source handle")


@dataclass(frozen=True)
class Gate:
    f: Scalar
    q: Scalar
    phase_column: object
    status: str
    lower: Fraction
    upper: Fraction
    graph_rows: tuple
    frame: object = field(repr=False, compare=False)
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("Gate is an owned source handle")


@dataclass(frozen=True)
class Definition:
    kind: str
    data: object


@dataclass(frozen=True)
class Conv:
    input_columns: tuple
    input_shape: tuple
    kernel: tuple
    kernel_shape: tuple
    bias: tuple
    stride: tuple
    padding: tuple
    dilation: tuple
    groups: int
    output_shape: tuple
    output_columns: tuple


@dataclass(frozen=True)
class BN:
    input_column: int
    output_column: int
    error_column: object
    mean: Fraction
    beta: Fraction
    scale_interval: tuple
    ahat: Fraction
    bhat: Fraction
    E: Fraction


def _conv_terms(op, flat, m):
    cin, height, width = op.input_shape
    _, oh, ow = op.output_shape
    cout, per_group, kh, kw = op.kernel_shape
    oc, rem = divmod(flat, oh * ow)
    oy, ox = divmod(rem, ow)
    group = oc // (cout // op.groups)
    m.charge(work=12)
    for local_c in range(per_group):
        ic = group * per_group + local_c
        for ky in range(kh):
            iy = oy * op.stride[0] - op.padding[0] + ky * op.dilation[0]
            for kx in range(kw):
                ix = ox * op.stride[1] - op.padding[1] + kx * op.dilation[1]
                m.charge(work=12)
                if not (0 <= iy < height and 0 <= ix < width):
                    continue
                a = op.kernel[((oc * per_group + local_c) * kh + ky) * kw + kx]
                col = op.input_columns[(ic * height + iy) * width + ix]
                m.charge(entries=2)
                yield col, a


def _definition_form(state, col, m):
    definition = state.definitions[col]
    m.charge(work=1)
    if definition.kind == "affine":
        f = definition.data
        m.charge(work=len(f.terms) + 1, entries=len(f.terms) + 1)
        return f
    if definition.kind == "conv":
        op_index, flat = definition.data
        op = state.convolutions[op_index]
        oc = flat // (op.output_shape[1] * op.output_shape[2])
        return _form(op.bias[oc], _conv_terms(op, flat, m), state.n_columns, m)
    raise Rejected("this column has no affine source definition")


def _check_scalar(state, scalar, m):
    m.charge(work=2)
    registry = state._scalars if type(state) is Source else state.scalars
    if (type(scalar) is not Scalar or scalar.frame is not state.frame
            or not 0 <= scalar.column < state.n_columns
            or registry[scalar.column] is not scalar):
        raise Rejected("foreign or nonowned scalar handle")
    return scalar.column


def _check_frame(state, frame):
    if frame is not state.frame:
        raise Rejected("frame identity mismatch")


def _input_row(row, width, m):
    if type(row) is not Row:
        raise Rejected("predicates must be exact Row records")
    return _row(row.terms, row.rhs, width, m)


class Source:
    """Append-only owned declaration; seal is the sole publication path."""

    def __init__(self, input_bounds, *, source_kinds=None, eq=(), le=(),
                 enabled=False, budget=None):
        self._budget = Budget() if budget is None else budget
        if type(self._budget) is not Budget:
            raise Rejected("one shared D228 Budget is required")
        m = self._budget._branch()
        try:
            if enabled is not True:
                raise Rejected("Source is default-off")
            bounds = _tuple(input_bounds, name="input bounds")
            kinds = (("continuous",) * len(bounds) if source_kinds is None
                     else _tuple(source_kinds, len(bounds), "source kinds"))
            self._frame = object()
            self._sealed = False
            self._bounds, self._scalars, self._definitions = [], [], []
            self._eq_refs, self._extra_le, self._binary = [], [], []
            self._errors, self._gates, self._convolutions, self._bn_records = [], [], [], []
            m.charge(work=2 * len(bounds) + 10, entries=4 * len(bounds) + 18)
            for pair, kind in zip(bounds, kinds):
                lo, hi = (_exact._number(v, m) for v in _tuple(pair, 2, "input interval"))
                if _exact._cmp(lo, hi, m) > 0:
                    raise Rejected("reversed input interval")
                if type(kind) is not str or kind not in ("continuous", "binary"):
                    raise Rejected("unknown source kind")
                if kind == "binary" and (lo != NEG or hi != ONE):
                    raise Rejected("signed binary inputs require [-1,1]")
                scalar = self._append((lo, hi), Definition("input", len(self._scalars)), m)
                if kind == "binary":
                    self._binary.append(scalar.column)
            self._inputs = tuple(self._scalars)
            self._input_columns = tuple(s.column for s in self._inputs)
            self._source_kinds = tuple(kinds)
            for row in _tuple(eq, name="input EQ"):
                self._eq_refs.append(_input_row(row, self.n_columns, m))
            for row in _tuple(le, name="input LE"):
                self._extra_le.append(_input_row(row, self.n_columns, m))
            m.charge(work=len(eq) + len(le), entries=len(eq) + len(le))
        except Rejected as exc:
            self._budget._reject(str(exc))
        except MemoryError:
            self._budget._reject("allocation failed during source construction")
        except (TypeError, ValueError, IndexError, AttributeError) as exc:
            self._budget._reject("invalid Source: " + str(exc))

    @property
    def budget(self):
        return self._budget

    @property
    def frame(self):
        return self._frame

    @property
    def inputs(self):
        return self._inputs

    @property
    def n_columns(self):
        return len(self._bounds)

    @property
    @_op
    def error_columns(self, *, _meter):
        _meter.charge(work=len(self._errors), entries=len(self._errors) + 1)
        return tuple(self._errors)

    def _open(self):
        if self._sealed:
            raise Rejected("source was already sealed")

    def _append(self, bounds, definition, m):
        col = self.n_columns
        m.charge(work=4, entries=12)
        scalar = Scalar(col, self.frame, _OWNED)
        self._bounds.append(bounds)
        self._scalars.append(scalar)
        self._definitions.append(definition)
        return scalar

    def _affine(self, terms, bias, m):
        f = _form(bias, terms, self.n_columns, m)
        interval = _bound_terms(f.constant, f.terms, self._bounds, m)
        out = self._append(interval, Definition("affine", f), m)
        self._eq_refs.append(out.column)
        m.charge(work=1, entries=1)
        return out

    @_op
    def affine(self, terms, bias=ZERO, *, _meter):
        self._open()
        parsed = []
        for pair in _tuple(terms, name="affine terms"):
            scalar, value = _tuple(pair, 2, "affine term")
            col = _check_scalar(self, scalar, _meter)
            parsed.append((col, _exact._number(value, _meter)))
            _meter.charge(entries=2)
        return self._affine(parsed, bias, _meter)

    @_op
    def add(self, left, right, *, _meter):
        self._open()
        a, b = (_check_scalar(self, v, _meter) for v in (left, right))
        return self._affine(((a, ONE), (b, ONE)), ZERO, _meter)

    @_op
    def predicate(self, row, *, equality=False, _meter):
        self._open()
        if type(equality) is not bool:
            raise Rejected("equality must be bool")
        row = _input_row(row, self.n_columns, _meter)
        (self._eq_refs if equality else self._extra_le).append(row)
        _meter.charge(work=1, entries=1)

    @_op
    def relu(self, f, *, _meter):
        self._open()
        m = _meter
        col = _check_scalar(self, f, m)
        lo, hi = self._bounds[col]
        if _exact._cmp(lo, ZERO, m) >= 0:
            gate = Gate(f, f, None, "positive", lo, hi, (), self.frame, _OWNED)
        elif _exact._cmp(hi, ZERO, m) <= 0:
            q = self._affine((), ZERO, m)
            gate = Gate(f, q, None, "negative", lo, hi, (), self.frame, _OWNED)
        else:
            phase = self._append((NEG, ONE), Definition("phase", col), m)
            self._binary.append(phase.column)
            q = self._append((ZERO, hi), Definition("relu", (col, phase.column)), m)
            width = self.n_columns
            qf, ff = _var(q.column, width, m), _var(col, width, m)
            alpha = _form(HALF, ((phase.column, HALF),), width, m)
            rows = (
                _le(_linear(((NEG, qf),), width, m), ZERO, width, m),
                _le(_linear(((ONE, ff), (NEG, qf)), width, m), ZERO, width, m),
                _le(_linear(((ONE, qf), (_exact._neg(hi, m), alpha)), width, m), ZERO, width, m),
                _le(_linear(((ONE, qf), (NEG, ff), (_exact._neg(lo, m), alpha)), width, m),
                    _exact._neg(lo, m), width, m),
            )
            indices = tuple(range(len(self._extra_le), len(self._extra_le) + 4))
            self._extra_le.extend(rows)
            gate = Gate(f, q, phase.column, "crossing", lo, hi, indices, self.frame, _OWNED)
        self._gates.append(gate)
        m.charge(work=5, entries=18)
        return gate

    def _bn(self, x, mean, beta, scale_interval, ahat, m):
        col = _check_scalar(self, x, m)
        mean, beta = _exact._number(mean, m), _exact._number(beta, m)
        al, au = (_exact._number(v, m) for v in _tuple(scale_interval, 2, "BN scale interval"))
        if _exact._cmp(al, au, m) > 0:
            raise Rejected("reversed BN scale interval")
        ahat = (_exact._mul(HALF, _exact._add(al, au, m), m) if ahat is None
                else _exact._number(ahat, m))
        if _exact._cmp(ahat, al, m) < 0 or _exact._cmp(ahat, au, m) > 0:
            raise Rejected("BN nominal scale must lie in its declared interval")
        bhat = _exact._sub(beta, _exact._mul(ahat, mean, m), m)
        da = _max(_abs(_exact._sub(al, ahat, m), m), _abs(_exact._sub(au, ahat, m), m), m)
        lo, hi = self._bounds[col]
        radius = _max(_abs(_exact._sub(lo, mean, m), m), _abs(_exact._sub(hi, mean, m), m), m)
        error = _exact._mul(da, radius, m)
        epsilon = None
        terms = [(col, ahat)]
        if error != ZERO:
            epsilon = self._append((NEG, ONE), Definition("error", len(self._errors)), m)
            self._errors.append(epsilon.column)
            terms.append((epsilon.column, error))
        out = self._affine(terms, bhat, m)
        self._bn_records.append(BN(col, out.column, None if epsilon is None else epsilon.column,
                                   mean, beta, (al, au), ahat, bhat, error))
        m.charge(work=4, entries=18)
        return out

    @_op
    def bn(self, x, *, mean, beta, scale_interval, ahat=None, _meter):
        self._open()
        return self._bn(x, mean, beta, scale_interval, ahat, _meter)

    def _tensor(self, shape, values, m):
        shape = _tuple(shape, 3, "CHW shape")
        size = 1
        for n in shape:
            if type(n) is not int or n <= 0:
                raise Rejected("tensor dimensions must be positive integers")
            size = m.integer(size * n)
        values = _tuple(values, size, "tensor values")
        m.charge(work=size, entries=2 * size + 8)
        columns = tuple(_check_scalar(self, v, m) for v in values)
        return Tensor(shape, values, columns, self.frame, _OWNED)

    def _check_tensor(self, tensor, m):
        if type(tensor) is not Tensor or tensor.frame is not self.frame:
            raise Rejected("foreign tensor")
        m.charge(work=len(tensor.values), entries=1)
        for scalar in tensor.values:
            _check_scalar(self, scalar, m)
        return tensor

    @_op
    def tensor(self, shape, values=None, *, _meter):
        self._open()
        return self._tensor(shape, self.inputs if values is None else values, _meter)

    @_op
    def bn2d(self, tensor, *, mean, beta, scale_intervals, ahat=None, _meter):
        self._open()
        tensor = self._check_tensor(tensor, _meter)
        channels, height, width = tensor.shape
        mean = _tuple(mean, channels, "BN means")
        beta = _tuple(beta, channels, "BN betas")
        scale_intervals = _tuple(scale_intervals, channels, "BN intervals")
        centers = (None,) * channels if ahat is None else _tuple(ahat, channels, "BN nominal scales")
        values = []
        _meter.charge(work=channels, entries=len(tensor.values) + channels)
        for i, scalar in enumerate(tensor.values):
            c = i // (height * width)
            values.append(self._bn(scalar, mean[c], beta[c], scale_intervals[c], centers[c], _meter))
        return self._tensor(tensor.shape, tuple(values), _meter)

    @_op
    def conv2d(self, tensor, kernel, bias=None, *, stride=(1, 1), padding=(0, 0, 0, 0),
               dilation=(1, 1), groups=1, _meter):
        self._open()
        m = _meter
        tensor = self._check_tensor(tensor, m)
        cin, height, width = tensor.shape
        stride, dilation = _tuple(stride, 2, "stride"), _tuple(dilation, 2, "dilation")
        padding = _tuple(padding, 4, "top,left,bottom,right padding")
        if any(type(v) is not int or v <= 0 for v in stride + dilation):
            raise Rejected("stride and dilation must be positive integers")
        if any(type(v) is not int or v < 0 for v in padding):
            raise Rejected("padding must be nonnegative")
        if type(groups) is not int or groups <= 0 or cin % groups:
            raise Rejected("invalid convolution groups")
        for value in stride + dilation + padding + (groups,):
            m.integer(value)
        m.charge(work=24, entries=8)
        kernel = _tuple(kernel, name="kernel")
        cout = len(kernel)
        if cout == 0 or cout % groups:
            raise Rejected("invalid output channels")
        per = cin // groups
        first = _tuple(kernel[0], per, "kernel input channels")
        kh = len(_tuple(first[0], name="kernel rows"))
        if kh == 0:
            raise Rejected("empty kernel height")
        kw = len(_tuple(first[0][0], name="kernel columns"))
        if kw == 0:
            raise Rejected("empty kernel width")
        count = m.integer(cout * per * kh * kw)
        m.charge(work=count, entries=count + 20)
        flat_kernel = []
        for output in kernel:
            for channel in _tuple(output, per, "kernel input channels"):
                for row in _tuple(channel, kh, "kernel rows"):
                    flat_kernel.extend(_exact._number(a, m) for a in _tuple(row, kw, "kernel columns"))
        bias = (ZERO,) * cout if bias is None else _tuple(bias, cout, "conv bias")
        bias = tuple(_exact._number(a, m) for a in bias)
        effective_h = m.integer(dilation[0] * (kh - 1))
        effective_w = m.integer(dilation[1] * (kw - 1))
        span_h = m.integer(m.integer(height + padding[0] + padding[2]) - effective_h - 1)
        span_w = m.integer(m.integer(width + padding[1] + padding[3]) - effective_w - 1)
        oh = m.integer(span_h // stride[0] + 1)
        ow = m.integer(span_w // stride[1] + 1)
        if oh <= 0 or ow <= 0:
            raise Rejected("convolution has no output positions")
        population = m.integer(cout * oh * ow)
        m.charge(work=population + 20, entries=2 * population + cout + count + 20)
        start = self.n_columns
        op = Conv(tensor.columns, tensor.shape, tuple(flat_kernel), (cout, per, kh, kw), bias,
                  stride, padding, dilation, groups, (cout, oh, ow), tuple(range(start, start + population)))
        op_index = len(self._convolutions)
        self._convolutions.append(op)
        values = []
        for flat in range(population):
            oc = flat // (oh * ow)
            interval = _bound_terms(bias[oc], _conv_terms(op, flat, m), self._bounds, m)
            scalar = self._append(interval, Definition("conv", (op_index, flat)), m)
            self._eq_refs.append(scalar.column)
            values.append(scalar)
            m.charge(work=1, entries=2)
        return self._tensor(op.output_shape, tuple(values), m)

    @_op
    def seal(self, *, _meter):
        self._open()
        m = _meter
        population = (self.n_columns * 4 + len(self._eq_refs) + len(self._extra_le)
                      + len(self._gates) + len(self._binary) + len(self._errors)
                      + len(self._convolutions) + len(self._bn_records) + len(self.inputs) * 2)
        m.charge(work=population + 12, entries=population + 25)
        result = H(tuple(self._bounds), tuple(self._scalars), tuple(self._definitions),
                   tuple(self._eq_refs), tuple(self._extra_le), tuple(self._binary),
                   self._inputs, self._input_columns, self._source_kinds, tuple(self._errors),
                   tuple(self._gates), tuple(self._convolutions), tuple(self._bn_records),
                   self.frame, self.budget, _OWNED)
        self._sealed = True
        return result


def _eq_at(state, index, m):
    if type(index) is not int or not 0 <= index < state.n_eq:
        raise Rejected("EQ index out of range")
    ref = state.eq_refs[index]
    if type(ref) is Row:
        return _input_row(ref, state.n_columns, m)
    f = _definition_form(state, ref, m)
    terms = [(ref, ONE)]
    for col, a in f.terms:
        terms.append((col, _exact._neg(a, m)))
    m.charge(work=len(f.terms), entries=2 * len(f.terms) + 2)
    return _row(terms, f.constant, state.n_columns, m)


def _bound_row(bounds, index, width, m):
    col, side = divmod(index, 2)
    lo, hi = bounds[col]
    return _row(((col, NEG if side == 0 else ONE),),
                _exact._neg(lo, m) if side == 0 else hi, width, m)


def _le_at(state, index, m):
    if type(index) is not int or not 0 <= index < state.n_le:
        raise Rejected("LE index out of range")
    if index < 2 * state.n_columns:
        return _bound_row(state.column_bounds, index, state.n_columns, m)
    return _input_row(state.extra_le[index - 2 * state.n_columns], state.n_columns, m)


def _row_holds(row, values, equality, m):
    value = ZERO
    for col, a in row.terms:
        value = _exact._add(value, _exact._mul(a, values[col], m), m)
    comparison = _exact._cmp(value, row.rhs, m)
    return comparison == 0 if equality else comparison <= 0


def _holds_h(state, values, integral, m):
    if len(values) != state.n_columns:
        return False
    for value, (lo, hi) in zip(values, state.column_bounds):
        if _exact._cmp(value, lo, m) < 0 or _exact._cmp(value, hi, m) > 0:
            return False
    if integral:
        for col in state.binary_columns:
            m.charge(work=1)
            if values[col] not in (NEG, ONE):
                return False
    for i in range(state.n_eq):
        if not _row_holds(_eq_at(state, i, m), values, True, m):
            return False
    for row in state.extra_le:
        m.charge(work=1 + len(row.terms))
        if not _row_holds(row, values, False, m):
            return False
    return True


@dataclass(frozen=True)
class Materialized:
    eq: tuple
    le: tuple
    n_columns: int
    column_bounds: tuple
    binary_columns: tuple


def _materialize(state, m):
    m.charge(work=state.n_eq + state.n_le, entries=state.n_eq + state.n_le + 8)
    eq = tuple(_eq_at(state, i, m) for i in range(state.n_eq))
    le = tuple(_le_at(state, i, m) for i in range(state.n_le))
    return Materialized(eq, le, state.n_columns, state.column_bounds, state.binary_columns)


@dataclass(frozen=True)
class H:
    column_bounds: tuple
    scalars: tuple
    definitions: tuple
    eq_refs: tuple
    extra_le: tuple
    binary_columns: tuple
    inputs: tuple
    input_columns: tuple
    source_kinds: tuple
    error_columns: tuple
    gates: tuple
    convolutions: tuple
    bn_records: tuple
    frame: object = field(repr=False, compare=False)
    budget: Budget = field(repr=False, compare=False)
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("H is published only by Source.seal")

    @property
    def n_columns(self):
        return len(self.column_bounds)

    @property
    def n_eq(self):
        return len(self.eq_refs)

    @property
    def n_le(self):
        return 2 * self.n_columns + len(self.extra_le)

    @_op
    def eq_row(self, index, *, frame, _meter):
        _check_frame(self, frame)
        return _eq_at(self, index, _meter)

    @_op
    def le_row(self, index, *, frame, _meter):
        _check_frame(self, frame)
        return _le_at(self, index, _meter)

    @_op
    def materialize(self, *, frame, _meter):
        _check_frame(self, frame)
        return _materialize(self, _meter)

    @_op
    def satisfied(self, values, *, frame, integral=False, _meter):
        _check_frame(self, frame)
        if type(integral) is not bool:
            raise Rejected("integral must be bool")
        values = _exact._vector(values, self.n_columns, "complete H assignment", _meter)
        return _holds_h(self, values, integral, _meter)

    @_op
    def decode(self, values, *, frame, _meter):
        _check_frame(self, frame)
        values = _exact._vector(values, self.n_columns, "complete H assignment", _meter)
        if not _holds_h(self, values, True, _meter):
            raise Rejected("decoder requires a complete integer H witness")
        _meter.charge(work=len(self.input_columns), entries=len(self.input_columns) + 1)
        return tuple(values[i] for i in self.input_columns)

    @_op
    def canonical(self, input_values, *, errors=None, zero_phases=(), frame, _meter):
        _check_frame(self, frame)
        m = _meter
        inputs = _exact._vector(input_values, len(self.inputs), "input assignment", m)
        errors = ((ZERO,) * len(self.error_columns) if errors is None else errors)
        errors = _exact._vector(errors, len(self.error_columns), "BN error assignment", m)
        labels = {}
        phase_columns = {g.phase_column for g in self.gates if g.phase_column is not None}
        m.charge(work=len(self.gates), entries=len(self.gates) + 1)
        for pair in _tuple(zero_phases, name="phase choices"):
            col, value = _tuple(pair, 2, "phase choice")
            value = _exact._number(value, m)
            m.charge(work=3, entries=2)
            if type(col) is not int or col not in phase_columns or col in labels or value not in (NEG, ONE):
                raise Rejected("invalid or duplicate original phase choice")
            labels[col] = value
        values = []
        m.charge(work=self.n_columns, entries=self.n_columns + 1)
        for col, definition in enumerate(self.definitions):
            kind, data = definition.kind, definition.data
            if kind == "input":
                value = inputs[data]
            elif kind == "error":
                value = errors[data]
            elif kind in ("affine", "conv"):
                value = _at(_definition_form(self, col, m), values, m)
            elif kind == "phase":
                f = values[data]
                expected = ONE if f.numerator >= 0 else NEG
                value = labels.get(col, expected)
                if f != ZERO and value != expected:
                    raise Rejected("phase choice contradicts nonzero preactivation")
            elif kind == "relu":
                value = _max(values[data[0]], ZERO, m)
            else:
                raise Rejected("unrecognized owned definition")
            values.append(value)
        result = tuple(values)
        m.charge(entries=len(result) + 1)
        if not _holds_h(self, result, True, m):
            raise Rejected("input/error choices do not satisfy the complete original H")
        return result


def _alias(state, col, m):
    while True:
        m.charge(work=1)
        definition = state.definitions[col]
        if definition.kind != "affine":
            return col
        f = definition.data
        if f.constant != ZERO or len(f.terms) != 1 or f.terms[0][1] != ONE:
            return col
        col = f.terms[0][0]


def _expand(state, target, stops, m, memo):
    """Exact DAG substitution, with only literal identity aliases quotiented."""
    definitions, stack = {}, [(target, False)]
    m.charge(entries=3)
    while stack:
        col, ready = stack.pop()
        m.charge(work=2)
        if col in memo:
            continue
        representative = _alias(state, col, m)
        if representative in stops:
            memo[col] = _var(stops[representative], state.n_columns, m)
            m.charge(entries=2)
            continue
        definition = state.definitions[col]
        if definition.kind not in ("affine", "conv"):
            memo[col] = _var(col, state.n_columns, m)
            m.charge(entries=2)
            continue
        if not ready:
            form = _definition_form(state, col, m)
            definitions[col] = form
            stack.append((col, True))
            m.charge(work=len(form.terms), entries=2 * len(form.terms) + 4)
            for child, coefficient in reversed(form.terms):
                if child not in memo:
                    stack.append((child, False))
        else:
            form = definitions[col]
            memo[col] = _linear(((a, memo[child]) for child, a in form.terms),
                                state.n_columns, m, constant=form.constant)
            m.charge(entries=2)
    return memo[target]


@dataclass(frozen=True)
class Product:
    column: int
    phase_index: int
    source: Affine
    lower: Fraction
    upper: Fraction


@dataclass(frozen=True)
class Relation:
    parent: H
    parents: tuple
    children: tuple
    normalized: tuple
    child_forms: tuple
    tau: Fraction
    a: Fraction
    b: Fraction
    z: Affine
    d: Affine
    r: Affine
    e: Affine
    r_bounds: tuple
    e_bounds: tuple
    K: Affine
    products: tuple
    additional_le: tuple
    product_rows: tuple
    defect_rows: tuple
    capacity_rows: tuple
    column_bounds: tuple
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("Relation is produced only by attach")

    @property
    def frame(self):
        return self.parent.frame

    @property
    def budget(self):
        return self.parent.budget

    @property
    def anchor_kind(self):
        return "x"

    @property
    def n_columns(self):
        return len(self.column_bounds)

    def _holds(self, values, integral, m):
        n = self.parent.n_columns
        m.charge(work=n, entries=n)
        if not _holds_h(self.parent, values[:n], integral, m):
            return False
        for value, (lo, hi) in zip(values[n:], self.column_bounds[n:]):
            if _exact._cmp(value, lo, m) < 0 or _exact._cmp(value, hi, m) > 0:
                return False
        return all(_row_holds(row, values, False, m) for row in self.additional_le)

    @_op
    def canonical_extension(self, values, *, frame, _meter):
        _check_frame(self, frame)
        m = _meter
        values = _exact._vector(values, self.parent.n_columns, "original H witness", m)
        if not _holds_h(self.parent, values, True, m):
            raise Rejected("extension requires a complete original integer witness")
        products = []
        for product in self.products:
            alpha = _exact._mul(HALF, _exact._add(values[product.phase_index], ONE, m), m)
            products.append(_exact._mul(alpha, _at(product.source, values, m), m))
        m.charge(work=self.n_columns, entries=self.n_columns + 5)
        result = values + tuple(products)
        if not self._holds(result, True, m):
            raise Rejected("canonical extension failed its complete predicates")
        return result

    @_op
    def satisfied(self, values, *, frame, integral=False, _meter):
        _check_frame(self, frame)
        if type(integral) is not bool:
            raise Rejected("integral must be bool")
        values = _exact._vector(values, self.n_columns, "extended assignment", _meter)
        return self._holds(values, integral, _meter)

    @_op
    def decode(self, values, *, frame, _meter):
        _check_frame(self, frame)
        values = _exact._vector(values, self.n_columns, "extended assignment", _meter)
        if not self._holds(values, True, _meter):
            raise Rejected("decoder requires a complete extended integer witness")
        _meter.charge(work=len(self.parent.input_columns), entries=len(self.parent.input_columns) + 1)
        return tuple(values[i] for i in self.parent.input_columns)

    @_op
    def materialize(self, *, frame, _meter):
        _check_frame(self, frame)
        old = _materialize(self.parent, _meter)
        _meter.charge(work=len(old.eq) + len(old.le) + 29,
                      entries=len(old.eq) + len(old.le) + 37)
        new_bounds = tuple(_bound_row(self.column_bounds, 2 * i + side, self.n_columns, _meter)
                           for i in range(self.parent.n_columns, self.n_columns) for side in (0, 1))
        return Materialized(old.eq, old.le + new_bounds + self.additional_le,
                            self.n_columns, self.column_bounds, self.parent.binary_columns)


def attach(h, *, parents, children, enabled=False, frame):
    """Append the one fixed X relation to a sealed same-source declaration."""
    if type(h) is not H or h._authority is not _OWNED:
        raise Rejected("attach requires a sealed owned H")
    m = h.budget._branch()
    try:
        if enabled is not True:
            raise Rejected("relation attachment is default-off")
        _check_frame(h, frame)
        parents = _tuple(parents, 2, "parent gates")
        children = _tuple(children, 2, "child gates")
        m.charge(work=4 * len(h.gates) + 12, entries=16)
        for gate in parents + children:
            if (type(gate) is not Gate or gate.frame is not h.frame
                    or not any(old is gate for old in h.gates)):
                raise Rejected("foreign or nonowned gate")
            _check_scalar(h, gate.f, m)
            _check_scalar(h, gate.q, m)
        if len({id(g) for g in parents + children}) != 4:
            raise Rejected("four distinct original gates are required")
        if any(g.status != "crossing" or g.phase_column is None for g in parents):
            raise Rejected("the two original parents must be crossing")
        scales = tuple(_max(_exact._neg(g.lower, m), g.upper, m) for g in parents)
        representatives = tuple(_alias(h, g.f.column, m) for g in parents)
        all_stops = representatives + tuple(g.q.column for g in parents)
        if len(set(all_stops)) != 4:
            raise Rejected("parent anchor identities cannot be separated")
        stops = {representatives[i]: parents[i].f.column for i in range(2)}
        stops.update({g.q.column: g.q.column for g in parents})
        memo = {}
        m.charge(entries=1)
        g1, g2 = tuple(_expand(h, g.f.column, stops, m, memo) for g in children)
        old_width, width = h.n_columns, h.n_columns + 4
        x1, x2 = tuple(_linear(((_exact._div(ONE, scales[i], m),
                                _var(parents[i].f.column, width, m)),), width, m) for i in range(2))
        q1, q2 = tuple(_linear(((_exact._div(ONE, scales[i], m),
                                _var(parents[i].q.column, width, m)),), width, m) for i in range(2))
        y1, y2 = tuple(_var(g.q.column, width, m) for g in children)
        alpha1, alpha2 = tuple(_form(HALF, ((g.phase_column, HALF),), width, m) for g in parents)
        z = _linear(((HALF, g1), (HALF, g2)), width, m)
        d = _linear(((HALF, g1), (_exact._neg(HALF, m), g2)), width, m)
        A, B = tuple(_exact._mul(_coef(d, parents[i].q.column, m), scales[i], m) for i in range(2))
        tau = _exact._mul(HALF, _exact._sub(A, B, m), m)
        if _exact._cmp(tau, ZERO, m) <= 0:
            raise Rejected("fixed child coefficients do not give positive tau")
        a, b = tuple(_exact._mul(_coef(z, parents[i].f.column, m), scales[i], m) for i in range(2))
        r = _linear(((ONE, z), (_exact._neg(a, m), x1), (_exact._neg(b, m), x2)), width, m)
        e = _linear(((ONE, d), (_exact._neg(tau, m), q1), (tau, q2)), width, m)
        rb = _bound_terms(r.constant, r.terms, h.column_bounds, m)
        eb = _bound_terms(e.constant, e.terms, h.column_bounds, m)
        ap, am = _max(a, ZERO, m), _max(_exact._neg(a, m), ZERO, m)
        bp, bm = _max(b, ZERO, m), _max(_exact._neg(b, m), ZERO, m)
        magnitude = _max(_exact._add(ap, bm, m), _exact._add(am, bp, m), m)
        if (_exact._cmp(_exact._sub(rb[0], magnitude, m), _exact._neg(tau, m), m) < 0
                or _exact._cmp(_exact._add(rb[1], magnitude, m), tau, m) > 0):
            raise Rejected("the complete same-H X mismatch guard is not certified")
        sources, phases = (x2, x1, r, r), (parents[0].phase_column, parents[1].phase_column,
                                             parents[0].phase_column, parents[1].phase_column)
        intervals = ((NEG, ONE), (NEG, ONE), rb, rb)
        products, bounds, rows, product_rows = [], [], [], []
        for offset, (source, phase, interval) in enumerate(zip(sources, phases, intervals)):
            lo, hi = interval
            col = old_width + offset
            products.append(Product(col, phase, source, lo, hi))
            bounds.append((_min(ZERO, lo, m), _max(ZERO, hi, m)))
            value, alpha = _var(col, width, m), _form(HALF, ((phase, HALF),), width, m)
            forms = (
                _linear(((NEG, value), (lo, alpha)), width, m),
                _linear(((ONE, value), (_exact._neg(hi, m), alpha)), width, m),
                _linear(((NEG, value), (ONE, source), (hi, alpha)), width, m),
                _linear(((ONE, value), (NEG, source), (_exact._neg(lo, m), alpha)), width, m),
            )
            product_rows.append(tuple(range(len(rows), len(rows) + 4)))
            rows.extend(_le(f, rhs, width, m) for f, rhs in
                        zip(forms, (ZERO, ZERO, hi, _exact._neg(lo, m))))
            m.charge(work=4, entries=24)
        c12, c21, v1, v2 = tuple(_var(old_width + i, width, m) for i in range(4))
        K = _linear(((tau, alpha1), (_exact._neg(tau, m), alpha2), (a, q1),
                     (_exact._neg(a, m), c21), (b, c12), (_exact._neg(b, m), q2),
                     (ONE, v1), (NEG, v2)), width, m)
        difference = _linear(((ONE, y1), (NEG, y2), (NEG, K)), width, m)
        p1, p2 = (_linear(((ONE, alpha), (NEG, q)), width, m)
                  for alpha, q in ((alpha1, q1), (alpha2, q2)))
        two_tau = _exact._mul(TWO, tau, m)
        rows.append(_le(_linear(((ONE, difference), (_exact._neg(two_tau, m), p2)), width, m),
                        _exact._mul(TWO, _max(eb[1], ZERO, m), m), width, m))
        rows.append(_le(_linear(((NEG, difference), (_exact._neg(two_tau, m), p1)), width, m),
                        _exact._neg(_exact._mul(TWO, _min(eb[0], ZERO, m), m), m), width, m))
        T = _linear(((ONE, q1), (NEG, q2), (ONE, c12), (NEG, c21)), width, m)
        d1, d2 = tuple(_linear(((_exact._neg(TWO, m), q), (ONE, x)), width, m, constant=ONE)
                       for q, x in ((q1, x1), (q2, x2)))
        U = _linear(((ONE, q1), (ONE, q2), (NEG, c12), (NEG, c21)), width, m)
        D = _linear(((TWO, q1), (NEG, x1), (TWO, q2), (NEG, x2)), width, m)
        rows.extend(_le(f, ZERO, width, m) for f in (
            _linear(((ONE, T), (NEG, d2)), width, m),
            _linear(((NEG, T), (NEG, d1)), width, m),
            _linear(((ONE, U), (NEG, D)), width, m)))
        m.charge(work=width + 40, entries=width + 110)
        return Relation(h, parents, children, (x1, x2, q1, q2), (g1, g2), tau, a, b,
                        z, d, r, e, rb, eb, K, tuple(products), tuple(rows), tuple(product_rows),
                        (16, 17), (18, 19, 20), h.column_bounds + tuple(bounds), _OWNED)
    except Rejected:
        raise
    except (TypeError, ValueError, IndexError, KeyError, AttributeError, OverflowError) as exc:
        raise Rejected("invalid relation attachment: " + str(exc)) from exc
