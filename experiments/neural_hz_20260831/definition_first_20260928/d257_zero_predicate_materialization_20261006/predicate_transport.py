"""Default-off predicate transport through authentic production lazy views.

Only this module's private function namespace is redirected. No production
module, class, allocator or registry is patched. The logical Budget covers
these added checks/copies, not uninstrumented legacy propagation loops.
Inputs are trusted, synchronously owned in-memory objects; no concurrent or
malicious mutation, untrusted deserialization, or online admission is claimed.
"""
from dataclasses import dataclass, field
from types import FunctionType, MappingProxyType
import hashlib

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import tf_cnn as production
from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from act.back_end.solver.solver_hz import SparseHZono
from experiments.neural_hz_20260831.definition_first_20260928.d255_rebase_relation_transport_20261006 import native_rebased as nm

Budget, Limits, Rejected = nm.Budget, nm.Limits, nm.Rejected
SparseHZAffineExpr = production.SparseHZAffineExpr
SparseHZAffineTerm = production.SparseHZAffineTerm
_OWNED = object()


def _enabled(value):
    if type(value) is not bool:
        raise Rejected("enabled must be bool")
    return value


def _budget(value):
    if type(value) is not Budget:
        raise Rejected("the shared Budget is required")
    return value


def _failure(exc, budget):
    if isinstance(exc, MemoryError):
        budget._reject("predicate transport allocation failed")
    nm._legacy_failure(exc, budget)
    if isinstance(exc, Rejected):
        raise exc
    raise Rejected("invalid predicate transport: " + str(exc)) from exc


_ERRORS = (MemoryError, nm.nb.KernelError, TypeError, ValueError, IndexError,
           KeyError, AttributeError, OverflowError)


def _array_stamp(array, m, *, floating=False):
    if type(array) is not np.ndarray or array.ndim != 1:
        raise Rejected("one-dimensional owned NumPy payload required")
    m.charge(work=4 * int(array.size) + int(array.nbytes) + 8,
             entries=2 * int(array.size) + 8)
    if floating and (array.dtype != np.dtype("float64")
                     or not np.all(np.isfinite(array))):
        raise Rejected("finite float64 payload required")
    digest = hashlib.sha256()
    digest.update(repr((array.shape, array.dtype.str)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _csr_stamp(matrix, m):
    if type(matrix) is not sp.csr_matrix:
        raise Rejected("an actual canonical CSR operator is required")
    rows, cols = matrix.shape
    m.integer(int(rows)), m.integer(int(cols))
    size = int(matrix.data.size + matrix.indices.size + matrix.indptr.size)
    m.charge(work=4 * size + 16, entries=size + 16)
    nm.nb._csr(matrix, rows, cols)
    return (matrix.shape, _array_stamp(matrix.data, m, floating=True),
            _array_stamp(matrix.indices, m), _array_stamp(matrix.indptr, m))


def _implicit_stamp(operator, m):
    # Reconstruct the ORIGINAL descriptor, not its expanded convolution. Its
    # constructor visits all spatial stencil positions, which are prepaid.
    if type(operator) is not ImplicitConv2DOp:
        raise Rejected("unsupported pure linear operator type")
    kernel = operator._kernel
    if (type(kernel) is not np.ndarray or kernel.dtype != np.dtype("float64")
            or kernel.ndim != 4 or min(kernel.shape) <= 0):
        raise Rejected("invalid original implicit kernel")
    m.charge(work=4 * int(kernel.size) + 64, entries=2 * int(kernel.size) + 64)
    if not np.all(np.isfinite(kernel)):
        raise Rejected("non-finite implicit kernel")
    shape = operator._input_shape
    stride, padding, dilation = operator._stride, operator._padding, operator._dilation
    if (type(shape) is not tuple or len(shape) != 4
            or any(type(v) is not int or v < 1 for v in shape)
            or any(type(pair) is not tuple or len(pair) != 2
                   for pair in (stride, padding, dilation))
            or any(type(v) is not int or v < lower
                   for pair, lower in ((stride, 1), (padding, 0), (dilation, 1))
                   for v in pair)
            or type(operator._groups) is not int or operator._groups < 1):
        raise Rejected("invalid original implicit geometry")
    for value in (*shape, *stride, *padding, *dilation, operator._groups,
                  *kernel.shape):
        m.integer(int(value))
    n, ci, height, width = shape
    co, cig, kh, kw = (int(v) for v in kernel.shape)
    if ci != cig * operator._groups or co % operator._groups:
        raise Rejected("inconsistent grouped implicit geometry")
    oh = (height + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (width + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    if oh < 0 or ow < 0:
        raise Rejected("negative implicit output geometry")
    expected = (n, co, oh, ow)
    if operator._output_shape != expected:
        raise Rejected("implicit output geometry was changed")
    nr, nc = n * co * oh * ow, n * ci * height * width
    m.integer(nr), m.integer(nc)
    mask = operator._row_mask
    if mask is not None and (type(mask) is not np.ndarray or mask.ndim != 1
                            or mask.dtype != np.dtype("bool") or mask.size != nr):
        raise Rejected("invalid original implicit row mask")
    mask_size = 0 if mask is None else int(mask.size)
    geometry_visits = oh * ow * kh * kw
    m.integer(geometry_visits)
    # Includes the constructor's copies, finite checks, mask count_nonzero,
    # geometry integer operations and content-key byte copies/hash scans.
    m.charge(work=96 * (geometry_visits + int(kernel.size) + mask_size + oh * ow + 1),
             entries=12 * (int(kernel.size) + mask_size) + 128)
    rebuilt = ImplicitConv2DOp(kernel, shape, stride=stride, padding=padding,
        dilation=dilation, groups=operator._groups, row_mask=mask)
    if (operator.shape != rebuilt.shape
            or operator.logical_expanded_nnz != rebuilt.logical_expanded_nnz
            or operator.resident_entries != rebuilt.resident_entries
            or operator.resident_bytes != rebuilt.resident_bytes
            or operator.content_key != rebuilt.content_key):
        raise Rejected("implicit descriptor content or cached identity changed")
    return rebuilt.content_key


def _operator_stamp(operator, m):
    if type(operator) is sp.csr_matrix:
        return ("csr", _csr_stamp(operator, m))
    if type(operator) is ImplicitConv2DOp:
        return ("implicit", _implicit_stamp(operator, m))
    raise Rejected("only original CSR and ImplicitConv2DOp are supported")


def _inspect(expr, m, *, widths=None, frame=None):
    if (type(expr) is not SparseHZAffineExpr or type(expr.terms) is not tuple
            or not expr.terms or type(expr.n_out) is not int or expr.n_out < 0
            or type(expr.frame_id) is not int or expr.frame_id < 0):
        raise Rejected("an actual immutable production lazy expression is required")
    m.integer(expr.n_out), m.integer(expr.frame_id)
    if frame is not None and expr.frame_id != frame:
        raise Rejected("all views must retain the original shared frame")
    bias = _array_stamp(expr.bias, m, floating=True)
    if expr.bias.size != expr.n_out:
        raise Rejected("lazy bias has the wrong complete output width")
    m.charge(work=8 * len(expr.terms) + 8, entries=8 * len(expr.terms) + 8)
    sources, operators, layout = {}, {}, []
    for term in expr.terms:
        if type(term) is not SparseHZAffineTerm or type(term.operators) is not tuple:
            raise Rejected("original immutable production terms required")
        source = term.source
        if (type(source) is not SparseHZono or not source.exact
                or source.frame_id != expr.frame_id):
            raise Rejected("term source lost its exact shared frame")
        if widths is not None and (source.n_cont > widths[0] or source.n_bin > widths[1]):
            raise Rejected("a registered source exceeds the complete old high water")
        sid = id(source)
        if sid not in sources:
            sources[sid] = (source, nm._stamp(source, m))
        width = source.n_out
        m.charge(work=6 * len(term.operators) + 2,
                 entries=4 * len(term.operators) + 2)
        chain = []
        for operator in term.operators:
            oid = id(operator)
            if oid not in operators:
                operators[oid] = _operator_stamp(operator, m)
            if operator.shape[1] != width:
                raise Rejected("lazy operator chain has a mismatched input width")
            width = int(operator.shape[0])
            chain.append((oid, operators[oid]))
        if width != expr.n_out:
            raise Rejected("lazy operator chain has a mismatched output width")
        layout.append((sid, sources[sid][1], tuple(chain)))
    return (expr.n_out, expr.frame_id, bias, tuple(layout))


def _zero_operator(rows, cols, m):
    m.integer(rows), m.integer(cols)
    m.charge(work=4 * (rows + 1) + 16, entries=3 * (rows + 1) + 16)
    return sp.csr_matrix((rows, cols), dtype=np.float64)


def _normalize(expr, m):
    before = _inspect(expr, m)
    m.charge(work=4 * len(expr.terms) + 8, entries=4 * len(expr.terms) + 8)
    terms, count = [], 0
    for term in expr.terms:
        source = term.source
        m.charge(work=int(source.c.size) + 4, entries=int(source.c.size))
        if source.Gc.nnz == 0 and source.Gb.nnz == 0 and np.all(source.c == 0.0):
            terms.append(SparseHZAffineTerm(source,
                (_zero_operator(expr.n_out, source.n_out, m),)))
            count += 1
        else:
            terms.append(term)
    # Reuses the authentic class and the original bias; never drops a source.
    normalized = (SparseHZAffineExpr(tuple(terms), expr.bias, expr.n_out, expr.frame_id)
                  if count else expr)
    if _inspect(expr, m) != before:
        raise Rejected("lazy expression changed during zero normalization")
    return normalized, count


def normalize(expr, *, budget, enabled=False):
    if not _enabled(enabled):
        return expr
    budget = _budget(budget)
    m = budget._branch()
    try:
        return _normalize(expr, m)[0]
    except _ERRORS as exc:
        _failure(exc, budget)


@dataclass(frozen=True)
class Batch:
    views: tuple
    anchor: SparseHZono
    new_widths: tuple
    budget: Budget = field(repr=False, compare=False)
    source: SparseHZono = field(repr=False, compare=False)
    result: nm.Result = field(repr=False, compare=False)
    _authority: object = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self._authority is not _OWNED:
            raise Rejected("a detached batch must be returned by transport")


def transport(result, source, views, *, global_widths, enabled=False):
    """Return a complete detached batch; do not reserve or publish live slots."""
    if not _enabled(enabled):
        return None
    if type(result) is not nm.Result or result._authority is not nm._OWNED:
        raise Rejected("an owned D255 Result is required")
    budget = _budget(result.budget)
    m = budget._branch()
    try:
        result._verify(m)
        snapshot = result.snapshot
        expected = (snapshot.hz.n_cont, snapshot.hz.n_bin)
        if (type(global_widths) is not tuple or len(global_widths) != 2
                or any(type(v) is not int or v < 0 for v in global_widths)
                or global_widths != expected):
            raise Rejected("global high water must equal the complete captured old widths")
        if (type(source) is not SparseHZono
                or (source.n_cont, source.n_bin) != snapshot.original_dimensions):
            raise Rejected("the reachable source must match original capture dimensions")
        source_stamp = nm._stamp(source, m)
        if (source.n_cont, source.n_bin) == expected:
            padded_stamp = source_stamp
        else:
            # Ordinary lazy siblings may predate later slot reservations. This
            # is the same paid zero padding used by capture, not a source edit.
            nm._charge_hz(source, m, 4)
            m.charge(work=sum(expected) + 20, entries=sum(expected) + 20)
            padded = SparseHZono(source.c.copy(), nm._pad(source.Gc, expected[0]),
                nm._pad(source.Gb, expected[1]), nm._pad(source.Ac, expected[0]),
                nm._pad(source.Ab, expected[1]), source.b.copy(),
                nm._pad(source.Auc, expected[0]), nm._pad(source.Aub, expected[1]),
                source.ub.copy(), frame_id=source.frame_id, exact=source.exact)
            padded_stamp = nm._stamp(padded, m)
            del padded
        if padded_stamp != snapshot.fingerprint:
            raise Rejected("reachable source content differs from the Result snapshot")
        if type(views) is not tuple or not views:
            raise Rejected("a nonempty complete tuple of registered views is required")
        m.charge(work=8 * len(views) + 8, entries=8 * len(views) + 8)
        stamps, reachable = [], False
        for view in views:
            stamps.append(_inspect(view, m, widths=expected, frame=source.frame_id))
            reachable = reachable or any(term.source is source for term in view.terms)
        if not reachable:
            raise Rejected("the actual local source must be reachable in the registered union")
        enhanced = result.hz
        new_widths = (enhanced.n_cont, enhanced.n_bin)
        if new_widths[0] < expected[0] or new_widths[1] != expected[1]:
            raise Rejected("enhancement changed original phase widths")
        # All anchor arrays are independent owned copies, including predicates.
        nm._charge_hz(enhanced, m, 4)
        anchor = SparseHZono(np.zeros(1, dtype=np.float64),
            _zero_operator(1, enhanced.n_cont, m), _zero_operator(1, enhanced.n_bin, m),
            enhanced.Ac.copy(), enhanced.Ab.copy(), enhanced.b.copy(),
            enhanced.Auc.copy(), enhanced.Aub.copy(), enhanced.ub.copy(),
            frame_id=enhanced.frame_id, exact=enhanced.exact)
        nm._readonly(anchor)
        nm._stamp(anchor, m)
        published = []
        for view in views:
            m.charge(work=6 * len(view.terms) + 8, entries=4 * len(view.terms) + 8)
            term = SparseHZAffineTerm(anchor, (_zero_operator(view.n_out, 1, m),))
            published.append(SparseHZAffineExpr((*view.terms, term), view.bias,
                                               view.n_out, view.frame_id))
        # Before publishing ANY view, recheck all original sources/operators,
        # the original local source and the owned D255 result. No allocator writes.
        result._verify(m)
        if nm._stamp(source, m) != source_stamp:
            raise Rejected("source changed during detached transport")
        for view, old_stamp in zip(views, stamps):
            if _inspect(view, m, widths=expected, frame=source.frame_id) != old_stamp:
                raise Rejected("a registered view changed during detached transport")
        return Batch(tuple(published), anchor, new_widths, budget, source, result, _OWNED)
    except _ERRORS as exc:
        _failure(exc, budget)


class Adapter:
    """Trusted private call closure; it is never installed on HybridzTF."""
    __slots__ = ("budget", "namespace", "materializations", "normalized_terms",
                 "_names", "_authority", "_active_meter")

    def __init__(self, budget, namespace, names, authority):
        if authority is not _OWNED:
            raise Rejected("adapter must be returned by make_adapter")
        self.budget = budget
        self.namespace = MappingProxyType(namespace)
        self.materializations = 0
        self.normalized_terms = 0
        self._names = frozenset(names)
        self._authority = authority
        self._active_meter = None

    def call(self, name, *args, **kwargs):
        previous = self._active_meter
        m = self.budget._branch() if previous is None else previous
        self._active_meter = m
        try:
            if type(name) is not str or name not in self._names:
                raise Rejected("only authenticated module-local entry functions are exposed")
            m.charge(work=len(name) + len(args) + len(kwargs) + 4,
                     entries=len(args) + len(kwargs) + 4)
            answer = self.namespace[name](*args, **kwargs)
            # Legacy selective/deferred functions catch ValueError/MemoryError.
            # A caught resource rejection MUST NOT turn into apparent success.
            m.charge(work=1)
            return answer
        except _ERRORS as exc:
            _failure(exc, self.budget)
        finally:
            self._active_meter = previous


def make_adapter(*, budget, enabled=False):
    if not _enabled(enabled):
        return None
    budget = _budget(budget)
    m = budget._branch()
    try:
        original = vars(production)
        m.charge(work=8 * len(original) + 16, entries=8 * len(original) + 16)
        private = dict(original)
        clones, names = {}, []
        for name, function in original.items():
            if (type(function) is not FunctionType
                    or function.__globals__ is not original
                    or function.__module__ != production.__name__):
                continue
            m.charge(work=len(name) + 20, entries=len(name) + 20)
            key = id(function)
            if key not in clones:
                clone = FunctionType(function.__code__, private, function.__name__,
                                     function.__defaults__, function.__closure__)
                meta = len(function.__annotations__) + len(function.__dict__)
                kw = function.__kwdefaults__
                meta += 0 if kw is None else len(kw)
                m.charge(work=4 * meta + 12, entries=4 * meta + 12)
                clone.__kwdefaults__ = None if kw is None else dict(kw)
                clone.__annotations__ = dict(function.__annotations__)
                clone.__dict__.update(function.__dict__)
                clone.__qualname__, clone.__module__ = function.__qualname__, function.__module__
                clone.__doc__ = function.__doc__
                clones[key] = clone
            private[name] = clones[key]
            names.append(name)
        if "_lazy_materialize" not in names:
            raise Rejected("production materializer is not a module-local function")
        raw_materialize = private["_lazy_materialize"]
        m.charge(work=len(names) + 1, entries=len(names) + 1)
        adapter = Adapter(budget, private, names, _OWNED)

        def materialize(expr, keep_rows, limit, *, allow_transient_sum=False):
            meter = adapter._active_meter
            if meter is None:
                meter = budget._branch()
            try:
                normalized, count = _normalize(expr, meter)
                meter.charge(work=8, entries=2)
                adapter.materializations += 1
                adapter.normalized_terms += count
                answer = raw_materialize(normalized, keep_rows, limit,
                    allow_transient_sum=allow_transient_sum)
                meter.charge(work=1)
                return answer
            except _ERRORS as exc:
                _failure(exc, budget)

        # Preserve aliases of the materializer as well: no local name may
        # accidentally retain a bypass to its unnormalized clone.
        m.charge(work=2 * len(names) + 1)
        for name in names:
            if private[name] is raw_materialize:
                private[name] = materialize
        # Check the exact references rather than trusting a module-name string.
        for name in names:
            if private[name] is not materialize and private[name].__globals__ is not private:
                raise Rejected("private function namespace is not closed")
        return adapter
    except _ERRORS as exc:
        _failure(exc, budget)
