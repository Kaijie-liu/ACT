"""Opt-in native fixed-query Attention fiber and certified direction query.

One unchanged System and one joint score/value table define the native graph.
The temporary polygons are exact images of source BOXES, not a replacement
domain.  Source predicates/overlap make the product query only an outer bound.
No model binding, LP/MILP, attack, source split, GPU, or ADV decoder is added.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from functools import cmp_to_key

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef
from experiments.neural_hz_20260831.definition_first_20260928.d119_curvature_component_20261002 import curvature_transfer as ct
from experiments.neural_hz_20260831.definition_first_20260928.d127_native_attention_component_20261002 import exp_interval as ei


Form, System = ef.Form, ef.System
MAX_ENTRIES = 64_000_000
MAX_WORK = ei.MAX_WORK
ZERO, ONE = F(0), F(1)


@dataclass(frozen=True)
class Fiber:
    system: System
    scores: tuple
    values: tuple
    frames: tuple
    used_columns: tuple
    exact_product: bool
    entries: int
    entry_upper: int
    max_entries: int


def build(system, scores, values, *, frames, enabled=False,
          max_entries=MAX_ENTRIES):
    """Keep the original objects; frames authenticate metadata, not a model.

    There is one frame per token, covering its score and all value channels.
    The source-usage test is deliberately direction-independent.  Even a fixed
    shared column conservatively prevents an exact_product claim here.
    """
    if enabled is False:
        return None
    if enabled is not True:
        raise ValueError('strict boolean opt-in required')
    source_entries, binary = ct._schema(system, max_entries)
    if (type(scores) is not tuple or not scores or type(values) is not tuple
            or len(values) != len(scores) or type(frames) is not tuple
            or len(frames) != len(scores)):
        raise ValueError('immutable token shape required')
    if any(type(frame) is not int or frame != system.frame for frame in frames):
        raise ValueError('source/frame mismatch')
    if any(type(row) is not tuple for row in values) or not values[0]:
        raise ValueError('nonempty immutable value channels required')
    channels = len(values[0])
    if any(len(row) != channels for row in values):
        raise ValueError('value channel shape')
    entries = source_entries + len(frames) + 16
    seen, used_columns = set(), []
    exact_product = not system.eq and not system.le
    for score, row in zip(scores, values):
        columns = set()
        for form in (score,) + row:
            ct._form(system, form)
            entries += 1 + 2*len(form.terms)
            if entries > max_entries:
                raise ValueError('entry cap')
            for index, _ in form.terms:
                if index in binary:
                    raise ValueError('binary score/value source is unsupported')
                columns.add(index)
        exact_product = exact_product and not bool(seen.intersection(columns))
        seen.update(columns)
        ordered = tuple(sorted(columns))
        used_columns.append(ordered)
        entries += len(ordered)
    # Reservation includes all returned two-coordinate vertices and the
    # direction query's working forms/vectors.  This is a symbolic-entry
    # bound, NOT the memory size of Python/Fraction objects or a GPU claim.
    dimension = sum(map(len, used_columns))
    entry_upper = entries + 16*dimension + 64*len(scores) + 8*channels + 256
    if entry_upper > max_entries:
        raise ValueError('entry cap')
    return Fiber(system, scores, values, frames, tuple(used_columns),
                 bool(exact_product), entries, entry_upper, max_entries)


def _fraction(value):
    if type(value) is not F:
        raise ValueError('Fraction required')
    return ei.rational(value)


def _direction_form(values, direction, budget):
    bias, terms = ZERO, {}
    for weight, form in zip(direction, values):
        budget.charge()
        if not weight:
            continue
        bias = ei.add(bias, ei.mul(weight, form.bias, budget), budget)
        for index, coefficient in form.terms:
            value = ei.add(terms.get(index, ZERO),
                           ei.mul(weight, coefficient, budget), budget)
            if value:
                terms[index] = value
            else:
                terms.pop(index, None)
    # Canonical integer-index sorting is conservatively charged before Form
    # construction; actual angular comparisons are counted in _polygon.
    budget.charge(len(terms)*max(1, len(terms).bit_length()))
    return Form(bias, tuple(terms.items()))


def _score_upper(system, form, budget):
    result = form.bias
    for index, coefficient in form.terms:
        lo, hi = system.bounds[index]
        budget.charge()
        endpoint = hi if coefficient > ZERO else lo
        result = ei.add(result, ei.mul(coefficient, endpoint, budget), budget)
    return result


def _polygon(system, score, value, columns, budget):
    """Exact 2D zonotope boundary, without source preimage duplication.

    [lo,hi] source segments are oriented into one closed upper half-plane.
    Angularly ordered positive segments followed by their negatives trace
    the boundary.  Equal-angle segments merge; point and line images remain.
    """
    sa, va = dict(score.terms), dict(value.terms)
    base_s, base_v = score.bias, value.bias
    vectors = []
    for index in columns:
        a, b = sa.get(index, ZERO), va.get(index, ZERO)
        lo, hi = system.bounds[index]
        base_s = ei.add(base_s, ei.mul(a, lo, budget), budget)
        base_v = ei.add(base_v, ei.mul(b, lo, budget), budget)
        width = ei.sub(hi, lo, budget)
        ds, dv = ei.mul(a, width, budget), ei.mul(b, width, budget)
        budget.charge(3)
        if not ds and not dv:
            continue
        if dv < ZERO or (dv == ZERO and ds < ZERO):
            base_s = ei.add(base_s, ds, budget)
            base_v = ei.add(base_v, dv, budget)
            ds, dv = -ds, -dv
        vectors.append((ds, dv))
    if not vectors:
        return ((base_s, base_v),)

    def cross(left, right):
        return ei.sub(ei.mul(left[0], right[1], budget),
                      ei.mul(left[1], right[0], budget), budget)

    def compare(left, right):
        product = cross(left, right)
        budget.charge(2)
        return -1 if product > ZERO else (1 if product < ZERO else 0)

    vectors.sort(key=cmp_to_key(compare))
    merged = []
    for vector in vectors:
        budget.charge()
        if merged and not cross(merged[-1], vector):
            old = merged[-1]
            merged[-1] = (ei.add(old[0], vector[0], budget),
                          ei.add(old[1], vector[1], budget))
        else:
            merged.append(vector)
    vertices = [(base_s, base_v)]
    current = (base_s, base_v)
    for ds, dv in merged:
        current = (ei.add(current[0], ds, budget),
                   ei.add(current[1], dv, budget))
        vertices.append(current)
    for ds, dv in merged[:-1]:
        current = (ei.sub(current[0], ds, budget),
                   ei.sub(current[1], dv, budget))
        vertices.append(current)
    # The omitted last negative edge closes the polygon to vertices[0].
    return tuple(vertices)


def _point_interval(point, threshold, shift, budget):
    score, value = point
    el, eu = ei.exp_bounds(ei.sub(score, shift, budget), budget=budget)
    residual = ei.sub(value, threshold, budget)
    budget.charge()
    if residual >= ZERO:
        return ei.mul(el, residual, budget), ei.mul(eu, residual, budget)
    return ei.mul(eu, residual, budget), ei.mul(el, residual, budget)


def _maximum_interval(vertices, threshold, shift, budget):
    lower = upper = None
    candidates = 0

    def consume(point):
        nonlocal lower, upper, candidates
        lo, hi = _point_interval(point, threshold, shift, budget)
        budget.charge(2)
        lower = lo if lower is None else max(lower, lo)
        upper = hi if upper is None else max(upper, hi)
        candidates += 1

    for point in vertices:
        consume(point)
    if len(vertices) > 1:
        for index, first in enumerate(vertices):
            second = vertices[(index + 1) % len(vertices)]
            a = ei.sub(second[0], first[0], budget)
            b = ei.sub(second[1], first[1], budget)
            budget.charge()
            if ei.mul(a, b, budget) >= ZERO:
                continue
            parameter = ei.sub(ei.div(-ONE, a, budget),
                               ei.div(ei.sub(first[1], threshold, budget), b, budget),
                               budget)
            budget.charge(2)
            if ZERO < parameter < ONE:
                consume((ei.add(first[0], ei.mul(parameter, a, budget), budget),
                         ei.add(first[1], ei.mul(parameter, b, budget), budget)))
    return (lower, upper), candidates


def certify(fiber, direction, threshold, *, max_work=MAX_WORK):
    """Certify one threshold; never interpret a positive lower bound as ADV.

    The reported interval encloses exp(-shift)*F_product_box(threshold),
    including when the native source is constrained or shared.  The lower
    endpoint then need not bound the actual correlated maximum from below.
    The query installs no terminal rows and does not modify the native graph.
    Revalidation and its conservative working-entry reservation must fit
    min(fiber.max_entries, max_work).  This can reject before the arithmetic
    counter is exhausted; neither budget represents a wall-clock guarantee.
    """
    budget = ei.Budget(max_work)
    if type(fiber) is not Fiber:
        raise ValueError('built native Fiber required')
    if (type(fiber.max_entries) is not int
            or not 0 < fiber.max_entries <= MAX_ENTRIES):
        raise ValueError('entry cap')
    # Fiber is a public dataclass; replacing metadata must not confer source
    # independence or hide altered forms, frames, widths, or entry budgets.
    checked = build(fiber.system, fiber.scores, fiber.values,
                    frames=fiber.frames, enabled=True,
                    max_entries=min(fiber.max_entries, max_work))
    if (type(fiber.exact_product) is not bool
            or fiber.used_columns != checked.used_columns
            or fiber.exact_product != checked.exact_product
            or type(fiber.entries) is not int or fiber.entries != checked.entries
            or type(fiber.entry_upper) is not int or fiber.entry_upper != checked.entry_upper):
        raise ValueError('inconsistent Fiber metadata')
    # Charge the bounded source/metadata validation pass.  These are symbolic
    # entry units; the remaining counter charges rational operations, rounding,
    # comparisons and a conservative canonical-index sort reservation.
    budget.charge(checked.entries)
    if type(direction) is not tuple or len(direction) != len(checked.values[0]):
        raise ValueError('joint value direction shape')
    direction = tuple(_fraction(weight) for weight in direction)
    threshold = _fraction(threshold)
    budget.charge(len(direction) + 1)
    score_uppers = tuple(_score_upper(checked.system, form, budget)
                         for form in checked.scores)
    budget.charge(len(score_uppers))
    shift = max(score_uppers)
    polygons, intervals = [], []
    total_lo = total_hi = ZERO
    candidates = 0
    for score, values, columns in zip(checked.scores, checked.values, checked.used_columns):
        value = _direction_form(values, direction, budget)
        vertices = _polygon(checked.system, score, value, columns, budget)
        interval, count = _maximum_interval(vertices, threshold, shift, budget)
        total_lo = ei.add(total_lo, interval[0], budget)
        total_hi = ei.add(total_hi, interval[1], budget)
        polygons.append(vertices)
        intervals.append(interval)
        candidates += count
    entries = (checked.entries + len(direction) + 4
               + sum(2*len(vertices) + 2 for vertices in polygons))
    if entries > checked.max_entries or entries > checked.entry_upper:
        raise ValueError('entry cap')
    budget.charge()
    return dict(certified=total_hi <= ZERO, threshold=threshold,
                f_interval=(total_lo, total_hi),
                quantity='shifted_product_box_F', shift=shift,
                exact_product=checked.exact_product,
                token_intervals=tuple(intervals), polygon_vertices=tuple(polygons),
                candidate_points=candidates, work=budget.work, entries=entries,
                entry_upper=checked.entry_upper, query_entry_cap=checked.max_entries,
                work_count_kind='charged_scalar_units_not_bit_complexity_or_runtime',
                original_system_retained=True, original_bits_deleted=0,
                source_metadata_is_model_binding=False,
                native_binding_qualified=False, actual_model_qualified=False,
                complete_physical_qualification=False, gpu_qualified=False,
                whole_work_qualified=False, adv_witness_returned=False)
