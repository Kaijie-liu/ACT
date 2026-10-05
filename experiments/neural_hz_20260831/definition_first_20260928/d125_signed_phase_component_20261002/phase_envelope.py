"""Opt-in source-conditioned projection on a mathematical signed-bit HZ.

This emits fewer parent-amplitude columns, not just additional cuts.  The
original System is deliberately retained for mathematical decoder checking;
emitted sparsity is therefore NOT a net-memory or native-verifier claim.
Literal gate rows establish semantics relative to that System, not binding to
an external neural network.  All visible readouts must be supplied by caller.
"""
from dataclasses import dataclass
from fractions import Fraction as F

from experiments.neural_hz_20260831.definition_first_20260928.d112_shared_endpoint_forward_20261002 import endpoint_forward as ef
from experiments.neural_hz_20260831.definition_first_20260928.d119_curvature_component_20261002 import curvature_transfer as ct


Form, System, Gate = ef.Form, ef.System, ct.Gate
ZERO, ONE = F(0), F(1)
MAX_ENTRIES = 64_000_000


@dataclass(frozen=True)
class Projection:
    system: object
    readouts: object
    retained_ids: tuple
    removed_ids: tuple
    original: object
    parents: tuple
    receipt: dict


def _active(bit):
    return Form(F(1, 2), ((bit, F(1, 2)),))


def _gate_rows(gate):
    lower, upper = min(ZERO, gate.lo), max(ZERO, gate.hi)
    active = _active(gate.bit)
    return (
        ct._linear(((-ONE, gate.q),)),
        ct._linear(((ONE, gate.g), (-ONE, gate.q))),
        ct._linear(((ONE, gate.q), (-upper, active))),
        ct._linear(((ONE, gate.q), (-ONE, gate.g), (-lower, active)), lower),
    )


def _gate(system, gate, binary):
    if type(gate) is not Gate:
        raise ValueError('Gate required')
    ct._form(system, gate.g)
    output = ct._output(system, gate.q, binary)
    if type(gate.bit) is not int or gate.bit not in binary:
        raise ValueError('original signed gate bit required')
    if ct._fraction(gate.lo) > ct._fraction(gate.hi):
        raise ValueError('reversed gate bounds')
    return output


def _remap(form, mapping):
    if any(index not in mapping for index, _ in form.terms):
        raise ValueError('uncovered removed-column consumer')
    return Form(form.bias, tuple((mapping[index], value) for index, value in form.terms))


def _nnz(system):
    return sum(len(row.terms) for row in (*system.eq, *system.le))


def project(system, parents, target, *, tau, kappa, readouts=(), frames,
            enabled=False, max_entries=MAX_ENTRIES):
    """Sound outer projection; no source/model, solver or runtime dispatch.

    ``retained_ids[new_index]`` is the original semantic column identity.
    Gates may have tighter preactivation bounds than their independent box:
    their four literal inequalities are INPUT semantics, not an oracle claim.
    Unlisted consumers outside ``system``/``readouts`` remain a native-binding
    obligation and are never certified by an empty readout tuple.
    """
    if enabled is False:
        return Projection(system, readouts, (), (), system, (), {'enabled': False})
    if enabled is not True:
        raise ValueError('strict boolean opt-in required')
    old_entries, binary = ct._schema(system, max_entries)
    if (type(parents) is not tuple or not parents or type(tau) is not tuple
            or type(kappa) is not tuple or len(tau) != len(parents)
            or len(kappa) != len(parents) or type(readouts) is not tuple
            or type(frames) is not tuple or len(frames) != len(parents) + 1):
        raise ValueError('immutable block shape required')
    if any(type(frame) is not int or frame != system.frame for frame in frames):
        raise ValueError('source/frame mismatch')
    if any(type(reference) is not int or reference not in (0, 1) for reference in tau):
        raise ValueError('fixed Boolean references required')
    kappas = tuple(ct._fraction(value) for value in kappa)
    if any(value <= ZERO for value in kappas):
        raise ValueError('positive finite multipliers required')
    gates = parents + (target,)
    outputs = tuple(_gate(system, gate, binary) for gate in gates)
    phases = tuple(gate.bit for gate in gates)
    if len(set(outputs)) != len(outputs) or len(set(phases)) != len(phases):
        raise ValueError('distinct original gates required')
    removed = set(outputs[:-1])
    forbidden = removed | {outputs[-1]} | set(phases)
    if any(index in forbidden for gate in parents for index, _ in gate.g.terms):
        raise ValueError('cyclic or phase-dependent parent source')
    target_terms = dict(target.g.terms)
    coefficients = tuple(target_terms.get(index, ZERO) for index in outputs[:-1])
    if any(not value for value in coefficients):
        raise ValueError('every removed parent must feed target')
    rest = Form(target.g.bias,
                tuple((index, value) for index, value in target.g.terms if index not in removed))
    if any(index in forbidden for index, _ in rest.terms):
        raise ValueError('cyclic or phase-dependent target remainder')
    for form in readouts:
        ct._form(system, form)
        if any(index in removed for index, _ in form.terms):
            raise ValueError('public readout still consumes removed amplitude')

    # This conservative retained symbolic-entry reservation includes BOTH
    # original and emitted systems, forms and certificate metadata.  It is not
    # a Python-memory, rational bit-complexity or complete-work certificate.
    supports = (sum(len(gate.g.terms) + len(gate.q.terms) for gate in gates)
                + sum(len(form.terms) for form in readouts))
    entry_upper = 2*old_entries + 16*(supports + len(system.bounds)) + 160*len(gates) + 256
    if entry_upper > max_entries:
        raise ValueError('entry cap')

    # Consume separate literal occurrences; an extra duplicate using q is an
    # unhandled predicate and is conservatively rejected below.
    available = {}
    for index, row in enumerate(system.le):
        available.setdefault(row, []).append(index)
    consumed = set()
    used_counts = {}
    removed_rows = []
    for gate in gates:
        for row in _gate_rows(gate):
            count = used_counts.get(row, 0)
            indices = available.get(row, ())
            if count >= len(indices):
                raise ValueError('missing literal original gate inequality')
            consumed.add(indices[count])
            used_counts[row] = count + 1
            removed_rows.append(row)
    remaining_le = tuple(row for index, row in enumerate(system.le) if index not in consumed)
    for row in (*system.eq, *remaining_le):
        if any(index in removed for index, _ in row.terms):
            raise ValueError('uncovered predicate or side consumer')

    h = ct._linear(((ONE, rest),) + tuple(
        (ct._mul(coefficient, F(reference)), gate.g)
        for coefficient, reference, gate in zip(coefficients, tau, parents)))
    terms = tuple(ct._linear(((F(1 - 2*reference), gate.g),))
                  for reference, gate in zip(tau, parents))
    rho = tuple(max(ZERO, ct._box(system, ct._linear(((ONE, h), (value, term))))[1])
                for value, term in zip(kappas, terms))
    tp = tm = rp = rm = ZERO
    for coefficient, value, cap in zip(coefficients, kappas, rho):
        weight = ct._div(abs(coefficient), value)
        amount = ct._mul(weight, cap)
        if coefficient > ZERO:
            tp, rp = ct._add(tp, weight), ct._add(rp, amount)
        else:
            tm, rm = ct._add(tm, weight), ct._add(rm, amount)
    if tp >= ONE:
        raise ValueError('positive phase budget must be strictly below one')
    ep = ct._div(rp, ct._add(ONE, -tp))
    em = ct._add(rm, ct._mul(tm, ep))
    lh, _ = ct._box(system, h)
    ur = max(ZERO, min(max(ZERO, target.hi), system.bounds[outputs[-1]][1]))
    active = _active(target.bit)
    added = []
    for gate, output in zip(parents, outputs[:-1]):
        lower, upper = min(ZERO, gate.lo), max(ZERO, gate.hi)
        alpha = _active(gate.bit)
        added.extend((ct._linear(((ONE, gate.g), (-upper, alpha))),
                      ct._linear(((-ONE, gate.g), (-lower, alpha)), lower)))
        qlo, qhi = system.bounds[output]
        if qhi < ZERO:
            # Original nonnegative q plus negative upper bound is infeasible.
            added.append(Form(ONE))
        elif qhi < upper:
            added.append(ct._linear(((ONE, gate.g),), -qhi))
        if qlo > ZERO:
            added.append(ct._linear(((-ONE, gate.g),), qlo))
    added.extend((
        ct._linear(((-ONE, target.q),)),
        ct._linear(((ONE, target.q), (-ur, active))),
        ct._linear(((ONE, h), (-ONE, target.q), (ct._add(rm, -em), active)), -rm),
        ct._linear(((ONE, target.q), (-ONE, h), (-ct._add(ep, lh), active)), lh),
    ))
    retained_ids = tuple(index for index in range(len(system.bounds)) if index not in removed)
    mapping = {old: new for new, old in enumerate(retained_ids)}
    added_rows = tuple(_remap(row, mapping) for row in added)
    result = System(tuple(system.bounds[index] for index in retained_ids),
                    tuple(mapping[index] for index in system.binary),
                    tuple(_remap(row, mapping) for row in system.eq),
                    tuple(_remap(row, mapping) for row in remaining_le) + added_rows,
                    system.frame)
    emitted_entries, _ = ct._schema(result, max_entries)
    if old_entries + emitted_entries > max_entries:
        raise ValueError('entry cap including retained original')
    mapped_h = _remap(h, mapping)
    c = ct._add(-ep, -lh)
    if ur == ZERO:
        forward_upper = Form()
    elif c > ZERO:
        forward_upper = ct._linear(((ct._div(ur, ct._add(ur, c)), mapped_h),),
                                   -ct._mul(ct._div(ur, ct._add(ur, c)), lh))
    else:
        forward_upper = ct._linear(((ONE, mapped_h),), ep)
    forward_lower = ct._linear(((ONE, mapped_h),), -em)

    # Zero error proves target-label equivalence, but dropping an artificially
    # tight original preactivation restriction can still enlarge the System.
    # This conservative independent reconstruction box supplies the additional
    # condition needed before claiming equality of the entire projection.
    reconstructed_lo, reconstructed_hi = ct._box(system, rest)
    for coefficient, gate in zip(coefficients, parents):
        flo, fhi = ct._box(system, gate.g)
        qlo, qhi = max(ZERO, flo), max(ZERO, fhi)
        pair = (ct._mul(coefficient, qlo), ct._mul(coefficient, qhi))
        reconstructed_lo = ct._add(reconstructed_lo, min(pair))
        reconstructed_hi = ct._add(reconstructed_hi, max(pair))
    bounds_proved = (min(ZERO, target.lo) <= reconstructed_lo
                     and reconstructed_hi <= max(ZERO, target.hi))
    zero_error = all(value == ZERO for value in rho)
    receipt = dict(
        enabled=True, rho=rho, T_plus=tp, T_minus=tm, R_plus=rp, R_minus=rm,
        E_plus=ep, E_minus=em, h=mapped_h,
        t=tuple(_remap(term, mapping) for term in terms), coefficients=coefficients,
        kappa=kappas, tau=tau, L_h=lh, U_r=ur,
        forward_upper=forward_upper, forward_lower=forward_lower,
        removed_rows=tuple(removed_rows), removed_row_indices=tuple(sorted(consumed)),
        added_rows=added_rows, old_rows=len(system.eq)+len(system.le),
        new_rows=len(result.eq)+len(result.le), old_columns=len(system.bounds),
        new_columns=len(result.bounds), old_nnz=_nnz(system), new_nnz=_nnz(result),
        old_continuous=len(system.bounds)-len(system.binary),
        new_continuous=len(result.bounds)-len(result.binary),
        original_bit_ids=system.binary,
        resulting_bit_ids=tuple(retained_ids[index] for index in result.binary),
        original_bits_deleted=0, rho_zero=zero_error,
        exact_target_relation_when_rho_zero=zero_error,
        exact_when_rho_zero=zero_error and bounds_proved,
        reconstructed_target_bounds=(reconstructed_lo, reconstructed_hi),
        original_target_bounds_source_proved=bounds_proved,
        retained_original_system=True, original_entries=old_entries,
        emitted_entries=emitted_entries,
        original_plus_emitted_entries=old_entries+emitted_entries,
        entry_upper=entry_upper, max_entries=max_entries,
        exact_rational_rows=True, decoder_is_network_adv=False,
        public_readouts_are_caller_declared=True,
        external_consumers_qualified=False, native_binding_qualified=False,
        actual_model_qualified=False, complete_physical_qualification=False,
        whole_work_qualified=False, gpu_qualified=False,
    )
    return Projection(result, tuple(_remap(form, mapping) for form in readouts),
                      retained_ids, tuple(sorted(removed)), system, parents, receipt)


def _evaluate(form, values):
    total = form.bias
    for index, coefficient in form.terms:
        total = ct._add(total, ct._mul(coefficient, values[index]))
    return total


def _validate_point(system, values):
    if type(values) is not tuple or len(values) != len(system.bounds):
        raise ValueError('complete immutable point required')
    for value, (lo, hi) in zip(values, system.bounds):
        if not lo <= ct._fraction(value) <= hi:
            raise ValueError('point outside bounds')
    if any(values[index] not in (F(-1), ONE) for index in system.binary):
        raise ValueError('decoder requires original integer bits')
    if any(_evaluate(row, values) != ZERO for row in system.eq):
        raise ValueError('point violates equality')
    if any(_evaluate(row, values) > ZERO for row in system.le):
        raise ValueError('point violates inequality')


def decode(projection, values):
    """Return checked old mathematical coordinates, never a network ADV."""
    if type(projection) is not Projection or projection.receipt.get('enabled') is not True:
        raise ValueError('enabled Projection required')
    _validate_point(projection.system, values)
    old_values = [ZERO] * len(projection.original.bounds)
    for index, old in enumerate(projection.retained_ids):
        old_values[old] = values[index]
    for gate in projection.parents:
        old_values[gate.q.terms[0][0]] = max(ZERO, _evaluate(gate.g, old_values))
    old_values = tuple(old_values)
    _validate_point(projection.original, old_values)
    return old_values
