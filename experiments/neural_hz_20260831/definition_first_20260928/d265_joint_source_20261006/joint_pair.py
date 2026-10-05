"""Complete registered pair population on the original declared source graph.

No model is run and no HZ, source axis, phase bit or decoder is replaced here.
The caller has authenticated the complete states/phases and their original
Conv ledgers/BN carriers.  Group boxes only bound existing scalar readouts.
All new arrays and scalar evidence use the caller's one cumulative meter.

One full-stencil Gram choice is shared by all nine masks of a channel pair.
It chooses constants, not extra physical padded sources: actual readouts omit
each invalid tap.  A stride-two shortcut's noncentral coarse sources are
disjoint from the selected parents' fine stencil; only the center is expanded
and merged with the parent A coefficients.  Original parent/final BN errors
remain distinct axes.  Trest can include the other collinear bias/error terms
without duplicating them or their original semantic identities.
"""

import math

import numpy as np

from experiments.neural_hz_20260831.definition_first_20260928.d259_shared_difference_source_20261006 import source_bounds as frozen
from experiments.neural_hz_20260831.definition_first_20260928.d261_mask_matched_reference_20261006.source_observer import geometry_classes
from . import source_arithmetic as ar
from .jp_certificate import certify


Rejected = frozen.Rejected
I = frozen.Interval


def _pay(budget, work, entries=0):
    budget.charge(int(work), entries=int(entries))


def _require(condition, reason):
    if not condition:
        raise Rejected(reason)


def _view(value, key, budget):
    # Scalar indexing gets a zero-dimensional wrapper; vector indexing uses
    # actual ndarray views.  No copied dense payload is hidden in this helper.
    _pay(budget, 24, 16)
    return I(np.asarray(value.lo[key]), np.asarray(value.hi[key]))


def _flat(value, budget):
    _pay(budget, 12, 12)
    _require(value.lo.flags.c_contiguous and value.hi.flags.c_contiguous,
             "complete contiguous Gram readout required")
    return I(value.lo.reshape(-1), value.hi.reshape(-1))


def _zeros(shape, budget):
    n = math.prod(shape)
    _pay(budget, n + 16, n + 16)
    values = np.zeros(shape, dtype=np.float64)
    return I(values, values)


def _raw(values, budget):
    # Authenticated producer arrays are scanned once, not once per mask.
    frozen._borrow_point_array(values, budget)
    _pay(budget, 8, 8)
    return I(values, values)


def _bounds_record(value, budget):
    _pay(budget, 4 * value.lo.size + 16, 2 * value.lo.size + 16)
    return (value.lo.tolist(), value.hi.tolist())


def _scalar(value, budget):
    _pay(budget, 16, 8)
    frozen._check(value, budget)
    _require(value.lo.shape == (), "scalar source interval required")
    return float(value.lo), float(value.hi)


def _conv(state, cin, cout, kernel, stride, padding, budget):
    _pay(budget, 96, 32)
    _require(state["kind"] == "conv" and state["weight_shape"] == (cout, cin) + kernel
             and state["strides"] == stride and state["pads"] == padding
             and state["dilations"] == (1, 1) and state["group"] == 1,
             "unregistered complete Conv geometry")
    ledger = state["ledger"]
    _require(type(ledger) is frozen.ConvLedger and ledger._token is frozen._TOKEN
             and ledger.weights.shape == state["weight_shape"],
             "original owned complete Conv ledger required")
    weights = state["weights"]
    frozen._borrow_point_array(weights, budget)
    _pay(budget, 2 * weights.size + 16, weights.size + 16)
    _require(np.array_equal(weights, ledger.weights), "Conv source and ledger coefficients differ")
    return _raw(ledger.weights, budget)


def _carrier(state, channels, budget):
    _pay(budget, 48, 24)
    carrier = state["carrier"]
    _require(state["kind"] == "batchnorm" and type(carrier) is frozen.BNCarrier
             and carrier._token is frozen._TOKEN and state["bounds"] is carrier.bounds,
             "original declared BN carrier required")
    for values in (carrier.nominal_a, carrier.nominal_b, carrier.error):
        _require(values.shape == (channels,), "complete original BN channel population required")
    return tuple(_raw(values, budget) for values in
                 (carrier.nominal_a, carrier.nominal_b, carrier.error))


def _removal_cache(ledger, weights, budget):
    """Precompute the two directed tap payments, not fresh source variables."""
    n = ledger.weights.size
    _pay(budget, 5 * n + 48, 3 * n + 48)
    positive = ledger.weights >= 0.0
    low = ledger.input_bounds.lo[None, :, None, None]
    high = ledger.input_bounds.hi[None, :, None, None]
    lower_operand = np.where(positive, low, high)
    upper_operand = np.where(positive, high, low)
    minimum = ar.mul(weights, I(lower_operand, lower_operand), budget)
    maximum = ar.mul(weights, I(upper_operand, upper_operand), budget)
    # Removing min requires its UPPER enclosure; removing max requires LOWER.
    _pay(budget, 16, 16)
    return I(minimum.hi, minimum.hi), I(maximum.lo, maximum.lo)


def _peel(context, channel, ky, kx, budget):
    ledger = context["middle"]["ledger"]
    minimum, maximum = context["removal"]
    _pay(budget, 24, 24)
    low = I(ledger.bounds.lo, ledger.bounds.lo)
    high = I(ledger.bounds.hi, ledger.bounds.hi)
    for selected in (channel, channel + 1):
        key = (slice(None), selected, 2 - ky, 2 - kx)
        low = ar.sub(low, _view(minimum, key, budget), budget)
        high = ar.sub(high, _view(maximum, key, budget), budget)
    _pay(budget, 8, 8)
    return I(low.lo, high.hi)


def _prepare(states, phases, sb, budget):
    _pay(budget, 512, 192)
    _require(type(phases) in (list, tuple) and len(phases) == 3,
             "the complete three-ReLU prefix is required")
    first, parent, child = phases
    shape = tuple(parent["shape"])
    _require(shape == tuple(child["shape"]) and len(shape) == 4 and shape[0] == 1,
             "registered parent and child shapes differ")
    channels, height, width = shape[1:]
    _require(channels >= 2 and height >= 3 and width >= 3,
             "registered full banks need at least two channels and nine masks")
    outer_bn = states[child["preactivation"]]
    outer = states[outer_bn["input"]]
    add = states[outer["input"]]
    _require(add["kind"] == "add" and len(add["inputs"]) == 2,
             "complete original residual Add required")
    candidates = []
    for port in add["inputs"]:
        bn = states[port]
        if bn["kind"] == "batchnorm":
            conv = states[bn["input"]]
            if conv["kind"] == "conv" and conv["input"] == parent["output"]:
                candidates.append((port, bn, conv))
    _require(len(candidates) == 1, "unique complete second-ReLU main branch required")
    main_port, middle_bn, middle = candidates[0]
    skip_port = next(port for port in add["inputs"] if port != main_port)
    skip = states[skip_port]
    a_port = first["output"]
    a_state = states[a_port]
    ca = a_state["shape"][1]
    parent_bn = states[parent["preactivation"]]
    parent_conv = states[parent_bn["input"]]
    _require(parent_conv["input"] == a_port, "parents do not share the registered first-ReLU source")
    identity = skip_port == a_port
    if identity:
        _require(tuple(a_state["shape"]) == shape, "identity skip source shape differs")
        parent_stride = (1, 1)
    else:
        _require(skip["kind"] == "batchnorm", "registered projection skip must end in BN")
        skip_conv = states[skip["input"]]
        _require(skip_conv["input"] == a_port
                 and tuple(a_state["shape"][2:]) == (2 * height - 1, 2 * width - 1),
                 "projection skip and parent fine-stencil source differ")
        parent_stride = (2, 2)
    wp = _conv(parent_conv, ca, channels, (3, 3), parent_stride, (1, 1, 1, 1), budget)
    wm = _conv(middle, channels, channels, (3, 3), (1, 1), (1, 1, 1, 1), budget)
    wo = _conv(outer, channels, channels, (3, 3), (1, 1), (1, 1, 1, 1), budget)
    pa, pb, pe = _carrier(parent_bn, channels, budget)
    ma, mb, me = _carrier(middle_bn, channels, budget)
    oa, ob, oe = _carrier(outer_bn, channels, budget)
    ppre = parent_bn["bounds"]
    frozen._check(ppre, budget)
    _pay(budget, 5 * channels + 32, 3 * channels + 32)
    scales = np.maximum(np.negative(ppre.lo), ppre.hi)
    # Constant-zero parents are recorded ineligible.  A unit placeholder is
    # used only in the shared cache and is never emitted as a normalization.
    safe_scales = np.where(scales > 0.0, scales, 1.0)
    scale_iv = _raw(safe_scales, budget)
    factor = ar.div(pa, scale_iv, budget)
    pcoeff = ar.mul(wp, _view(factor, (slice(None), None, None, None), budget), budget)
    pbias = ar.div(ar.add(ar.mul(pa, _raw(parent_conv["ledger"].bias, budget), budget),
                          pb, budget), scale_iv, budget)
    peps = ar.div(pe, scale_iv, budget)
    out = ar.mul(wo, _view(oa, (slice(None), None, None, None), budget), budget)
    out_bias = ar.add(ar.mul(oa, _raw(outer["ledger"].bias, budget), budget), ob, budget)
    gamma = ar.mul(ar.mul(wm, _view(ma, (slice(None), None, None, None), budget), budget),
                   _view(scale_iv, (None, slice(None), None, None), budget), budget)
    frozen._check(a_state["bounds"], budget)
    a_center, a_radius = ar.center_radius(a_state["bounds"], budget)
    _pay(budget, 24, 16)
    radius4 = a_radius[None, :, None, None]
    parent_axes = ar.scale(pcoeff, radius4, budget)
    zero = ar.point(0.0, budget)
    skip_center = skip_bias_noise = None
    if not identity:
        ws = _conv(skip_conv, ca, channels, (1, 1), (2, 2), (0, 0, 0, 0), budget)
        sa, sbias, se = _carrier(skip, channels, budget)
        basis = ar.mul(_view(ws, (slice(None), slice(None), 0, 0), budget),
                       _view(sa, (slice(None), None), budget), budget)
        skip_constant = ar.add(ar.mul(sa, _raw(skip_conv["ledger"].bias, budget), budget),
                               sbias, budget)
        _pay(budget, channels + 16, channels + 16)
        skip_bias_noise = ar.add(skip_constant, I(np.negative(se.hi), se.hi), budget)
        # Complete output rows are composed once; adjacent pairs reuse them.
        rows = []
        for output in range(channels):
            row = _view(out, (output, slice(None), 1, 1), budget)
            products = ar.mul(_view(row, (slice(None), None), budget), basis, budget)
            rows.append(ar.sum_axis(products, 0, budget))
        skip_center = ar.stack(tuple(rows), budget)
    classes = geometry_classes(height, width, budget=budget, enabled=True)
    reference = sb.masked_child_bounds(outer["ledger"], add["bounds"],
                                       outer_bn["carrier"], budget=budget, enabled=True)
    _pay(budget, 256 + 24 * len(classes) + 3 * reference.upper.size,
         96 + 24 * len(classes) + reference.upper.size)
    reference_ids = {mask: index for index, mask in enumerate(reference.masks)}
    _require(len(reference_ids) == len(classes) == 9
             and set(reference_ids) == {item["mask"] for item in classes}
             and reference.upper.shape == (9, channels), "incomplete same-mask child reference")
    _require(bool(np.equal(reference.old_upper, states[child["output"]]["bounds"].hi).all()),
             "same-H original child range differs")
    return dict(parent=parent, child=child, first=first, channels=channels, ca=ca,
                height=height, width=width, parent_bn=parent_bn, outer_bn=outer_bn,
                middle=middle, outer=outer, main_port=main_port, skip_port=skip_port,
                skip=skip, identity=identity, classes=classes, reference=reference,
                reference_ids=reference_ids, scales=scales, pcoeff=pcoeff,
                parent_axes=parent_axes, pbias=pbias, peps=peps,
                a_bounds=a_state["bounds"], a_center=a_center, a_radius=a_radius,
                middle_a=ma, middle_b=mb, middle_e=me, out=out,
                out_bias=out_bias, out_error=oe, gamma=gamma,
                removal=_removal_cache(middle["ledger"], wm, budget),
                skip_center=skip_center, skip_bias_noise=skip_bias_noise, zero=zero,
                qbounds=states[parent["output"]]["bounds"])


def _fixed_gram(context, channel, mean, budget):
    ca = context["ca"]
    if context["identity"]:
        ztemplate = mean
    else:
        # Two zero payloads plus the two center-column assignments below.
        _pay(budget, 20 * ca + 48, 18 * ca + 48)
        zl = np.zeros((ca, 3, 3), dtype=np.float64)
        zu = np.zeros((ca, 3, 3), dtype=np.float64)
        center = ar.half(ar.add(_view(context["skip_center"], channel, budget),
                                _view(context["skip_center"], channel + 1, budget), budget), budget)
        zl[:, 1, 1], zu[:, 1, 1] = center.lo, center.hi
        ztemplate = I(zl, zu)
    _pay(budget, 16, 16)
    zaxes = ar.scale(ztemplate, context["a_radius"][:, None, None], budget)
    pi = _view(context["parent_axes"], channel, budget)
    pj = _view(context["parent_axes"], channel + 1, budget)
    ei, ej = (_view(context["peps"], k, budget) for k in (channel, channel + 1))
    zero = context["zero"]
    p1 = ar.concat((_flat(pi, budget), ar.stack((ei, zero), budget)), budget)
    p2 = ar.concat((_flat(pj, budget), ar.stack((zero, ej), budget)), budget)
    z = ar.concat((_flat(zaxes, budget), ar.stack((zero, zero), budget)), budget)
    return ar.gram(p1, p2, z, budget)


def _offset(context, channel, ky, kx, mean, difference, chosen_a, budget):
    m = _view(mean, (slice(None), ky, kx), budget)
    h = _view(difference, (slice(None), ky, kx), budget)
    crest = _peel(context, channel, ky, kx, budget)
    tail = ar.add(ar.mul(context["middle_a"], crest, budget), context["middle_b"], budget)
    _pay(budget, context["channels"] + 16, context["channels"] + 16)
    error = context["middle_e"].hi
    tail = ar.add(tail, I(np.negative(error), error), budget)
    if not context["identity"]:
        # Original skip epsilon/bias occur here exactly once.  Only at the
        # central tap do we expose its A dependence for the parent subtraction.
        skip_tail = (context["skip_bias_noise"] if (ky, kx) == (1, 1)
                     else context["skip"]["bounds"])
        tail = ar.add(tail, skip_tail, budget)
    rc, ec, tail_stats = ar.group_stats(m, h, tail, budget)

    pi = _view(context["pcoeff"], (channel, slice(None), ky, kx), budget)
    pj = _view(context["pcoeff"], (channel + 1, slice(None), ky, kx), budget)
    parent_term = ar.add(ar.scale(pi, chosen_a[0], budget),
                         ar.scale(pj, chosen_a[1], budget), budget)
    if context["identity"]:
        za, da = m, h
    elif (ky, kx) == (1, 1):
        first = _view(context["skip_center"], channel, budget)
        second = _view(context["skip_center"], channel + 1, budget)
        za = ar.half(ar.add(first, second, budget), budget)
        da = ar.half(ar.sub(first, second, budget), budget)
    else:
        za = da = _zeros((context["ca"],), budget)
    ra = ar.sub(za, parent_term, budget)
    ar0, ae0, a_stats = ar.group_stats(ra, da, context["a_bounds"], budget,
        center_radius=(context["a_center"], context["a_radius"]))
    # In stride two, this A coordinate is 2p+(ky-1,kx-1), not the noncentral
    # shortcut's 2p+2*(ky-1,kx-1).  They must not be identified as one axis.
    rc, ec = ar.add(rc, ar0, budget), ar.add(ec, ae0, budget)
    gamma = _view(context["gamma"],
                  (slice(None), slice(channel, channel + 2), 2 - ky, 2 - kx), budget)
    bm = ar.sum_axis(ar.mul(_view(m, (slice(None), None), budget), gamma, budget), 0, budget)
    bd = ar.sum_axis(ar.mul(_view(h, (slice(None), None), budget), gamma, budget), 0, budget)
    return (rc, ec, ar.combine_stats((tail_stats, a_stats), budget), ar.stack((bm, bd), budget))


def _pair(context, channel, budget):
    first, second = (_view(context["out"], k, budget) for k in (channel, channel + 1))
    mean = ar.half(ar.add(first, second, budget), budget)
    difference = ar.half(ar.sub(first, second, budget), budget)
    chosen_a = _fixed_gram(context, channel, mean, budget)
    if chosen_a is None:
        return None
    offsets = tuple(_offset(context, channel, ky, kx, mean, difference, chosen_a, budget)
                    for ky in range(3) for kx in range(3))
    c1, c2 = (_view(context["out_bias"], k, budget) for k in (channel, channel + 1))
    r0, e0 = ar.half(ar.add(c1, c2, budget), budget), ar.half(ar.sub(c1, c2, budget), budget)
    for offset, selected in enumerate((channel, channel + 1)):
        r0 = ar.sub(r0, ar.scale(_view(context["pbias"], selected, budget),
                                 chosen_a[offset], budget), budget)
    outer_e = tuple(_view(context["out_error"], k, budget) for k in (channel, channel + 1))
    rho_eps = ar.concat((ar.half(ar.stack(outer_e, budget), budget),
        ar.stack(tuple(ar.scale(_view(context["peps"], k, budget), -chosen_a[t], budget)
                       for t, k in enumerate((channel, channel + 1))), budget)), budget)
    e_eps = ar.concat((ar.half(ar.stack((outer_e[0], ar.neg(outer_e[1], budget)), budget), budget),
                       _zeros((2,), budget)), budget)
    common_stats = ar.statistics(rho_eps, e_eps, budget)
    return dict(a=chosen_a, offsets=offsets, r0=r0, e0=e0, common_stats=common_stats)


def audit(states, phases, sb, budget, enabled=False):
    """Return all registered adjacent-pair/mask records, never a selected subset."""
    if type(enabled) is not bool:
        raise Rejected("enabled must be bool")
    if not enabled:
        return None
    context = _prepare(states, phases, sb, budget)
    channels = context["channels"]
    reference = context["reference"]
    records = []
    counts = dict(registered=0, ineligible=0, excluded=0, not_excluded=0)
    improved_classes = improved_population = 0
    ppre = context["parent_bn"]["bounds"]
    cpre = context["outer_bn"]["bounds"]
    for channel in range(channels - 1):
        i, j = channel, channel + 1
        _pay(budget, 128, 64)
        crossing = all(float(bounds.lo[k]) < 0.0 < float(bounds.hi[k])
                       for bounds in (ppre, cpre) for k in (i, j))
        scales = tuple(float(context["scales"][k]) for k in (i, j))
        valid = crossing and all(math.isfinite(s) and s > 0.0 for s in scales)
        pair = _pair(context, channel, budget) if valid else None
        reason = "parent_or_child_not_crossing" if not valid else "uncertified_template_gram"
        _pay(budget, 64, 32)
        qlo = np.zeros(2, dtype=np.float64)
        qu = []
        if pair is not None:
            for k, scale in zip((i, j), scales):
                upper = ar.div(ar.point(float(context["qbounds"].hi[k]), budget), scale, budget)
                qu.append(min(1.0, float(upper.hi)))
        for cls in context["classes"]:
            _pay(budget, 256, 128)
            population = cls["population"]
            row = dict(pair=(i, j), class_id=cls["class_id"], population=population)
            counts["registered"] += population
            if pair is None:
                row.update(status="ineligible", reason=reason)
                counts["ineligible"] += population
                records.append(row)
                continue
            active = tuple(offset for offset in range(9)
                           if cls["mask"][offset // 3][offset % 3])
            selected = tuple(pair["offsets"][offset] for offset in active)
            coefficients = ar.sum_axis(ar.stack(tuple(item[3] for item in selected), budget), 0, budget)
            b_iv, d_iv = _view(coefficients, 0, budget), _view(coefficients, 1, budget)
            b_mid, d_mid = ar.midpoint(b_iv, budget), ar.midpoint(d_iv, budget)
            _pay(budget, 32, 24)
            nominal = float(d_mid[0]) * 0.5 - float(d_mid[1]) * 0.5
            if not math.isfinite(nominal) or nominal == 0.0:
                row.update(status="ineligible", reason="zero_or_nonfinite_nominal_tau")
                counts["ineligible"] += population
                records.append(row)
                continue
            order = (0, 1) if nominal > 0.0 else (1, 0)
            tau = abs(nominal)
            targets = np.asarray((nominal, -nominal), dtype=np.float64)
            rq = ar.sub(b_iv, I(b_mid, b_mid), budget)
            eq = ar.sub(d_iv, I(targets, targets), budget)
            qbounds = I(qlo, np.asarray(qu, dtype=np.float64))
            qr0, qe0, qstats = ar.group_stats(rq, eq, qbounds, budget)
            r0 = ar.add(pair["r0"], ar.sum_iv(ar.stack(tuple(item[0] for item in selected), budget), budget), budget)
            e0 = ar.add(pair["e0"], ar.sum_iv(ar.stack(tuple(item[1] for item in selected), budget), budget), budget)
            r0, e0 = ar.add(r0, qr0, budget), ar.add(e0, qe0, budget)
            stats = ar.combine_stats((pair["common_stats"], qstats)
                                     + tuple(item[2] for item in selected), budget)
            supported = ar.support(stats, r0, e0, budget)
            a = tuple(float(pair["a"][index]) for index in order)
            b = tuple(float(b_mid[index]) for index in order)
            ref_index = context["reference_ids"][cls["mask"]]
            child_upper = tuple(float(reference.upper[ref_index, k]) for k in (i, j))
            proof = certify(a, b, float(tau), tuple(supported["r_bounds"]),
                float(supported["upper"]), float(supported["independent_upper"]), child_upper,
                budget=budget, enabled=True)
            status = proof["status"]
            counts[status] += population
            if proof["payment_credit"]:
                improved_classes += 1
                improved_population += population
            _pay(budget, 256, 192)
            row.update(status=status, parent_order=tuple((i, j)[index] for index in order),
                parent_scales=tuple(scales[index] for index in order),
                parent_u=tuple(qu[index] for index in order), a=a, b=b, tau=float(tau),
                r_bounds=supported["r_bounds"], e_bounds=supported["e_bounds"],
                residual_support=supported, certificate=proof, reference_index=ref_index,
                child_upper=child_upper, old_child_upper=tuple(float(reference.old_upper[k]) for k in (i, j)),
                selected_mean_coefficients=_bounds_record(b_iv, budget),
                selected_difference_coefficients=_bounds_record(d_iv, budget),
                gram_policy="one certified nominal-point full-stencil Gram choice shared by all masks",
                source_policy="original Trest groups plus intersecting A and original BN errors",
                same_source_axes=True, all_residual_terms_paid=True, relation_installed=False,
                source_axis_binding_scope="authenticated declared source graph; not native columns")
            records.append(row)
    expected = (channels - 1) * context["height"] * context["width"]
    _require(counts["registered"] == expected
             and sum(counts[key] for key in ("ineligible", "excluded", "not_excluded")) == expected
             and len(records) == 9 * (channels - 1), "incomplete joint pair population")
    _pay(budget, 4 * reference.upper.size + 4 * reference.old_upper.size + 256,
         reference.upper.size + reference.old_upper.size + 192)
    table = dict(masks=reference.masks, upper=reference.upper.tolist(),
        old_upper=reference.old_upper.tolist(), preactivation=_bounds_record(reference.preactivation, budget),
        same_original_bn_carrier=True, original_bn_error_retained=True,
        scope="same-H ordinary mask-matched child ranges, not a new relation")
    return dict(parent_bank=context["parent"], child_bank=context["child"],
        middle_conv=context["middle"]["node"], outer_conv=context["outer"]["node"],
        main_port=context["main_port"], skip_port=context["skip_port"],
        boundary_classes=context["classes"], records=tuple(records), counts=counts,
        reference_table=table,
        summary_metrics=dict(joint_payment_improved_classes=improved_classes,
            joint_payment_improved_population=improved_population, strict_gain_certified=False),
        full_remaining_source_terms_paid=True, exact_complete_inner_coefficients_materialized=False,
        relation_installed=False, native_relation_extraction_qualified=False)
