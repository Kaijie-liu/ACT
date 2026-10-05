"""Opt-in complete-prefix, same-H mask-reference observer; no relation install.

All modules, original bytes and the D241/D179 evidence supplied to ``extract``
must be authenticated by the caller.  This file performs no file I/O, solver
call, native-HZ construction, phase search or model runtime call.  Array bounds
describe the complete implicit D242 graph, including its free BN error terms;
they do not replace that graph by independent intervals.  The observer only
tests D258's sufficient redundancy condition for the registered adjacent pair
family against both the old and mask-matched ordinary child ranges.  The old
difference, peeling, tau and parent normalization path remains unchanged.
Escaping either condition is NOT a capability/verification result.
"""

from fractions import Fraction
import hashlib
import math

import numpy as np


SCHEMA = "d261_mask_matched_reference_v1"
REFERENCE_SHA256 = {
    "5747c00f20d8458b60da85c6ae446b4689409307146ca02f439277fbb7d89f16":
        "9494cb57122809d61f62a7c75f7ee4533c8b41e9683cda86a668321fbef20dc5",
    "aba117ad0ad4abdd630c220beca70cd58825e72e7bada5dffdda10bb725cece4":
        "5374d47e99db2a34e2099603b469dcb50d828050e536900e1d832267bf8222e0",
    "234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776":
        "ccac731af636a70c1b6210fb8ea6bdd6108fd72ab0949547263fee1bd48b7bed",
}


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _pay(budget, work, entries=0):
    budget.charge(int(work), entries=int(entries))


class _WorkOnly:
    """No new budget: adapt the existing D241 metadata helpers' work API."""

    def __init__(self, meter):
        self.meter = meter

    def charge(self, amount):
        _pay(self.meter, amount)


def _fraction(pair, budget):
    _pay(budget, 12, 3)
    _require(type(pair) in (tuple, list) and len(pair) == 2
             and all(type(v) is int for v in pair), "invalid rational evidence")
    n, d = pair
    _require(d > 0 and max(abs(n).bit_length(), d.bit_length()) <= 512,
             "rational evidence exceeds the original bit cap")
    return Fraction(n, d)


def _point(sb, values, budget):
    _pay(budget, 4)
    return sb.point(values, budget=budget, enabled=True)


def _slice(sb, value, key, budget):
    # The only caller retains the full channel axis and fixes two kernel axes.
    # NumPy creates two views, not two copied K-element payloads.  The fused
    # consumer pays its actual element reads; this helper pays view metadata.
    _pay(budget, 24, 16)
    lo, hi = value.lo[key], value.hi[key]
    _require(type(lo) is np.ndarray and type(hi) is np.ndarray
             and lo.dtype == np.float64 and hi.dtype == np.float64
             and lo.shape == hi.shape, "channel views required")
    return sb.Interval(lo, hi)


def _scalar(value):
    _require(np.shape(value.lo) == () and np.shape(value.hi) == (),
             "a scalar interval is required")
    return float(value.lo), float(value.hi)


def _record_interval(value, budget):
    size = int(value.lo.size)
    _pay(budget, 8 * size + 16, 2 * size + 2)
    return [value.lo.tolist(), value.hi.tolist()]


def _node_record(node):
    return dict(node, inputs=list(node["inputs"]), outputs=list(node["outputs"]))


def geometry_classes(height, width, *, budget, enabled=False):
    """All same-pad 3x3 masks, with an explicit whole-position census."""
    if not enabled:
        return None
    _require(type(height) is int and type(width) is int
             and 0 < height <= 65536 and 0 < width <= 65536,
             "positive bounded spatial dimensions required")
    _pay(budget, 128 + 24 * height * width, 128)
    by_mask = {}
    for y in range(height):
        for x in range(width):
            mask = tuple(tuple(0 <= y + ky - 1 < height
                               and 0 <= x + kx - 1 < width
                               for kx in range(3)) for ky in range(3))
            if mask not in by_mask:
                _pay(budget, 32, 24)
                by_mask[mask] = dict(class_id=len(by_mask), mask=mask,
                                     representative=(y, x), population=0)
            by_mask[mask]["population"] += 1
    result = tuple(by_mask.values())
    _require(sum(item["population"] for item in result) == height * width
             and len(result) <= 9, "incomplete spatial-mask partition")
    return result


def payment_classification(e_bounds, tau, parent_u, child_upper, *, budget,
                           enabled=False, fair_child_upper=None):
    """Compare chosen dyadic constants exactly, not rounded inequalities."""
    if not enabled:
        return None
    _pay(budget, 256, 64)
    values = tuple(e_bounds) + (tau,) + tuple(parent_u) + tuple(child_upper)
    _require(len(values) == 7 and all(type(v) is float and math.isfinite(v)
                                    for v in values), "finite scalar payment data required")
    exact = tuple(Fraction.from_float(v) for v in values)
    _require(all(max(abs(v.numerator).bit_length(), v.denominator.bit_length()) <= 512
                 for v in exact), "payment constant exceeds the original bit cap")
    le, ue, t, u1, u2, y1, y2 = exact
    _require(le <= ue and t > 0 and 0 <= u1 <= 1 and 0 <= u2 <= 1
             and y1 >= 0 and y2 >= 0, "invalid D258 comparison ranges")
    plus, minus = 2 * max(ue, 0), -2 * min(le, 0)
    tp, tm = y1 + 2 * t * u2, y2 + 2 * t * u1
    _require(all(max(abs(v.numerator).bit_length(), v.denominator.bit_length()) <= 512
                 for v in (plus, minus, tp, tm)), "payment result exceeds the original bit cap")
    answer = dict(excluded_by_d258=(plus >= tp and minus >= tm),
                  plus_condition=plus >= tp, minus_condition=minus >= tm)
    answer["exact_payment_and_threshold"] = tuple((v.numerator, v.denominator)
                                                   for v in (plus, minus, tp, tm))
    if fair_child_upper is not None:
        # Reuse the original seven exact constants and the two old thresholds.
        # Do not repeat the full conversion/payment computation for this view.
        _pay(budget, 128, 40)
        _require(type(fair_child_upper) in (tuple, list) and len(fair_child_upper) == 2
                 and all(type(v) is float and math.isfinite(v) and v >= 0.0
                         for v in fair_child_upper), "finite nonnegative fair child ranges required")
        fy1, fy2 = tuple(Fraction.from_float(v) for v in fair_child_upper)
        _require(fy1 <= y1 and fy2 <= y2, "fair reference may not widen the old child range")
        ftp, ftm = tp - y1 + fy1, tm - y2 + fy2
        _require(all(max(abs(v.numerator).bit_length(), v.denominator.bit_length()) <= 512
                     for v in (fy1, fy2, ftp, ftm)), "fair threshold exceeds the original bit cap")
        fair = dict(excluded_by_d258=(plus >= ftp and minus >= ftm),
                    plus_condition=plus >= ftp, minus_condition=minus >= ftm,
                    child_upper=tuple(fair_child_upper),
                    exact_payment_and_threshold=tuple((v.numerator, v.denominator)
                                                       for v in (plus, minus, ftp, ftm)))
        _require(not answer["excluded_by_d258"] or fair["excluded_by_d258"],
                 "ordinary-range refinement reversed exclusion")
        answer["fair_reference"] = fair
    return answer


def _packed(tensor, descriptor, onnx, budget):
    """Full authenticated FLOAT32 payload; no Fraction weight decoder."""
    shape = tuple(int(v) for v in tensor.dims)
    _pay(budget, 128 + 8 * len(shape))
    _require(shape and all(0 < v <= 65536 for v in shape), "invalid initializer shape")
    count = math.prod(shape)
    _require(count <= 1_000_000 and descriptor.get("name") == tensor.name
             and descriptor.get("shape") == list(shape)
             and descriptor.get("scalar_count") == count,
             "complete initializer descriptor differs")
    _require(tensor.data_type == onnx.TensorProto.FLOAT
             and descriptor.get("dtype") == 1
             and descriptor.get("payload_encoding") == "raw_data"
             and descriptor.get("scalar_bytes") == 4
             and not tensor.external_data
             and tensor.data_location != onnx.TensorProto.EXTERNAL,
             "only the authenticated internal FLOAT32 raw payload is registered")
    size = tensor.ByteSize()
    _pay(budget, 16 * count + 3 * size + 128, 3 * count + 16)
    raw = tensor.raw_data
    _require(len(raw) == 4 * count and not tensor.float_data and not tensor.double_data
             and not tensor.int32_data and not tensor.int64_data
             and not tensor.uint64_data and not tensor.string_data,
             "ambiguous or malformed original payload")
    encoded = tensor.SerializeToString(deterministic=True)
    _require(len(encoded) == size and descriptor.get("tensor_proto_bytes") == size
             and hashlib.sha256(encoded).hexdigest() == descriptor.get("tensor_proto_sha256")
             and hashlib.sha256(raw).hexdigest() == descriptor.get("raw_data_sha256")
             and descriptor.get("raw_data_bytes") == len(raw),
             "complete initializer bytes differ from D241")
    values = np.frombuffer(raw, dtype="<f4").astype(np.float64).reshape(shape)
    _require(bool(np.isfinite(values).all()), "nonfinite original parameter")
    values.setflags(write=False)
    return values


def _bn_intervals(record, channels, sb, budget):
    post = record.get("outward_post_affine")
    _require(type(post) is list and len(post) == channels,
             "complete authenticated BN envelope required")
    _pay(budget, 32 + 8 * channels, 4 * channels)
    a, b = [], []
    for item in post:
        _require(type(item) is list and len(item) == 2
                 and all(type(v) is list and len(v) == 2 for v in item),
                 "malformed BN coefficient interval")
        a.append((_fraction(item[0][0], budget), _fraction(item[0][1], budget)))
        b.append((_fraction(item[1][0], budget), _fraction(item[1][1], budget)))
    return (sb.from_rationals(a, budget=budget, enabled=True),
            sb.from_rationals(b, budget=budget, enabled=True))


def _op(sb, name, *args, budget):
    return getattr(sb, name)(*args, budget=budget, enabled=True)


def _sum(sb, value, budget):
    return sb.sum_axis(value, axis=0, budget=budget, enabled=True)


def _pair_audit(states, phases, sb, budget):
    """Complete registered family; reuse masks, never sample its members."""
    parent, child = phases[1], phases[2]
    pshape, cshape = parent["shape"], child["shape"]
    _require(pshape == cshape and pshape[0] == 1,
             "registered parent/child banks must have identical complete shape")
    channels, height, width = pshape[1:]
    outer_bn = states[child["preactivation"]]
    _require(outer_bn["kind"] == "batchnorm", "last original preactivation must be BN")
    outer = states[outer_bn["input"]]
    _require(outer["kind"] == "conv", "last BN must directly consume original Conv")
    add = states[outer["input"]]
    _require(add["kind"] == "add", "last Conv must directly consume original Add")
    main_candidates = []
    for port in add["inputs"]:
        possible = states[port]
        if possible["kind"] == "batchnorm":
            previous = states[possible["input"]]
            if previous["kind"] == "conv" and previous["input"] == parent["output"]:
                main_candidates.append((port, possible, previous))
    _require(len(main_candidates) == 1, "unique original second-ReLU main branch required")
    main_port, middle_bn, middle = main_candidates[0]
    skip_port = next(port for port in add["inputs"] if port != main_port)
    for conv in (middle, outer):
        _require(conv["weight_shape"] == (channels, channels, 3, 3)
                 and conv["strides"] == (1, 1) and conv["dilations"] == (1, 1)
                 and conv["pads"] == (1, 1, 1, 1) and conv["group"] == 1,
                 "registered equal-shape two-convolution geometry differs")
    classes = geometry_classes(height, width, budget=budget, enabled=True)
    ppre = states[parent["preactivation"]]["bounds"]
    cpre = states[child["preactivation"]]["bounds"]
    qbounds = states[parent["output"]]["bounds"]
    ybounds = states[child["output"]]["bounds"]
    oa, ob, oe = (outer_bn["carrier"].nominal_a,
                  outer_bn["carrier"].nominal_b, outer_bn["carrier"].error)
    weights = outer["weights"]
    skip_bounds = states[skip_port]["bounds"]
    # Compute every output channel and every actual mask once, on the SAME
    # nominal BN carrier and original E.  In particular, do not rebuild BN on
    # a smaller Conv range: that would change the declared source graph H.
    reference = sb.masked_child_bounds(outer["ledger"], add["bounds"],
        outer_bn["carrier"], budget=budget, enabled=True)
    _pay(budget, 128 + 32 * len(classes) + 4 * channels,
         64 + 16 * len(classes) + channels)
    _require(len(reference.masks) == len(classes)
             and reference.upper.shape == (len(classes), channels)
             and reference.old_upper.shape == (channels,)
             and bool(np.equal(reference.old_upper, ybounds.hi).all()),
             "complete same-H ordinary child reference differs")
    reference_ids = {mask: index for index, mask in enumerate(reference.masks)}
    _require(len(reference_ids) == len(classes)
             and set(reference_ids) == {item["mask"] for item in classes},
             "ordinary reference masks omit or duplicate a spatial class")
    _pay(budget, 2 * reference.upper.size + 2 * reference.old_upper.size + 32,
         reference.upper.size + reference.old_upper.size + 32)
    reference_table = dict(masks=reference.masks,
        preactivation=_record_interval(reference.preactivation, budget),
        upper=reference.upper.tolist(), old_upper=reference.old_upper.tolist(),
        same_original_bn_carrier=True, original_bn_error_retained=True,
        scope="ordinary child ranges on the unchanged H; not a new relation")
    records = []
    counts = dict(registered=0, ineligible=0, excluded=0, not_excluded=0)
    for channel in range(channels - 1):
        i, j = channel, channel + 1
        _pay(budget, 128, 64)
        scales = (max(-float(ppre.lo[i]), float(ppre.hi[i])),
                  max(-float(ppre.lo[j]), float(ppre.hi[j])))
        crossing = (float(ppre.lo[i]) < 0 < float(ppre.hi[i])
                    and float(ppre.lo[j]) < 0 < float(ppre.hi[j])
                    and float(cpre.lo[i]) < 0 < float(cpre.hi[i])
                    and float(cpre.lo[j]) < 0 < float(cpre.hi[j]))
        if not crossing or not all(math.isfinite(s) and s > 0 for s in scales):
            for cls in classes:
                _pay(budget, 64, 32)
                count = cls["population"]
                records.append(dict(pair=(i, j), class_id=cls["class_id"], population=count,
                                    status="ineligible", reason="parent_or_child_not_crossing"))
                counts["registered"] += count
                counts["ineligible"] += count
            continue
        _pay(budget, 16, 16)
        h = sb.fused_h(weights[i:j + 1], oa[i:j + 1], budget=budget, enabled=True)
        # The original outer Conv bias is part of g, even when a particular
        # registered model happened to encode an implicit zero bias.
        child_constants = []
        for child_channel in (i, j):
            product = _op(sb, "mul", _point(sb, float(oa[child_channel]), budget),
                _point(sb, float(outer["ledger"].bias[child_channel]), budget), budget=budget)
            child_constants.append(_op(sb, "add", product,
                                       _point(sb, float(ob[child_channel]), budget), budget=budget))
        constant = _op(sb, "scale_half", _op(sb, "sub", child_constants[0],
                       child_constants[1], budget=budget), budget=budget)
        outer_error = _op(sb, "scale_half", _op(sb, "add",
            _point(sb, float(oe[i]), budget), _point(sb, float(oe[j]), budget), budget=budget), budget=budget)
        error_radius = float(outer_error.hi)
        constant = _op(sb, "add", constant,
                       sb.Interval(np.asarray(-error_radius), np.asarray(error_radius)), budget=budget)
        offset_records = []
        for ky in range(3):
            for kx in range(3):
                _pay(budget, 32, 16)
                chosen = ((i, 2 - ky, 2 - kx), (j, 2 - ky, 2 - kx))
                hs = _slice(sb, h, (slice(None), ky, kx), budget)
                residual, raw_coefficients = sb.fused_offset(middle["ledger"], chosen, hs,
                    middle_bn["carrier"], skip_bounds, budget=budget, enabled=True)
                coefficients = tuple(_op(sb, "mul", raw_coefficient,
                    _point(sb, float(scale), budget), budget=budget)
                    for raw_coefficient, scale in zip(raw_coefficients, scales))
                offset_records.append((residual, tuple(coefficients)))
        _pay(budget, 128, 32)
        # Ratios are bounded outward by the numerical module, then capped at 1
        # using the independently proved normalization Q<=max(-L,U).
        pu = []
        for p, scale in zip((i, j), scales):
            exact = Fraction.from_float(float(qbounds.hi[p])) / Fraction.from_float(scale)
            upper = float(exact)
            if Fraction.from_float(upper) < exact:
                upper = math.nextafter(upper, math.inf)
            pu.append(min(1.0, upper))
        for cls in classes:
            _pay(budget, 256, 64)
            erest = constant
            coeff = [_point(sb, 0.0, budget), _point(sb, 0.0, budget)]
            for offset, (residual, two_coeffs) in enumerate(offset_records):
                if cls["mask"][offset // 3][offset % 3]:
                    erest = _op(sb, "add", erest, residual, budget=budget)
                    coeff = [_op(sb, "add", old, new, budget=budget)
                             for old, new in zip(coeff, two_coeffs)]
            endpoints = [_scalar(value) for value in coeff]
            midpoint = [lo * 0.5 + hi * 0.5 for lo, hi in endpoints]
            nominal = (midpoint[0] - midpoint[1]) * 0.5
            count = cls["population"]
            row = dict(pair=(i, j), class_id=cls["class_id"], population=count)
            counts["registered"] += count
            if not math.isfinite(nominal) or nominal == 0.0:
                row.update(status="ineligible", reason="zero_or_nonfinite_nominal_tau")
                counts["ineligible"] += count
                records.append(row)
                continue
            order = (0, 1) if nominal > 0 else (1, 0)
            tau = abs(nominal)
            c1, c2 = coeff[order[0]], coeff[order[1]]
            normalized_upper = (float(pu[order[0]]), float(pu[order[1]]))
            residual1 = _op(sb, "sub", c1, _point(sb, tau, budget), budget=budget)
            residual2 = _op(sb, "add", c2, _point(sb, tau, budget), budget=budget)
            for residual, upper in zip((residual1, residual2), normalized_upper):
                q = sb.Interval(np.asarray(0.0), np.asarray(upper))
                erest = _op(sb, "add", erest, _op(sb, "mul", residual, q, budget=budget), budget=budget)
            eb = _scalar(erest)
            yu = (float(ybounds.hi[i]), float(ybounds.hi[j]))
            reference_index = reference_ids[cls["mask"]]
            fair_yu = (float(reference.upper[reference_index, i]),
                       float(reference.upper[reference_index, j]))
            verdict = payment_classification(eb, float(tau), normalized_upper, yu,
                budget=budget, enabled=True, fair_child_upper=fair_yu)
            fair_verdict = verdict["fair_reference"]
            status = "excluded" if fair_verdict["excluded_by_d258"] else "not_excluded"
            counts[status] += count
            row.update(status=status, parent_order=tuple((i, j)[v] for v in order),
                       tau=tau, parent_u=normalized_upper,
                       parent_scales=tuple(scales[v] for v in order), e_bounds=eb,
                       selected_coefficients=tuple(endpoints[v] for v in order),
                       old_child_upper=yu,
                       old_status="excluded" if verdict["excluded_by_d258"] else "not_excluded",
                       old_excluded_by_d258=verdict["excluded_by_d258"],
                       old_plus_condition=verdict["plus_condition"],
                       old_minus_condition=verdict["minus_condition"],
                       old_exact_payment_and_threshold=verdict["exact_payment_and_threshold"],
                       reference_index=reference_index, **fair_verdict)
            records.append(row)
    population = (channels - 1) * height * width
    _require(counts["registered"] == population
             and counts["ineligible"] + counts["excluded"] + counts["not_excluded"] == population
             and len(records) == (channels - 1) * len(classes), "incomplete pair-family census")
    # One paid metadata pass records old/fair counts, including ineligible rows;
    # no second numerical bound path or reread of an old source JSON is used.
    _pay(budget, 48 * len(records) + 128, 8 * len(records) + 96)
    old_counts = dict(registered=0, ineligible=0, excluded=0, not_excluded=0)
    regional_counts = {name: dict(registered=0, ineligible=0, old_excluded=0,
        old_not_excluded=0, fair_excluded=0, fair_not_excluded=0, newly_excluded=0)
        for name in ("interior", "boundary")}
    class_masks = {item["class_id"]: item["mask"] for item in classes}
    for row in records:
        count, status = row["population"], row["status"]
        old_status = row.setdefault("old_status", "ineligible")
        _require(old_status in old_counts and old_status != "registered",
                 "missing original classification")
        _require(old_status != "excluded" or status == "excluded",
                 "mask-matched exclusion is not monotone")
        _require((old_status == "ineligible") == (status == "ineligible"),
                 "mask comparison changed eligibility or population")
        old_counts["registered"] += count
        old_counts[old_status] += count
        mask = class_masks[row["class_id"]]
        region = regional_counts["interior" if all(all(r) for r in mask) else "boundary"]
        region["registered"] += count
        if status == "ineligible":
            region["ineligible"] += count
        else:
            region["old_" + old_status] += count
            region["fair_" + status] += count
            if old_status == "not_excluded" and status == "excluded":
                region["newly_excluded"] += count
    _require(old_counts["registered"] == counts["registered"]
             and old_counts["ineligible"] == counts["ineligible"],
             "old and fair populations differ")
    return dict(parent_bank=parent, child_bank=child, middle_conv=middle["node"],
                outer_conv=outer["node"], main_port=main_port, skip_port=skip_port,
                boundary_classes=classes, counts=counts, records=tuple(records),
                old_counts=old_counts, regional_counts=regional_counts,
                reference_table=reference_table, fair_reference_monotonicity_verified=True,
                exact_complete_inner_coefficients_materialized=False,
                full_remaining_source_terms_paid=True, relation_installed=False,
                qg_source_proposal_only=True, native_relation_extraction_qualified=False,
                tau_semantics="positive nominal midpoint difference; all coefficient uncertainty paid")


def extract(raw, spec, source, structure, meter, k, helper, base, *, reference,
            reference_identity, geometry, bounds, enabled=False):
    """One complete original source, with caller-owned shared budgets/identity."""
    if not enabled:
        return None
    del k  # BN envelopes are authenticated D241 evidence, not recomputed here.
    _require(type(raw) is bytes and 0 < len(raw) <= 64 * 1024 * 1024
             and type(spec) is bytes and spec, "bounded original raw bytes required")
    _pay(meter, 4096 + len(raw) + 8 * len(spec))
    model_sha, spec_sha = hashlib.sha256(raw).hexdigest(), hashlib.sha256(spec).hexdigest()
    _require(source.get("model_sha256") == model_sha and source.get("spec_sha256") == spec_sha,
             "original model/property identity differs")
    _require(reference.get("schema") == "d241_complete_residual_source_v1"
             and reference.get("source_binding_complete") is True
             and reference.get("source") == source
             and reference_identity.get("sha256") == REFERENCE_SHA256.get(model_sha),
             "authenticated complete D241 source evidence is required")
    _require(structure.get("schema") == "d179_preterminal_domain_v1"
             and structure.get("structure_complete") is True
             and structure.get("metadata_complete") is True
             and structure.get("source", {}).get("model_sha256") == model_sha,
             "authenticated complete D179 original structure required")
    import onnx  # The runner verifies find_spec and actual __file__ before entry.

    model = onnx.ModelProto()
    model.ParseFromString(raw)
    _require(len(model.graph.input) == 1, "one original graph input is required")
    original = model.graph.input[0]
    declared = geometry._dimensions(original)
    frame = reference["frame_identity"]
    _require(list(declared) == frame["original_declared_dimensions"],
             "original input declaration differs")
    shape = tuple(frame["input_shape"])
    _require(shape == tuple(structure["input_shape"]) and len(shape) == 4
             and shape[0] == 1 and shape[1] == 3
             and original.name == frame["input_name"]
             and int(original.type.tensor_type.elem_type) == frame["input_dtype"],
             "original complete NCHW input binding differs")
    reader = base._Reader(onnx, model)
    work = _WorkOnly(meter)
    original_nodes, consumers = geometry._complete_metadata(reader, structure, work)
    relu_indices = tuple(i for i, node in enumerate(reader.nodes) if node.op_type == "Relu")
    _require(len(relu_indices) >= 3, "third original ReLU is missing")
    last = relu_indices[2]
    evidence_nodes = reference["prefix_nodes"]
    _require(len(evidence_nodes) == last + 1, "complete prefix population differs")
    descriptions = {item["name"]: item for item in reference["initializer_descriptors"]}
    required = []
    for node in reader.nodes[:last + 1]:
        _pay(meter, 32 + 8 * len(node.input))
        _require(node.op_type in ("Conv", "BatchNormalization", "Relu", "Add"),
                 "unregistered complete-prefix operator")
        if node.op_type in ("Conv", "BatchNormalization"):
            for name in node.input[1:]:
                if name and name not in required:
                    required.append(name)
    _require(set(required) == set(descriptions), "complete initializer population differs")
    packed = {}
    for name in required:
        _require(name in reader.initializers, "original initializer is missing")
        packed[name] = _packed(reader.initializers[name], descriptions[name], onnx, meter)
    scalar_count = sum(int(value.size) for value in packed.values())
    _require(scalar_count == reference["decoded_scalar_count"], "complete parameter scalar count differs")
    box = helper.input_box(spec, shape)
    size = shape[1] * shape[2] * shape[3]
    _pay(meter, 64 * size + 64, 6 * size)
    _require(len(box) == size, "complete property coordinate population differs")
    pairs = []
    for index in range(size):
        lo, hi = box[index]
        _require(reference["input_box"][str(index)] == [[lo.numerator, lo.denominator],
                                                       [hi.numerator, hi.denominator]],
                 "original exact property endpoint differs")
        pairs.append((lo, hi))
    pixel_bounds = bounds.from_rationals(pairs, budget=meter, enabled=True)
    _pay(meter, 4 * size + 32, 6)
    ilo = pixel_bounds.lo.reshape(shape[1], -1).min(axis=1)
    ihi = pixel_bounds.hi.reshape(shape[1], -1).max(axis=1)
    states = {original.name: dict(kind="input", shape=shape,
                                  bounds=bounds.Interval(ilo, ihi))}
    phases, port_records, bn_records = [], [], []
    for index, node in enumerate(reader.nodes[:last + 1]):
        _pay(meter, 128 + 16 * len(node.input), 64)
        expected, old = structure["nodes"][index], evidence_nodes[index]
        geometry._attributes(onnx, reader, node, expected, work)
        output = reader.one_output(node)
        _require(output not in states, "duplicate original prefix port")
        if node.op_type == "Conv":
            _require(len(node.input) in (2, 3) and node.input[0] in states
                     and old.get("kind") == "conv", "unbound original Conv")
            incoming = states[node.input[0]]
            weight = packed[node.input[1]]
            ws = tuple(weight.shape)
            strides, pads, dilations = tuple(old["strides"]), tuple(old["pads"]), tuple(old["dilations"])
            group = int(old["group"])
            _require(len(ws) == 4 and group == 1 and ws[1] == incoming["shape"][1],
                     "unsupported original Conv channel geometry")
            sy, sx = strides
            dy, dx = dilations
            top, left, bottom, right = pads
            oh = (incoming["shape"][2] + top + bottom - dy * (ws[2] - 1) - 1) // sy + 1
            ow = (incoming["shape"][3] + left + right - dx * (ws[3] - 1) - 1) // sx + 1
            outshape = (1, ws[0], oh, ow)
            _require(min(sy, sx, dy, dx, oh, ow) > 0
                     and list(outshape) == old["output_shape"]
                     and list(incoming["shape"]) == old["input_shape"], "Conv shape differs")
            geometry._geometry(dict(old, weight_shape=ws), expected, work)
            _pay(meter, 8 * ws[0] + 16, ws[0])
            bias = packed[node.input[2]] if len(node.input) == 3 else np.zeros(ws[0], dtype=np.float64)
            ledger = bounds.conv_channel(weight, bias, incoming["bounds"], pads, meter, enabled=True)
            state = dict(kind="conv", node=index, input=node.input[0], shape=outshape,
                         weights=weight, weight_shape=ws, strides=strides, pads=pads,
                         dilations=dilations, group=group, ledger=ledger, bounds=ledger.bounds)
        elif node.op_type == "BatchNormalization":
            _require(len(node.input) == 5 and node.input[0] in states
                     and old.get("kind") == "batchnorm", "unbound original BN")
            incoming = states[node.input[0]]
            outshape = incoming["shape"]
            a, b = _bn_intervals(old, outshape[1], bounds, meter)
            carrier = bounds.bn_carrier(incoming["bounds"], a, b, budget=meter, enabled=True)
            state = dict(kind="batchnorm", node=index, input=node.input[0], shape=outshape,
                         carrier=carrier, bounds=carrier.bounds)
            _pay(meter, 16 * outshape[1] + 32, 3 * outshape[1])
            bn_records.append(dict(node=index, output=output,
                nominal_a=carrier.nominal_a.tolist(), nominal_b=carrier.nominal_b.tolist(),
                error=carrier.error.tolist(),
                error_identity="(model_sha,spec_sha,BN output,channel,row,column)",
                error_independent_per_actual_scalar=True, shared_by_all_consumers=True))
        elif node.op_type == "Relu":
            _require(len(node.input) == 1 and node.input[0] in states,
                     "unbound original ReLU")
            incoming = states[node.input[0]]
            outshape = incoming["shape"]
            state = dict(kind="relu", node=index, input=node.input[0], shape=outshape,
                         bounds=_op(bounds, "relu", incoming["bounds"], budget=meter))
            phases.append(dict(node=index, preactivation=node.input[0], output=output,
                               shape=outshape, scalar_population=math.prod(outshape[1:]),
                               actual_native_phase_columns_bound=False))
        else:
            _require(len(node.input) == 2 and all(port in states for port in node.input),
                     "both dynamic residual sources required")
            left_state, right_state = (states[port] for port in node.input)
            _require(left_state["shape"] == right_state["shape"], "Add broadcasting is unsupported")
            outshape = left_state["shape"]
            state = dict(kind="add", node=index, inputs=tuple(node.input), shape=outshape,
                         bounds=_op(bounds, "add", left_state["bounds"], right_state["bounds"], budget=meter))
        _require(list(outshape) == expected["output_shape"]
                 and list(outshape) == reference["prefix_port_shapes"][output],
                 "complete original port shape differs")
        states[output] = state
        port_records.append(dict(node=index, output=output, shape=outshape,
                                 channel_bounds=_record_interval(state["bounds"], meter)))
    _require(len(phases) == 3, "complete three-ReLU population differs")
    pairs_audit = _pair_audit(states, phases, bounds, meter)
    counts = pairs_audit["counts"]
    prefix_ports = tuple(states)
    later = tuple(dict(source=port, consumer=_node_record(original_nodes[idx]), input_slot=slot)
                  for port in prefix_ports for idx, slot in consumers.get(port, ()) if idx > last)
    _require(list(later) == reference["later_consumers"], "complete future-consumer boundary differs")
    summary = dict(prefix_node_count=last + 1, relu_banks=3,
        relu_scalar_populations=tuple(item["scalar_population"] for item in phases),
        final_relu_output=phases[-1]["output"], input_coordinates=size,
        initializer_count=len(required), packed_scalar_count=scalar_count,
        registered_pair_population=counts["registered"],
        geometry_class_count=len(pairs_audit["boundary_classes"]),
        classified_population=counts["excluded"] + counts["not_excluded"],
        excluded_population=counts["excluded"], not_excluded_population=counts["not_excluded"],
        ineligible_population=counts["ineligible"], pair_class_records=len(pairs_audit["records"]),
        old_excluded_population=pairs_audit["old_counts"]["excluded"],
        old_not_excluded_population=pairs_audit["old_counts"]["not_excluded"],
        fair_newly_excluded_population=counts["excluded"] - pairs_audit["old_counts"]["excluded"],
        fair_reference_monotonicity_verified=True,
        reference_output_mask_count=len(pairs_audit["boundary_classes"]) * phases[2]["shape"][1],
        reference_regional_counts=pairs_audit["regional_counts"],
        temp_numeric_entries_upper=0,
        numeric_entries_scope="all allocations/visits prepaid cumulatively by shared meter; no second peak prepayment",
        complete_channel_bounds_propagated=True, full_source_residual_terms_paid=True,
        outgoing_consumer_slots=len(later), selected_windows=None, model_runtime_calls=0,
        solver_calls=0, relation_installed=False, actual_native_phase_columns_bound=False,
        native_HZ_admitted=False, actual_model_binding_qualified=False,
        qg_source_proposal_only=True, native_relation_extraction_qualified=False,
        complete_physical_qualification=False, new_domain_qualified=False,
        new_capability_qualified=False, formal_gain=0, independent_e0_gain=0)
    record = dict(schema=SCHEMA, source=source, source_binding_complete=True,
        raw_parameters_verified=True, through_third_relu=True,
        all_prefix_consumers_accounted=True, frame_identity=frame,
        source_reference=reference_identity, raw_model=dict(sha256=model_sha, bytes=len(raw)),
        raw_spec=dict(sha256=spec_sha, bytes=len(spec)), input_box=reference["input_box"],
        initializer_descriptors=reference["initializer_descriptors"],
        original_graph_node_count=len(original_nodes), original_opsets=dict(reader.opsets),
        prefix_nodes=tuple(original_nodes[:last + 1]), prefix_port_shapes=reference["prefix_port_shapes"],
        prefix_all_graph_consumers=reference["prefix_all_graph_consumers"], later_consumers=later,
        original_graph_outputs=reference["original_graph_outputs"], logical_phases=tuple(phases),
        channel_bounds=tuple(port_records), bn_carriers=tuple(bn_records), pair_audit=pairs_audit,
        summary=summary, actual_model_binding_qualified=False, native_HZ_admitted=False,
        complete_physical_qualification=False, new_domain_qualified=False,
        new_capability_qualified=False, formal_gain=0, independent_e0_gain=0)
    roots = dict(source=source, model_raw=raw, spec_raw=spec, structure=structure,
                 reference=reference, evidence=record)
    # All packed arrays, ledgers and bound temporaries expire on return.  Their
    # entire real lifetime remains inside the caller's unreset process peaks.
    return roots, record


def validate_record(record, source, structure, meter):
    """Small complete-population check; no rereading or numerical inference."""
    _pay(meter, 256, 64)
    _require(type(record) is dict and record.get("schema") == SCHEMA
             and record.get("source") == source
             and record.get("source_binding_complete") is True
             and record.get("raw_parameters_verified") is True
             and record.get("through_third_relu") is True
             and record.get("all_prefix_consumers_accounted") is True,
             "incomplete source observer record")
    summary = record.get("summary")
    _require(type(summary) is dict, "source summary missing")
    original_relus = tuple(node for node in structure["nodes"] if node["op"] == "Relu")[:3]
    _require(len(original_relus) == 3, "original third-ReLU record missing")
    shapes = tuple(tuple(node["output_shape"]) for node in original_relus)
    _require(shapes[1] == shapes[2] and len(shapes[1]) == 4,
             "registered source shapes differ")
    channels, height, width = shapes[1][1:]
    expected_population = (channels - 1) * height * width
    audit = record.get("pair_audit", {})
    classes, rows = audit.get("boundary_classes"), audit.get("records")
    _require(type(classes) in (tuple, list) and type(rows) in (tuple, list)
             and len(classes) <= 9 and len(rows) == (channels - 1) * len(classes),
             "source class-record population differs")
    _pay(meter, 64 * len(classes) + 96 * len(rows), 2 * len(rows) + 128)
    masks, populations, masks_by_id = set(), {}, {}
    for item in classes:
        key, count = item["class_id"], item["population"]
        mask = tuple(tuple(row) for row in item["mask"])
        _require(type(key) is int and key not in populations and mask not in masks
                 and type(count) is int and count > 0,
                 "duplicate or invalid spatial class")
        masks.add(mask)
        populations[key] = count
        masks_by_id[key] = mask
    _require(sum(populations.values()) == height * width, "spatial classes omit positions")
    seen, totals = set(), dict(registered=0, ineligible=0, excluded=0, not_excluded=0)
    old_totals = dict(registered=0, ineligible=0, excluded=0, not_excluded=0)
    table = audit.get("reference_table", {})
    _require(table.get("same_original_bn_carrier") is True
             and table.get("original_bn_error_retained") is True
             and len(table.get("masks", ())) == len(classes)
             and len(table.get("upper", ())) == len(classes)
             and len(table.get("old_upper", ())) == channels
             and all(len(values) == channels for values in table["upper"]),
             "complete same-H output/mask table missing")
    reference_masks = tuple(tuple(tuple(row) for row in mask) for mask in table["masks"])
    for item in rows:
        pair, cls, status = tuple(item["pair"]), item["class_id"], item["status"]
        _require(len(pair) == 2 and all(type(v) is int for v in pair)
                 and 0 <= pair[0] < channels - 1 and pair[1] == pair[0] + 1
                 and cls in populations and item["population"] == populations[cls]
                 and (pair, cls) not in seen and status in ("ineligible", "excluded", "not_excluded"),
                 "pair/class omission or duplication")
        seen.add((pair, cls))
        totals["registered"] += populations[cls]
        totals[status] += populations[cls]
        old_status = item.get("old_status")
        _require(old_status in old_totals and old_status != "registered"
                 and (old_status == "ineligible") == (status == "ineligible")
                 and (old_status != "excluded" or status == "excluded"),
                 "old/fair exclusion or eligibility is not monotone")
        old_totals["registered"] += populations[cls]
        old_totals[old_status] += populations[cls]
        if status != "ineligible":
            reference_index = item.get("reference_index")
            _require(type(reference_index) is int and 0 <= reference_index < len(classes)
                     and reference_masks[reference_index] == masks_by_id[cls]
                     and tuple(item["child_upper"]) == tuple(table["upper"][reference_index][p] for p in pair)
                     and tuple(item["old_child_upper"]) == tuple(table["old_upper"][p] for p in pair),
                     "pair readout differs from shared mask reference table")
    _require(totals == audit.get("counts") and totals["registered"] == expected_population
             and summary.get("registered_pair_population") == expected_population
             and summary.get("geometry_class_count") == len(classes)
             and summary.get("pair_class_records") == len(rows)
             and summary.get("classified_population") == totals["excluded"] + totals["not_excluded"]
             and summary.get("excluded_population") == totals["excluded"]
             and summary.get("not_excluded_population") == totals["not_excluded"]
             and summary.get("ineligible_population") == totals["ineligible"],
             "complete source classification counts differ")
    _require(old_totals == audit.get("old_counts")
             and old_totals["registered"] == totals["registered"]
             and old_totals["ineligible"] == totals["ineligible"]
             and summary.get("old_excluded_population") == old_totals["excluded"]
             and summary.get("old_not_excluded_population") == old_totals["not_excluded"]
             and summary.get("fair_newly_excluded_population") == totals["excluded"] - old_totals["excluded"]
             and summary.get("fair_reference_monotonicity_verified") is True
             and audit.get("fair_reference_monotonicity_verified") is True
             and summary.get("reference_output_mask_count") == len(classes) * channels
             and summary.get("reference_regional_counts") == audit.get("regional_counts"),
             "old/fair reference summary differs")
    _require(summary.get("prefix_node_count") == original_relus[-1]["index"] + 1
             and summary.get("relu_banks") == 3
             and tuple(summary.get("relu_scalar_populations", ())) == tuple(math.prod(s[1:]) for s in shapes)
             and summary.get("final_relu_output") == original_relus[-1]["output"]
             and summary.get("initializer_count") == len(record["initializer_descriptors"])
             and summary.get("packed_scalar_count") == sum(v["scalar_count"] for v in record["initializer_descriptors"])
             and summary.get("temp_numeric_entries_upper") == 0
             and summary.get("outgoing_consumer_slots") == len(record["later_consumers"]),
             "complete prefix source population differs")
    for name in ("actual_native_phase_columns_bound", "native_HZ_admitted",
                 "actual_model_binding_qualified", "complete_physical_qualification",
                 "new_domain_qualified", "new_capability_qualified", "relation_installed",
                 "native_relation_extraction_qualified"):
        _require(summary.get(name) is False, "observer qualification overclaim: " + name)
    _require(summary.get("qg_source_proposal_only") is True
             and summary.get("complete_channel_bounds_propagated") is True
             and summary.get("full_source_residual_terms_paid") is True
             and summary.get("formal_gain") == 0 and summary.get("independent_e0_gain") == 0
             and summary.get("model_runtime_calls") == 0 and summary.get("solver_calls") == 0,
             "source observer boundary differs")
    return summary
