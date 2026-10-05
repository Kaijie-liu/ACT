"""Read-only structural inspection of an already authenticated C34 archive.

This is not a candidate, range certificate, loader, or solver.  The caller
authenticates/restores the complete archive before calling inspect_cut; this
module neither unpickles nor imports any ACT/numerical module.  No numeric
array is copied, mutated, expanded, or multiplied.  Only the 200-entry root
value map is read numerically; parent weights and bound endpoints are not.
"""

import math


ENTRY_CAP = 64_000_000
WIDTH = 200
EXPR_FIELDS = {"terms", "bias", "n_out", "frame_id"}


def _type_name(value):
    cls = type(value)
    return cls.__module__ + "." + cls.__name__


def inspect_cut(saved, lifted, *, pool, branch):
    """Return complete prerequisite metadata; every candidate gate stays false.

    branch.charge(name, amount) also charges the caller's whole pool.  Metadata
    entries count retained result containers/keys/scalars, not referenced old
    arrays, whose resident memory belongs to the caller's restored-state bill.
    Exceptions are fail-closed, not permission to return a successful subset.
    """
    if pool is None or branch is None:
        raise ValueError("explicit whole and branch accounting required")

    def charge(name, amount):
        branch.charge("d206_" + name, amount)

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def shape(value):
        result = tuple(int(n) for n in value.shape)
        charge("shape_metadata", 8 + 4 * len(result))
        require(all(n >= 0 for n in result), "negative metadata dimension")
        return result

    def expression(value):
        charge("expression_header", 32)
        require(_type_name(value) == "act.back_end.hybridz_tf.tf_cnn.SparseHZAffineExpr",
                "unknown expression class")
        require(set(vars(value)) == EXPR_FIELDS, "unknown expression fields")
        require(type(value.terms) is tuple and value.terms, "empty/non-tuple terms")
        require(type(value.n_out) is int and value.n_out > 0, "invalid expression width")
        require(shape(value.bias) == (value.n_out,), "incomplete expression bias shape")
        for term in value.terms:
            charge("term_header", 16)
            require(_type_name(term) == "act.back_end.hybridz_tf.tf_cnn.SparseHZAffineTerm"
                    and set(vars(term)) == {"source", "operators"}
                    and type(term.operators) is tuple, "unknown affine term schema")

    def bounds_metadata(value, width):
        charge("bounds_header", 24)
        require(_type_name(value) == "act.back_end.core.Bounds"
                and set(vars(value)) == {"lb", "ub"}, "unknown Bounds schema")
        tensors = []
        for name in ("lb", "ub"):
            tensor = getattr(value, name)
            require(_type_name(tensor) == "torch.Tensor", "unknown bound tensor class")
            dims = shape(tensor)
            count = math.prod(dims)
            tensors.append({"field": name, "shape": list(dims),
                            "dtype": str(tensor.dtype), "device": str(tensor.device),
                            "layout": str(tensor.layout), "entries": count})
        complete = (tensors[0]["shape"] == tensors[1]["shape"]
                    and all(t["entries"] == width for t in tensors)
                    and all(t["dtype"] == "torch.float64" for t in tensors)
                    and all(t["device"] == "cpu" and t["layout"] == "torch.strided"
                            for t in tensors))
        return {"tensors": tensors, "complete_shape_and_dtype": complete,
                "endpoint_values_read": False, "finite_order_checked": False,
                "reliability_flag_in_bounds_schema": False,
                "whole_parent_range_certificate": False,
                "status": "stored_native_bounds_only" if complete else "premise_missing"}

    charge("archive_header", 128)
    require(type(saved) is dict
            and saved.get("schema") == "c34_reconstructable_final_native_checkpoint_v1",
            "wrong authenticated archive schema")
    roots = saved["runtime_numeric_roots"]
    require(type(roots) is dict, "runtime roots must be the complete saved dictionary")
    fields = lifted.original_fields
    require(roots["original_source_fields"] is fields and roots["hz"] is lifted.hz,
            "restored runtime/source identity changed")
    pre, expr = fields["hz"], roots["expression"]
    require(expr is fields["expression"], "runtime expression is not the original source expression")
    expression(expr)
    require(pre.n_out == expr.n_out == WIDTH and pre.frame_id == expr.frame_id,
            "complete registered 200-row parent required")
    gc, gb = pre.Gc, pre.Gb
    require(_type_name(gc) == "scipy.sparse._csr.csr_matrix"
            and _type_name(gb) == "scipy.sparse._csr.csr_matrix", "root maps are not original CSR")
    require(shape(gc) == (WIDTH, pre.n_cont) and shape(gb) == (WIDTH, pre.n_bin)
            and int(gc.nnz) == WIDTH and int(gb.nnz) == 0,
            "not a complete one-root-per-output value map")
    require(str(gc.dtype) == "float64" and shape(pre.c) == (WIDTH,), "wrong root dtype/bias")
    columns = set()
    for row in range(WIDTH):
        charge("complete_root_row", 40)
        start, stop = int(gc.indptr[row]), int(gc.indptr[row + 1])
        require(stop == start + 1, "root row not singleton")
        col, value = int(gc.indices[start]), float(gc.data[start])
        require(0 <= col < pre.n_cont and col not in columns, "root columns not distinct")
        require(math.isfinite(value) and value > 0 and math.frexp(value)[0] == 0.5,
                "root coefficient not a positive power of two")
        bias, expected = float(pre.c[row]), float(expr.bias[row])
        require(math.isfinite(bias) and bias.hex() == expected.hex(), "root full bias differs")
        columns.add(col)

    require(all(term.operators for term in expr.terms), "missing final shared operator")
    final_op = expr.terms[0].operators[-1]
    require(_type_name(final_op) == "scipy.sparse._csr.csr_matrix", "final operator not CSR")
    final_shape = shape(final_op)
    require(len(final_shape) == 2 and final_shape[0] == WIDTH, "final operator output mismatch")
    for term in expr.terms:
        charge("common_final_operator", 8)
        require(term.operators[-1] is final_op, "not one shared final operator")

    # HybridzTF.apply(L, input_bounds, net, before, after); L is stored separately.
    args, kwargs = roots["apply_args"], roots["apply_kwargs"]
    require(type(args) is tuple and type(kwargs) is dict and len(args) <= 4,
            "unknown saved native apply calling convention")
    names = ("input_bounds", "net", "before", "after")
    require(set(kwargs) <= set(names), "unknown native apply keyword")
    arguments = {}
    for position, name in enumerate(names):
        charge("native_apply_binding", 8)
        require(not (position < len(args) and name in kwargs), "duplicate native apply argument")
        require(position < len(args) or name in kwargs, "missing native apply argument")
        arguments[name] = args[position] if position < len(args) else kwargs[name]
    net = saved["net"]
    require(arguments["net"] is net, "saved apply net differs from authenticated net")
    facts = {name: arguments[name] for name in ("before", "after")}
    for name, population in facts.items():
        require(type(population) is dict, "saved facts are not complete dictionaries")
        for key, fact in population.items():
            charge("complete_fact_header", 16)
            require(type(key) is int and _type_name(fact) == "act.back_end.core.Fact"
                    and set(vars(fact)) == {"bounds", "cons"}, "unknown saved Fact schema")
    selected = saved["selected_layer"]
    require(roots["layer"] is selected and net.by_id.get(selected.id) is selected
            and selected.kind == "RELU", "selected native ReLU identity changed")

    cache = saved["expr_cache"]
    require(type(cache) is dict and len(cache) <= ENTRY_CAP, "invalid complete expression cache")
    matches = []
    for layer_id, parent in cache.items():
        charge("complete_cache_item", 24)
        require(type(layer_id) is int and layer_id in net.by_id, "unknown cached graph identity")
        expression(parent)
        matching = (parent.frame_id == expr.frame_id and parent.n_out == final_shape[1]
                    and len(parent.terms) == len(expr.terms))
        if len(parent.terms) == len(expr.terms):
            for old, new in zip(parent.terms, expr.terms):
                charge("prefix_source_and_length", 12)
                matching &= old.source is new.source and len(old.operators) + 1 == len(new.operators)
                if len(old.operators) + 1 == len(new.operators):
                    for index, operator in enumerate(old.operators):
                        charge("prefix_operator_identity", 4)
                        matching &= operator is new.operators[index]
        if matching:
            bound_records = []
            for name, population in facts.items():
                if layer_id in population:
                    bound_records.append({"fact_dictionary": name, "layer_id": layer_id,
                                          **bounds_metadata(population[layer_id].bounds, parent.n_out)})
            matches.append({"layer_id": layer_id, "kind": net.by_id[layer_id].kind,
                            "frame_id": parent.frame_id, "n_out": parent.n_out,
                            "term_count": len(parent.terms), "all_source_and_prefix_identities": True,
                            "shared_final_operator_shape": list(final_shape),
                            "shared_final_operator_nnz": int(final_op.nnz),
                            "bounds": bound_records, "stored_bounds_found": bool(bound_records),
                            "whole_parent_range_certificate": False,
                            "bias_matvec_or_affine_identity_evaluated": False})

    widths, slots, old_slots = saved["frame_widths"], saved["relu_slots"], roots["entry_slots"]
    require(type(widths) is dict and type(slots) is dict and type(old_slots) is dict,
            "missing complete original slot/frame dictionaries")
    selected_slots = {}
    for key, slot in slots.items():
        charge("complete_original_slot", 40)
        require(type(key) is tuple and len(key) == 3 and all(type(v) is int for v in key)
                and type(slot) is tuple and len(slot) == 3 and all(type(v) is int for v in slot),
                "unknown original ReLU slot schema")
        frame, layer_id, row = key
        require(frame in widths and layer_id in net.by_id and row >= 0, "invalid original slot identity")
        nc, nb = widths[frame]
        require(0 <= slot[0] < nc and 0 <= slot[1] < nc and 0 <= slot[2] < nb,
                "original phase slot outside final frame")
        if frame == pre.frame_id and layer_id == selected.id:
            require(0 <= row < WIDTH and row not in selected_slots and key not in old_slots,
                    "selected phase is incomplete, duplicate, or pre-existing")
            selected_slots[row] = slot
    for key, slot in old_slots.items():
        charge("all_older_slots_preserved", 12)
        require(key in slots and slots[key] == slot, "older original phase slot changed")
    require(len(selected_slots) == WIDTH and set(selected_slots) == set(range(WIDTH)),
            "selected original phase population is not all 200 rows")
    continuous, binaries = set(), set()
    for row in range(WIDTH):
        charge("selected_slot_disjointness", 16)
        first, second, binary = selected_slots[row]
        require(first != second and first not in continuous and second not in continuous
                and binary not in binaries and first >= pre.n_cont and second >= pre.n_cont
                and binary >= pre.n_bin, "selected phase overlaps its parent or another phase")
        continuous.update((first, second))
        binaries.add(binary)

    result = {
        "schema": "d206_authenticated_parent_cut_metadata_v1",
        "diagnostic_complete": True, "status": "premise_missing",
        "candidate_executed": False, "candidate_qualification": False,
        "source_qualification": False, "native_qualification": False, "formal_gain": 0,
        "root_map": {"rows": WIDTH, "distinct_root_columns": len(columns), "gb_nnz": 0,
                     "positive_power_two_coefficients": True, "complete_bias_equal": True,
                     "independent_root_box_common_deficit_is_zero": True},
        "parent_cache_population": len(cache), "all_cache_items_scanned": True,
        "matches": matches, "match_count": len(matches),
        "saved_fact_population": {name: len(value) for name, value in facts.items()},
        "selected_input_bounds": bounds_metadata(arguments["input_bounds"], WIDTH),
        "phase_slots": {"all_original_slot_population": len(slots),
                        "all_older_slot_population": len(old_slots), "all_older_slots_preserved": True,
                        "selected_layer_id": selected.id, "selected_rows": WIDTH,
                        "selected_unique_bits": len(binaries), "selected_unique_continuous": len(continuous),
                        "active_label": "beta=(1-original_binary_z)/2",
                        "orientation_evidence": "c32_native_blocks_v1.py:60-89; authenticated native phase equations",
                        "orientation_new_numerical_test": False},
        "premises_missing": ["whole-parent reliability of stored parent bounds not certified here",
                             "parent-port affine identity and source-scale binding not certified here",
                             "no D205 arithmetic, source census, or terminal encoding executed"],
        "all_original_state_references_retained_by_caller": True,
        "old_array_bytes_copied": 0, "old_arrays_mutated": False,
    }
    if not matches:
        result["premises_missing"].append("no full common-final-operator parent expression found")
    entries = 0

    def count_metadata(value):
        nonlocal entries
        entries += 1
        require(entries <= ENTRY_CAP, "retained metadata entry cap exceeded")
        charge("retained_metadata_field", 4)
        if type(value) is dict:
            for key, item in value.items():
                count_metadata(key)
                count_metadata(item)
        elif type(value) in (list, tuple):
            for item in value:
                count_metadata(item)
        else:
            require(type(value) in (str, int, bool, float, type(None)), "nonmetadata result object")

    count_metadata(result)
    result["retained_result_entries_upper"] = entries + 3
    require(entries + 3 <= ENTRY_CAP, "retained metadata entry cap exceeded")
    charge("retained_entry_counter", 12)
    return result
