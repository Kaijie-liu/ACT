"""Read-only opset-12 CNN shape/consumer diagnostic, never a verifier.

One authenticated in-memory protobuf parse; no weight-value decoding, forward,
checker, ONNX shape inference, solver, or domain implementation. The caller owns
authentication, once-only execution, resource limits and exclusive evidence writes.
"""
import io
import math


def require(condition, message):
    if not condition:
        raise ValueError(message)


class NoMatch(ValueError):
    pass


def broadcast(left, right):
    width = max(len(left), len(right))
    a, b = (1,) * (width - len(left)) + left, (1,) * (width - len(right)) + right
    require(all(x == y or x == 1 or y == 1 for x, y in zip(a, b)), "broadcast mismatch")
    return tuple(max(x, y) for x, y in zip(a, b))


def inspect_model(onnx, raw, inventory, meter):
    require(type(raw) is bytes and 0 < len(raw) <= 64 * 1024**2, "bounded model bytes required")
    meter.charge(4096 + len(raw))
    model = onnx.load_model(io.BytesIO(raw), format="protobuf", load_external_data=False)
    graph = model.graph
    require(not model.functions and not model.training_info and not graph.sparse_initializer,
            "functions, training graphs and sparse initializers unsupported")
    require([(x.domain, int(x.version)) for x in model.opset_import] == [("", 12)], "opset 12 required")
    require(len(graph.input) == len(graph.output) == 1 and 0 < len(graph.node) <= 1000,
            "single-input/output bounded graph required")
    require(inventory.get("metadata_complete") is True and inventory.get("all_top_level_node_ports_retained") is True,
            "complete D152 metadata required")
    nodes, old = tuple(graph.node), inventory.get("nodes")
    require(type(old) is list and len(old) == len(nodes), "saved node population mismatch")
    meter.charge(4096 + 256 * len(nodes), entries=64 * len(nodes))
    inp = graph.input[0]
    require(inp.name and inp.type.HasField("tensor_type") and inp.type.tensor_type.elem_type == 1,
            "FLOAT graph input required")
    dims = inp.type.tensor_type.shape.dim
    require(len(dims) == 4, "NCHW graph input required")
    declared = [int(d.dim_value) if d.HasField("dim_value") else {"parameter": d.dim_param} for d in dims]
    symbol = dims[0].dim_param if dims[0].HasField("dim_param") else None
    require((symbol is not None and bool(symbol)) or (dims[0].HasField("dim_value") and dims[0].dim_value == 1),
            "batch must be symbolic or one; diagnostic explicitly binds one sample")
    require(all(d.HasField("dim_value") and 0 < d.dim_value <= 1000000 for d in dims[1:]),
            "positive static spatial/channel dimensions required")
    input_shape = (1,) + tuple(int(d.dim_value) for d in dims[1:])
    require(inventory.get("graph_inputs") == [dict(name=inp.name, data_type=1,
            declared_shape=declared, shape_binding_verified=False)], "saved graph input mismatch")
    shapes, initializers, producers, consumers = {inp.name: input_shape}, {}, {inp.name: -1}, {}
    saved_init = inventory.get("initializers")
    require(type(saved_init) is list and len(saved_init) == len(graph.initializer), "initializer population mismatch")
    for tensor, saved in zip(graph.initializer, saved_init):
        name, shape = tensor.name, tuple(int(x) for x in tensor.dims)
        require(name and name not in shapes and tensor.data_type == 1 and len(shape) <= 4
                and all(0 < x <= 1000000 for x in shape), "initializer identity/type/dimensions")
        require(tensor.data_location == 0 and not tensor.external_data and not tensor.HasField("segment"),
                "external or segmented initializer forbidden")
        count = math.prod(shape)
        require(4 * count <= 64 * 1024**2, "initializer payload cap")
        typed = {key: len(getattr(tensor, key)) for key in
                 ("float_data", "int32_data", "string_data", "int64_data", "double_data", "uint64_data")}
        raw_size = len(tensor.raw_data)
        require((raw_size == 4 * count and not any(typed.values())) or
                (raw_size == 0 and typed["float_data"] == count and
                 not any(v for k, v in typed.items() if k != "float_data")), "malformed FLOAT payload")
        require(saved.get("name") == name and saved.get("dims") == list(shape) and saved.get("data_type") == 1
                and saved.get("raw_payload_bytes") == raw_size and saved.get("typed_payload_counts") == typed,
                "saved initializer metadata mismatch")
        meter.charge(256 + 16 * len(shape), entries=32 + len(shape))
        shapes[name], initializers[name] = shape, tensor
    versions = {"Conv": 11, "BatchNormalization": 9, "Relu": 6, "Add": 7, "Flatten": 11, "Gemm": 11}
    allowed = {"Conv": {"auto_pad": 3, "dilations": 7, "group": 2, "kernel_shape": 7, "pads": 7, "strides": 7},
               "BatchNormalization": {"epsilon": 1, "momentum": 1}, "Relu": {}, "Add": {},
               "Flatten": {"axis": 2}, "Gemm": {"alpha": 1, "beta": 1, "transA": 2, "transB": 2}}
    records, decoded = [], []
    for index, (node, saved) in enumerate(zip(nodes, old)):
        op = node.op_type
        require(op in versions and node.domain == "" and len(node.output) == 1 and bool(node.output[0]),
                "unsupported domain/operator/output grammar")
        require(saved.get("index") == index and saved.get("name") == node.name and saved.get("domain") == node.domain
                and saved.get("op_type") == op and saved.get("inputs") == list(node.input)
                and saved.get("outputs") == list(node.output), "saved original node/port mismatch")
        require(node.output[0] not in shapes, "duplicate output identity")
        require(node.input and node.input[0] in shapes and node.input[0] not in initializers, "dynamic first input required")
        require(all(not x or x in shapes for x in node.input), "topologically missing input")
        schema = onnx.defs.get_schema(op, 12, "")
        require(schema.since_version == versions[op] and set(schema.attributes) == set(allowed[op]),
                "authenticated operator schema differs")
        attrs, evidence = {}, {}
        for attr in node.attribute:
            require(attr.name in allowed[op] and attr.name not in attrs and not attr.ref_attr_name
                    and attr.type == allowed[op][attr.name], "unknown, duplicate or mistyped attribute")
            attrs[attr.name] = attr
        require(saved.get("attributes") == [dict(name=a.name, protobuf_type=int(a.type), value_decoded=False)
                for a in node.attribute], "saved attribute population mismatch")
        values = {}
        for name, kind in allowed[op].items():
            attr = attrs.get(name, schema.attributes[name].default_value)
            if attr.type == 0:
                continue
            require(attr.type == kind and not attr.ref_attr_name, "attribute default/type mismatch")
            field = {1: "f", 2: "i", 3: "s", 7: "ints"}[kind]
            require(all(f.name in {"name", "type", "doc_string", field} for f, _ in attr.ListFields()),
                    "ambiguous attribute payload")
            value = (float(attr.f) if kind == 1 else int(attr.i) if kind == 2 else
                     attr.s.decode("ascii") if kind == 3 else tuple(int(x) for x in attr.ints))
            require(kind != 1 or math.isfinite(value), "nonfinite FLOAT attribute")
            values[name] = value
            evidence[name] = dict(present=name in attrs, value_hex=value.hex()) if kind == 1 else dict(
                present=name in attrs, value=list(value) if kind == 7 else value)
        x = shapes[node.input[0]]
        if op == "Conv":
            require(len(node.input) in (2, 3) and len(x) == 4 and node.input[1] in initializers, "Conv arity/weight")
            w = shapes[node.input[1]]
            require(len(w) == 4, "Conv rank-four weight")
            group = values["group"]
            kernel = values.get("kernel_shape", w[2:])
            stride, dilation, pads = values.get("strides", (1, 1)), values.get("dilations", (1, 1)), values.get("pads", (0,) * 4)
            require(values["auto_pad"] == "NOTSET" and not ("auto_pad" in attrs and "pads" in attrs), "explicit Conv padding grammar")
            require(group > 0 and x[1] == w[1] * group and w[0] % group == 0 and kernel == w[2:], "Conv channel/kernel mismatch")
            require(len(stride) == len(dilation) == 2 and len(pads) == 4
                    and all(v > 0 for v in stride + dilation) and all(v >= 0 for v in pads), "Conv spatial attributes")
            y = (x[0], w[0]) + tuple((x[j + 2] + pads[j] + pads[j + 2]
                    - dilation[j] * (kernel[j] - 1) - 1) // stride[j] + 1 for j in range(2))
            if len(node.input) == 3 and node.input[2]:
                require(node.input[2] in initializers and shapes[node.input[2]] == (w[0],), "Conv bias shape")
            values.update(kernel_shape=kernel, strides=stride, dilations=dilation, pads=pads)
        elif op == "BatchNormalization":
            require(len(node.input) == 5 and len(x) >= 2 and all(k in initializers and shapes[k] == (x[1],)
                    for k in node.input[1:]), "single-output inference BN parameter shapes")
            require(values["epsilon"] >= 0, "negative BN epsilon")
            y = x
        elif op == "Relu":
            require(len(node.input) == 1, "Relu arity")
            y = x
        elif op == "Add":
            require(len(node.input) == 2 and node.input[1] in shapes, "Add arity")
            y = broadcast(x, shapes[node.input[1]])
        elif op == "Flatten":
            require(len(node.input) == 1 and -len(x) <= values["axis"] <= len(x), "Flatten axis/arity")
            axis = values["axis"] if values["axis"] >= 0 else values["axis"] + len(x)
            y = (math.prod(x[:axis]), math.prod(x[axis:]))
        else:
            require(len(node.input) in (2, 3) and len(x) == 2 and node.input[1] in initializers, "Gemm arity/weight")
            w = shapes[node.input[1]]
            require(len(w) == 2 and values["transA"] in (0, 1) and values["transB"] in (0, 1), "Gemm transpose/rank")
            a, b = x[::-1] if values["transA"] else x, w[::-1] if values["transB"] else w
            require(a[1] == b[0], "Gemm contraction dimension mismatch")
            y = (a[0], b[1])
            if len(node.input) == 3 and node.input[2]:
                require(node.input[2] in initializers and broadcast(shapes[node.input[2]], y) == y, "Gemm bias broadcasting")
        require(all(0 < v <= 1000000 for v in y) and math.prod(y) <= 64000000, "output shape cap")
        out = node.output[0]
        shapes[out], producers[out] = y, index
        for port, name in enumerate(node.input):
            if name:
                consumers.setdefault(name, []).append([index, port])
        records.append(dict(index=index, name=node.name, op=op, inputs=list(node.input), output=out,
                            output_shape=list(y), attributes=evidence, schema_since_version=int(schema.since_version)))
        decoded.append(values)
        meter.charge(1024 + 64 * (len(node.input) + len(evidence)), entries=64 + 8 * len(evidence))
        meter.check()
    output = graph.output[0]
    require(output.name in producers and output.type.HasField("tensor_type") and output.type.tensor_type.elem_type == 1,
            "missing/FLOAT graph output")
    out_dims = output.type.tensor_type.shape.dim
    require(len(out_dims) == len(shapes[output.name]), "declared graph output rank")
    for dim, value in zip(out_dims, shapes[output.name]):
        require((dim.HasField("dim_value") and dim.dim_value == value) or
                (dim.HasField("dim_param") and symbol is not None and dim.dim_param == symbol and value == 1),
                "declared graph output shape mismatch")
    require(inventory.get("graph_outputs", [{}])[0].get("name") == output.name, "saved graph output identity")

    def next_node(port, op, input_port=0):
        uses = consumers.get(port, [])
        if port == output.name or len(uses) != 1:
            raise NoMatch("graph-output bypass or multiple/no consumer at " + port)
        index, used_port = uses[0]
        if nodes[index].op_type != op or (input_port is not None and used_port != input_port):
            raise NoMatch("expected sole " + op + " consumer at " + port)
        return index, nodes[index], used_port

    population, matches = [], []
    for ri, relu in enumerate(nodes):
        if relu.op_type != "Relu":
            continue
        row = dict(relu_index=ri, output=relu.output[0], matched=False)
        population.append(row)
        meter.charge(1024, entries=128)
        try:
            ci, conv, _ = next_node(relu.output[0], "Conv")
            bi, bn, _ = next_node(conv.output[0], "BatchNormalization")
            ai, add, branch_port = next_node(bn.output[0], "Add", None)
            skip = add.input[1 - branch_port]
            pre_bn_i = producers.get(relu.input[0], -1)
            pre_bn = nodes[pre_bn_i] if pre_bn_i >= 0 else None
            pre_conv_i = producers.get(pre_bn.input[0], -1) if pre_bn is not None else -1
            pre_conv = nodes[pre_conv_i] if pre_conv_i >= 0 else None
            if not (skip in producers and producers[skip] < ri and pre_bn is not None
                    and pre_bn.op_type == "BatchNormalization" and pre_conv is not None
                    and pre_conv.op_type == "Conv" and pre_conv.input[0] == skip
                    and shapes[skip] == shapes[bn.output[0]]):
                raise NoMatch("Add skip is not the same earlier parent of the pre-ReLU Conv/BN")
            fi, flat, _ = next_node(add.output[0], "Flatten")
            gi, gemm, _ = next_node(flat.output[0], "Gemm")
            ni, following, _ = next_node(gemm.output[0], "Relu")
            if decoded[fi]["axis"] != 1 or decoded[gi]["transA"] != 0 or shapes[gemm.output[0]][0] != 1:
                raise NoMatch("consumer bundle does not preserve the single-sample axis")
            m, r = math.prod(shapes[relu.output[0]][1:]), shapes[gemm.output[0]][1]
            p = r + 1
            item = dict(relu_index=ri, nodes=dict(conv=ci, bn=bi, add=ai, flatten=fi, gemm=gi, next_relu=ni),
                        q=relu.output[0], q_shape=list(shapes[relu.output[0]]), skip=skip, skip_shape=list(shapes[skip]),
                        skip_preconv=pre_conv_i, skip_prebn=pre_bn_i, receiver=gemm.output[0],
                        m=m, r=r, p_conservative=p, mass_row_policy="append; no numerical equality test",
                        complete_graph_consumer_chain=True, candidate_guard_bits_required=m,
                        phase_identity_retention_verified=False, phase_count_is_design_requirement_only=True,
                        candidate_generated_rows=2 * m + 18 * p + 2 * r, old_gate_rows=4 * m,
                        row_count_window_only=(m > 9 * p + r), C_dense_entry_upper=p * m,
                        coefficient_validity_verified=False, affine_bundle_coefficients_bound=False,
                        actual_rank_computed=False, nnz_computed=False, precision_or_speed_gain_claimed=False)
            matches.append(item)
            row.update(matched=True, match_index=len(matches) - 1, reason="complete ordinary preterminal consumer chain")
        except NoMatch as error:
            row["reason"] = str(error)
        meter.check()
    return dict(schema="d179_preterminal_structure_v1", structure_complete=True, metadata_complete=True, sample_batch_binding=1,
                original_batch_declaration=declared[0], original_bytes_or_annotations_modified=False,
                input_shape=list(input_shape), output_shape=list(shapes[output.name]), nodes=records,
                consumers=consumers, relu_population=population, match_count=len(matches), matches=matches,
                initializer_values_decoded=False, coefficient_validity_verified=False,
                native_domain_state_bound=False, candidate_executed=False, complete_physical_qualification=False,
                structural_shape_binding_only=True, formal_gain=0)
