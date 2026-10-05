"""UNEXECUTED D015 draft: bounded, default-off, raw first-bank ONNX extraction.

``extract_model(raw, enabled=False)`` returns None without importing ONNX.
With explicit ``enabled=True`` it returns a plain-Python packet:

* input_shape / input_name / input_dtype and pre_affine {scale,bias};
* first_conv: exact raw weights/bias and explicit NCHW convolution geometry;
* first_post_ops: ordered channel-affine or raw BatchNormalization records;
* first_relu: its original node and input/output tensor identities;
* branches: each immediate Conv branch, ordered post_ops, optional target_relu,
  and stop metadata when a residual merge or graph output ends the local scope;
* side_consumers: all encountered dynamic Add/Sub merges and graph outputs;
* graph: original node ports, operation counts and opsets for provenance.

All stored scalars are exact Fractions of the original floating payloads.
Channel-affine composition is a REAL-ARITHMETIC identity, not a claim of IEEE
execution equivalence. BN gamma/beta/mean/variance/epsilon remain raw: no square
root, approximate fold, bound propagation or network evaluation occurs here.
The caller must prove its numerical semantics and outward bounds separately.

The supported local grammar is input -> channel affine/Identity -> Conv ->
channel affine/BN/Identity -> Relu, followed by immediate Conv branches.
Identity aliases are followed. Dynamic Add/Sub merges are recorded and NOT
traversed. Unknown first-path/branch operators fail closed, never disappear.
Graph nodes outside this local scope are audited, not interpreted or qualified.
This module neither constructs HZ nor emits native rows, changes original bits,
loads external tensor files, imports torch, simplifies ONNX, or runs a solver.

Internal rejection caps supplement, never replace, the frozen worker limits.
No execution or component qualification is claimed by this draft.
"""

from fractions import Fraction
import hashlib
import math
import struct


MAX_RAW_BYTES = 64 * 1024 * 1024
MAX_NODES = 10000
MAX_DECODED_SCALARS = 1000000
MAX_DIMENSION = 65536
MAX_RATIONAL_BITS = 512


class SourcePacketError(ValueError):
    """The requested local source packet was not established."""


def _check(condition, reason):
    if not condition:
        raise SourcePacketError(reason)


def _rational(value):
    _check(math.isfinite(value), "non-finite source scalar")
    result = Fraction.from_float(float(value))
    return _bounded(result)


def _bounded(value):
    _check(
        value.numerator.bit_length() <= MAX_RATIONAL_BITS
        and value.denominator.bit_length() <= MAX_RATIONAL_BITS,
        "source rational exceeds the registered bit limit",
    )
    return value


def _product(values):
    answer = 1
    for value in values:
        _check(0 < value <= MAX_DIMENSION, "unsupported tensor dimension")
        answer *= value
        _check(answer <= MAX_DECODED_SCALARS, "needed tensor is too large")
    return answer


def _node_record(index, node):
    return {
        "index": index,
        "name": node.name,
        "op": node.op_type,
        "domain": node.domain,
        "inputs": tuple(node.input),
        "outputs": tuple(node.output),
    }


class _Reader:
    def __init__(self, onnx, model):
        self.onnx = onnx
        self.model = model
        self.nodes = tuple(model.graph.node)
        _check(len(self.nodes) <= MAX_NODES, "too many graph nodes")
        self.initializers = {}
        for tensor in model.graph.initializer:
            _check(tensor.name and tensor.name not in self.initializers,
                   "duplicate or unnamed initializer")
            self.initializers[tensor.name] = tensor
        _check(not any(value.name in self.initializers for value in model.graph.input),
               "overridable initializer inputs are outside the fixed-source grammar")
        self.producers = {}
        self.consumers = {}
        for index, node in enumerate(self.nodes):
            for output in node.output:
                if not output:
                    continue
                _check(output not in self.producers and output not in self.initializers,
                       "duplicate tensor producer")
                self.producers[output] = index
            for name in dict.fromkeys(node.input):
                if name:
                    self.consumers.setdefault(name, []).append(index)
        self.graph_outputs = {value.name for value in model.graph.output}
        self.constants = {}
        self.decoded_scalars = 0
        self.side_consumers = []
        self.side_keys = set()
        self.opsets = {}
        for item in model.opset_import:
            domain = "" if item.domain in ("", "ai.onnx") else item.domain
            _check(domain not in self.opsets, "duplicate opset domain")
            self.opsets[domain] = int(item.version)
        _check(self.opsets.get("", 0) > 0, "missing standard ONNX opset")

    def attrs(self, node, allowed):
        _check(node.domain in ("", "ai.onnx"), "custom-domain local operator")
        attributes = {}
        for attr in node.attribute:
            _check(attr.name in allowed and attr.name not in attributes,
                   "unknown or duplicate local attribute")
            attributes[attr.name] = attr
        return attributes

    def int_attr(self, attributes, name, default):
        if name not in attributes:
            return default
        attr = attributes[name]
        _check(attr.type == self.onnx.AttributeProto.INT, "expected integer attribute")
        return int(attr.i)

    def ints_attr(self, attributes, name, default):
        if name not in attributes:
            return tuple(default)
        attr = attributes[name]
        _check(attr.type == self.onnx.AttributeProto.INTS, "expected integer-list attribute")
        return tuple(int(value) for value in attr.ints)

    def one_output(self, node):
        _check(len(node.output) == 1 and bool(node.output[0]),
               "local operator must have exactly one output")
        return node.output[0]

    def tensor(self, tensor):
        _check(not tensor.external_data
               and tensor.data_location != self.onnx.TensorProto.EXTERNAL,
               "external tensor data is not bound by the raw model")
        dimensions = tuple(int(value) for value in tensor.dims)
        count = _product(dimensions)
        _check(self.decoded_scalars + count <= MAX_DECODED_SCALARS,
               "decoded scalar budget exceeded")
        self.decoded_scalars += count
        dtype = int(tensor.data_type)
        if dtype == self.onnx.TensorProto.FLOAT:
            code, width, repeated = "<f", 4, tuple(tensor.float_data)
        elif dtype == self.onnx.TensorProto.DOUBLE:
            code, width, repeated = "<d", 8, tuple(tensor.double_data)
        else:
            raise SourcePacketError("needed tensor is not FLOAT/DOUBLE")
        if tensor.raw_data:
            _check(not repeated and len(tensor.raw_data) == count * width,
                   "ambiguous or malformed raw tensor bytes")
            values = tuple(_rational(item[0]) for item in struct.iter_unpack(code, tensor.raw_data))
        else:
            _check(len(repeated) == count, "malformed repeated tensor fields")
            values = tuple(_rational(value) for value in repeated)
        return {"shape": dimensions, "values": values, "dtype": dtype}

    def constant(self, name, visiting=None):
        if name in self.constants:
            return self.constants[name]
        if visiting is None:
            visiting = set()
        _check(name not in visiting, "cyclic constant alias")
        visiting.add(name)
        if name in self.initializers:
            result = self.tensor(self.initializers[name])
        elif name in self.producers:
            node = self.nodes[self.producers[name]]
            if node.op_type == "Identity":
                self.attrs(node, set())
                _check(len(node.input) == 1 and self.one_output(node) == name,
                       "malformed constant Identity")
                result = self.constant(node.input[0], visiting)
            elif node.op_type == "Constant":
                attributes = self.attrs(node, {"value"})
                _check(not node.input and self.one_output(node) == name
                       and set(attributes) == {"value"},
                       "only tensor-valued Constant is supported")
                attr = attributes["value"]
                _check(attr.type == self.onnx.AttributeProto.TENSOR,
                       "Constant value is not a tensor")
                result = self.tensor(attr.t)
            else:
                result = None
        else:
            result = None
        visiting.remove(name)
        self.constants[name] = result
        return result

    def needed(self, name, dtype):
        value = self.constant(name)
        _check(value is not None, "needed coefficient is not a supported constant")
        _check(value["dtype"] == dtype, "mixed source tensor dtypes are not qualified")
        return value

    def channel_constant(self, name, shape, dtype):
        tensor = self.needed(name, dtype)
        dims = tensor["shape"]
        _check(len(dims) <= 4, "constant rank exceeds NCHW broadcast rank")
        aligned = (1,) * (4 - len(dims)) + dims
        _check(all(d == 1 or d == extent for d, extent in zip(aligned, shape)),
               "constant does not broadcast to the source shape")
        _check(aligned[0] == aligned[2] == aligned[3] == 1
               and aligned[1] in (1, shape[1]),
               "affine constant is not scalar/channelwise")
        values = tensor["values"]
        if aligned[1] == 1:
            _check(len(values) == 1, "invalid scalar constant")
            return values * shape[1]
        _check(len(values) == shape[1], "invalid channel constant")
        return values

    def affine(self, index, port, shape, dtype):
        node = self.nodes[index]
        channels = shape[1]
        zeros = (Fraction(0),) * channels
        ones = (Fraction(1),) * channels
        if node.op_type == "Identity":
            self.attrs(node, set())
            _check(tuple(node.input) == (port,), "invalid data Identity")
            return {"kind": "affine", "node": _node_record(index, node),
                    "scale": ones, "bias": zeros, "output": self.one_output(node)}
        _check(node.op_type in {"Add", "Sub", "Mul", "Div"},
               "unsupported channel-affine operator")
        self.attrs(node, set())
        _check(len(node.input) == 2 and port in node.input, "invalid binary affine ports")
        if node.input[0] == node.input[1] == port:
            if node.op_type == "Add":
                scales = (Fraction(2),) * channels
            elif node.op_type == "Sub":
                scales = zeros
            else:
                raise SourcePacketError("non-affine repeated dynamic operand")
            offsets = zeros
        else:
            dynamic_left = node.input[0] == port
            other = node.input[1 if dynamic_left else 0]
            if self.constant(other) is None:
                _check(node.op_type in {"Add", "Sub"}, "non-affine dynamic binary operator")
                return None  # A residual merge, not a channel-affine chain.
            constant = self.channel_constant(other, shape, dtype)
            if node.op_type == "Add":
                scales, offsets = ones, constant
            elif node.op_type == "Sub":
                scales = ones if dynamic_left else (Fraction(-1),) * channels
                offsets = tuple(-value for value in constant) if dynamic_left else constant
            elif node.op_type == "Mul":
                scales, offsets = constant, zeros
            else:
                _check(dynamic_left and all(value != 0 for value in constant),
                       "division is not by a nonzero channel constant")
                scales = tuple(_bounded(1 / value) for value in constant)
                offsets = zeros
        return {"kind": "affine", "node": _node_record(index, node),
                "scale": scales, "bias": offsets, "output": self.one_output(node)}

    def bn(self, index, port, shape, dtype):
        node = self.nodes[index]
        attrs = self.attrs(node, {"epsilon", "momentum", "training_mode", "is_test", "spatial"})
        _check(len(node.input) == 5 and node.input[0] == port,
               "invalid BatchNormalization inputs")
        _check(self.opsets[""] >= 7, "pre-opset7 BatchNormalization is not qualified")
        _check(self.int_attr(attrs, "training_mode", 0) == 0
               and self.int_attr(attrs, "is_test", 1) == 1
               and self.int_attr(attrs, "spatial", 1) == 1,
               "training/non-spatial BatchNormalization is unsupported")
        parameters = []
        for name in node.input[1:]:
            tensor = self.needed(name, dtype)
            _check(tensor["shape"] == (shape[1],), "BatchNormalization parameter shape mismatch")
            parameters.append(tensor["values"])
        if "epsilon" in attrs:
            attr = attrs["epsilon"]
            _check(attr.type == self.onnx.AttributeProto.FLOAT, "invalid BN epsilon attribute")
        else:
            schema = self.onnx.defs.get_schema("BatchNormalization", self.opsets[""], "")
            attr = schema.attributes["epsilon"].default_value
            _check(attr.type == self.onnx.AttributeProto.FLOAT, "invalid schema epsilon default")
        epsilon = _rational(attr.f)
        _check(epsilon >= 0 and all(value + epsilon > 0 for value in parameters[3]),
               "BatchNormalization variance/epsilon is not positive")
        return {
            "kind": "batchnorm", "node": _node_record(index, node),
            "gamma": parameters[0], "beta": parameters[1],
            "mean": parameters[2], "variance": parameters[3],
            "epsilon": epsilon, "epsilon_from_schema": "epsilon" not in attrs,
            "output": self.one_output(node),
        }

    def conv(self, index, port, shape, dtype):
        node = self.nodes[index]
        attrs = self.attrs(node, {"strides", "pads", "dilations", "group", "kernel_shape", "auto_pad"})
        _check(node.op_type == "Conv" and len(node.input) in (2, 3)
               and node.input[0] == port, "invalid Conv input ports")
        weight = self.needed(node.input[1], dtype)
        _check(len(weight["shape"]) == 4, "Conv kernel is not rank4")
        out_channels, kernel_channels, kh, kw = weight["shape"]
        groups = self.int_attr(attrs, "group", 1)
        _check(groups > 0 and shape[1] % groups == 0 and out_channels % groups == 0
               and kernel_channels == shape[1] // groups, "invalid Conv group/channel geometry")
        strides = self.ints_attr(attrs, "strides", (1, 1))
        dilations = self.ints_attr(attrs, "dilations", (1, 1))
        kernel_shape = self.ints_attr(attrs, "kernel_shape", (kh, kw))
        _check(kernel_shape == (kh, kw) and len(strides) == len(dilations) == 2
               and all(value > 0 for value in strides + dilations), "invalid Conv spatial geometry")
        auto_pad = "NOTSET"
        if "auto_pad" in attrs:
            attr = attrs["auto_pad"]
            _check(attr.type == self.onnx.AttributeProto.STRING, "invalid auto_pad attribute")
            try:
                auto_pad = attr.s.decode("ascii") or "NOTSET"
            except UnicodeDecodeError as exc:
                raise SourcePacketError("non-ASCII auto_pad") from exc
        _check(auto_pad in {"NOTSET", "VALID", "SAME_UPPER", "SAME_LOWER"},
               "unsupported auto_pad")
        pads = self.ints_attr(attrs, "pads", (0, 0, 0, 0))
        _check(len(pads) == 4 and all(value >= 0 for value in pads), "invalid Conv padding")
        if auto_pad != "NOTSET":
            _check("pads" not in attrs, "explicit pads and auto_pad are ambiguous")
            before, after = [], []
            for extent, kernel, stride, dilation in zip(shape[2:], (kh, kw), strides, dilations):
                if auto_pad == "VALID":
                    total = 0
                else:
                    out = (extent + stride - 1) // stride
                    total = max(0, (out - 1) * stride + dilation * (kernel - 1) + 1 - extent)
                start = total // 2 if auto_pad != "SAME_LOWER" else (total + 1) // 2
                before.append(start)
                after.append(total - start)
            pads = tuple(before + after)
        out_hw = tuple((extent + pads[axis] + pads[axis + 2]
                        - dilation * (kernel - 1) - 1) // stride + 1
                       for axis, (extent, kernel, stride, dilation)
                       in enumerate(zip(shape[2:], (kh, kw), strides, dilations)))
        _check(all(0 < value <= MAX_DIMENSION for value in out_hw), "invalid Conv output shape")
        if len(node.input) == 3 and node.input[2]:
            bias = self.needed(node.input[2], dtype)
            _check(bias["shape"] == (out_channels,), "Conv bias shape mismatch")
            bias_values = bias["values"]
        else:
            bias_values = (Fraction(0),) * out_channels
        return {
            "node": _node_record(index, node), "input": port,
            "output": self.one_output(node), "input_shape": shape,
            "output_shape": (1, out_channels, *out_hw),
            "weight_source": node.input[1], "weight_shape": weight["shape"],
            "weights": weight["values"], "bias": bias_values,
            "strides": strides, "pads": pads, "dilations": dilations,
            "group": groups, "auto_pad": auto_pad,
        }

    def side(self, port, index, reason):
        key = (port, index, reason)
        if key in self.side_keys:
            return
        self.side_keys.add(key)
        self.side_consumers.append({
            "source": port, "reason": reason,
            "node": None if index is None else _node_record(index, self.nodes[index]),
        })

    def follow_to_relu(self, port, shape, dtype, allow_side):
        operations = []
        visited = set()
        while True:
            _check(port not in visited, "cyclic local dataflow")
            visited.add(port)
            if port in self.graph_outputs:
                _check(allow_side, "first bank reaches graph output before Relu")
                self.side(port, None, "graph_output")
            consumers = self.consumers.get(port, [])
            chain = []
            for index in consumers:
                node = self.nodes[index]
                if node.op_type in {"Add", "Sub", "Mul", "Div", "Identity"}:
                    operation = self.affine(index, port, shape, dtype)
                    if operation is None:
                        _check(allow_side, "first bank has an unsupported dynamic merge")
                        self.side(port, index, "dynamic_merge")
                    else:
                        chain.append((index, operation))
                elif node.op_type == "BatchNormalization":
                    chain.append((index, self.bn(index, port, shape, dtype)))
                elif node.op_type == "Relu":
                    self.attrs(node, set())
                    _check(tuple(node.input) == (port,), "invalid Relu input")
                    self.one_output(node)
                    chain.append((index, None))
                else:
                    raise SourcePacketError("unsupported operator on a direct Conv-to-Relu branch")
            if not chain:
                _check(allow_side and (consumers or port in self.graph_outputs),
                       "local path ends without an accounted consumer")
                return operations, None, {"port": port, "reason": "side_boundary"}
            _check(len(chain) == 1, "ambiguous affine-to-Relu fanout")
            index, operation = chain[0]
            if operation is None:
                record = _node_record(index, self.nodes[index])
                record.update(kind="relu", input=record["inputs"][0], output=record["outputs"][0])
                return operations, record, None
            operations.append(operation)
            port = operation["output"]

    def run(self):
        inputs = [value for value in self.model.graph.input if value.name not in self.initializers]
        _check(len(inputs) == 1, "expected one non-initializer graph input")
        original = inputs[0]
        tensor_type = original.type.tensor_type
        shape = tuple(int(dim.dim_value) for dim in tensor_type.shape.dim)
        _check(len(shape) == 4 and shape[0] == 1 and shape[1] == 3
               and all(0 < dim <= MAX_DIMENSION for dim in shape),
               "expected static batch1 RGB NCHW input")
        _check(all(not dim.dim_param for dim in tensor_type.shape.dim), "symbolic input dimension")
        dtype = int(tensor_type.elem_type)
        _check(dtype in {self.onnx.TensorProto.FLOAT, self.onnx.TensorProto.DOUBLE},
               "unsupported input dtype")
        port = original.name
        _check(port and port not in self.producers, "invalid original input identity")
        scale, bias = (Fraction(1),) * 3, (Fraction(0),) * 3
        pre_ops, visited = [], set()
        while True:
            _check(port not in visited and port not in self.graph_outputs,
                   "invalid first affine path")
            visited.add(port)
            consumers = self.consumers.get(port, [])
            _check(len(consumers) == 1, "ambiguous first input-to-Conv path")
            index = consumers[0]
            node = self.nodes[index]
            if node.op_type == "Conv":
                first_conv = self.conv(index, port, shape, dtype)
                break
            _check(node.op_type in {"Add", "Sub", "Mul", "Div", "Identity"},
                   "unsupported operator before first Conv")
            operation = self.affine(index, port, shape, dtype)
            _check(operation is not None, "dynamic merge before first Conv")
            scale = tuple(_bounded(a * s) for a, s in zip(operation["scale"], scale))
            bias = tuple(_bounded(a * b + c)
                         for a, b, c in zip(operation["scale"], bias, operation["bias"]))
            pre_ops.append(operation)
            port = operation["output"]
        first_post, first_relu, stop = self.follow_to_relu(
            first_conv["output"], first_conv["output_shape"], dtype, False)
        _check(first_relu is not None and stop is None, "first Relu was not established")
        relu_port = first_relu["outputs"][0]
        branches = []
        pending = [(relu_port, ())]
        seen_ports, seen_conv = set(), set()
        while pending:
            branch_port, aliases = pending.pop(0)
            _check(branch_port not in seen_ports, "duplicate or cyclic first-Relu alias")
            seen_ports.add(branch_port)
            if branch_port in self.graph_outputs:
                self.side(branch_port, None, "graph_output")
            consumers = self.consumers.get(branch_port, [])
            _check(consumers or branch_port in self.graph_outputs, "unconsumed first Relu output")
            for index in consumers:
                node = self.nodes[index]
                if node.op_type == "Identity":
                    operation = self.affine(index, branch_port, first_conv["output_shape"], dtype)
                    pending.append((operation["output"], aliases + (operation["node"],)))
                elif node.op_type == "Conv":
                    _check(index not in seen_conv, "duplicate immediate Conv branch")
                    seen_conv.add(index)
                    conv = self.conv(index, branch_port, first_conv["output_shape"], dtype)
                    post, target, branch_stop = self.follow_to_relu(
                        conv["output"], conv["output_shape"], dtype, True)
                    branches.append({"conv": conv, "post_ops": post,
                                     "input_aliases": aliases,
                                     "target_relu": target, "stop": branch_stop})
                elif node.op_type in {"Add", "Sub"}:
                    operation = self.affine(index, branch_port, first_conv["output_shape"], dtype)
                    _check(operation is None, "non-Identity affine before consumer Conv is unsupported")
                    self.side(branch_port, index, "dynamic_merge")
                else:
                    raise SourcePacketError("unsupported immediate first-Relu consumer")
        counts = {}
        records = []
        for index, node in enumerate(self.nodes):
            counts[node.op_type] = counts.get(node.op_type, 0) + 1
            records.append(_node_record(index, node))
        return {
            "schema": "d015_raw_first_bank_v1", "input_name": original.name,
            "input_shape": shape, "input_dtype": dtype,
            "pre_affine": {"scale": scale, "bias": bias}, "pre_ops": pre_ops,
            "first_conv": first_conv, "first_post_ops": first_post,
            "first_relu": first_relu, "branches": branches,
            "side_consumers": self.side_consumers,
            "decoded_scalar_count": self.decoded_scalars,
            "graph": {"node_count": len(self.nodes), "op_counts": counts,
                      "opsets": dict(self.opsets), "nodes": records},
        }


def extract_model(raw, enabled=False):
    """Return a certified-shape packet or fail closed; disabled means no work.

    A returned packet is source metadata, NOT a neural/HZ correctness verdict.
    Reading the raw protobuf is intentionally delayed until explicit opt-in.
    """
    if enabled is not True:
        return None
    _check(type(raw) is bytes and 0 < len(raw) <= MAX_RAW_BYTES,
           "expected bounded immutable ONNX bytes")
    import onnx

    model = onnx.ModelProto()
    try:
        model.ParseFromString(raw)
        result = _Reader(onnx, model).run()
    except SourcePacketError:
        raise
    except Exception as exc:
        raise SourcePacketError("raw source packet extraction failed") from exc
    result["raw_model_sha256"] = hashlib.sha256(raw).hexdigest()
    return result
