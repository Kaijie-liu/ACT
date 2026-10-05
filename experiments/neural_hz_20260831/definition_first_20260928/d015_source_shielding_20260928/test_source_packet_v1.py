"""UNEXECUTED D015 draft: twelve explicit source-packet tests.

No real model is decoded here. Synthetic ONNX objects are built only when a
test calls its local helper, never during module collection. Root must freeze
the complete qualification inventory before importing or running this file.
"""

from fractions import Fraction
import importlib
import struct


def _reader():
    return importlib.import_module(
        "experiments.neural_hz_20260831.definition_first_20260928."
        "d015_source_shielding_20260928.source_packet_v1")


def _tensor(onnx, name, dims, values, raw=False, double=False):
    tensor = onnx.TensorProto()
    tensor.name = name
    tensor.dims.extend(dims)
    tensor.data_type = onnx.TensorProto.DOUBLE if double else onnx.TensorProto.FLOAT
    if raw:
        code = "<d" if double else "<f"
        tensor.raw_data = b"".join(struct.pack(code, value) for value in values)
    elif double:
        tensor.double_data.extend(values)
    else:
        tensor.float_data.extend(values)
    return tensor


def _base(double=False, raw=False):
    import onnx

    helper = onnx.helper
    dtype = onnx.TensorProto.DOUBLE if double else onnx.TensorProto.FLOAT
    tensors = [_tensor(onnx, "w0", (2, 3, 1, 1), (1, 2, 3, -1, .5, 2), raw, double),
               _tensor(onnx, "b0", (2,), (.25, -.5), raw, double),
               _tensor(onnx, "w1", (2, 2, 1, 1), (2, -1, -1, 2), raw, double)]
    nodes = [helper.make_node("Conv", ["x", "w0", "b0"], ["f0"], name="conv0"),
             helper.make_node("Relu", ["f0"], ["r0"], name="relu0"),
             helper.make_node("Conv", ["r0", "w1"], ["f1"], name="conv1"),
             helper.make_node("Relu", ["f1"], ["y"], name="relu1")]
    graph = helper.make_graph(nodes, "test", [helper.make_tensor_value_info("x", dtype, [1, 3, 4, 4])],
                              [helper.make_tensor_value_info("y", dtype, [1, 2, 4, 4])], tensors)
    return onnx, helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _reject(module, model):
    try:
        module.extract_model(model.SerializeToString(), enabled=True)
    except module.SourcePacketError:
        return
    raise AssertionError("unsupported source was admitted")


def test_disabled_does_not_import_onnx_or_read_bytes(monkeypatch):
    import builtins

    module = _reader()
    original = builtins.__import__
    def forbidden(name, *args, **kwargs):
        if name == "onnx":
            raise AssertionError("disabled extractor imported ONNX")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", forbidden)
    assert module.extract_model(object()) is None
    assert module.extract_model(b"not a model", enabled="true") is None


def test_simple_first_bank_and_branch_repeated_float_payloads():
    module = _reader()
    _, model = _base()
    packet = module.extract_model(model.SerializeToString(), enabled=True)
    assert packet["input_shape"] == (1, 3, 4, 4)
    assert packet["first_conv"]["weights"] == tuple(map(Fraction, (1, 2, 3, -1, .5, 2)))
    assert packet["first_conv"]["bias"] == (Fraction(1, 4), Fraction(-1, 2))
    assert packet["first_conv"]["output_shape"] == (1, 2, 4, 4)
    assert packet["branches"][0]["target_relu"]["name"] == "relu1"
    assert packet["side_consumers"] == []


def test_raw_float_and_double_decoding_are_exact():
    module = _reader()
    for double in (False, True):
        _, model = _base(double=double, raw=True)
        packet = module.extract_model(model.SerializeToString(), enabled=True)
        assert packet["first_conv"]["weights"][4] == Fraction(1, 2)
        assert packet["first_conv"]["bias"][1] == Fraction(-1, 2)


def test_constant_tensor_and_identity_weight_aliases():
    module = _reader()
    onnx, model = _base()
    weight = model.graph.initializer[2]
    saved = onnx.TensorProto()
    saved.CopyFrom(weight)
    del model.graph.initializer[2]
    model.graph.node[2].input[1] = "w_alias"
    model.graph.node.extend([
        onnx.helper.make_node("Constant", [], ["constant_weight"], value=saved),
        onnx.helper.make_node("Identity", ["constant_weight"], ["w_alias"])])
    packet = module.extract_model(model.SerializeToString(), enabled=True)
    assert packet["branches"][0]["conv"]["weights"] == tuple(map(Fraction, (2, -1, -1, 2)))


def test_input_channel_affine_composition_and_operand_order():
    module = _reader()
    onnx, model = _base()
    model.graph.initializer.extend([
        _tensor(onnx, "scale", (1, 3, 1, 1), (2, -1, .5)),
        _tensor(onnx, "offset", (), (3,))])
    model.graph.node[0].input[0] = "shifted"
    model.graph.node.extend([
        onnx.helper.make_node("Mul", ["scale", "x"], ["scaled"]),
        onnx.helper.make_node("Sub", ["offset", "scaled"], ["shifted"])])
    packet = module.extract_model(model.SerializeToString(), enabled=True)
    assert packet["pre_affine"] == {"scale": (Fraction(-2), Fraction(1), Fraction(-1, 2)),
                                     "bias": (Fraction(3),) * 3}


def test_batchnorm_raw_parameters_precede_original_relu():
    module = _reader()
    onnx, model = _base()
    for name, values in (("gamma", (2, -1)), ("beta", (.25, 0)),
                         ("mean", (1, -2)), ("var", (4, 9))):
        model.graph.initializer.append(_tensor(onnx, name, (2,), values))
    model.graph.node[1].input[0] = "normalized"
    model.graph.node.append(onnx.helper.make_node(
        "BatchNormalization", ["f0", "gamma", "beta", "mean", "var"],
        ["normalized"], epsilon=.125))
    packet = module.extract_model(model.SerializeToString(), enabled=True)
    op = packet["first_post_ops"][0]
    assert op["kind"] == "batchnorm"
    assert op["gamma"] == (Fraction(2), Fraction(-1))
    assert op["variance"] == (Fraction(4), Fraction(9))
    assert op["epsilon"] == Fraction(1, 8)
    assert "scale" not in op


def test_residual_side_consumer_is_recorded_without_following_it():
    module = _reader()
    onnx, model = _base()
    model.graph.node.append(onnx.helper.make_node("Add", ["r0", "y"], ["merged"], name="residual"))
    model.graph.output[0].name = "merged"
    packet = module.extract_model(model.SerializeToString(), enabled=True)
    assert len(packet["branches"]) == 1
    assert packet["side_consumers"][0]["node"]["name"] == "residual"
    assert packet["side_consumers"][0]["reason"] == "dynamic_merge"


def test_all_immediate_conv_branches_include_projection_merge():
    module = _reader()
    onnx, model = _base()
    model.graph.node.extend([
        onnx.helper.make_node("Conv", ["r0", "w1"], ["projection"], name="projection_conv"),
        onnx.helper.make_node("Add", ["projection", "y"], ["merged"], name="merge")])
    model.graph.output[0].name = "merged"
    packet = module.extract_model(model.SerializeToString(), enabled=True)
    assert len(packet["branches"]) == 2
    projection = packet["branches"][1]
    assert projection["conv"]["node"]["name"] == "projection_conv"
    assert projection["target_relu"] is None
    assert projection["stop"]["reason"] == "side_boundary"


def test_geometry_preserves_explicit_padding_stride_and_dilation():
    module = _reader()
    onnx, model = _base()
    node = model.graph.node[0]
    node.attribute.extend([onnx.helper.make_attribute("pads", [1, 2, 3, 4]),
                           onnx.helper.make_attribute("strides", [2, 3]),
                           onnx.helper.make_attribute("dilations", [2, 1])])
    packet = module.extract_model(model.SerializeToString(), enabled=True)
    conv = packet["first_conv"]
    assert conv["pads"] == (1, 2, 3, 4)
    assert conv["strides"] == (2, 3)
    assert conv["dilations"] == (2, 1)
    assert conv["output_shape"] == (1, 2, 4, 4)


def test_malformed_nonfinite_external_and_duplicate_sources_reject():
    module = _reader()
    onnx, model = _base(raw=True)
    model.graph.initializer[0].raw_data = b"short"
    _reject(module, model)
    _, model = _base()
    model.graph.initializer[0].float_data[0] = float("nan")
    _reject(module, model)
    _, model = _base()
    model.graph.initializer[0].data_location = onnx.TensorProto.EXTERNAL
    _reject(module, model)
    _, model = _base()
    model.graph.node[2].output[0] = "f0"
    _reject(module, model)


def test_dynamic_shape_spatial_constants_and_unknown_branches_reject():
    module = _reader()
    onnx, model = _base()
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_param = "height"
    _reject(module, model)
    _, model = _base()
    model.graph.initializer.append(_tensor(onnx, "spatial", (4,), (1, 2, 3, 4)))
    model.graph.node[0].input[0] = "changed"
    model.graph.node.append(onnx.helper.make_node("Add", ["x", "spatial"], ["changed"]))
    _reject(module, model)
    _, model = _base()
    model.graph.node.append(onnx.helper.make_node("Sigmoid", ["r0"], ["unexpected"]))
    _reject(module, model)


def test_training_batchnorm_and_dynamic_division_reject():
    module = _reader()
    onnx, model = _base()
    for name, values in (("gamma", (1, 1)), ("beta", (0, 0)),
                         ("mean", (0, 0)), ("var", (1, 1))):
        model.graph.initializer.append(_tensor(onnx, name, (2,), values))
    model.graph.node[1].input[0] = "normalized"
    model.graph.node.append(onnx.helper.make_node(
        "BatchNormalization", ["f0", "gamma", "beta", "mean", "var"],
        ["normalized"], training_mode=1))
    _reject(module, model)
    _, model = _base()
    model.graph.node[0].input[0] = "changed"
    model.graph.node.append(onnx.helper.make_node("Div", ["x", "x"], ["changed"]))
    _reject(module, model)
