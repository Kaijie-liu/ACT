"""Explicit batch-one specialization, reusing the frozen D015 local parser.

Only an in-memory copy's input type annotation is specialized. Original bytes,
all node ports/parameters, original declared dimensions and binding are retained.
This is not a network transform, property verdict or general symbolic executor.
"""
import hashlib
from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import source_packet_v1 as base

SourcePacketError = base.SourcePacketError


def extract_model(raw, enabled=False, input_batch=None):
    if enabled is not True:
        return None
    base._check(type(input_batch) is int and input_batch == 1,
                'explicit single-input batch binding required')
    base._check(type(raw) is bytes and 0 < len(raw) <= base.MAX_RAW_BYTES,
                'expected bounded immutable ONNX bytes')
    import onnx
    model = onnx.ModelProto()
    try:
        model.ParseFromString(raw)
        base._check(len(model.graph.input) == 1, 'expected one original graph input')
        value = model.graph.input[0]
        dims = value.type.tensor_type.shape.dim
        declared = tuple(dict(kind=d.WhichOneof('value'), value=int(d.dim_value),
                              symbol=d.dim_param) for d in dims)
        base._check(len(dims) == 4, 'expected rank-four NCHW input')
        base._check(all(d.WhichOneof('value') == 'dim_value'
                        and 0 < d.dim_value <= base.MAX_DIMENSION for d in dims[1:]),
                    'only the batch dimension may be symbolic')
        base._check(dims[1].dim_value == 3, 'expected RGB channels')
        kind = dims[0].WhichOneof('value')
        if kind == 'dim_param':
            symbol = dims[0].dim_param
            base._check(bool(symbol), 'empty batch symbol is not a binding')
            dims[0].dim_value = input_batch  # protobuf oneof clears dim_param.
            mapping = {symbol: input_batch}
        else:
            base._check(kind == 'dim_value' and dims[0].dim_value == input_batch,
                        'declared batch disagrees with the explicit property batch')
            mapping = {}
        packet = base._Reader(onnx, model).run()
        packet['original_declared_input_dimensions'] = declared
        packet['input_batch_binding'] = dict(batch=input_batch, symbols=mapping,
            scope='local single-sample source; exact VNNLIB input population checked by caller',
            original_bytes_unchanged=True, numeric_parameters_changed=False)
        packet['raw_model_sha256'] = hashlib.sha256(raw).hexdigest()
        return packet
    except SourcePacketError:
        raise
    except Exception as exc:
        raise SourcePacketError('batch-bound source extraction failed') from exc
