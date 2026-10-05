"""Frozen-source extraction through the third original ReLU, not a verifier.

The caller authenticates this file, the supplied D015 modules, D179 metadata,
and all three registered source records before invoking extract.  This module
uses one caller-owned WorkBudget.  It reads the complete fixed prefix, never
selects spatial windows/channels from data, and never executes a network,
propagates activation bounds, calls D240, or constructs a native HZ.

Every required initializer is decoded and fully checked by the frozen D015
Reader.  Its temporary numeric peak is prepaid before any scalar decoding;
the caller also measures process peaks across the complete extraction.  After
this source-only audit, large decoded arrays expire.  Returned roots preserve
their lossless location in unchanged SHA-bound ONNX bytes, not another numeric
consumer or a repeated million-Fraction JSON.  BN raw parameters and outward
channel-affine intervals are retained explicitly.  This is real-arithmetic
source evidence, not IEEE equivalence or an exact rational BN fold.
"""

from fractions import Fraction
import hashlib
import math


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _dimensions(value):
    return tuple(dict(kind=dim.WhichOneof('value'), value=int(dim.dim_value),
                      symbol=dim.dim_param) for dim in value.type.tensor_type.shape.dim)


def _node(index, node):
    return dict(index=index, name=node.name, op=node.op_type, domain=node.domain,
                inputs=tuple(node.input), outputs=tuple(node.output))


def _complete_metadata(reader, structure, budget):
    """Check all ports and input slots, including consumers beyond the prefix."""
    stored = structure.get('nodes')
    _require(type(stored) is list and len(stored) == len(reader.nodes),
             'D179 complete node population differs')
    budget.charge(64 + 32 * len(reader.nodes)
                  + 8 * sum(len(node.input) + len(node.output) for node in reader.nodes))
    consumers, records = {}, []
    for index, (node, expected) in enumerate(zip(reader.nodes, stored)):
        _require(type(expected) is dict and expected.get('index') == index,
                 'D179 node index differs')
        _require(node.domain in ('', 'ai.onnx'), 'custom-domain model node is unsupported')
        _require(expected.get('op') == node.op_type
                 and expected.get('name') == node.name
                 and expected.get('inputs') == list(node.input)
                 and len(node.output) == 1 and expected.get('output') == node.output[0],
                 'D179 original node/port identity differs')
        records.append(_node(index, node))
        for slot, port in enumerate(node.input):
            if port:
                consumers.setdefault(port, []).append([index, slot])
    _require(consumers == structure.get('consumers'),
             'D179 whole-graph consumer slots differ')
    return tuple(records), consumers


def _attributes(onnx, reader, node, expected, budget):
    """Compare original attribute presence/values, not just output dimensions."""
    recorded = expected.get('attributes')
    _require(type(recorded) is dict, 'D179 attribute record is missing')
    actual = {attr.name: attr for attr in node.attribute}
    _require(len(actual) == len(node.attribute), 'duplicate original attribute')
    budget.charge(32 + 16 * (len(actual) + len(recorded)))
    _require(set(actual).issubset(recorded), 'unrecorded prefix attribute')
    for name, record in recorded.items():
        _require(type(record) is dict and type(record.get('present')) is bool,
                 'invalid D179 attribute presence')
        _require(record['present'] == (name in actual), 'prefix attribute presence differs')
        if name not in actual:
            continue
        attr = actual[name]
        if attr.type == onnx.AttributeProto.INT:
            value = int(attr.i)
        elif attr.type == onnx.AttributeProto.INTS:
            budget.charge(4 * len(attr.ints))
            value = list(attr.ints)
        elif attr.type == onnx.AttributeProto.STRING:
            budget.charge(len(attr.s))
            value = attr.s.decode('ascii')
        elif attr.type == onnx.AttributeProto.FLOAT:
            _require(math.isfinite(attr.f), 'nonfinite original attribute')
            _require(record.get('value_hex') == float(attr.f).hex(),
                     'prefix float attribute differs')
            continue
        else:
            raise ValueError('unsupported prefix attribute encoding')
        _require(record.get('value') == value, 'prefix attribute value differs')
    schema = onnx.defs.get_schema(node.op_type, reader.opsets[''], '')
    _require(expected.get('schema_since_version') == schema.since_version,
             'D179 original operator schema differs')


def _plan_initializer(name, reader, onnx, descriptors, budget, base):
    """Pay for the full upcoming decoder and later scalar statistics first."""
    if name in descriptors:
        return
    _require(name in reader.initializers,
             'registered prefix parameter must name an original initializer')
    tensor = reader.initializers[name]
    shape = tuple(int(value) for value in tensor.dims)
    budget.charge(64 + 8 * len(shape) + len(name))
    count = 1
    for extent in shape:
        _require(0 < extent <= base.MAX_DIMENSION, 'invalid original parameter extent')
        count *= extent
        _require(count <= base.MAX_DECODED_SCALARS, 'required parameter exceeds scalar cap')
    _require(not tensor.external_data and tensor.data_location != onnx.TensorProto.EXTERNAL,
             'external initializer data is not covered by the original bytes')
    dtype = int(tensor.data_type)
    if dtype == onnx.TensorProto.FLOAT:
        scalar_bytes, repeated_name = 4, 'float_data'
    elif dtype == onnx.TensorProto.DOUBLE:
        scalar_bytes, repeated_name = 8, 'double_data'
    else:
        raise ValueError('required initializer must be FLOAT or DOUBLE')
    # ByteSize and serialization inspect this frozen protobuf, not a model run.
    # The raw-model prepayment also covers parsing the descriptor itself.
    proto_size = tensor.ByteSize()
    budget.charge(128 + 64 * count + 3 * proto_size)
    raw_data = tensor.raw_data
    repeated_count = len(getattr(tensor, repeated_name))
    if raw_data:
        _require(repeated_count == 0 and len(raw_data) == scalar_bytes * count,
                 'ambiguous or malformed initializer payload')
        encoding = 'raw_data'
    else:
        _require(repeated_count == count, 'malformed repeated initializer payload')
        encoding = repeated_name
    serialized = tensor.SerializeToString(deterministic=True)
    _require(len(serialized) == proto_size, 'TensorProto size changed during extraction')
    descriptors[name] = dict(
        name=name, dtype=dtype, shape=shape, scalar_count=count,
        tensor_proto_bytes=proto_size, tensor_proto_sha256=_digest(serialized),
        tensor_proto_digest_semantics='deterministic serialization of original TensorProto, not a raw file slice',
        payload_encoding=encoding, scalar_bytes=scalar_bytes,
        raw_data_bytes=len(raw_data),
        raw_data_sha256=_digest(raw_data) if raw_data else None,
        decoder='frozen D015 source_packet_v1._Reader.tensor; exact Fraction of stored FLOAT/DOUBLE',
    )


def _geometry(conv, expected, budget):
    attributes = expected['attributes']
    values = dict(strides=list(conv['strides']), pads=list(conv['pads']),
                  dilations=list(conv['dilations']), group=conv['group'],
                  kernel_shape=list(conv['weight_shape'][2:]), auto_pad=conv['auto_pad'])
    budget.charge(64)
    for name, value in values.items():
        _require(attributes.get(name, {}).get('value') == value,
                 'D179 Conv geometry differs: ' + name)


def extract(raw, spec, source, structure, budget, k, helper, base):
    """Return pure-Python held roots and reconstructible evidence for one source.

    No independent budget is created here.  Caller registration fixes the three
    source files and supplies already authenticated dependency modules/metadata.
    Every original node up to the THIRD original ReLU is covered, including the
    shortcut branch.  All later consumers of prefix ports are retained as ports,
    not interpreted or silently claimed to have been verified.
    """
    _require(type(raw) is bytes and 0 < len(raw) <= base.MAX_RAW_BYTES,
             'bounded original ONNX bytes required')
    _require(type(spec) is bytes and len(spec) > 0, 'original VNNLIB bytes required')
    _require(type(source) is dict and type(structure) is dict,
             'frozen source and D179 metadata are required')
    budget.charge(4096 + len(raw) + 8 * len(spec))
    model_sha, spec_sha = _digest(raw), _digest(spec)
    _require(source.get('model_sha256') == model_sha and source.get('spec_sha256') == spec_sha,
             'original model/spec bytes differ from frozen source')
    _require(structure.get('schema') == 'd179_preterminal_domain_v1'
             and structure.get('structure_complete') is True
             and structure.get('metadata_complete') is True,
             'complete D179 structure metadata is required')
    structure_source = structure.get('source', {})
    _require(structure_source.get('model_sha256') == model_sha
             and structure_source.get('model_path') == source.get('model_path'),
             'D179 structure belongs to a different original model')

    import onnx

    model = onnx.ModelProto()
    model.ParseFromString(raw)
    _require(len(model.graph.input) == 1, 'one original graph input is required')
    original = model.graph.input[0]
    declared = _dimensions(original)
    dims = original.type.tensor_type.shape.dim
    _require(len(dims) == 4 and all(d.WhichOneof('value') == 'dim_value'
             and 0 < d.dim_value <= base.MAX_DIMENSION for d in dims[1:]),
             'only the original NCHW batch dimension may be symbolic')
    _require(dims[1].dim_value == 3, 'original input must have RGB channels')
    if dims[0].WhichOneof('value') == 'dim_param':
        symbol = dims[0].dim_param
        _require(bool(symbol), 'empty symbolic batch is unsupported')
        _require(structure.get('original_batch_declaration') == {'parameter': symbol},
                 'D179 batch symbol differs')
        dims[0].dim_value = 1
        batch_symbols = {symbol: 1}
    else:
        _require(dims[0].WhichOneof('value') == 'dim_value' and dims[0].dim_value == 1,
                 'the declared property batch must be one')
        _require(structure.get('original_batch_declaration') == {'value': 1},
                 'D179 fixed batch differs')
        batch_symbols = {}
    input_shape = tuple(int(d.dim_value) for d in dims)
    _require(list(input_shape) == structure.get('input_shape')
             and structure.get('sample_batch_binding') == 1,
             'D179 specialized input shape differs')
    dtype = int(original.type.tensor_type.elem_type)
    _require(dtype in (onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE),
             'unsupported original input dtype')
    reader = base._Reader(onnx, model)
    _require(original.name and original.name not in reader.producers,
             'invalid original input port')
    records, all_consumers = _complete_metadata(reader, structure, budget)
    relus = tuple(i for i, node in enumerate(reader.nodes) if node.op_type == 'Relu')
    _require(len(relus) >= 3, 'the original graph lacks its third ReLU')
    last = relus[2]
    # Plan every required original parameter before decoding any scalar.  This
    # is the complete declared prefix, not five windows or selected channels.
    descriptors = {}
    planned_bn_channels = 0
    for node in reader.nodes[:last + 1]:
        budget.charge(64 + 8 * len(node.input))
        _require(node.op_type in ('Conv', 'BatchNormalization', 'Relu', 'Add'),
                 'unsupported operator in the complete registered prefix: ' + node.op_type)
        if node.op_type == 'Conv':
            _require(len(node.input) in (2, 3), 'invalid original Conv input count')
            for name in node.input[1:]:
                if name:
                    _plan_initializer(name, reader, onnx, descriptors, budget, base)
        elif node.op_type == 'BatchNormalization':
            _require(len(node.input) == 5, 'invalid original BN input count')
            for name in node.input[1:]:
                _plan_initializer(name, reader, onnx, descriptors, budget, base)
            planned_bn_channels += descriptors[node.input[1]]['scalar_count']
    needed_scalars = sum(value['scalar_count'] for value in descriptors.values())
    _require(0 < needed_scalars <= base.MAX_DECODED_SCALARS,
             'complete required prefix exceeds the existing decoded-scalar cap')
    input_coordinates = input_shape[1] * input_shape[2] * input_shape[3]
    # This is an identity-deduplicated peak numeric-entry upper bound, NOT the
    # cumulative number of short-lived arithmetic allocations.  One parameter
    # contributes Fraction/num/den (3N), with at most N extra repeated-payload
    # floats while tensor() decodes.  Conv/BN reuse the exact same values tuple.
    # 8N leaves room for parameter shape/descriptor counts and implicit zero
    # Conv biases.  Separate terms cover BN output/intermediate fractions,
    # complete box bounds/parser records, node numeric metadata and fixed
    # scratch.  Raw/protobuf bytes are storage, not numeric tensor entries;
    # their storage is still measured by the caller's full process peaks.
    temp_numeric_entries_upper = (8 * needed_scalars + 128 * planned_bn_channels
                                  + 32 * input_coordinates
                                  + 64 * len(reader.nodes) + 65536)
    _require(temp_numeric_entries_upper <= 64_000_000,
             'prepaid temporary numeric entries exceed the existing 64M cap')
    budget.charge(8 * temp_numeric_entries_upper)
    box = helper.input_box(spec, input_shape)
    budget.charge(64 + 8 * len(box))
    _require(len(box) == input_shape[1] * input_shape[2] * input_shape[3],
             'complete original input-box population differs')

    shapes = {original.name: input_shape}
    decoded_nodes, evidence_nodes, logical_phases = [], [], []
    bn_channels = conv_count = add_count = 0
    for index, node in enumerate(reader.nodes[:last + 1]):
        budget.charge(128 + 16 * len(node.input))
        expected = structure['nodes'][index]
        _attributes(onnx, reader, node, expected, budget)
        output = reader.one_output(node)
        _require(output not in shapes, 'duplicate declared prefix output')
        if node.op_type == 'Conv':
            _require(len(node.input) in (2, 3) and node.input[0] in shapes,
                     'prefix Conv source has not been constructed')
            for name in node.input[1:]:
                if name:
                    _plan_initializer(name, reader, onnx, descriptors, budget, base)
            conv = reader.conv(index, node.input[0], shapes[node.input[0]], dtype)
            shape = conv['output_shape']
            _geometry(conv, expected, budget)
            decoded_nodes.append(dict(kind='conv', value=conv))
            geometry = {key: value for key, value in conv.items() if key not in ('weights', 'bias')}
            geometry.update(kind='conv', weights_ref=node.input[1],
                            bias_ref=node.input[2] if len(node.input) == 3 and node.input[2]
                            else dict(kind='implicit_zero', channels=shape[1]),
                            padding_semantics='zero after the complete input feature, at this Conv only')
            evidence_nodes.append(geometry)
            conv_count += 1
        elif node.op_type == 'BatchNormalization':
            _require(len(node.input) == 5 and node.input[0] in shapes,
                     'prefix BN source has not been constructed')
            for name in node.input[1:]:
                _plan_initializer(name, reader, onnx, descriptors, budget, base)
            shape = shapes[node.input[0]]
            bn = reader.bn(index, node.input[0], shape, dtype)
            budget.charge(64 + 16 * shape[1])
            post = tuple(helper.post_affine((bn,), channel, k, budget)
                         for channel in range(shape[1]))
            decoded_nodes.append(dict(kind='batchnorm', value=bn, outward_post_affine=post))
            evidence_nodes.append(dict(kind='batchnorm', raw=bn, input_shape=shape,
                output_shape=shape, parameter_refs=tuple(node.input[1:]),
                outward_post_affine=post,
                numerical_semantics='raw real-arithmetic BN enclosed outward; no midpoint network'))
            bn_channels += shape[1]
        elif node.op_type == 'Relu':
            reader.attrs(node, set())
            _require(len(node.input) == 1 and node.input[0] in shapes,
                     'prefix ReLU source has not been constructed')
            shape = shapes[node.input[0]]
            phase = dict(node=index, preactivation=node.input[0], output=output,
                         shape=shape, scalar_population=shape[1] * shape[2] * shape[3],
                         logical_identity='(model_sha256,spec_sha256,output_port,channel,row,column)',
                         actual_native_phase_columns_bound=False)
            logical_phases.append(phase)
            evidence_nodes.append(dict(kind='relu', node=records[index],
                                       input_shape=shape, output_shape=shape, phases=phase))
        elif node.op_type == 'Add':
            reader.attrs(node, set())
            _require(len(node.input) == 2 and all(port in shapes for port in node.input),
                     'prefix residual Add requires both original dynamic sources')
            shape = shapes[node.input[0]]
            _require(shapes[node.input[1]] == shape,
                     'broadcast/differently shaped residual Add is unsupported')
            evidence_nodes.append(dict(kind='add', node=records[index],
                input_shapes=tuple(shapes[port] for port in node.input), output_shape=shape,
                both_original_sources_retained=True))
            add_count += 1
        else:
            raise ValueError('unsupported operator in the complete registered prefix: ' + node.op_type)
        _require(list(shape) == expected.get('output_shape'),
                 'computed prefix shape differs from D179')
        shapes[output] = shape
    _require(len(logical_phases) == 3 and logical_phases[-1]['node'] == last,
             'complete three-ReLU population differs')

    statistics, decoded_constants, total_scalars, total_nonzero = [], {}, 0, 0
    for name, descriptor in descriptors.items():
        value = reader.constants.get(name)
        _require(type(value) is dict and value.get('dtype') == descriptor['dtype']
                 and tuple(value.get('shape', ())) == descriptor['shape'],
                 'decoded original initializer descriptor differs')
        values = value.get('values')
        _require(type(values) is tuple and len(values) == descriptor['scalar_count'],
                 'decoded original initializer population differs')
        # The 64*count prepayment precedes the Reader decode and this full pass.
        zero = positive = negative = numerator_bits = denominator_bits = 0
        for item in values:
            _require(type(item) is Fraction, 'decoder did not preserve exact stored scalars')
            if item.numerator == 0:
                zero += 1
            elif item.numerator > 0:
                positive += 1
            else:
                negative += 1
            numerator_bits = max(numerator_bits, abs(item.numerator).bit_length())
            denominator_bits = max(denominator_bits, item.denominator.bit_length())
        _require(zero + positive + negative == len(values)
                 and max(numerator_bits, denominator_bits) <= base.MAX_RATIONAL_BITS,
                 'decoded scalar statistics/bit cap differ')
        record = dict(descriptor, zero=zero, positive=positive, negative=negative,
                      nonzero=positive + negative, max_numerator_bits=numerator_bits,
                      max_denominator_bits=denominator_bits)
        statistics.append(record)
        decoded_constants[name] = value
        total_scalars += len(values)
        total_nonzero += positive + negative
    _require(total_scalars == reader.decoded_scalars,
             'some decoded parameter is missing from the complete source audit')
    _require(total_scalars == needed_scalars and bn_channels == planned_bn_channels,
             'decoded population differs from the prepaid full-prefix population')
    budget.charge(128 + 16 * len(statistics) + 16 * len(shapes))
    prefix_ports = tuple(shapes)
    consumers = {port: tuple(tuple(pair) for pair in all_consumers.get(port, ()))
                 for port in prefix_ports}
    graph_outputs = tuple(dict(name=value.name, dtype=int(value.type.tensor_type.elem_type),
                               original_declared_dimensions=_dimensions(value))
                          for value in model.graph.output)
    outgoing = tuple(dict(source=port, consumer=records[index], input_slot=slot)
                     for port in prefix_ports for index, slot in consumers[port] if index > last)
    frame = dict(model_sha256=model_sha, spec_sha256=spec_sha,
                 input_name=original.name, input_dtype=dtype, input_shape=input_shape,
                 input_layout='NCHW', explicit_batch=1, batch_symbols=batch_symbols,
                 original_declared_dimensions=declared,
                 scope='one complete declared original-model/spec source, not a native HZ frame')
    summary = dict(prefix_node_count=last + 1, final_relu_index=last,
        final_relu_output=logical_phases[-1]['output'], relu_banks=3,
        relu_scalar_populations=tuple(item['scalar_population'] for item in logical_phases),
        conv_nodes=conv_count, bn_channels=bn_channels, residual_add_nodes=add_count,
        input_coordinates=len(box), initializer_count=len(statistics),
        decoded_parameter_scalars=total_scalars, nonzero_parameter_scalars=total_nonzero,
        simultaneously_decoded_parameter_scalars=total_scalars,
        temp_numeric_entries_upper=temp_numeric_entries_upper,
        temp_numeric_entry_work_prepaid=8 * temp_numeric_entries_upper,
        decoded_weight_arrays_released_before_return=True,
        retained_root_ledger_includes_expired_decoder_arrays=False,
        outgoing_consumer_slots=len(outgoing), selected_windows=None,
        activation_bounds_propagated=False, actual_complete_child_affine_map_constructed=False,
        actual_native_phase_columns_bound=False, native_HZ_admitted=False,
        actual_model_verification_qualified=False, d240_executed=False,
        model_forward_calls=0, solver_calls=0, formal_gain=0)
    evidence_record = dict(schema='d241_complete_residual_source_v1', source=source,
        source_binding_complete=True, raw_parameters_verified=True,
        through_third_relu=True, all_prefix_consumers_accounted=True,
        source_binding_scope='raw declared prefix provenance/geometry only; no native HZ or activation bounds',
        decoded_scalar_count=total_scalars, prefix_node_count=last + 1,
        third_relu_port=logical_phases[-1]['output'],
        temp_numeric_entries_upper=temp_numeric_entries_upper,
        temporary_numeric_scope='identity-deduplicated peak upper bound, not cumulative allocation count',
        frame_identity=frame, input_box=box,
        raw_model=dict(byte_count=len(raw), sha256=model_sha),
        raw_spec=dict(byte_count=len(spec), sha256=spec_sha),
        reconstruction=dict(model_reference='source.model_path authenticated by model_sha256',
            tensor_reference='unique original initializer name plus dtype/shape and TensorProto digest',
            all_weight_values_reconstructible=True, weight_values_duplicated_in_json=False,
            decoder_module='D015 source_packet_v1', decoded_values_retained_in_roots=False,
            raw_bn_and_outward_post_values_retained=True,
            expired_decoder_lifetime='complete decode and scalar audit, then release before evidence writing'),
        batch_binding=dict(explicit_batch=1, symbols=batch_symbols,
                           only_in_memory_input_annotation_specialized=True,
                           original_model_bytes_unchanged=True, original_parameters_unchanged=True),
        original_graph_node_count=len(records), original_opsets=dict(reader.opsets),
        prefix_nodes=tuple(evidence_nodes), prefix_port_shapes=shapes,
        prefix_all_graph_consumers=consumers, later_consumers=outgoing,
        original_graph_outputs=graph_outputs,
        prefix_ports_also_graph_outputs=tuple(port for port in prefix_ports if port in reader.graph_outputs),
        initializer_descriptors=tuple(statistics), logical_phases=tuple(logical_phases), summary=summary)
    roots = dict(source=source, model_raw=raw, spec_raw=spec, structure=structure,
                 frame_identity=frame, input_box=box,
                 all_original_nodes=records, all_original_consumers=all_consumers,
                 evidence_record=evidence_record)
    # No subsequent stage consumes decoded weights: source_audit is not a
    # verifier/candidate adapter.  BN records remain intentionally in evidence.
    # Delete every local owner of large decoded tuples before returning only
    # reconstructible raw source roots.  Caller process peaks include them.
    del decoded_constants, decoded_nodes, reader, model, conv, value, values, item
    return roots, evidence_record
