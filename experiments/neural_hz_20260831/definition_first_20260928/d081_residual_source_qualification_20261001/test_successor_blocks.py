"""Four unexecuted source-extraction checks; no top-level candidate imports.

Small synthetic ONNX graphs are built only inside the complete frozen pytest
invocation.  These are parameter/identity checks, not model evaluations, HZ
phase-column binding, physical resource qualification, or new formal solves.
"""


def _parts():
    import importlib
    prefix = 'experiments.neural_hz_20260831.definition_first_20260928.'
    module = importlib.import_module(prefix +
        'd081_residual_source_qualification_20261001.source_block')
    fixtures = importlib.import_module(prefix +
        'd015_source_shielding_20260928.test_source_packet_v1')
    return module, fixtures


def _outputs(onnx, model, names):
    dtype = model.graph.input[0].type.tensor_type.elem_type
    del model.graph.output[:]
    model.graph.output.extend(onnx.helper.make_tensor_value_info(name, dtype, [1, 2, 4, 4])
                              for name in names)


def _residual():
    _, fixtures = _parts()
    onnx, model = fixtures._base()
    tensor, node = fixtures._tensor, onnx.helper.make_node
    model.graph.initializer.extend([
        tensor(onnx, 'w2', (2, 2, 3, 3), (1, -.5, .25, -1) * 9),
        tensor(onnx, 'b2', (2,), (.125, -.25)),
        tensor(onnx, 'gamma2', (2,), (2, -1)),
        tensor(onnx, 'beta2', (2,), (.25, 0)),
        tensor(onnx, 'mean2', (2,), (1, -2)),
        tensor(onnx, 'var2', (2,), (4, 9)),
        tensor(onnx, 'offset2', (1, 2, 1, 1), (.5, -.25)),
    ])
    model.graph.node.extend([
        node('Conv', ['y', 'w2', 'b2'], ['c2'], name='wide_conv',
             pads=[1, 1, 1, 1], strides=[1, 1], dilations=[1, 1]),
        node('BatchNormalization', ['c2', 'gamma2', 'beta2', 'mean2', 'var2'],
             ['n2'], name='main_bn', epsilon=.125),
        node('Conv', ['r0', 'w1'], ['sc0'], name='shortcut_conv'),
        node('BatchNormalization', ['sc0', 'gamma2', 'beta2', 'mean2', 'var2'],
             ['sc'], name='shortcut_bn', epsilon=.125),
        node('Add', ['n2', 'sc'], ['joined'], name='residual_add'),
        node('Conv', ['joined', 'w1'], ['c3'], name='next_conv'),
        node('Identity', ['c3'], ['id3'], name='preserved_identity'),
        node('Sub', ['offset2', 'id3'], ['shift3'], name='constant_left_sub'),
        node('Relu', ['shift3'], ['out'], name='next_relu'),
    ])
    _outputs(onnx, model, ('out',))
    return onnx, model


def _multiple_origins():
    _, fixtures = _parts()
    onnx, model = fixtures._base()
    node = onnx.helper.make_node
    model.graph.node.extend([
        node('Conv', ['r0', 'w1'], ['fB'], name='second_origin_conv'),
        node('Relu', ['fB'], ['b'], name='second_origin_relu'),
        node('Conv', ['y', 'w1'], ['left'], name='left_conv'),
        node('Conv', ['b', 'w1'], ['right'], name='right_conv'),
        node('Add', ['left', 'right'], ['merged'], name='two_origin_add'),
        node('Relu', ['merged'], ['out'], name='joined_relu'),
        node('Identity', ['left'], ['tap'], name='side_identity'),
        node('Conv', ['y', 'w1'], ['other'], name='other_conv'),
        node('Relu', ['other'], ['out2'], name='other_relu'),
    ])
    _outputs(onnx, model, ('out', 'out2', 'tap'))
    return onnx, model


def _extract(module, model):
    return module.extract_successor_blocks(model.SerializeToString(),
                                           enabled=True, input_batch=1)


def _reject(module, raw, **options):
    try:
        module.extract_successor_blocks(raw, enabled=True,
                                         **dict({'input_batch': 1}, **options))
    except module.SourceBlockError:
        return
    raise AssertionError('unsupported successor packet was admitted')


def _named_node(model, name):
    return next(node for node in model.graph.node if node.name == name)


def _check_edges(packet, model):
    frontier = {item['port'] for item in packet['frontier']}
    records = {item['index']: item for item in packet['linear_nodes']}
    order = {item['index']: i for i, item in enumerate(packet['linear_nodes'])}
    producers = {port: i for i, node in enumerate(model.graph.node) for port in node.output}
    assert len(records) == len(packet['linear_nodes'])
    for index, record in records.items():
        for port in record['dynamic_inputs']:
            assert port in frontier or order[producers[port]] < order[index]
    for block in packet['blocks']:
        assert block['complete_local_edge_coverage'] is True
        edges = {(e['source'], e['consumer']): e for e in block['dynamic_edges']}
        for port in block['covered_ports']:
            for index, node in enumerate(model.graph.node):
                positions = tuple(i for i, value in enumerate(node.input) if value == port)
                if positions:
                    assert edges[(port, index)]['input_positions'] == positions
        assert all(edge['covered'] for edge in block['frontier_consumer_edges']
                   if edge['source'] == block['origin_port'])
        assert all(index in records for index in block['ordered_linear_nodes'])


def test_disabled_binding_and_source_caps():
    import builtins
    import hashlib
    module, _ = _parts()
    original_import = builtins.__import__

    def forbidden(name, *args, **kwargs):
        if name == 'onnx' or 'd015_source_shielding' in name:
            raise AssertionError('disabled extractor imported the source decoder')
        return original_import(name, *args, **kwargs)

    try:
        builtins.__import__ = forbidden
        for flag in (False, None, 'true', 1):
            assert module.extract_successor_blocks(object(), enabled=flag,
                                                    input_batch=object()) is None
    finally:
        builtins.__import__ = original_import
    for batch in (None, True, 0, 2):
        _reject(module, b'not a model', input_batch=batch)
    for raw in (b'', bytearray(b'x'), object()):
        _reject(module, raw)

    _, model = _residual()
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch'
    raw = model.SerializeToString()
    packet = module.extract_successor_blocks(raw, enabled=True, input_batch=1)
    assert model.SerializeToString() == raw
    assert packet['raw_model_sha256'] == hashlib.sha256(raw).hexdigest()
    assert packet['source_frame'] == (packet['raw_model_sha256'], 'x', 1)
    assert packet['input_batch_binding']['symbols'] == {'batch': 1}
    assert packet['prefix']['original_declared_input_dimensions'][0]['kind'] == 'dim_param'
    assert packet['prefix']['input_shape'] == (1, 3, 4, 4)
    assert packet['decoded_scalar_count'] == 60
    for field in ('model_execution_performed', 'actual_phase_column_binding_verified',
                  'source_census_qualified', 'complete_physical_qualification',
                  'full_model_qualified', 'gpu_computation_completed'):
        assert packet[field] is False
    assert packet['formal_gain'] == packet['original_bits_deleted'] == 0

    # A declared >1M coefficient tensor rejects before allocating its values.
    _, model = _residual()
    weight = next(t for t in model.graph.initializer if t.name == 'w2')
    del weight.dims[:]
    weight.dims.extend((512, 512, 2, 2))
    _reject(module, model.SerializeToString())


def test_ordered_residual_parameters_and_boundaries():
    from fractions import Fraction as F
    module, _ = _parts()
    _, model = _residual()
    packet = _extract(module, model)
    _check_edges(packet, model)
    records = {item['node']['name']: item for item in packet['linear_nodes']}
    assert set(records) == {'wide_conv', 'main_bn', 'shortcut_conv', 'shortcut_bn',
                            'residual_add', 'next_conv', 'preserved_identity',
                            'constant_left_sub'}
    wide = records['wide_conv']['parameters']
    assert wide['weight_shape'] == (2, 2, 3, 3)
    assert wide['weights'] == (F(1), F(-1, 2), F(1, 4), F(-1)) * 9
    assert wide['bias'] == (F(1, 8), F(-1, 4))
    assert wide['pads'] == (1, 1, 1, 1)
    assert wide['strides'] == wide['dilations'] == (1, 1)
    assert wide['input_shape'] == wide['output_shape'] == (1, 2, 4, 4)
    bn = records['main_bn']['parameters']
    assert bn['gamma'] == (F(2), F(-1)) and bn['beta'] == (F(1, 4), F(0))
    assert bn['mean'] == (F(1), F(-2)) and bn['variance'] == (F(4), F(9))
    assert bn['epsilon'] == F(1, 8) and 'scale' not in bn
    assert bn['gamma'] is records['shortcut_bn']['parameters']['gamma']
    assert records['residual_add']['parameters']['inputs'] == ('n2', 'sc')
    assert records['constant_left_sub']['parameters']['scale'] == (F(-1), F(-1))
    assert records['constant_left_sub']['parameters']['bias'] == (F(1, 2), F(-1, 4))
    block, = packet['blocks']
    assert block['origin_port'] == 'y' and block['frontier'] == ('r0', 'y')
    assert [item['output'] for item in block['next_relus']] == ['out']
    assert [item['port'] for item in block['graph_outputs']] == ['out']
    assert block['next_relus'][0]['successors_interpreted'] is False
    shortcut = next(b for b in packet['prefix']['branches']
                    if b['conv']['node']['name'] == 'shortcut_conv')
    assert shortcut['target_relu'] is None
    assert shortcut['stop']['reason'] == 'side_boundary'
    _, model = _residual()
    _named_node(model, 'residual_add').op_type = 'Sub'
    packet = _extract(module, model)
    _check_edges(packet, model)
    merge = next(item for item in packet['linear_nodes']
                 if item['node']['name'] == 'residual_add')
    assert merge['parameters']['operator'] == 'Sub'
    assert merge['parameters']['inputs'] == ('n2', 'sc')


def test_all_consumers_shared_cache_and_frontier_identity():
    module, fixtures = _parts()
    _, model = _multiple_origins()
    packet = _extract(module, model)
    _check_edges(packet, model)
    assert packet['decoded_scalar_count'] == 12  # One w1, not one copy per edge/block.
    assert [b['origin_port'] for b in packet['blocks']] == ['y', 'b']
    assert {f['port'] for f in packet['frontier']} == {'r0', 'y', 'b'}
    records = {item['node']['name']: item for item in packet['linear_nodes']}
    assert set(records) == {'left_conv', 'right_conv', 'two_origin_add',
                            'side_identity', 'other_conv'}
    weights = packet['prefix']['branches'][0]['conv']['weights']
    assert packet['prefix']['branches'][1]['conv']['weights'] is weights
    assert all(records[name]['parameters']['weights'] is weights
               for name in ('left_conv', 'right_conv', 'other_conv'))
    first, second = packet['blocks']
    assert {r['output'] for r in first['next_relus']} == {'out', 'out2'}
    assert {r['output'] for r in second['next_relus']} == {'out'}
    assert {r['port'] for r in first['graph_outputs']} == {'out', 'out2', 'tap'}
    assert {r['port'] for r in second['graph_outputs']} == {'out', 'tap'}
    other_index = records['other_conv']['index']
    assert any(e['source'] == 'y' and e['consumer'] == other_index
               and e['status'] == 'outside_frontier_cut' and e['covered'] is False
               for e in second['frontier_consumer_edges'])

    # Ordinary original-preactivation bypass: the already-declared q=R(g)
    # remains an input frontier gate, not a newly discovered next-layer gate.
    onnx, model = fixtures._base()
    model.graph.node.extend([
        onnx.helper.make_node('Add', ['y', 'f1'], ['bypass'], name='preactivation_add'),
        onnx.helper.make_node('Relu', ['bypass'], ['out'], name='bypass_next_relu'),
    ])
    _outputs(onnx, model, ('out',))
    packet = _extract(module, model)
    _check_edges(packet, model)
    block, = packet['blocks']
    assert {r['output'] for r in block['frontier_relus']} == {'y'}
    assert {r['output'] for r in block['next_relus']} == {'out'}
    assert any(e['source'] == 'f1' and e['consumer'] == 3
               and e['input_positions'] == (0,) and e['kind'] == 'frontier_relu_boundary'
               for e in block['dynamic_edges'])


def test_unsupported_edges_shapes_and_cycles_fail_closed():
    module, _ = _parts()
    onnx, model = _residual()
    model.graph.node.append(onnx.helper.make_node('Sigmoid', ['n2'], ['unhandled'],
                                                 name='unknown_side_consumer'))
    _reject(module, model.SerializeToString())
    _, model = _residual()
    _named_node(model, 'residual_add').op_type = 'Mul'
    _reject(module, model.SerializeToString())
    _, model = _residual()
    _named_node(model, 'shortcut_conv').attribute.append(
        onnx.helper.make_attribute('strides', [2, 2]))
    _reject(module, model.SerializeToString())
    _, model = _residual()
    model.graph.output[0].type.tensor_type.elem_type = onnx.TensorProto.DOUBLE
    _reject(module, model.SerializeToString())
    _, model = _residual()
    _named_node(model, 'main_bn').attribute.append(
        onnx.helper.make_attribute('training_mode', 1))
    _reject(module, model.SerializeToString())

    _, model = _multiple_origins()
    _named_node(model, 'two_origin_add').input[1] = 'unbound_other_frame'
    model.graph.output.append(onnx.helper.make_tensor_value_info(
        'right', onnx.TensorProto.FLOAT, [1, 2, 4, 4]))
    _reject(module, model.SerializeToString())
    _, model = _multiple_origins()
    nodes = list(model.graph.node)
    where = next(i for i, n in enumerate(nodes) if n.name == 'two_origin_add')
    nodes[where].input[1] = 'unregistered_relu'
    nodes.insert(where, onnx.helper.make_node('Relu', ['right'], ['unregistered_relu'],
                                             name='nonfrontier_relu'))
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    _reject(module, model.SerializeToString())
    _, model = _multiple_origins()
    _named_node(model, 'left_conv').input[0] = 'merged'
    _reject(module, model.SerializeToString())
    for version in (6, onnx.defs.onnx_opset_version() + 1):
        _, model = _residual()
        model.opset_import[0].version = version
        _reject(module, model.SerializeToString())
