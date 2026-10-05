"""UNEXECUTED, default-off D081 source-only successor-block draft.

Reuse the frozen D015 reader on ONE parsed model and ONE constant cache.  Its
original prefix is unchanged.  Extend each confirmed second-bank ReLU through
all linear successors, stopping at the next ReLU or a graph-output boundary.
Dynamic Add/Sub operands must resolve through supported linear operators to
the declared first/second-ReLU frontier of that same original model.

This is parameter/shape extraction, not network evaluation, interval bound
propagation, a native HZ binding, or an innovation/verification claim.  Conv
and BN stay ordered operators with raw parameters: neither is composed/folded.
Original node/tensor identities are retained; no original phase is deleted.
The old 64 MiB raw, 1M decoded-scalar, dimension and 512-bit caps are unchanged.
New traversal, protobuf lifetime, evidence, metadata and complete work/memory
costs are NOT certified.  No supervisor, files, logging or decoder is added.

All outputs of internal linear nodes are followed, including those introduced
to account for a residual operand.  A frontier is an intentional input cut:
its unrelated outgoing consumers are explicitly recorded, NOT interpreted.
The origin's outgoing edges are never left on that cut.  Unknown operators
encountered inside the successor/dependency closure reject the whole request.
This conservative grammar rejects dynamic broadcasting and constant-only
arithmetic DAGs; it is not a general ONNX interpreter.  A ReLU already on the
declared frontier is recorded as a frontier cut, not as a new receiver.  The
standard opset must be between 7 and the installed ONNX supported version.
"""

from collections import deque
import hashlib


class SourceBlockError(ValueError):
    """No complete successor-block packet was established."""


def _check(condition, message):
    if not condition:
        raise SourceBlockError(message)


class _Successors:
    def __init__(self, reader, base, frontier):
        self.reader, self.base, self.frontier = reader, base, frontier
        self.metadata = {port: (item['shape'], item['dtype'])
                         for port, item in frontier.items()}
        self.records, self.order, self.visiting = {}, [], set()
        self.output_declarations = {item.name: item for item in reader.model.graph.output}

    def metadata_for(self, port):
        if port in self.metadata:
            return self.metadata[port]
        _check(port and port not in self.visiting, 'cyclic/empty successor dependency')
        _check(port in self.reader.producers,
               'dynamic operand is outside the declared same-model frontier')
        self.visiting.add(port)
        try:
            index = self.reader.producers[port]
            node = self.reader.nodes[index]
            _check(self.reader.one_output(node) == port, 'mismatched successor output')
            record, shape, dtype = self.linear_node(index)
            self.records[index] = record
            self.order.append(index)  # All dynamic dependencies were inserted first.
            self.metadata[port] = (shape, dtype)
            return shape, dtype
        finally:
            self.visiting.remove(port)

    def linear_node(self, index):
        reader = self.reader
        node = reader.nodes[index]
        op = node.op_type
        _check(op in {'Conv', 'BatchNormalization', 'Identity', 'Add', 'Sub', 'Mul', 'Div'},
               'unsupported operator or undeclared nonlinear operand in successor block')
        _check(bool(node.input), 'successor operator has no input')
        if op in {'Conv', 'BatchNormalization', 'Identity'}:
            port = node.input[0]
            shape, dtype = self.metadata_for(port)
            if op == 'Conv':
                data = reader.conv(index, port, shape, dtype)
                output_shape = data['output_shape']
                kind = 'conv'
            elif op == 'BatchNormalization':
                data = reader.bn(index, port, shape, dtype)
                output_shape, kind = shape, 'batchnorm'
            else:
                data = reader.affine(index, port, shape, dtype)
                output_shape, kind = shape, 'channel_affine'
            dynamic = (port,)
        else:
            reader.attrs(node, set())
            _check(len(node.input) == 2 and all(node.input), 'invalid binary successor inputs')
            constant = tuple(reader.constant(port) for port in node.input)
            dynamic = tuple(port for port, value in zip(node.input, constant) if value is None)
            _check(dynamic, 'constant-only successor arithmetic is not qualified')
            shape, dtype = self.metadata_for(dynamic[0])
            if len(dynamic) == 1:
                data = reader.affine(index, dynamic[0], shape, dtype)
                _check(data is not None, 'channel-affine operation was not established')
                output_shape, kind = shape, 'channel_affine'
            else:
                _check(op in {'Add', 'Sub'}, 'nonlinear dynamic Mul/Div is unsupported')
                other_shape, other_dtype = self.metadata_for(dynamic[1])
                _check(shape == other_shape and dtype == other_dtype,
                       'dynamic merge shape/dtype mismatch or unsupported broadcasting')
                output_shape, kind = shape, 'dynamic_merge'
                data = dict(operator=op, node=self.base._node_record(index, node),
                            inputs=tuple(node.input), output=reader.one_output(node))
        result = dict(index=index, kind=kind, node=self.base._node_record(index, node),
                      dynamic_inputs=dynamic, input_shapes=tuple(self.metadata_for(p)[0]
                                                               for p in dynamic),
                      output=reader.one_output(node), output_shape=output_shape,
                      dtype=dtype, parameters=data)
        return result, output_shape, dtype

    def output_boundary(self, port, known_metadata=None):
        shape, dtype = self.metadata_for(port) if known_metadata is None else known_metadata
        declaration = self.output_declarations[port].type.tensor_type
        _check(int(declaration.elem_type) == dtype, 'graph output dtype disagrees')
        dims = declaration.shape.dim
        _check(len(dims) == len(shape), 'graph output rank disagrees')
        for item, extent in zip(dims, shape):
            if item.WhichOneof('value') == 'dim_value':
                _check(int(item.dim_value) == extent, 'graph output dimension disagrees')
        return dict(port=port, shape=shape, dtype=dtype,
                    original_declared_dimensions=tuple(
                        dict(kind=d.WhichOneof('value'), value=int(d.dim_value), symbol=d.dim_param)
                        for d in dims))

    def block(self, origin):
        reader = self.reader
        pending, enqueued, processed = deque([origin]), {origin}, set()
        used_frontier, internal, next_relus, outputs = set(), set(), {}, {}
        frontier_relus = {}
        edges = {}

        def enqueue(port):
            if port not in enqueued:
                enqueued.add(port)
                pending.append(port)

        def include(port):
            self.metadata_for(port)
            if port in self.frontier:
                used_frontier.add(port)
                return
            index = reader.producers[port]
            if index in internal:
                return
            record = self.records[index]
            internal.add(index)
            for dependency in record['dynamic_inputs']:
                include(dependency)
            enqueue(port)

        used_frontier.add(origin)
        while pending:
            port = pending.popleft()
            if port in processed:
                continue
            processed.add(port)
            shape, dtype = self.metadata_for(port)
            if port in reader.graph_outputs:
                outputs[port] = self.output_boundary(port)
            consumers = reader.consumers.get(port, ())
            _check(consumers or port in reader.graph_outputs,
                   'unaccounted dead successor output')
            for index in consumers:
                node = reader.nodes[index]
                positions = tuple(i for i, value in enumerate(node.input) if value == port)
                _check(positions, 'consumer edge identity mismatch')
                if node.op_type == 'Relu':
                    reader.attrs(node, set())
                    _check(tuple(node.input) == (port,), 'invalid boundary Relu inputs')
                    out = reader.one_output(node)
                    record = dict(node=self.base._node_record(index, node),
                        input=port, output=out, shape=shape, dtype=dtype,
                        original_relu_identity_only=True,
                        outgoing_consumers=tuple(reader.consumers.get(out, ())),
                        successors_interpreted=False)
                    if out in self.frontier:
                        known = self.frontier[out]
                        original = known['relu_node']
                        _check(reader.producers.get(out) == index
                               and original['index'] == index
                               and original['op'] == 'Relu'
                               and original['inputs'] == (port,)
                               and original['outputs'] == (out,),
                               'known frontier Relu identity disagrees')
                        _check(known['shape'] == shape and known['dtype'] == dtype,
                               'known frontier Relu shape/dtype disagrees')
                        record.update(frontier_role=known['role'],
                                      status='existing_frontier_relu_cut')
                        frontier_relus[index] = record
                        used_frontier.add(out)
                        kind = 'frontier_relu_boundary'
                        # This existing cut is not a new receiver and does not
                        # enqueue its output. The origin was already queued.
                    else:
                        next_relus[index] = record
                        kind = 'next_relu_boundary'
                    if out in reader.graph_outputs:
                        # Do not cache a new nonlinear port as an admissible operand.
                        outputs[out] = self.output_boundary(out, (shape, dtype))
                else:
                    include(reader.one_output(node))
                    _check(index in internal, 'successor bypasses the declared block boundary')
                    kind = 'linear'
                edges[(port, index)] = dict(source=port, consumer=index,
                    input_positions=positions, kind=kind)

        # Record every dynamic edge of required side operands, even when its
        # frontier source is an intentional cut rather than the starting port.
        for index in internal:
            record = self.records[index]
            node = reader.nodes[index]
            for port in record['dynamic_inputs']:
                positions = tuple(i for i, value in enumerate(node.input) if value == port)
                edges[(port, index)] = dict(source=port, consumer=index,
                    input_positions=positions, kind='linear')
        for port in processed:
            _check(all((port, i) in edges for i in reader.consumers.get(port, ())),
                   'missing successor edge coverage')
        _check(next_relus or outputs, 'successor block has no accounted output boundary')
        cuts = []
        for port in sorted(used_frontier):
            for index in reader.consumers.get(port, ()):
                covered = (port, index) in edges
                _check(port != origin or covered, 'origin has an uncovered side edge')
                cuts.append(dict(source=port, consumer=index, covered=covered,
                    node=self.base._node_record(index, reader.nodes[index]),
                    status='inside_block' if covered else 'outside_frontier_cut'))
        return dict(origin_port=origin, origin_relu=self.frontier[origin]['relu_node'],
                    frontier=tuple(sorted(used_frontier)),
                    ordered_linear_nodes=tuple(i for i in self.order if i in internal),
                    next_relus=tuple(next_relus[i] for i in sorted(next_relus)),
                    frontier_relus=tuple(frontier_relus[i] for i in sorted(frontier_relus)),
                    graph_outputs=tuple(outputs[p] for p in sorted(outputs)),
                    covered_ports=tuple(sorted(processed)),
                    dynamic_edges=tuple(edges[key] for key in sorted(edges)),
                    frontier_consumer_edges=tuple(cuts), complete_local_edge_coverage=True)


def extract_successor_blocks(raw, *, enabled=False, input_batch=None):
    """Return a source packet or reject; disabled calls inspect no task input.

    The caller must authenticate this draft, the frozen reader/import chain,
    original model/spec bytes, population, and all resource costs separately.
    A returned packet does not establish native phases, bounds or verification.
    """
    if enabled is not True:
        return None
    from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import source_packet_v1 as base

    _check(type(input_batch) is int and input_batch == 1, 'explicit batch-one binding required')
    _check(type(raw) is bytes and 0 < len(raw) <= base.MAX_RAW_BYTES,
           'expected bounded immutable original model bytes')
    import onnx

    try:
        model = onnx.ModelProto()
        model.ParseFromString(raw)
        _check(len(model.graph.input) == 1, 'expected one original graph input')
        value = model.graph.input[0]
        dims = value.type.tensor_type.shape.dim
        declared = tuple(dict(kind=d.WhichOneof('value'), value=int(d.dim_value), symbol=d.dim_param)
                         for d in dims)
        _check(len(dims) == 4 and all(d.WhichOneof('value') == 'dim_value'
               and 0 < d.dim_value <= base.MAX_DIMENSION for d in dims[1:]),
               'only batch may be symbolic; static NCHW dimensions required')
        _check(dims[1].dim_value == 3, 'expected original RGB input')
        if dims[0].WhichOneof('value') == 'dim_param':
            symbol = dims[0].dim_param
            _check(bool(symbol), 'empty original batch symbol')
            dims[0].dim_value = input_batch
            mapping = {symbol: input_batch}
        else:
            _check(dims[0].WhichOneof('value') == 'dim_value'
                   and dims[0].dim_value == input_batch, 'original batch disagrees')
            mapping = {}
        reader = base._Reader(onnx, model)
        _check(7 <= reader.opsets[''] <= int(onnx.defs.onnx_opset_version()),
               'standard ONNX opset is outside the supported broadcast grammar')
        # ONNX's ordered-DAG contract also rejects cycles without executing it.
        for index, node in enumerate(reader.nodes):
            _check(all(name not in reader.producers or reader.producers[name] < index
                       for name in node.input if name), 'non-topological or cyclic original dataflow')
        prefix = reader.run()
        digest = hashlib.sha256(raw).hexdigest()
        binding = dict(batch=input_batch, symbols=mapping, original_bytes_unchanged=True,
                       numeric_parameters_changed=False,
                       scope='single sample; exact VNNLIB population must be checked by caller')
        prefix.update(raw_model_sha256=digest, original_declared_input_dimensions=declared,
                      input_batch_binding=binding)
        dtype = prefix['input_dtype']
        first = prefix['first_relu']
        frontier = {first['output']: dict(port=first['output'],
                    shape=prefix['first_conv']['output_shape'], dtype=dtype,
                    relu_node=first, role='first_relu')}
        origins = []
        for branch in prefix['branches']:
            target = branch['target_relu']
            if target is None:
                continue  # Its original stop/side metadata remains in prefix.
            port = target['output']
            item = dict(port=port, shape=branch['conv']['output_shape'], dtype=dtype,
                        relu_node=target, role='confirmed_target_relu')
            _check(port not in frontier or frontier[port] == item,
                   'conflicting original frontier identity')
            if port not in frontier:
                frontier[port] = item
                origins.append(port)
        _check(origins, 'no confirmed second-ReLU origin in original prefix')
        successor = _Successors(reader, base, frontier)
        blocks = tuple(successor.block(port) for port in origins)
        return dict(schema='d081_successor_blocks_draft_v1', raw_model_sha256=digest,
                    source_frame=(digest, prefix['input_name'], input_batch),
                    input_batch_binding=binding, prefix=prefix,
                    frontier=tuple(frontier[p] for p in sorted(frontier)),
                    linear_nodes=tuple(successor.records[i] for i in successor.order),
                    blocks=blocks, decoded_scalar_count=reader.decoded_scalars,
                    formal_gain=0, model_execution_performed=False,
                    original_bits_deleted=0, actual_phase_column_binding_verified=False,
                    source_census_qualified=False, complete_physical_qualification=False,
                    full_model_qualified=False, frontier_cut_edges_interpreted=False,
                    gpu_computation_completed=False, status='unqualified_source_only_draft')
    except SourceBlockError:
        raise
    except Exception as exc:
        raise SourceBlockError('complete successor-block extraction was not established') from exc
