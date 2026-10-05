"""Bounded synthetic ID-consumer diagnostic, NOT a source/HZ admission adapter.

This calls the real Layer/Net validators, stride reader and ConSet.add_op.
The separately labelled closure is only the ID metadata constructed here.
No production schema, binary-operand guard or existing LIVE ledger is changed.
"""
import hashlib
import struct
import sys

from act.back_end.core import Con, ConSet, Layer, Net
from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
from act.back_end.layer_util import validate_graph
from experiments.neural_hz_20260831.c44_var_ids_v1 import VarIds, span


def fixture(width, consumers, *, compact, pool):
    if type(width) is not int or not 1 <= width <= 65536:
        raise ValueError('synthetic width outside frozen diagnostic domain')
    if type(consumers) is not int or consumers not in (1, 2, 4):
        raise ValueError('unregistered materializing consumer count')
    if type(compact) is not bool:
        raise ValueError('explicit diagnostic comparator required')
    pool.charge('consumer_fixture_metadata', 1024)
    if compact:
        x = span(1000, width, enabled=True)
        y = span(1000 + width, width, enabled=True)
    else:
        pool.charge('baseline_ID_allocation_and_shallow_copy', 12 * width)
        x = list(range(1000, 1000 + width))
        y = list(range(1000 + width, 1000 + 2 * width))
    # Match fresh source construction: copies share the original scalar ints.
    layers = [Layer(0, 'INPUT', {'shape': (1,width), 'dtype':'torch.float64'}, [], x),
              Layer(1, 'INPUT_SPEC', {'kind':'BOX','lb_val':0,'ub_val':1}, x, x),
              Layer(2, 'RELU', {}, x.copy(), y),
              Layer(3, 'ASSERT', {'kind':'TOP1_ROBUST','C':None,'thresholds':None,'M':None,'y_true':0}, y, y)]
    node_outputs = {'input': x.copy(), 'relu': y.copy()}
    pool.charge('unmodified_graph_and_stride_consumer_upper_bound', 128 * width)
    net = Net(layers, {0: [], 1: [0], 2:[1], 3:[2]}, {0: [1], 1: [2], 2:[3], 3:[]})
    validate_graph(layers)
    stride = HybridzTF._net_var_id_stride(net)
    if stride != 1000 + 2 * width:
        raise ValueError('unmodified global-ID stride differs')
    cons = ConSet()
    for index in range(consumers):
        # These are separate metadata consumers of identical values, NOT
        # different verification paths or independent latent variables.
        pool.charge('unmodified_ConSet_concat_tuple_signature', 16 * width)
        cons.add_op(f'relu:{index}', y + layers[2].in_vars)
    return dict(net=net, node_outputs=node_outputs, cons=cons)


def sequence_digest(value, pool):
    """Ordered fixed-width value image; no representation-specific object IDs."""
    pool.charge('ID_semantic_image', 4 * len(value) + 32)
    digest = hashlib.sha256()
    digest.update(struct.pack('<Q', len(value)))
    for item in value:
        if type(item) is not int or not 0 <= item < 2**63:
            raise ValueError('unregistered integer-value consumer')
        digest.update(struct.pack('<q', item))
    return digest.hexdigest()


def evidence(root, pool):
    net = root['net']; nodes = root['node_outputs']; cons = root['cons']
    streams = [*(v for layer in net.layers for v in (layer.in_vars, layer.out_vars)),
               *nodes.values(), *(c.var_ids for c in cons)]
    # All original sequence fields, including INPUT empty input, are included.
    images = [sequence_digest(v, pool) for v in streams]
    pool.charge('outer_alias_equivalence', len(streams)**2)
    aliases = [[a is b for b in streams] for a in streams]
    values = list(cons)
    sample = 0
    return dict(ordered_sequence_sha256=images, outer_alias_matrix=aliases,
        constraints=len(values),
        repeated_consumer_scalar_identity=(None if len(values) < 2 else
            values[0].var_ids[sample] is values[1].var_ids[sample]),
        baseline_scalar_shared_with_layer=(net.layers[2].out_vars[sample] is
            values[0].var_ids[sample]))


_FIELDS = {
    Net: {'layers', 'preds', 'succs', 'by_id', '_topo_cache'},
    Layer: {'id', 'kind', 'params', 'in_vars', 'out_vars', 'cache'},
    ConSet: {'S'}, Con: {'kind', 'var_ids', 'meta', 'A', 'b', 'C', 'd'},
    VarIds: {'runs', 'size'},
}


def diagnostic_python_closure(root, pool):
    """Exact unique sys.getsizeof sum for this closed Python-only fixture.

    Includes instance __dict__, dictionary keys, all sequence entries and
    integer identities. Numeric tensors/arrays and unknown roots are rejected.
    Does NOT replace the original C5 numeric or full-source LIVE ledger.
    """
    seen=set(); counts={}; sizes={}; refs=0

    def visit(value):
        nonlocal refs
        pool.charge('closed_python_reference_visit', 32); refs += 1
        if id(value) in seen:return
        seen.add(id(value)); kind=type(value); name=kind.__name__
        counts[name]=counts.get(name,0)+1
        sizes[name]=sizes.get(name,0)+sys.getsizeof(value)
        if value is None or kind in (int, str, bool):return
        if kind in (list, tuple):
            for item in value:visit(item)
        elif kind is dict:
            for key,item in value.items():visit(key);visit(item)
        elif kind in _FIELDS:
            if set(vars(value)) != _FIELDS[kind]:raise ValueError('unregistered diagnostic object fields')
            if kind is VarIds:value.validate()
            visit(vars(value))
        else:raise ValueError(f'unregistered diagnostic root: {name}')

    visit(root)
    return dict(scope='complete_synthetic_Python_ID_consumer_fixture_only',
        unique_python_shallow_bytes=sum(sizes.values()), unique_objects=len(seen),
        visited_references=refs, object_counts=counts, bytes_by_type=sizes,
        allocator_or_C_workspace_bytes_included=False, full_HZ_LIVE_gate_proved=False)
