"""Restore only structurally irrelevant zero lAs in branching metadata.

No backend source edits, bound replacement, omitted obligation or accepted
status override. Unsupported graphs, live missing coefficients and schema
changes fail closed. Enabled only by the separately frozen launcher.
"""
def graph_scope(net):
    final = net[net.output_name[0]]
    if type(final).__name__ != 'BoundConcat' or final.axis != 1:
        raise ValueError('repair supports a two-dimensional axis-1 final concat only')
    graph = {n.name: [i.name for i in n.inputs] for n in net.nodes()}
    if len(graph) != len(net.nodes()) or any(i not in graph for v in graph.values() for i in v):
        raise ValueError('invalid graph node identity')
    parts = []
    start = 0
    for n in final.inputs:
        if len(n.output_shape) != 2 or n.output_shape[1] <= 0:
            raise ValueError('unsupported concat input shape')
        end = start + int(n.output_shape[1])
        parts.append({'node': n.name, 'start': start, 'end': end})
        start = end
    return {'graph': graph, 'parts': parts, 'width': start, 'final': final.name}


def active_ancestors(scope, coefficients):
    """Exact zero/nonzero test; never threshold a small nonzero coefficient."""
    import math
    active = set()
    if not coefficients:
        raise ValueError('empty property batch')
    for batch in coefficients:
        if not batch:
            raise ValueError('empty property rows')
        for row in batch:
            if len(row) != scope['width'] or not all(math.isfinite(v) for v in row):
                raise ValueError('invalid property coefficients')
            for part in scope['parts']:
                if any(v != 0 for v in row[part['start']:part['end']]):
                    active.add(part['node'])
    todo = list(active)
    while todo:
        node = todo.pop()
        for parent in scope['graph'][node]:
            if parent not in active:
                active.add(parent)
                todo.append(parent)
    return active


def install(domain_class, solver_class, emit):
    import torch
    originals = domain_class.__init__, domain_class.add, solver_class.get_lA
    old_init, old_add, old_get = originals

    def init(self, ret, lAs, *args, **kwargs):
        old_init(self, ret, lAs, *args, **kwargs)
        net = self.net
        if hasattr(net, '_metamoe_las_scope'):
            raise ValueError('duplicate domain initialization on one solver')
        scope = graph_scope(net.net)
        C = net.c.detach().cpu()
        if C.ndim != 3 or C.dtype != torch.float64:
            raise ValueError('repair requires the frozen float64 rank-3 specification')
        active_ancestors(scope, C.tolist())
        nodes = {n.name:n for n in net.net.get_splittable_activations()}
        if set(lAs) != set(nodes):
            raise ValueError('initial full branching schema required')
        schema = {}
        for key, value in lAs.items():
            if (value.dtype != C.dtype or value.ndim < 3 or
                    list(value.shape[2:]) != list(nodes[key].output_shape[1:]) or
                    value.shape[1] != C.shape[1] or not torch.isfinite(value).all()):
                raise ValueError('initial lA schema mismatch')
            schema[key] = list(value.shape[1:])
        zero_keys = {k for k,v in lAs.items() if v.count_nonzero() == 0}
        net._metamoe_las_scope = (id(net.net), scope, schema, C.clone(), zero_keys)
        emit('SCHEMA', {'scope':scope, 'schema':schema, 'C':C.tolist(),
                       'initial_zero_keys':sorted(zero_keys)})

    def get(self, *args, **kwargs):
        result = old_get(self, *args, **kwargs)
        state = getattr(self, '_metamoe_las_scope', None)
        if state is None:
            return result
        identity, scope, schema, C, zero_keys = state
        if id(self.net) != identity or graph_scope(self.net) != scope or not torch.equal(self.c.cpu(), C):
            raise ValueError('branching graph or property scope changed')
        if set(result) - set(schema):
            raise ValueError('unexpected lA key; never discard coefficients')
        missing = sorted(set(schema) - set(result))
        if not missing:
            return result
        # Only the native update_bounds call shape is registered here.
        total = kwargs.get('tot_cells', args[1] if len(args) > 1 else None)
        transpose = kwargs.get('transpose', True)
        device = kwargs.get('device', None)
        if type(total) is not int or total <= 0 or not transpose or device != 'cpu':
            raise ValueError('unsupported lA restoration call')
        active = active_ancestors(scope, C.tolist())
        nodes = {n.name:n for n in self.net.get_splittable_activations()}
        for key in missing:
            if (key in active or key not in zero_keys or key not in nodes or getattr(nodes[key], 'lA', None) is not None
                    or list(nodes[key].output_shape[1:]) != schema[key][1:]):
                raise ValueError('missing active or mismapped lA; no zero substitution')
        for key, value in result.items():
            if list(value.shape) != [total, *schema[key]] or value.dtype != C.dtype:
                raise ValueError('native lA shape/dtype changed')
        restored = dict(result)
        for key in missing:
            restored[key] = torch.zeros((total, *schema[key]), dtype=C.dtype, device=device)
        emit('RESTORE_DISCONNECTED_ZERO', {'missing':missing, 'native_keys':sorted(result),
             'active_ancestors':sorted(active), 'batch':total, 'shapes':{k:list(v.shape) for k,v in restored.items()}})
        return restored

    def add(self, bounds, *args, **kwargs):
        if set(self.all_lAs) != set(bounds['lAs']):
            raise ValueError('unrepaired domain schema mismatch')
        result = old_add(self, bounds, *args, **kwargs)
        emit('DOMAIN_ADD_COMPLETE', {'keys':sorted(self.all_lAs), 'remaining':len(self)})
        return result

    domain_class.__init__, domain_class.add, solver_class.get_lA = init, add, get
    return originals
