"""Opt-in CSR representation admission, not a process-memory or soundness bound.

V1 covers sequential affine/ReLU CNNs only. Estimates intentionally overcount
shared arrays and intermediates. An external supervisor still bounds the whole
process (model, solver, Python allocator and objects held by callers included).
No mathematical encoding or numerical acceptance condition is changed here.
"""
import math

MATRICES = ('Gc', 'Gb', 'Ac', 'Ab', 'Auc', 'Aub')
KINDS = {'INPUT', 'INPUT_SPEC', 'ASSERT', 'CONV2D', 'AVGPOOL2D', 'RELU',
         'FLATTEN', 'RESHAPE', 'DENSE', 'BIAS', 'SCALE', 'BN'}


class SparseResourceLimit(RuntimeError):
    def __init__(self, event):
        self.event = event
        super().__init__('CSR representation admission: '+str(event))


def storage(hz):
    matrices = {name: {'nnz': int(getattr(hz, name).nnz),
                      'bytes': sum(int(a.nbytes) for a in
                                   (getattr(hz, name).data, getattr(hz, name).indices,
                                    getattr(hz, name).indptr)),
                      'index_bytes': int(getattr(hz, name).indices.dtype.itemsize)}
                for name in MATRICES}
    return {'bytes': sum(m['bytes'] for m in matrices.values()) +
                     sum(int(getattr(hz, name).nbytes) for name in ('c', 'b', 'ub')),
            'nnz': sum(m['nnz'] for m in matrices.values()), 'matrices': matrices,
            'n_out': hz.n_out, 'n_cont': hz.n_cont, 'n_bin': hz.n_bin,
            'n_eq': hz.n_eq, 'n_ineq': hz.n_ineq, 'frame_id': hz.frame_id}


def retained_bound(nnz, n_out, n_eq, n_ineq):
    # float64 data + worst-case int64 indices; all SIX row-pointer arrays.
    return 16*int(nnz) + 8*(2*(n_out+n_eq+n_ineq)+6) + 8*(n_out+n_eq+n_ineq)


def estimate(layer, hz, out_dim, *, unstable=None):
    kind = layer.kind.upper()
    if kind not in KINDS:
        return None
    n, e, u = int(out_dim), hz.n_eq, hz.n_ineq
    generators = int(hz.Gc.nnz+hz.Gb.nnz)
    constraints = sum(int(getattr(hz, name).nnz) for name in MATRICES[2:])
    operator_nnz = 0
    k = 0
    if kind == 'RELU':
        # Before support queries use all neurons; before slots use actual k.
        k = hz.n_out if unstable is None else int(unstable)
        nnz = generators + constraints + 8*k
        e, u = e+k, u+2*k
    elif kind in {'CONV2D', 'AVGPOOL2D', 'DENSE'}:
        if kind == 'CONV2D':
            w = layer.params['weight']
            groups = int(layer.params.get('groups', 1))
            fanin = int(w.shape[1])*int(w.shape[2])*int(w.shape[3])
            fanout = int(w.shape[0])//groups*int(w.shape[2])*int(w.shape[3])
        elif kind == 'DENSE':
            w = layer.params['weight']
            fanout, fanin = int(w.shape[0]), int(w.shape[1])
        else:
            kernel = layer.params.get('kernel_size', 1)
            fanin = int(kernel)**2 if isinstance(kernel, int) else math.prod(int(x) for x in kernel)
            fanout = fanin
        # Integer combinatorial upper bounds: no weighted float sum used as a count.
        generator_nnz = min(n*hz.n_cont, int(hz.Gc.nnz)*fanout) + min(n*hz.n_bin, int(hz.Gb.nnz)*fanout)
        operator_nnz = n*fanin
        nnz = constraints + generator_nnz
    else:
        nnz = generators+constraints
    retained = retained_bound(nnz, n, e, u)
    # Explicit engineering reserve for copied/padded CSR, COO concatenations,
    # Python operator triplets, sparse-product column workspaces, and slot maps.
    work = 8*retained + 160*operator_nnz + 192*k + 16*(hz.n_cont+hz.n_bin+3*k)
    return {'estimated_retained_bytes': retained, 'workspace_reserve_bytes': work,
            'nnz_upper_bound': nnz, 'operator_nnz_upper_bound': operator_nnz,
            'unstable_upper_bound': k}


class SparseBudget:
    def __init__(self, limit):
        self.limit = int(limit)
        self.events = []

    def admit(self, stage, layer, cache, estimate_bytes, *, details=None, slot_count=0):
        unique = {id(hz): hz for hz in cache if hz is not None}
        live = sum(storage(hz)['bytes'] for hz in unique.values()) + 192*int(slot_count)
        total = live+int(estimate_bytes)
        event = {'stage': stage, 'layer': int(layer), 'cached_representation_bytes': live,
                 'requested_reserve_bytes': int(estimate_bytes), 'total_accounted_bytes': total,
                 'limit_bytes': self.limit, 'accepted': total <= self.limit}
        if details is not None:
            event['details'] = details
        self.events.append(event)
        if not event['accepted']:
            raise SparseResourceLimit(event)
        return event
