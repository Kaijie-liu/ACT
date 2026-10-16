"""Optional native local-row factorization, never an accepting proof backend.

Import only in a separately admitted CPU production actor after hard limits.
The unchanged V3 COO operator and the row-block operator share this optimizer.
All exact acceptance is performed against the original CheckedSharedForest.
"""
import math
import struct

import torch

from scoped_source.native_loader import reference_operator
from scoped_source.check_shared_source import CheckedSharedForest
from scoped_source.check_scaled_references import checked
from scoped_source.shared_source import SourceLocal
from scoped_source.direct_lifecycle import RootLease
from scoped_source.hz_request_store import require
from scoped_source.native_method import validate

Operator = reference_operator()


class SourceRowOperator(Operator):
    """C=(I-R)^-1 (E_aux + sum S_B W_B P) D, including every tail."""
    def __init__(self, forest, work):
        checked(forest, work)
        require(type(forest) is CheckedSharedForest and torch.get_num_threads() == 1,
                'checked shared forest and one CPU thread')
        self.work, self.forest = work, forest
        self.shape = (len(forest.nodes), len(forest.factors))
        inputs = forest.views[0]['output']
        for j, (_, n) in enumerate(inputs):
            work.visit('row_visits')
            require(not forest.nodes[n][2] and forest.nodes[n][1] == ((j, 1),),
                    'checked input factor map')
        rr, cc, vv, depths, edges, banks = [], [], [], [], {}, {}
        aliases = 0
        for n, (_, local, refs) in enumerate(forest.nodes):
            work.visit('row_visits')
            depth = 1 + max((depths[p] for p, _ in refs), default=0)
            depths.append(depth)
            if type(local) is SourceLocal:
                require(local.work is work and not refs and local.row.width == len(inputs),
                        'source local owner/input map')
                graph, layer, row = local.row.key
                group = banks.setdefault((graph, layer), {})
                require(row not in group, 'source row represented exactly once')
                group[row] = (n, local.row)
                aliases += 1
                terms = local.tail
            else:
                terms = local
            for j, q in terms:
                work.visit('coefficient_visits')
                rr.append(n); cc.append(j); vv.append(float(q))
            for p, s in refs:
                work.visit('coefficient_visits')
                edges.setdefault(depth, []).append((n, p, float(s)))
        scales = []
        for q in forest.scales:
            work.visit('coefficient_visits'); scales.append(float(q))
        self.scales = self.val(scales)[:, None]
        self.local = torch.sparse_coo_tensor(self.ix([rr, cc]), self.val(vv),
                                             self.shape, device='cpu').coalesce()
        self.transpose = self.local.transpose(0, 1)
        self.levels = tuple((self.ix([n for n, p, s in edges[d]]),
                             self.ix([p for n, p, s in edges[d]]),
                             self.val([s for n, p, s in edges[d]])[:, None])
                            for d in sorted(edges))
        self.blocks = []
        weights = 0
        for key, group in banks.items():
            work.tick()
            require(set(group) == set(range(len(group))), 'complete natural row block')
            nodes = []
            # Owned tensor allocation; never expose source bytes as a writable alias.
            bank = torch.empty((len(group), len(inputs)), dtype=torch.float64, device='cpu')
            for row in range(len(group)):
                work.visit('row_visits')
                n, source = group[row]; nodes.append(n)
                require(source.key == (*key, row), 'natural source row identity')
                values = []
                for j in range(source.width):
                    work.visit('coefficient_visits')
                    values.append(struct.unpack_from('<d', source.raw, 8*(j+1))[0])
                    weights += 1
                work.visit('coefficient_visits', source.width)
                bank[row].copy_(self.val(values))
                del values
            require(bool(torch.isfinite(bank).all()), 'finite private source bank')
            self.blocks.append((self.ix(nodes), bank))
        self.blocks = tuple(self.blocks)
        self.input_width = len(inputs)
        self.stats = dict(algorithm='CPU_SCALED_REFERENCE_PROJECTED_SUBGRADIENT_128_V3',
            representation='native_source_row_blocks', device='cpu', dtype='float64',
            nodes=self.shape[0], factors=self.shape[1], local_nnz=len(vv),
            references=sum(map(len, edges.values())), levels=len(self.levels),
            source_blocks=len(self.blocks), source_rows=aliases, source_values=weights,
            source_bank_payload_bytes=8*weights, expanded_constraint_entries=0)
        work.tick()

    def forward(self, x):
        self.validate(x, self.shape[1])
        scaled = self.scales*x
        result = torch.sparse.mm(self.local, scaled)
        for nodes, bank in self.blocks:
            self.work.visit('row_visits')
            result.index_add_(0, nodes, bank @ scaled[:self.input_width])
        for children, parents, scale in self.levels:
            self.work.visit('row_visits')
            result.index_add_(0, children, result.index_select(0, parents)*scale)
        self.work.tick()
        require(bool(torch.isfinite(result).all()), 'finite source forward')
        return result

    def adjoint(self, alpha):
        self.validate(alpha, self.shape[0])
        values = alpha.clone()
        for children, parents, scale in reversed(self.levels):
            self.work.visit('row_visits')
            values.index_add_(0, parents, values.index_select(0, children)*scale)
        residual = torch.sparse.mm(self.transpose, values)
        for nodes, bank in self.blocks:
            self.work.visit('row_visits')
            residual[:self.input_width] += bank.T @ values.index_select(0, nodes)
        result = self.scales*residual
        self.work.tick()
        require(bool(torch.isfinite(result).all()), 'finite source adjoint')
        return result


class DualTile:
    """One owned best-dual tile; no optimizer stack, strings or borrowed roots."""
    def __init__(self, driver, tasks, token, equality, inequality):
        self.driver, self.tasks, self.token = driver, tasks, token
        self.binding = (id(driver.forest), driver.forest.request, driver.forest.identity,
                        driver.contract, driver.ordinal, id(driver.work))
        self.equality, self.inequality = equality, inequality
        self.live = True

    def check(self, driver, tasks, token):
        require(self.live and driver is self.driver and tasks is self.tasks and token is self.token
                and not driver.closed and driver.pending is not None
                and driver.pending[0] is tasks and driver.pending[1] is token
                and self.binding == (id(driver.forest), driver.forest.request, driver.forest.identity,
                                     driver.contract, driver.ordinal, id(driver.work)),
                'owned native tile identity/lifetime')
        for name, kind in (('equality', 'eq'), ('inequality', 'ine')):
            t = getattr(self, name)
            require(type(t) is torch.Tensor and t.device.type == 'cpu' and t.dtype == torch.float64
                    and tuple(t.shape) == (len(driver.forest.rows(tasks[0].pair, kind)), len(tasks))
                    and not t.requires_grad, 'owned complete native dual dimensions')
        driver.work.tick()

    def scalar(self, key, row, col):
        require(self.live and key in ('equality', 'inequality'), 'live scalar dual')
        value = float(getattr(self, key)[row, col].item())
        require(math.isfinite(value), 'finite native candidate')
        return 0.0 if value == 0 else value

    def close(self):
        self.live = False
        self.driver = self.tasks = self.token = self.equality = self.inequality = None


class Proposer:
    """Same V3 optimization for both representations; setup/propose/close once."""
    def __init__(self, config, *, enabled=False):
        require(enabled is True, 'native proposer opt-in')
        self.method = validate(config)
        self.state = 'NEW'
        self.operator = self.forest = self.work = self.tile = None
        self.stats = dict(setup_calls=0, propose_calls=0, close_calls=0)

    def setup(self, forest, work):
        require(self.state == 'NEW' and type(forest) is CheckedSharedForest, 'one native setup')
        self.state = 'STARTING'; self.stats['setup_calls'] += 1
        self.forest, self.work = forest, work
        op = Operator if self.method['producer']['representation'] == 'native_coo_reference' else SourceRowOperator
        self.operator = op(forest, work)
        self.stats['operator'] = dict(self.operator.stats,
            representation=self.method['producer']['representation'])
        self.state = 'READY'

    def propose(self, tasks, roots):
        require(self.state == 'READY' and self.tile is None and type(roots) is RootLease,
                'native ready/issued roots')
        roots._check()
        driver = roots._driver
        require(driver.forest is self.forest and roots._work is self.work and tasks is roots._tasks,
                'native forest/Work/task binding')
        k = len(tasks)
        require(0 < k <= 8 and len(roots) == k and len({t.pair for t in tasks}) == 1,
                'native one-pair bounded tile')
        self.state = 'PROPOSING'; self.stats['propose_calls'] += 1
        op, work = self.operator, self.work
        alpha = torch.zeros((op.shape[0], k), dtype=torch.float64, device='cpu')
        offsets = []
        for col, (d, weights) in enumerate(roots):
            work.visit('row_visits'); offsets.append(float(d))
            for n, q in weights.items():
                work.visit('coefficient_visits'); alpha[n, col] = float(q)
        d = op.val(offsets); groups = []
        for kind in ('ine', 'eq'):
            ix, rhs = [], []
            for row in self.forest.rows(tasks[0].pair, kind):
                work.visit('row_visits'); c, n = self.forest.row(row); ix.append(n)
                work.visit('coefficient_visits'); rhs.append(float(c))
            z = torch.zeros((len(rhs), k), dtype=torch.float64, device='cpu')
            groups.append(dict(kind=kind, ix=op.ix(ix), rhs=op.val(rhs)[:, None], z=z, best=z.clone()))
        best = torch.full((k,), -math.inf, dtype=torch.float64, device='cpu')
        with torch.no_grad():
            for step in range(129):
                work.visit('row_visits'); a, value = alpha.clone(), d.clone()
                for g in groups:
                    require(bool(torch.isfinite(g['z']).all()), 'finite dual iterate')
                    a.index_add_(0, g['ix'], -g['z']); value += (g['rhs']*g['z']).sum(0)
                residual = op.adjoint(a); value -= residual.abs().sum(0)
                require(bool(torch.isfinite(value).all()), 'finite candidate value')
                good = value > best; best = torch.where(good, value, best)
                for g in groups: g['best'][:, good] = g['z'][:, good]
                if step == 128: break
                points = op.forward(-torch.sign(residual)); rate = .125/math.sqrt(step+1)
                for g in groups:
                    z = g['z'] + rate*(g['rhs'] - points.index_select(0, g['ix']))
                    g['z'] = torch.minimum(z, torch.zeros_like(z, device='cpu')) if g['kind'] == 'ine' else z
        roots._check(); work.tick()
        # Explicitly map optimizer ine/eq order to the wire eq/ine order.
        self.tile = DualTile(driver, tasks, roots._token, groups[1]['best'], groups[0]['best'])
        self.state = 'TILE'
        return self.tile

    def release(self):
        require(self.state == 'TILE' and self.tile is not None, 'one native tile release')
        self.tile.close(); self.tile = None; self.state = 'READY'; self.work.tick()

    def close(self):
        require(self.state != 'CLOSED', 'one native close')
        self.stats['close_calls'] += 1; self.state = 'CLOSED'
        if self.tile is not None: self.tile.close()
        self.tile = self.operator = self.forest = self.work = None
