"""Complete finite route-history verification in a shared ACT/HybridZ graph.

All histories are obligations, including ties and histories whose feasibility
is undecided. This v1 deliberately trades scalability for an explicit coverage
contract. It proves a real-arithmetic model of stored CPU/float64 coefficients
under ACT's existing lowering/solver policy, NOT deployed float execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import inspect
import itertools
import json
import math
import operator
from pathlib import Path
import time

import torch
from torch import fx, nn


class RoutedLayer(nn.Module):
    """Single per-input top-k layer, usable anywhere in a static acyclic model.

    Nested routers inside a router/expert and token/patch routing are not v1.
    Residual connections and repeated calls to the same layer ARE supported.
    """
    def __init__(self, router, experts, *, k=1, mode='hard_top1', epsilon=1e-5):
        super().__init__()
        self.router, self.experts = router, nn.ModuleList(experts)
        self.k, self.mode, self.epsilon = k, mode, epsilon
        if (type(k) is not int or not 1 <= k <= len(experts) or
                mode not in {'hard_top1', 'selected_softmax', 'raw_epsilon', 'raw', 'nonzero_ste'}
                or mode in {'hard_top1', 'nonzero_ste'} and k != 1
                or not math.isfinite(epsilon) or epsilon <= 0):
            raise ValueError('invalid routing semantics')

    def forward(self, x):
        scores = self.router(x)
        indices = scores.topk(self.k, dim=1).indices
        selected = scores.gather(1, indices)
        if self.mode == 'selected_softmax':
            weights = selected.softmax(dim=1)
        elif self.mode == 'raw_epsilon':
            weights = selected / (selected.sum(1, keepdim=True) + self.epsilon)
        elif self.mode == 'raw':
            weights = selected
        elif self.mode == 'nonzero_ste':
            weights = (selected != 0).to(selected)
        else:
            weights = torch.ones_like(selected)
        outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        flat = outputs.flatten(start_dim=2)
        chosen = flat.gather(1, indices[:, :, None].expand(-1, -1, flat.shape[2]))
        return (chosen * weights[:, :, None]).sum(1).reshape(outputs.shape[0], *outputs.shape[2:])


def _author_layer(module):
    return (type(module).__module__ == 'src.models.nn.moe.layer'
            and type(module).__name__ in {'MOELayer', 'SkipMOELayer'})


def _routed(module):
    return type(module) is RoutedLayer or _author_layer(module)


@dataclass(frozen=True)
class Site:
    name: str
    target: str
    experts: int
    k: int
    mode: str
    epsilon: float
    rank: int
    residual: bool


def _semantics(module, source_hashes):
    if type(module) is RoutedLayer:
        return module.router, module.experts, module.k, module.mode, module.epsilon, False
    if not _author_layer(module):
        raise NotImplementedError('unsupported routing layer')
    gate = module.gate
    if (type(gate).__module__ != 'src.models.nn.moe.gate.topk'
            or type(gate).__name__ != 'TopKGate' or module.fixed_expert is not None
            or module.variance_alpha != 0):
        raise NotImplementedError('author fixed/variance/other gate semantics not supported')
    reviewed = {'MOELayer': '2988b8ee8a8a5fa0681fcd86c3ac7e2415ee2e4ef0f18345561194004dcad114',
        'SkipMOELayer': '2988b8ee8a8a5fa0681fcd86c3ac7e2415ee2e4ef0f18345561194004dcad114',
        'TopKGate': '9e3e480d7e5ccfec02c19ce8078596122b1fd6b7818b157674bc8c5a144da70c'}
    for obj in [module, gate]:
        path = str(Path(inspect.getsourcefile(type(obj))).resolve())
        if source_hashes.get(path) != reviewed[type(obj).__name__] or 'forward' in obj.__dict__:
            raise ValueError('unbound author source or overridden forward')
    mode = ('nonzero_ste' if gate.use_straight_through_estimator else
            'raw_epsilon' if gate.normalize_routing else 'raw')
    # The author's float32 STE cast is not represented as a floating-point proof.
    return gate.network, module.experts, gate.k, mode, 1e-5, type(module).__name__ == 'SkipMOELayer'


class _Tracer(fx.Tracer):
    def is_leaf_module(self, module, qualified_name):
        return _routed(module) or super().is_leaf_module(module, qualified_name)


class _Branch(nn.Module):
    def __init__(self, module, site, selected, source_hashes):
        super().__init__()
        router, experts, *_ = _semantics(module, source_hashes)
        self.router = router
        self.experts = nn.ModuleList([experts[i] for i in selected])
        self.selected, self.site = tuple(selected), site
        self.softmax = nn.Softmax(dim=1)
        with torch.random.fork_rng(devices=[]):
            self.select = nn.Linear(site.experts, len(selected), bias=False, dtype=torch.float64)
            self.denominator = nn.Linear(len(selected), len(selected), dtype=torch.float64)
            self.weight_rows = nn.ModuleList([nn.Linear(len(selected),1,bias=False,dtype=torch.float64) for _ in selected])
        with torch.no_grad():
            self.denominator.weight.fill_(1.)
            self.denominator.bias.fill_(site.epsilon)
            self.select.weight.zero_()
            for j, i in enumerate(selected):
                self.select.weight[j, i] = 1.
                self.weight_rows[j].weight.zero_()
                self.weight_rows[j].weight[0,j] = 1.

    def forward(self, x):
        scores = self.router(x)
        selected_scores = self.select(scores)
        mode = self.site.mode
        if mode == 'selected_softmax':
            weights = self.softmax(selected_scores)
        elif mode == 'raw_epsilon':
            weights = selected_scores / self.denominator(selected_scores)
        else:
            weights = selected_scores
        out = None
        for j, expert in enumerate(self.experts):
            value = expert(x)
            if mode not in {'hard_top1', 'nonzero_ste'}:
                weight = self.weight_rows[j](weights)
                if self.site.rank != 2:
                    weight = weight.reshape((1, 1) + (1,)*(self.site.rank-2))
                value = value * weight
            out = value if out is None else out + value
        if self.site.residual:
            out = x + out
        return out, scores


class HistoryPlan:
    def __init__(self, model, sample, *, source_hashes=None):
        self.model, self.source_hashes = model, dict(source_hashes or {})
        if any(m.training for m in model.modules()):
            raise ValueError('eval required; mode is never changed implicitly')
        for path, value in self.source_hashes.items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != value:
                raise ValueError('author source changed')
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm) and not module.track_running_stats:
                raise NotImplementedError('batch-dependent eval BN is not a per-input function')
            if type(module).__name__ in {'LoRAMoEAdapterMLP', 'LoRAMoEAdapterLinear'}:
                raise NotImplementedError('RoME dense continuous routing is not discrete top-k')
            if 'forward' in module.__dict__ or module._forward_hooks or module._forward_pre_hooks:
                raise ValueError('overridden forward / active hooks unsupported')
            if _routed(module):
                router, experts, *_ = _semantics(module, self.source_hashes)
                if any(_routed(child) for part in [router, *experts] for child in part.modules()):
                    raise NotImplementedError('nested/token routing requires a separate compiler')
        self.graph = fx.GraphModule(model, _Tracer().trace(model))
        self.sites = []
        owner = self
        class Shapes(fx.Interpreter):
            def run_node(self, node):
                if node.op == 'call_module' and _routed(self.fetch_attr(node.target)):
                    module = self.fetch_attr(node.target)
                    args, kwargs = self.fetch_args_kwargs_from_env(node)
                    if len(args) != 1 or kwargs or args[0].shape[0] != 1:
                        raise NotImplementedError('one per-input routing tensor required')
                    router, experts, k, mode, epsilon, residual = _semantics(module, owner.source_hashes)
                    scores = router(args[0])
                    outputs = [expert(args[0]) for expert in experts]
                    if (scores.shape != (1, len(experts)) or not torch.isfinite(scores).all()
                            or any(not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
                                   or v.shape != outputs[0].shape for v in outputs)):
                        raise ValueError('nonfinite or shape-changing components')
                    owner.sites.append(Site(node.name, node.target, len(experts), k, mode,
                        epsilon, outputs[0].ndim, residual))
                return super().run_node(node)
        with torch.no_grad():
            out = Shapes(self.graph).run(sample)
        if not isinstance(out, torch.Tensor) or out.ndim != 2 or out.shape[0] != 1 or not torch.isfinite(out).all():
            raise NotImplementedError('single batch classification/vector output required')
        self.output_width = out.shape[1]
        if not self.sites:
            raise NotImplementedError('no supported discrete routing sites')
        self.total_histories = math.prod(math.comb(s.experts,s.k) for s in self.sites)

    def histories(self):
        # Recursive generator does not materialize enormous combination pools
        # before the verifier has had a chance to apply its explicit cap.
        def walk(i, prefix):
            if i == len(self.sites):
                yield tuple(prefix)
            else:
                s = self.sites[i]
                for choice in itertools.combinations(range(s.experts),s.k):
                    yield from walk(i+1,[*prefix,choice])
        return walk(0,[])

    def compile(self, history):
        history = tuple(tuple(s) for s in history)
        if len(history) != len(self.sites) or any(len(h)!=s.k or tuple(sorted(set(h)))!=h
                or any(type(i) is not int or not 0 <= i < s.experts for i in h) for h,s in zip(history,self.sites)):
            raise ValueError('invalid complete history')
        root = nn.Module()
        root.add_module('original', self.graph)
        graph, env, score_nodes, score_rows = fx.Graph(), {}, [], []
        by_name = {s.name: (i, s) for i, s in enumerate(self.sites)}
        cursor = self.output_width
        for node in self.graph.graph.nodes:
            if node.op == 'output':
                original_out = node.args[0]
                if isinstance(original_out, (tuple, list)):
                    original_out = original_out[0]
                value = fx.map_arg(original_out, lambda n: env[n])
                graph.output(graph.call_function(torch.cat, args=([value, *score_nodes],), kwargs={'dim': 1}))
            elif node.name in by_name:
                i, site = by_name[node.name]
                module = self.graph.get_submodule(site.target)
                root.add_module('branch'+str(i), _Branch(module, site, history[i], self.source_hashes))
                pair = graph.call_module('branch'+str(i), args=fx.map_arg(node.args, lambda n: env[n]))
                env[node] = graph.call_function(operator.getitem, (pair, 0))
                score_nodes.append(graph.call_function(operator.getitem, (pair, 1)))
                score_rows.append(tuple(range(cursor, cursor+site.experts)))
                cursor += site.experts
            else:
                copied = graph.node_copy(node, lambda n: env[n])
                if node.op in {'call_module', 'get_attr'}:
                    copied.target = 'original.'+node.target
                env[node] = copied
        # ACT consumes an existing GraphModule without retracing it. Inline the
        # static branch wrappers now so it sees ordinary operators, not opaque
        # routing modules that its converter cannot lower.
        compiled = fx.symbolic_trace(fx.GraphModule(root, graph).eval()).eval()
        compiled.graph.eliminate_dead_code()
        compiled.recompile()
        return compiled, score_rows


def _hash_tensor(tensor):
    tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(tensor.shape), str(tensor.dtype))).encode()+tensor.numpy().tobytes()).hexdigest()


def _lower(compiled, center, lower, upper, width):
    from act.front_end.spec_creator_base import LabeledInputTensor
    from act.front_end.specs import InputSpec, InKind, OutKind, OutputSpec
    from act.front_end.verifiable_model import InputLayer, InputSpecLayer, OutputSpecLayer, VerifiableModel
    from act.pipeline.verification.torch2act import TorchToACT, _LayerGraphBuilder
    # Local additions: do not alter frozen production or author execution paths.
    class HistoryBuilder(_LayerGraphBuilder):
        def _handle_call_function(self, node):
            if node.target in {operator.add, torch.add} and len(node.args) == 2:
                a, b = node.args
                if isinstance(a, (int, float)):
                    a, b = b, a
                if isinstance(b, (int, float)):
                    self._emit_const_bias(node, a, self._resolve_const_tensor(b), negate=False)
                    return
            if node.target in {operator.truediv, torch.div}:
                a, b = node.args
                if a.name not in self.node_outputs or b.name not in self.node_outputs:
                    raise NotImplementedError('division requires two explicit tensor operands')
                av, bv = self.node_outputs[a.name], self.node_outputs[b.name]
                shape = self.node_shapes[a.name]
                if shape != self.node_shapes[b.name]:
                    raise NotImplementedError('division shape mismatch')
                out = self._alloc_ids(len(av))
                layer_id = self._add_layer('DIV', {'x_vars':av,'y_vars':bv,
                    'input_shape':shape,'output_shape':shape}, av+bv, out)
                self.prev_out, self.shape = out, shape
                self._register_node(node.name, layer_id)
                return
            if node.target is operator.getitem:
                raise NotImplementedError('tensor indexing must be lowered to an explicit selector')
            super()._handle_call_function(node)
    class HistoryLowering(TorchToACT):
        def _build_inner_model(self):
            layers, preds, succs = HistoryBuilder(self._find_inner_model(), self.shape,
                dtype=torch.float64, sample_input=self.sample_input).build_layer_graph()
            offset = len(self.layers)
            for layer in layers:
                layer.id += offset
                for key in ('q_src','k_src','w_src','v_src'):
                    if isinstance(layer.params.get(key), int):
                        layer.params[key] += offset
            self.layers.extend(layers)
            self.prev_out = layers[-1].out_vars
            self._model_preds = {k+offset:[v+offset for v in values] for k,values in preds.items()}
            self._model_succs = {k+offset:[v+offset for v in values] for k,values in succs.items()}
            self._wrapper_offset = offset
    wrapper = VerifiableModel(input_layer=InputLayer(LabeledInputTensor(center, label=None),
        shape=tuple(center.shape), dtype=center.dtype),
        input_spec=InputSpecLayer(InputSpec(kind=InKind.BOX, lb=lower, ub=upper)), model=compiled,
        output_spec=OutputSpecLayer(OutputSpec(kind=OutKind.LINEAR_LE,
            c=torch.zeros(1, width, dtype=center.dtype), d=torch.zeros(1, dtype=center.dtype))))
    return HistoryLowering(wrapper, sample_input=center).run()


def verify_multilayer_box(model, *, center, lower, upper, rows, thresholds,
                          total_seconds=300., max_histories=4096, source_hashes=None, progress=None):
    """All-history HZ-policy certificate; use a process supervisor for hard limits.

    No clean-route restriction, no incomplete-enumeration positives, no expert-
    only UNSAFE. Unknown feasibility is retained as an obligation. A deliberately
    full-history compiler preserves dependence on all earlier routing decisions.
    """
    from act.back_end.moe.class_separated_top1 import validate_replay
    from act.back_end.moe.hz_routing import condition_topk_set
    from act.back_end.solver.solver_hz import (SparseHZono, HZ_NUMERICAL_POLICY,
        hz_check_feasibility, hz_minimize_output, hz_multiply, sparse_hz_linear,
        hz_add_const, hz_support_bounds, hz_numerical_policy_manifest)
    from act.pipeline.moe.experiment1 import _propagate_component
    from act.config.config import HybridZConfig
    from act.util.device_manager import get_default_device
    started = time.monotonic()
    if not math.isfinite(total_seconds) or total_seconds <= 0 or type(max_histories) is not int or max_histories < 1:
        raise ValueError('finite positive budget and history cap required')
    deadline = started+total_seconds
    if (get_default_device().type != 'cpu' or any(t.device.type != 'cpu' or t.dtype != torch.float64
            or not torch.isfinite(t).all() for t in [center, lower, upper, rows, thresholds])
            or center.shape != lower.shape or center.shape != upper.shape or center.ndim < 2
            or center.shape[0] != 1 or (lower > center).any() or (center > upper).any()
            or rows.ndim != 2 or rows.shape[0] < 1 or thresholds.shape != (rows.shape[0],)):
        raise ValueError('explicit finite CPU/float64 domain and linear properties required')
    if any(p.device.type != 'cpu' or (p.is_floating_point() and p.dtype != torch.float64)
           for p in itertools.chain(model.parameters(), model.buffers())):
        raise ValueError('explicit CPU/float64 model snapshot required')
    state_id = {k: _hash_tensor(v) for k, v in model.state_dict().items()}
    result = {'schema': 'multilayer-histories-v1', 'status': 'UNKNOWN', 'records': [],
        'complete': False, 'evidence_grade': 'NONE', 'source_complete': False,
        'semantics': 'real_arithmetic_stored_coefficients_ANY_LEGAL_TOPK',
        'numerical_policy': hz_numerical_policy_manifest(), 'properties': len(rows),
        'trusted': ['FX/static_history_lowering', 'network_input_guard_to_HZ', 'HZ_solver_numerical_policy'],
        'identity': {'state': state_id, 'sources': dict(source_hashes or {}),
            'properties': len(rows), 'budget_seconds': total_seconds, 'history_cap': max_histories,
            'model_structure': repr(model), 'implementation': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'policy': hz_numerical_policy_manifest(),
            'tensors': {k: _hash_tensor(v) for k, v in [('center',center),('lower',lower),('upper',upper),('rows',rows),('thresholds',thresholds)]}}}
    def finish(status, reason):
        if {k: _hash_tensor(v) for k,v in model.state_dict().items()} != state_id:
            status, reason = 'ERROR', 'model_state_changed_during_request'
        if any(hashlib.sha256(Path(p).read_bytes()).hexdigest() != h for p,h in (source_hashes or {}).items()):
            status, reason = 'ERROR', 'source_changed_during_request'
        if any(_hash_tensor(v) != result['identity']['tensors'][k] for k,v in
               [('center',center),('lower',lower),('upper',upper),('rows',rows),('thresholds',thresholds)]):
            status, reason = 'ERROR', 'request_changed_during_execution'
        result.update(status=status, reason=reason, seconds=time.monotonic()-started)
        if time.monotonic() >= deadline:
            result.update(status='TIMEOUT', reason='request_deadline', complete=False)
        result['evidence_grade'] = ('HZ_POLICY_ACCEPTED' if result['status']=='POSITIVE' else
            'FULL_MODEL_REPLAY' if result['status']=='UNSAFE_REPLAYED' else 'NONE')
        return result
    def tick():
        if time.monotonic() >= deadline:
            raise TimeoutError('request_deadline')
    def publish():
        if progress:
            progress(copy.deepcopy(result))
    try:
        plan = HistoryPlan(model, center, source_hashes=source_hashes)
        if rows.shape[1] != plan.output_width:
            raise ValueError('property dimension mismatch')
        result.update(sites=[asdict(s) for s in plan.sites], expected_histories=plan.total_histories)
        result['identity']['graph'] = hashlib.sha256(plan.graph.code.encode()).hexdigest()
        result['identity']['sites'] = result['sites']
        result['request_id'] = hashlib.sha256(json.dumps(result['identity'],sort_keys=True).encode()).hexdigest()
        tick()
        if validate_replay(model, center, lower, upper, rows, thresholds):
            result['witness'] = center.tolist()
            return finish('UNSAFE_REPLAYED', 'complete_model_center')
        # In the original author dispatcher, all selected raw scores zero can
        # leave expert_output_shape undefined. A nonzero normalization
        # denominator alone does NOT prove this source dispatch is defined.
        # Static compilation supports its expression, but positive certification
        # currently admits only the separately checked nonzero-STE contract.
        if any(_author_layer(plan.graph.get_submodule(s.target)) and s.mode!='nonzero_ste' for s in plan.sites):
            return finish('UNSUPPORTED', 'author_raw_dispatch_definedness_not_supported')
        if plan.total_histories > max_histories:
            return finish('UNKNOWN', 'history_limit_no_histories_dropped')
        for history in plan.histories():
            tick()
            began = time.monotonic()
            record = {'history': [list(h) for h in history], 'status': 'UNKNOWN',
                'phase': 'compile', 'properties': [], 'definedness': [], 'seconds': None}
            result['records'].append(record)
            publish()
            compiled, score_rows = plan.compile(history)
            record['compiled_graph_sha256'] = hashlib.sha256(compiled.code.encode()).hexdigest()
            width = plan.output_width+sum(s.experts for s in plan.sites)
            net = _lower(compiled, center, lower, upper, width)
            record.update(compile_seconds=time.monotonic()-began, phase='propagation')
            publish()
            try:
                propagated = _propagate_component(net, hybridz_config=HybridZConfig(max_input_dim=1))
            except RuntimeError as exc:
                if str(exc) == 'HybridzTF did not retain an HZ at the component output':
                    raise NotImplementedError('backend_joint_HZ_not_retained; no independent-box substitution') from exc
                raise
            tick()
            record['propagation_seconds'] = propagated.elapsed
            phase_start = time.monotonic()
            hz = propagated.output_hz
            defined = True
            record['guard_seconds'] = 0.
            for prefix_index, (site, selected, indices) in enumerate(zip(plan.sites, history, score_rows)):
                guard_start = time.monotonic()
                hz = condition_topk_set(hz, selected, score_rows=indices).hz
                record['guard_seconds'] += time.monotonic()-guard_start
                # Establish score-definedness BEFORE later guards. Otherwise
                # downstream guards could hide inputs where the STE reduction
                # changed an earlier zero score into a unit contribution.
                if site.mode not in {'nonzero_ste', 'raw_epsilon'}:
                    continue
                groups = [selected] if site.mode=='raw_epsilon' else [(i,) for i in selected]
                for group in groups:
                    check_start = time.monotonic()
                    coefficients = torch.zeros(1, width, dtype=torch.float64)
                    coefficients[0, [indices[i] for i in group]] = 1.
                    offset = site.epsilon if site.mode=='raw_epsilon' else 0.
                    domain = (sparse_hz_linear(hz, coefficients.numpy(), [offset]) if isinstance(hz, SparseHZono)
                              else hz_add_const(hz_multiply(hz, coefficients), torch.tensor([[offset]],dtype=torch.float64)))
                    bounds = hz_support_bounds(domain, [0], time_limit=min(10., max(.001, deadline-time.monotonic())), relax_binaries=False)
                    lo, hi = bounds.bounds.lb.item(), bounds.bounds.ub.item()
                    accepted = math.isfinite(lo) and math.isfinite(hi) and lo <= hi and (lo > 0 or hi < 0)
                    record['definedness'].append({'site': site.name, 'selected': list(group),
                        'prefix_sites': prefix_index+1, 'lower': lo if math.isfinite(lo) else None,
                        'upper': hi if math.isfinite(hi) else None, 'accepted': accepted,
                        'seconds': time.monotonic()-check_start})
                    defined &= accepted
                    publish()
                    tick()
                if not defined:
                    break
            if not defined:
                record.update(phase='complete',reason='unproved_prefix_definedness',seconds=time.monotonic()-began)
                publish()
                continue
            record['phase'] = 'feasibility'
            publish()
            phase_start = time.monotonic()
            feasible = hz_check_feasibility(hz, time_limit=max(.001, min(10., deadline-time.monotonic())))
            tick()
            record.update(feasibility_seconds=time.monotonic()-phase_start, feasibility=feasible.status,
                status='EXCLUDED' if feasible.status=='infeasible' else 'UNKNOWN',phase='definedness')
            publish()
            if feasible.status != 'infeasible':
                if defined:
                    coefficients = torch.zeros(len(rows), width, dtype=torch.float64)
                    coefficients[:, :plan.output_width] = rows
                    objective = (sparse_hz_linear(hz, coefficients.numpy(), -thresholds.numpy()) if isinstance(hz, SparseHZono)
                        else hz_add_const(hz_multiply(hz, coefficients), -thresholds[:,None]))
                    for i in range(len(rows)):
                        tick()
                        record.update(phase='property', active_property=i)
                        publish()
                        bound = hz_minimize_output(objective, i,
                            input_hz=propagated.input_hz if isinstance(propagated.input_hz, SparseHZono) else None,
                            input_shape=tuple(center.shape), time_limit=max(.001,deadline-time.monotonic()))
                        tick()
                        record['properties'].append({'row': i, 'status': bound.status,
                            'lower': bound.minimum if bound.minimum is not None and math.isfinite(bound.minimum) else None,
                            'solver_status': bound.solver_status, 'bound_kind': bound.solver_bound_kind,
                            'seconds': bound.elapsed})
                        if validate_replay(model, bound.candidate_input, lower, upper, rows, thresholds):
                            result['witness'] = bound.candidate_input.reshape_as(center).tolist()
                            record['status'] = 'UNSAFE_REPLAYED'
                            record.update(phase='complete', seconds=time.monotonic()-began)
                            return finish('UNSAFE_REPLAYED', 'complete_model_candidate')
                    if all(p['status']=='optimal' and p['solver_status']==0 and p['lower'] is not None and math.isfinite(p['lower'])
                           and p['lower'] > HZ_NUMERICAL_POLICY.safe_positive_margin for p in record['properties']):
                        record['status'] = 'ACCEPTED'
            record['seconds'] = time.monotonic()-began
            record['phase'] = 'complete'
            publish()
        result['complete'] = len(result['records']) == plan.total_histories
        if (any(r['status']=='ACCEPTED' for r in result['records'])
                and all(r['status'] in {'ACCEPTED','EXCLUDED'} for r in result['records'])):
            return finish('POSITIVE', 'all_complete_histories_discharged')
        return finish('UNKNOWN', 'remaining_history_obligations')
    except TimeoutError:
        return finish('TIMEOUT', 'request_deadline')
    except NotImplementedError as exc:
        return finish('UNSUPPORTED', str(exc))
