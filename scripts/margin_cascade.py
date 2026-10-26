"""M-C bounded history phase; M-A removal keeps coupled margin-cell graphs.

Reuses the registered CROWN/GenBaB calls and outer owned process. No queue,
supervisor, new allowance, or mechanism admission is introduced here.
"""
import hashlib
import json
import math
import gate_decoupling_worker as W
import moe_aware_diagnostics as D
from scoped_proof.io import load, tick


def options(value):
    W.require(isinstance(value, dict) and set(value) == {'pieces', 'decouple', 'guards'}
        and type(value['pieces']) is int and value['pieces'] in (2, 4)
        and type(value['decouple']) is bool and type(value['guards']) is bool, 'fixed M-C/ablation options')
    return dict(value)


def scoped(parent, cell, layer, decouple):
    G = D.gate_module()
    identity = dict(parent=G.domain_digest(parent), layer=layer, lower=cell[0], upper=cell[1],
                    mode='MARGIN_CONDITIONAL_V2' if decouple else 'MARGIN_CONDITIONAL_COUPLED')
    sid = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    margins = list(parent.margins); margins[layer] = tuple(cell)
    return G.GateDomain(sid, parent.history, tuple(margins), tuple(G.sigmoid_interval(*m) for m in margins))


def check_priority(parent, priority):
    G = D.gate_module(); n = len(parent.history)
    W.require(priority['source_id'] == parent.source_id
        and priority['parent_domain_sha256'] == G.domain_digest(parent)
        and priority['heuristic_only'] is True and priority['certified_error_bound'] is False,
        'priority source, never acceptance evidence')
    widths = [hi-lo for lo, hi in parent.intervals]
    d = D.vector(priority['difference_absolute_amplitudes'], n)
    k = D.vector(priority['layer_lipschitz'], n)
    norm = priority['property_head_norm']
    W.require(priority['widths'] == widths and all(x >= 0 for x in d+k)
        and type(norm) in (int, float) and math.isfinite(norm) and norm >= 0, 'finite error ordering inputs')
    scores = [0.]*n; suffix = norm
    for i in reversed(range(n)):
        scores[i] = widths[i]*d[i]*suffix; suffix *= k[i]
    W.require(all(math.isfinite(v) for v in scores) and priority['scores'] == scores
        and priority['selected_layer'] == max(range(n), key=lambda i: (scores[i], -i))
        and priority['tie_rule'] == 'lowest layer index', 'fixed error ranking and tie policy')
    return priority['selected_layer']


def aggregate(parent, config, priority, rows, properties, experts, deadline):
    config = options(config); layer = check_priority(parent, priority); G = D.gate_module()
    cells = D.cells_for(parent, 'MC_'+str(config['pieces']), layer=layer)
    variants = ('V2_LO', 'V2_HI') if config['decouple'] else ('COUPLED',)
    expected = [(i, v) for i in range(len(cells)) for v in variants]
    seen, positive = set(), []
    for row in rows:
        tick(deadline); key = (row['cell_index'], row['variant'])
        W.require(key in expected and key not in seen, 'unique complete registered cell queries')
        seen.add(key); cell = cells[key[0]]; domain = scoped(parent, cell, layer, config['decouple'])
        W.require(row['layer'] == layer and row['cell'] == cell and row['source_id'] == domain.source_id
            and row['domain_sha256'] == G.domain_digest(domain)
            and row['parent_domain_sha256'] == G.domain_digest(parent), 'cell and coupled/decoupled identity')
        lower = row['initial_lower']; done = False
        if lower is not None:
            done = all(v > 0 for v in D.vector(lower, properties))
        support = row['guard_support']
        if support is not None:
            W.require(config['guards'], 'guard-dual ablation flag')
            result = D.aggregate_mb(support, domain.source_id, [list(p) for p in parent.history],
                'MB_ON', properties, sum(2*(e-2) for e in experts)+2, deadline,
                cells=[dict(layer=layer, lower=cell[0], upper=cell[1])])
            done = done or result['closed']
        if row['outcome'] is not None:
            outcome = row['outcome']
            W.require(json.loads(json.dumps(outcome['actual_raw_spec'])) == D.raw_spec(properties)
                and outcome['status'] in ('verified', 'unknown', 'falsified'), 'all node properties, original spec')
            done = done or outcome['status'] == 'verified'
        positive.append(done)
    complete = len(seen) == len(expected); closed = complete and all(positive)
    return dict(status='POSITIVE' if closed else 'UNKNOWN', closed=closed,
        original_history_closed=closed, complete_MoE=False, formal_SAFE=False,
        expected_nodes=len(expected), recorded_nodes=len(seen), closed_nodes=sum(positive),
        selected_layer=layer, variants=list(variants), complete_cover=complete)


def solve(model, tensors, parent, config, deadline, journal, replay, *, bound=None, verify=None, support=None):
    from act.back_end.moe import gate_decoupling as G
    from act.back_end.moe.moe_aware_bab import (margin_priority, margin_endpoint,
        history_objectives, MarginCell, GuardMaximum, conditional_bounds)
    from b2_observed_worker import verify_graph
    config = options(config)
    bound = D.initial_bounds if bound is None else bound
    verify = verify_graph if verify is None else verify
    support = conditional_bounds if support is None else support
    parent = G.GateDomain(parent.source_id, parent.history, parent.margins, parent.intervals)
    G.validate_domain(parent, parent.history, expected_source_id=parent.source_id)
    experts = [len(l.experts) for l in model.layers]; properties = len(tensors['rows'])
    # Ranking is an explicitly non-proof heuristic. Its mathematical/source
    # domain has already been checked; unavailable ranking permits fallback.
    try:
        priority = journal.call('margin_priority', lambda: margin_priority(model, parent,
            tensors['lower'], tensors['upper'], tensors['rows'], deadline=deadline))
    except (ValueError, OverflowError) as error:
        failure = dict(kind=type(error).__name__, message=str(error))
        journal.put('priority_unavailable.json', failure)
        return dict(status='UNKNOWN', closed=False, original_history_closed=False, complete_MoE=False,
            formal_SAFE=False, config=config, priority=None, priority_failure=failure,
            endpoints=[], full_model_replays=[], original_witness_found=False)
    layer = check_priority(parent, priority); journal.put('margin_priority.json', priority)
    cells = D.cells_for(parent, 'MC_'+str(config['pieces']), layer=layer)
    variants = ('V2_LO', 'V2_HI') if config['decouple'] else ('COUPLED',)
    def prepare(cell, variant):
        sd = scoped(parent, cell, layer, config['decouple']); condition = MarginCell(layer, *cell)
        if config['decouple']:
            joint, actual = margin_endpoint(model, parent, tensors['rows'], tensors['thresholds'], condition,
                variant=variant, input_features=tensors['center'][0].numel(), expected_source_id=parent.source_id)
            W.require(actual.source_id == sd.source_id, 'same conditional endpoint source')
            box = G.extend_box(tensors['center'], tensors['lower'], tensors['upper'], actual,
                variant=variant, expected_source_id=actual.source_id)
        else:
            joint = history_objectives(model, parent.history, tensors['rows'], tensors['thresholds'],
                                       source_id=sd.source_id, cells=(condition,))
            box = {k: tensors[k] for k in ('center', 'lower', 'upper')}
        return joint, GuardMaximum(joint).eval(), box, sd
    rows, witnesses = [], []
    for i, cell in enumerate(cells):
        for variant in variants:
            tick(deadline); name = f'branch_{i}_{variant}'
            joint, graph, box, sd = journal.call(name+':construct', lambda: prepare(cell, variant))
            guarded = None; values = None
            if config['guards']:
                guarded = journal.call(name+':guard_support', lambda: support(joint, box['center'],
                    box['lower'], box['upper'], deadline=deadline, use_guards=True, iterations=20))
            if guarded is None or not all(v > 0 for v in guarded['lower']):
                values = journal.call(name+':initial', lambda: bound(graph, box, deadline))
                D.vector(values, properties)
            row = dict(cell_index=i, layer=layer, cell=cell, variant=variant, source_id=sd.source_id,
                domain_sha256=G.domain_digest(sd), parent_domain_sha256=G.domain_digest(parent),
                guard_support=guarded, initial_lower=values, outcome=None)
            # Receiver validates either kind of lower-bound obligation separately;
            # a conditional property bound is not a global guard-max bound.
            aggregate(parent, config, priority, rows+[row], properties, experts, deadline)
            journal.put(name+'_bounded.json', row); rows.append(row)
            del joint, graph, box
    for row in rows:
        tick(deadline)
        one = aggregate(parent, config, priority, [row], properties, experts, deadline)
        if one['closed_nodes'] == 1:
            continue
        name = f"branch_{row['cell_index']}_{row['variant']}"
        joint, graph, box, _ = journal.call(name+':reconstruct', lambda: prepare(row['cell'], row['variant']))
        outcome, attacks = journal.call(name+':solve',
            lambda: verify(graph, box['lower'], box['upper'], properties, deadline))
        row['outcome'] = outcome; journal.put(name+'_solved.json', row)
        replayed = journal.call(name+':replay', lambda: replay(attacks, box))
        journal.put(name+'_replay.json', replayed); witnesses.extend(replayed)
        del joint, graph, box, attacks
    result = aggregate(parent, config, priority, rows, properties, experts, deadline)
    return dict(**result, config=config, priority=priority, endpoints=rows, full_model_replays=witnesses,
        original_witness_found=any(r['valid'] for r in witnesses),
        operation_counts=dict(nodes=len(rows), guard_support_calls=sum(r['guard_support'] is not None for r in rows),
            initial_alpha_calls=sum(r['initial_lower'] is not None for r in rows),
            residual_bab_calls=sum(r['outcome'] is not None for r in rows)))


def receive(folder, request, value, domain, deadline):
    config = options(request['margin_branch'])
    W.require(value['config'] == config, 'same admitted components and ablation')
    if value['priority'] is None:
        W.require(value['priority_failure'] == load(folder/'priority_unavailable.json')
            and value['status'] == 'UNKNOWN' and not any(value[k] for k in
            ('closed', 'original_history_closed', 'complete_MoE', 'formal_SAFE', 'endpoints', 'full_model_replays')),
            'missing heuristic cannot close an obligation')
        return
    W.require(value['priority'] == load(folder/'margin_priority.json'), 'saved ranking receipt')
    for row in value['endpoints']:
        suffix = '_bounded.json' if row['outcome'] is None else '_solved.json'
        W.require(row == load(folder/f"branch_{row['cell_index']}_{row['variant']}{suffix}"), 'saved node evidence')
    experts = [request['source_request']['model']['architecture']['experts']]*len(domain.history)
    checked = aggregate(domain, config, value['priority'], value['endpoints'], request['properties'], experts, deadline)
    W.require(all(value[k] == v for k, v in checked.items()), 'complete closed-cell and property aggregation')
