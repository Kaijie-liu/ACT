"""Stage5a mechanism calls in the existing 300-second diagnostic supervisor.

No model/torch import on the receiving side. These receipts check identity and
coverage under the numerical policy; they are not independent CROWN proofs.
"""
from fractions import Fraction
import importlib.util
import json
import math
import sys
import time

import gate_decoupling_worker as W
from scoped_proof.io import load, sha, tick

PIECES = {'MC_BASE': 1, 'MC_2': 2, 'MC_4': 4}


def gate_module():
    name = 'moe_aware_gate_receipts'
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, W.ROOT/'act/back_end/moe/gate_decoupling.py')
        module = importlib.util.module_from_spec(spec); sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def cells_for(domain, variant):
    """Exact equal cuts, converted outward; no gap at a float64 tie boundary."""
    n = PIECES[variant]
    lo, hi = domain.margins[0]
    if lo == hi:
        return [[lo, hi]]
    a, b = Fraction(lo), Fraction(hi)
    cuts = [a+(b-a)*i/n for i in range(n+1)]
    def rounded(v, up):
        f = float(v)
        if (up and Fraction(f) < v) or (not up and Fraction(f) > v):
            f = math.nextafter(f, math.inf if up else -math.inf)
        return f
    return [[max(lo, rounded(x, False)), min(hi, rounded(y, True))]
            for x, y in zip(cuts, cuts[1:])]


def conditional_domain(parent, cell):
    G = gate_module()
    identity = dict(parent=G.domain_digest(parent), layer=0, lower=cell[0], upper=cell[1],
                    mode='MARGIN_CONDITIONAL_V2')
    # Match the graph's explicit scoped identity, including JSON separators.
    import hashlib
    sid = hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
    margins = (tuple(cell), *parent.margins[1:])
    return G.GateDomain(sid, parent.history, margins, tuple(G.sigmoid_interval(*m) for m in margins))


def vector(value, n):
    W.require(isinstance(value, list) and len(value) == n
        and all(type(x) in (int, float) and math.isfinite(x) for x in value),
        'finite complete property vector')
    return value


def raw_spec(n):
    return [[[[float(i == j) for j in range(n)]], [0.]] for i in range(n)]


def aggregate_mc(parent, group, rows, properties, deadline):
    """Missing endpoints stay UNKNOWN; all cells and properties are obligations."""
    G = gate_module(); tick(deadline)
    expected = [(i, v) for i in range(len(cells_for(parent, group))) for v in ('V2_LO', 'V2_HI')]
    seen, bounds, closed = set(), [], []
    for row in rows:
        tick(deadline)
        key = (row['cell_index'], row['variant'])
        W.require(key in expected and key not in seen, 'unique registered cell/endpoint')
        seen.add(key)
        cell = cells_for(parent, group)[key[0]]
        domain = conditional_domain(parent, cell)
        W.require(row['cell'] == cell and row['parent_domain_sha256'] == G.domain_digest(parent)
            and row['source_id'] == domain.source_id
            and row['domain_sha256'] == G.domain_digest(domain), 'conditional range/source binding')
        values = vector(row['initial_lower'], properties)
        bounds.append(values)
        done = all(x > 0 for x in values)
        if row.get('outcome') is not None:
            outcome = row['outcome']
            W.require(json.loads(json.dumps(outcome['actual_raw_spec'])) == raw_spec(properties)
                and outcome['status'] in ('verified', 'unknown', 'falsified'), 'all raw property atoms and known status')
            done = done or outcome['status'] == 'verified'
        closed.append(done)
    complete = len(seen) == len(expected)
    lower = [min(v[j] for v in bounds) for j in range(properties)] if complete else None
    accepted = complete and all(closed)
    return dict(status='POSITIVE' if accepted else 'UNKNOWN', closed=accepted,
        original_history_closed=accepted, complete_MoE=False, formal_SAFE=False,
        metric_kind='GUARDED_SPEC_INITIAL_ALPHA_CROWN_LOWER_NOT_PROPERTY_LOWER',
        initial_lower=lower, score=min(lower) if lower is not None else None,
        expected_endpoints=len(expected), recorded_endpoints=len(seen),
        closed_endpoints=sum(closed), complete_initial_bounds=complete)


def aggregate_mb(row, source_id, history, group, properties, guards, deadline):
    tick(deadline)
    W.require(row['source_id'] == source_id and row['history'] == history and not row['cells']
        and row['objective_count'] == properties and row['guard_count'] == guards
        and row['use_guards'] is (group == 'MB_ON'), 'conditional support identity and mode')
    lower, zero = vector(row['lower'], properties), vector(row['zero_lower'], properties)
    W.require(all(a >= b for a, b in zip(lower, zero)), 'retain evaluated zero candidate')
    W.require(row['beta_count'] == properties*guards and len(row['selected_beta']) == properties,
        'all multipliers bound to all properties')
    betas = [v for line in row['selected_beta'] for v in vector(line, guards)]
    W.require(all(v >= 0 for v in betas) and row['beta_nonzero'] == sum(v > 0 for v in betas),
        'finite nonnegative multiplier candidates')
    W.require(group == 'MB_ON' or not any(betas), 'off has no guard multiplier')
    W.require(not row['formal_SAFE'] and not row['complete_MoE'], 'numerical local scope')
    accepted = all(x > 0 for x in lower)
    return dict(status='POSITIVE' if accepted else 'UNKNOWN', closed=accepted,
        original_history_closed=accepted, complete_MoE=False, formal_SAFE=False,
        metric_kind='GUARD_CONDITIONAL_PROPERTY_CROWN_LOWER', lower=lower, score=min(lower))


def prepare_source(obj, group, spec, deadline, journal):
    """Same registered model and materialized input; no other arm's answers."""
    for binding in obj['original_sources']:
        tick(deadline)
        W.require(sha(binding['path']) == binding['sha256'], 'original object binding')
    source_id = W.digest(dict(object=obj, sources=spec['source_files'], backend=spec['backend_identity']))
    journal.put('source.json', dict(object=obj, source_id=source_id, group=group))
    from multilayer_stage1 import setup
    torch = journal.call('native_setup', lambda: setup(deadline))
    environment = journal.call('backend_import', W.backend.probe)
    W.require(W.digest(environment['sources']) == spec['backend_identity'], 'frozen author engine')
    journal.put('environment.json', dict(python=environment['python'], torch=environment['torch'],
        modules=environment['modules'], source_identity=spec['backend_identity']))
    model, tensors = journal.call('load_object', lambda: W.load_object(obj, torch))
    G = gate_module()
    provenance = dict(history_origin=obj['history_source_kind'])
    if obj['history'] is None:
        selection = journal.call('reconstruct_history',
            lambda: W.reconstruct_history(obj, model, tensors, source_id, deadline, journal))
        provenance = dict(history_origin=selection['mode'], selection_sha256=sha(journal.folder/'history_selection.json'))
        if selection['history'] is None:
            return None, None, None, dict(source_id=source_id, **provenance)
        history = G.canonical_history(selection['history'], obj['experts'])
        records = [r for r in selection['prefix_records']
                   if tuple(map(tuple, r['prefix'])) in {history[:d] for d in range(len(history))}]
        origin = 'REBUILT_ORIGINAL_PREFIX_CROWN_CHARGED_NEW_HISTORY'
    else:
        history = G.canonical_history(obj['history'], obj['experts'])
        # Stage5a recomputes ranges independently even if older cached bounds exist.
        local = dict(obj, gate_range_source=None)
        records, origin = journal.call('gate_ranges', lambda: W.original_ranges(local, model, tensors, deadline, journal))
    domain = G.domain_from_prefix_bounds(history, obj['experts'], records,
        source_id=source_id, expected_source_id=source_id)
    journal.put('domain.json', dict(source_id=source_id, history=domain.history,
        margins=domain.margins, intervals=domain.intervals, domain_sha256=G.domain_digest(domain),
        origin=origin, evidence_grade='CROWN_SIGMOID_NUMERICAL_POLICY_NOT_EXACT_ENCLOSURE'))
    return model, tensors, domain, dict(source_id=source_id, history=[list(p) for p in history],
        range_origin=origin, property_count=len(tensors['rows']), domain_sha256=G.domain_digest(domain), **provenance)


def initial_bounds(graph, box, deadline):
    """Complete all-property score, never a pruned/ambiguous BaB global bound."""
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
    tick(deadline)
    bounded = BoundedModule(graph, box['center'], device='cpu')
    # Fixed 20 iterations, same on/off; no result-dependent alpha tuning.
    bounded.set_bound_opts({'optimize_bound_args': {'iteration': 20}})
    x = BoundedTensor(box['center'], PerturbationLpNorm(norm=float('inf'), x_L=box['lower'], x_U=box['upper']))
    lower, _ = bounded.compute_bounds(x=(x,), method='alpha-CROWN', bound_upper=False)
    tick(deadline)
    return lower[0].detach().cpu().tolist()


def sliced_calls(model, tensors, domain, group, deadline, journal, *, bound=initial_bounds, verify=None):
    """All initial cell/endpoint vectors first, then fixed-order GenBaB residuals.

    Rebuilding a residual graph and the engine's repeated setup are charged;
    neither saved initial rows nor timeout prefixes constitute online success.
    """
    from act.back_end.moe.moe_aware_bab import MarginCell, sliced_endpoint, GuardMaximum
    from act.back_end.moe import gate_decoupling as G
    from act.back_end.moe.class_separated_top1 import validate_replay
    from b2_observed_worker import verify_graph
    # The standard-library receiver loads the DTO module without importing
    # ACT/Torch. Re-materialize its immutable fields in the native module's
    # GateDomain class; retain every field and repeat the ordinary validation.
    domain = G.GateDomain(domain.source_id, domain.history, domain.margins, domain.intervals)
    G.validate_domain(domain, domain.history, expected_source_id=domain.source_id)
    verify = verify_graph if verify is None else verify
    pixels, properties = tensors['center'][0].numel(), len(tensors['rows'])
    rows, witnesses = [], []
    def prepare(cell, variant):
        joint, scoped = sliced_endpoint(model, domain, tensors['rows'], tensors['thresholds'],
            MarginCell(0, *cell), variant=variant, input_features=pixels, expected_source_id=domain.source_id)
        box = G.extend_box(tensors['center'], tensors['lower'], tensors['upper'], scoped,
                          variant=variant, expected_source_id=scoped.source_id)
        return GuardMaximum(joint).eval(), box, scoped
    for index, cell in enumerate(cells_for(domain, group)):
        for variant in ('V2_LO', 'V2_HI'):
            tick(deadline); name = f'cell_{index}_{variant}'
            graph, box, scoped = journal.call(name+':construct', lambda: prepare(cell, variant))
            values = journal.call(name+':initial', lambda: bound(graph, box, deadline))
            vector(values, properties)
            row = dict(cell_index=index, cell=cell, variant=variant, source_id=scoped.source_id,
                parent_domain_sha256=G.domain_digest(domain), domain_sha256=G.domain_digest(scoped),
                initial_lower=values, outcome=None)
            journal.put(name+'_initial.json', row); rows.append(row)
            del graph, box
    initial = aggregate_mc(domain, group, rows, properties, deadline)
    journal.put('initial_aggregate.json', initial)
    for row in rows:
        tick(deadline)
        if all(x > 0 for x in row['initial_lower']):
            continue
        name = f"cell_{row['cell_index']}_{row['variant']}"
        graph, box, _ = journal.call(name+':reconstruct', lambda: prepare(row['cell'], row['variant']))
        outcome, attacks = journal.call(name+':solve',
            lambda: verify(graph, box['lower'], box['upper'], properties, deadline))
        row['outcome'] = outcome
        journal.put(name+'_solved.json', row)
        replayed = []
        if attacks is not None:
            width = box['lower'][0].numel()
            W.require(attacks.numel() % width == 0, 'extended counterexample dimensions')
            for point in attacks.detach().cpu().double().reshape(-1, width):
                tick(deadline)
                candidate = point[:pixels].reshape_as(tensors['center'])
                valid = validate_replay(model, candidate, tensors['lower'], tensors['upper'],
                                        tensors['rows'], tensors['thresholds'])
                replayed.append(dict(valid=valid, original_input=candidate.flatten().tolist() if valid else None,
                    scope='ORIGINAL_DYNAMIC_MODEL_REPLAY', dropped_independent_gates=width-pixels))
        journal.put(name+'_replay.json', replayed); witnesses.extend(replayed)
        del graph, box, attacks
    aggregate = aggregate_mc(domain, group, rows, properties, deadline)
    # A replay may execute another legal history. Keep it, but do not confuse
    # selected-history closure with complete-request safety or contradiction.
    return dict(**aggregate, endpoints=rows, full_model_replays=witnesses,
                operation_counts=dict(initial_support_calls=len(rows),
                    residual_bab_calls=sum(r['outcome'] is not None for r in rows),
                    construct_calls=len(rows)+sum(r['outcome'] is not None for r in rows)),
                original_witness_found=any(r['valid'] for r in witnesses))


def diagnose(obj, group, spec, deadline, journal):
    model, tensors, domain, identity = prepare_source(obj, group, spec, deadline, journal)
    if domain is None:
        return dict(**identity, group=group, status='NO_UNEXCLUDED_HISTORY', closed=False,
            original_history_closed=False, complete_MoE=False, formal_SAFE=False, endpoints=[])
    if group.startswith('MB_'):
        from act.back_end.moe.moe_aware_bab import history_objectives, conditional_bounds
        graph = journal.call('guard_graph', lambda: history_objectives(model, domain.history,
            tensors['rows'], tensors['thresholds'], source_id=domain.source_id))
        result = journal.call('conditional_bounds', lambda: conditional_bounds(graph, tensors['center'],
            tensors['lower'], tensors['upper'], deadline=deadline, use_guards=group == 'MB_ON', iterations=20))
        journal.put('conditional_bounds.json', result)
        aggregate = aggregate_mb(result, domain.source_id, identity['history'], group,
                                 len(tensors['rows']), graph.guard_count, deadline)
        return dict(**identity, **aggregate, group=group, bounds=result)
    result = sliced_calls(model, tensors, domain, group, deadline, journal)
    return dict(**identity, **result, group=group)


def check_received(folder, value, spec, call, deadline):
    """Identity/coverage only, in the original request deadline and cost ledger."""
    tick(deadline)
    obj = W.source_object(spec, call)
    sid = W.digest(dict(object=obj, sources=spec['source_files'], backend=spec['backend_identity']))
    group = call['variant']
    W.require(value['source_id'] == sid and value['group'] == group, 'object and variant binding')
    history = obj['history']
    if history is None:
        from gate_decoupling_followup import check_selection, MODE
        selection = load(folder/'history_selection.json')
        history = check_selection(selection, obj['experts'], sid, deadline)
        W.require(value['history_origin'] == MODE and value['selection_sha256'] == sha(folder/'history_selection.json'),
                  'explicitly reconstructed history, not recovered old query')
        if history is None:
            W.require(value['status'] == 'NO_UNEXCLUDED_HISTORY' and not any(value[k] for k in
                ('closed', 'original_history_closed', 'formal_SAFE', 'complete_MoE', 'endpoints')), 'no unexcluded history')
            return
    G = gate_module(); saved = load(folder/'domain.json')
    domain = G.GateDomain(saved['source_id'], tuple(map(tuple, saved['history'])),
        tuple(map(tuple, saved['margins'])), tuple(map(tuple, saved['intervals'])))
    G.validate_domain(domain, tuple(map(tuple, history)), expected_source_id=sid)
    properties = obj['original_request']['model']['architecture']['classes']-1
    W.require(value['history'] == history and value['property_count'] == properties
        and value['domain_sha256'] == saved['domain_sha256'] == G.domain_digest(domain), 'all properties and original domain')
    if group.startswith('MB_'):
        result = load(folder/'conditional_bounds.json')
        W.require(value['bounds'] == result, 'saved multiplier/result identity')
        aggregate = aggregate_mb(result, sid, history, group, properties,
                                  sum(2*(e-2) for e in obj['experts']), deadline)
    else:
        for row in value['endpoints']:
            name = f"cell_{row['cell_index']}_{row['variant']}"
            suffix = '_solved.json' if row['outcome'] is not None else '_initial.json'
            W.require(load(folder/(name+suffix)) == row, 'saved cell/result identity')
        aggregate = aggregate_mc(domain, group, value['endpoints'], properties, deadline)
    W.require(all(value[k] == v for k, v in aggregate.items()), 'recomputed complete local aggregation')
    tick(deadline)
