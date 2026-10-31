"""Opt-in S2a training routines; selected checkpoints keep old model semantics."""
import torch
from torch.nn import functional as F
from .route_envelope_ibp import (
    model_bounds, margin_rows, certified_loss, route_exclusion_penalty,
)


def build(item):
    if item['family'] == 'dense':
        from .architecture_dense import build as make
    elif item['family'] == 'moe':
        from .multilayer_training import build as make
    else:
        raise ValueError('S2a family')
    return make(item['architecture'], item['seed'])


def schedule(epoch, batch, batches, spec):
    """Zero-based batch; exactly five warmup plus forty ramp epochs."""
    if not (1 <= epoch <= spec['epochs'] and 0 <= batch < batches):
        raise ValueError('schedule position')
    position = epoch - 1 + batch / batches
    fraction = min(1., max(0., (position-spec['warmup_epochs'])/spec['ramp_epochs']))
    return spec['epsilon']*fraction, 1 - (1-spec['kappa_end'])*fraction


def objective(model, x, y, *, epsilon, kappa, route_weight, family):
    if family == 'moe':
        from .multilayer_training import balance_term
        logits, decisions = model.forward_with_routing(x)
        balance = balance_term(decisions, len(model.layers[0].experts), {})
    elif family == 'dense':
        logits = model(x); balance = x.new_zeros(())
    else:
        raise ValueError('S2a family')
    natural = F.cross_entropy(logits, y)
    robust = penalty = x.new_zeros(())
    if kappa < 1:
        rows = margin_rows(y, logits.shape[1], dtype=x.dtype)
        lb, _, envelope = model_bounds(model, (x-epsilon).clamp(0, 1),
                                       (x+epsilon).clamp(0, 1), rows)
        robust = certified_loss(lb)
        if family == 'moe' and route_weight:
            penalty = route_exclusion_penalty(envelope, decisions[0].indices)
    loss = kappa*natural + (1-kappa)*robust + route_weight*penalty + .1*balance
    if not torch.isfinite(loss):
        raise ValueError('nonfinite S2a objective')
    return loss, dict(natural=float(natural.detach()), robust=float(robust.detach()),
                     route_penalty=float(penalty.detach()), balance=float(balance.detach()),
                     correct=int((logits.detach().argmax(1) == y).sum()))


@torch.no_grad()
def evaluate_ibp(model, loader, epsilon):
    model.eval()
    clean = certified = single_route = total = 0
    width_sum = 0.
    for x, y in loader:
        logits = model(x)
        rows = margin_rows(y, logits.shape[1], dtype=x.dtype)
        lb, ub, envelope = model_bounds(model, (x-epsilon).clamp(0, 1),
                                       (x+epsilon).clamp(0, 1), rows)
        correct = logits.argmax(1) == y
        clean += int(correct.sum())
        certified += int((correct & (lb > 0).all(1)).sum())
        if envelope is not None:
            single_route += int(envelope.single_route.sum())
        width_sum += float((ub-lb).mean(1).sum())
        total += len(y)
    if not total:
        raise ValueError('empty evaluation')
    return dict(total=total, clean=clean, certified=certified,
                single_route=single_route if hasattr(model, 'layers') else None,
                mean_margin_width=width_sum/total,
                evidence_grade='IBP_NUMERICAL_POLICY')


def selection_key(metrics):
    """Frozen tie-break: certified count, clean count; caller keeps earliest tie."""
    return metrics['certified'], metrics['clean']


def tune_items(spec):
    return [dict(id=f'{arm["id"]}_seed0_config{i}', arm=arm['id'], family=arm['family'],
                 architecture=arm['architecture'], parameter_count=arm['parameter_count'],
                 seed=0, configuration=i, **config)
            for arm in spec['arms'] for i, config in enumerate(arm['configurations'])]


def selected_followups(spec, tuned):
    selected = {}
    for arm in spec['arms']:
        choices = [r for r in tuned if r['item']['arm'] == arm['id']]
        if len(choices) != 2 or {r['item']['configuration'] for r in choices} != {0, 1}:
            raise ValueError('two complete tuning runs per arm required')
        winner = max(choices, key=lambda r: (*selection_key(r['validation']),
                                            -r['item']['configuration']))
        selected[arm['id']] = winner
    followups = []
    for seed in spec['seeds'][1:]:
        for arm in spec['arms']:
            item = dict(selected[arm['id']]['item'])
            item.update(id=f'{arm["id"]}_seed{seed}', seed=seed)
            followups.append(item)
    return selected, followups
