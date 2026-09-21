"""Narrow, separately versioned functional-ReLU spelling adapter.

No BN folding, source mutation or bound change. The real-operator rewrite is
F.relu(x, inplace=False) -> nn.ReLU(inplace=False)(x), with the same input edge.
Original model remains the only counterexample replay target.
"""
import copy
import torch
from torch import nn, fx
from torch.nn import functional as F


def moduleize_relu(model):
    if any(m.training for m in model.modules()):
        raise ValueError('eval required')
    traced = fx.symbolic_trace(copy.deepcopy(model))
    replacements = []
    for node in traced.graph.nodes:
        if node.op != 'call_function' or node.target is not F.relu:
            continue
        inplace = node.kwargs.get('inplace', node.args[1] if len(node.args) > 1 else False)
        if inplace is not False or len(node.args) not in [1, 2] or set(node.kwargs) - {'inplace'}:
            raise ValueError('unsupported inplace/ambiguous ReLU')
        name = f'_act_explicit_relu_{len(replacements)}'
        if hasattr(traced, name):
            raise ValueError('module identity collision')
        traced.add_submodule(name, nn.ReLU(inplace=False))
        replacements.append({'node': node.name, 'module': name, 'input': node.args[0].name})
        node.op, node.target, node.args, node.kwargs = 'call_module', name, (node.args[0],), {}
    traced.graph.lint()
    traced.recompile()
    return traced.eval(), replacements


def adapted_class_separated(model, probe):
    from act.back_end.moe.class_separated_top1 import ClassSeparatedTop1
    original = ClassSeparatedTop1.from_metamoe(model)
    components, records = [], []
    for part in [original.router, *original.experts]:
        converted, mapping = moduleize_relu(part)
        with torch.no_grad():
            a, b = part(probe), converted(probe)
        if not torch.equal(a, b):
            raise ValueError('ReLU spelling adapter concrete mismatch')
        components.append(converted)
        records.append(mapping)
    adapted = ClassSeparatedTop1(components[0], components[1:], original.class_counts,
        original=model, source_sha256=original.source_sha256)
    return adapted, records


def center_witness(model, x, lower, upper, label, margin):
    if not torch.isfinite(x).all() or (x < lower).any() or (x > upper).any():
        raise ValueError('invalid materialized center')
    with torch.no_grad():
        out = model(x)[0]
    if not torch.isfinite(out).all() or not 0 <= label < out.shape[1]:
        raise ValueError('undefined center/label')
    margins = out[0, label] - torch.cat([out[0, :label], out[0, label+1:]])
    if (margins < margin).any():
        return {'status': 'UNSAFE_REPLAYED', 'evidence_grade': 'FULL_MODEL_REPLAY',
                'reason': 'common_original_model_center', 'witness': x.tolist(),
                'minimum_replayed_margin': float(margins.min())}
    return None
