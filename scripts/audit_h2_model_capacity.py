"""Metadata-only H2 capacity audit; no checkpoint, tensor, dataset or solver read."""
import argparse
import ast
import hashlib
import json
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = 'act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json'
TRAINING = 'act/pipeline/moe/configs/experiment1_multiseed_training_r1.json'
CONV = 'act/pipeline/moe/results/conv_training_review_20260915_r1.json'
FILES = (SELECTION, TRAINING, CONV, 'act/back_end/moe/factory.py',
         'act/back_end/moe/conv_factory.py', 'scoped_source/capture.py',
         'scoped_source/endpoint_source_build.py', 'scoped_source/endpoint_verify.py',
         'scoped_source/endpoint_supervised.py', 'scripts/audit_h2_model_capacity.py')


def integer(node):
    """Evaluate only nonnegative integer literal size expressions, never eval."""
    if isinstance(node, ast.Constant) and type(node.value) is int and node.value >= 0:
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Pow)):
        a, b = integer(node.left), integer(node.right)
        if a > 1024**3 or b > 1024**3 or isinstance(node.op, ast.Pow) and b > 32:
            raise ValueError('unbounded literal')
        return a*b if isinstance(node.op, ast.Mult) else a**b
    raise ValueError('nonliteral size policy')


def limits(reader, receiver):
    functions = [n for n in ast.parse(reader).body if isinstance(n, ast.FunctionDef) and n.name == 'read']
    if len(functions) != 1:
        raise ValueError('reader identity')
    sizes = [integer(n.comparators[0]) for n in ast.walk(functions[0])
             if isinstance(n, ast.Compare) and isinstance(n.left, ast.Attribute)
             and n.left.attr == 'st_size' and len(n.ops) == 1 and isinstance(n.ops[0], ast.Gt)]
    functions = [n for n in ast.parse(receiver).body if isinstance(n, ast.FunctionDef) and n.name == 'receive']
    if len(sizes) != 1 or len(functions) != 1:
        raise ValueError('ambiguous read policy')
    result = {'portable_member': sizes[0]}
    for name in ('source.json', 'proof.json'):
        calls = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'load' and n.args
                 and any(isinstance(v, ast.Constant) and v.value == 'bundle/'+name
                         for v in ast.walk(n.args[0]))]
        if len(calls) != 1:
            raise ValueError('receiver member policy')
        bound = [k.value for k in calls[0].keywords if k.arg == 'limit']
        if len(bound) != 1:
            raise ValueError('receiver size policy')
        result['receiver_'+name] = integer(bound[0])
    return result


def mlp(widths):
    if len(widths) < 2 or any(type(w) is not int or w < 1 for w in widths):
        raise ValueError('positive MLP dimensions')
    weights = sum(a*b for a, b in zip(widths, widths[1:]))
    affine = sum(widths[1:])
    return {'weights': weights, 'affine_outputs': affine, 'relu_outputs': sum(widths[1:-1]),
            'parameters': weights+affine, 'tensors': 2*(len(widths)-1)}


def footprint(parameters):
    if type(parameters) is not int or parameters < 0:
        raise ValueError('parameter count')
    raw = 8*parameters
    return {'binary64_bytes': raw, 'base64_bytes_lower_bound': 4*((raw+2)//3)}


def derive(root=ROOT):
    # Only these source/metadata paths are read; stored checkpoint paths are labels.
    raw = {name: (root/name).read_bytes() for name in FILES}
    selected = json.loads(raw[SELECTION]); training = json.loads(raw[TRAINING])
    cfg = training['training']; models = selected['models']; conv = json.loads(raw[CONV])
    if (cfg['dataset'] != 'CIFAR10' or cfg['gate'] != 'selected_softmax' or cfg['top_k'] != 2
            or set(models) != {'seed0', 'seed1', 'seed2'}):
        raise ValueError('registered family changed')
    # CIFAR-10 dimension/class contract, not inferred from a sampled tensor.
    inputs, classes, experts = 3*32*32, 10, cfg['num_experts']
    router = mlp([inputs, *cfg['router_hidden'], experts])
    expert = mlp([inputs, *cfg['expert_hidden'], classes])
    parameters = router['parameters']+experts*expert['parameters']
    tensors = router['tensors']+experts*expert['tensors']
    size_limits = limits(raw['scoped_source/endpoint_verify.py'].decode(),
                         raw['scoped_source/endpoint_supervised.py'].decode())
    records = []
    for name, model in models.items():
        state = model['model_state']
        if model['dataset'] != 'CIFAR10' or state['parameter_count'] != parameters or state['tensor_count'] != tensors:
            raise ValueError('metadata versus recipe count disagreement')
        size = footprint(state['parameter_count'])
        records.append({'model': name, 'checkpoint_sha256': model['checkpoint_sha256'],
            'registered_model_state': state, **size,
            'source_exceeds_limits': {k: size['base64_bytes_lower_bound'] > v for k, v in size_limits.items()
                                      if k != 'receiver_proof.json'},
            'checkpoint_opened': False, 'native_topology_revalidated': False})
    pairs = comb(experts, 2); duties = pairs*(classes-1)
    pair_affine = router['parameters']+2*expert['parameters']  # weight entries plus output LHS
    return {'schema': 'H2_STATIC_MODEL_CAPACITY_V1', 'status': 'NO_REAL_FREEZE_CAPACITY_NOT_ADMITTED',
        'source_bindings': {k: hashlib.sha256(v).hexdigest() for k, v in raw.items()},
        'recorded_models': records, 'read_limits_bytes': size_limits,
        'recipe_scope': 'seed1/2 registered training; seed0 metadata counts match, topology not reloaded',
        'dense_recipe_scenario': {'input_dimensions': inputs, 'experts': experts, 'classes': classes,
            'router': router, 'expert': expert, 'all_unordered_pairs': pairs,
            'properties_per_pair': classes-1, 'duties': duties, 'maximum_endpoint_LPs': 2*duties,
            'pair_variables': inputs+router['affine_outputs']+router['relu_outputs']+
                              2*(expert['affine_outputs']+expert['relu_outputs']),
            'pair_affine_equality_entries_if_all_weights_nonzero': pair_affine,
            'repeated_affine_entries_in_duty_bases_if_dense': duties*pair_affine,
            'scope': 'affine contribution only under dense weights; not measured nnz, RAM or runtime; ReLU/guard rows additional'},
        'conv_metadata': {'checkpoint_sha256': conv['landed']['checkpoint_sha256'],
            'factory': conv['config']['factory'], 'recorded_test_accuracy': conv['landed']['test']['accuracy'],
            'status': 'UNSUPPORTED_CONV_AND_POOL_OPERATORS', 'checkpoint_opened': False},
        'new_model_loads': 0, 'new_input_selections': 0, 'new_solves': 0,
        'next_gate': 'source/pair factorization with independent semantic expansion controls; no limit bump or real run implied'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--output', type=Path); g.add_argument('--check', type=Path)
    a = p.parse_args(); result = derive()
    if a.output:
        with a.output.open('x') as f:
            json.dump(result, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    elif json.loads(a.check.read_text()) != result:
        raise ValueError('capacity archive differs')
    print(json.dumps({'status': result['status'], 'models': len(result['recorded_models']),
                      'new_model_loads': 0, 'new_solves': 0}))
