"""Read source/registered metadata only; no model, propagation or optimizer imports."""
import argparse
import ast
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING = 'act/pipeline/moe/configs/experiment1_multiseed_training_r1.json'
MODELS = 'act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json'
CONV = 'act/pipeline/moe/results/conv_training_review_20260915_r1.json'
FILES = (TRAINING, MODELS, CONV, 'act/back_end/moe/factory.py',
         'act/back_end/moe/conv_factory.py', 'scoped_source/graph.py',
         'scoped_source/hz_source_build.py', 'scoped_source/hz_source_check.py',
         'source_enclosure/produce.py', 'source_enclosure/check.py',
         'act/back_end/moe/hz_endpoints.py', 'act/back_end/moe/check_hz_endpoints.py',
         'act/back_end/moe/batched_support.py', 'act/back_end/moe/check_batched_support.py',
         'scripts/hz_source_supervised.py', 'scoped_source/hz_portable_verify.py',
         'router_source/checker.py', 'scoped_source/capture.py',
         'scripts/audit_hz_real_intake.py', 'scripts/test_hz_real_intake.py')


def comparison(raw, function, expected):
    """Assert a named literal contract without importing/exec'ing checked code."""
    tree = ast.parse(raw)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function]
    if len(functions) != 1:
        raise ValueError('function contract missing')
    key = ast.dump(ast.parse(expected, mode='eval').body)
    found = [n for n in ast.walk(functions[0]) if isinstance(n, ast.Compare) and ast.dump(n) == key]
    if len(found) != 1:
        raise ValueError('changed or ambiguous literal capacity: '+expected)
    return {'function': function, 'line': found[0].lineno, 'expression': ast.unparse(found[0])}


def mlp(widths):
    return {'parameters': sum((a+1)*b for a,b in zip(widths,widths[1:])),
            'tensors': 2*(len(widths)-1), 'affine_outputs': sum(widths[1:]),
            'relu_outputs': sum(widths[1:-1]), 'layers': 2*len(widths)-2}


def roundtrips(value):
    try:
        candidate = float(value)
        return math.isfinite(candidate) and F(candidate) == value
    except OverflowError:
        return False


def arithmetic_witnesses():
    # All original operands are exact binary64. No network/fixture is executed.
    w = F(1)+F(1,2**52)
    generators = (w, w/F(2**96))
    residuals = [abs(w*g-F(float(w*g))) for g in generators]
    needed = sum(residuals,F(0))
    assert residuals == [F(1,2**104), F(1,2**200)]
    assert all(roundtrips(q) for q in (w,*generators,*residuals)) and not roundtrips(needed)
    radius = F(1)+F(1,2**54)
    relu_lower, relu_upper = F(1)-radius, F(1)+radius
    guard_rhs = F(1)-F(1,2**54)
    equality_rhs = F(1,2**56)-F(1,2)
    half_subnormal = F(1,2**1075)
    failures = {'affine_error_sum':needed, 'relu_upper':relu_upper,
                'guard_difference':guard_rhs, 'relu_equality_rhs':equality_rhs,
                'half_minimum_subnormal':half_subnormal}
    assert all(not roundtrips(v) for v in failures.values())
    return {'scope':'algebraic representability witnesses, not observed trained weights or executed HZ',
            'affine_weight':str(w), 'affine_generators':list(map(str,generators)),
            'individual_affine_residuals':list(map(str,residuals)),
            'relu_center':'1', 'relu_generators':['1',str(F(1,2**54))],
            'relu_lower':str(relu_lower),
            'relu_rhs_example':{'center':str(F(1,2**56)),'generator':'1/2','supplied_range':['-1','1']},
            'non_roundtrip_values':{k:str(v) for k,v in failures.items()},
            'exact_float_would_accept':{k:roundtrips(v) for k,v in failures.items()},
            'real_model_failure_frequency':None}


def derive(root=ROOT):
    raw = {n:(root/n).read_bytes() for n in FILES}
    cfg=json.loads(raw[TRAINING])['training']; selected=json.loads(raw[MODELS])['models']
    conv=json.loads(raw[CONV]); cf=conv['config']['factory']
    if (cfg['dataset'],cfg['gate'],cfg['top_k']) != ('CIFAR10','selected_softmax',2):
        raise ValueError('registered task changed')
    d,c,e=3072,10,cfg['num_experts']
    rw=[d,*cfg['router_hidden'],e]; ew=[d,*cfg['expert_hidden'],c]
    r,x=mlp(rw),mlp(ew)
    parameters=r['parameters']+e*x['parameters']; tensors=r['tensors']+e*x['tensors']
    if set(selected)!={'seed0','seed1','seed2'}:
        raise ValueError('registered model inventory')
    models={}
    for name,m in selected.items():
        state=m['model_state']
        if m['dataset']!='CIFAR10' or (state['parameter_count'],state['tensor_count'])!=(parameters,tensors):
            raise ValueError('recipe / registered state mismatch')
        models[name]={'checkpoint_sha256':m['checkpoint_sha256'],'state':state,
                      'checkpoint_loaded':False,'native_topology_revalidated':False}
    constraints=[
        ('scoped_source/hz_source_check.py','properties','2 <= e <= 4'),
        ('scoped_source/hz_source_check.py','properties','2 <= c <= 5'),
        ('scoped_source/hz_source_check.py','checked_state','1 <= len(c)+len(b) <= 128'),
        ('scoped_source/hz_source_check.py','checked_state',"len(h['c']) > 128"),
        ('scoped_source/hz_source_check.py','checked_state',"len(h['b'])+len(h['ub']) > 256"),
        ('act/back_end/moe/check_hz_endpoints.py','validate_request','1 <= len(props) <= 4'),
        ('act/back_end/moe/check_batched_support.py','validated_records','1 <= len(queries) <= 8')]
    contracts=[{'file':f,**comparison(raw[f].decode(),fn,expr)} for f,fn,expr in constraints]
    # Current portable scalar limits; not the earlier direct-node 64 MiB reader.
    p=ast.parse(raw['scoped_source/hz_portable_verify.py'])
    size=[n for n in p.body if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='(MEMBER_LIMIT, TOTAL_LIMIT)']
    if len(size)!=1 or ast.dump(size[0].value)!=ast.dump(ast.parse('(4*2**20,32*2**20)',mode='eval').body):
        raise ValueError('portable limit contract changed')
    pairs=math.comb(e,2); affine=r['affine_outputs']+2*x['affine_outputs']
    unstable=r['relu_outputs']+2*x['relu_outputs']; guards=2*(e-2)
    base64_min=4*((8*parameters+2)//3)
    return {'schema':'HZ_REAL_INTAKE_METADATA_V1','status':'REAL_INTAKE_NOT_ADMITTED',
        'source_bindings':{n:hashlib.sha256(v).hexdigest() for n,v in raw.items()},
        'models':models,'literal_contracts':contracts,
        'dense_recipe':{'input_dimensions':d,'experts':e,'classes':c,'router_widths':rw,'expert_widths':ew,
            'parameters':parameters,'parameter_tensors':tensors,'pairs':pairs,'duties':pairs*(c-1),
            'endpoint_targets_if_gate_nondegenerate':2*pairs*(c-1),
            'queries_per_pair_degenerate_gate':c-1,'queries_per_pair_nondegenerate_gate':2*(c-1),
            'input_factor_lower_bound':d,'router_layer_records':r['layers'],
            'expert_propagations_in_current_all_pair_loop':2*pairs,
            'all_layer_trace_records':r['layers']+2*pairs*x['layers'],
            'new_HZ_pair_scenario':{'all_unstable_relu_nodes':unstable,
                'max_nonzero_affine_error_factors':affine,
                'continuous_factors':d+affine+2*unstable,'binary_factors':unstable,
                'total_factors':d+affine+3*unstable,'equalities':unstable,
                'inequalities':2*unstable+guards,'total_constraint_rows':3*unstable+guards,
                'scope':'all-ReLU-unstable, every-affine-row-nonzero-error scenario; not measured sizes'},
            'registered_recipe_scope':'seed1/2 recipe; seed0 state counts only; no loaded topology',
            'definite_rejections':['experts','classes','input factors and output rows','property roster','query roster','portable source bytes']},
        'source_bytes':{'parameter_base64_lower_bound':base64_min,'portable_member_limit':4*2**20,
            'portable_bundle_limit':32*2**20,'source_lower_bound_exceeds_both':base64_min>32*2**20,
            'actual_source_or_proof_bytes':None},
        'conv':{'factory':cf,'checkpoint_sha256':conv['landed']['checkpoint_sha256'],
            'recorded_test_accuracy':conv['landed']['test']['accuracy'],
            'unsupported_operators':['Conv2d','AvgPool2d'],'input_dimensions':math.prod(cf['input_shape']),
            'all_pairs':math.comb(cf['num_experts'],2),
            'duties':math.comb(cf['num_experts'],2)*(cf['num_classes']-1),
            'checkpoint_loaded':False},
        'binary64_representation':arithmetic_witnesses(),
        'new_model_loads':0,'new_input_selections':0,'new_propagations':0,'new_solves':0,'cuda_calls':0,
        'source_or_output_certificate':False,'all_six_goal_gates_remain_open':True,
        'next_decision':'separate exact-reference to binary64 outer-enclosure controls before capacity changes; no cap bump or real execution'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); a=p.add_mutually_exclusive_group(required=True)
    a.add_argument('--output',type=Path); a.add_argument('--check',type=Path); args=p.parse_args()
    report=derive()
    if args.output:
        if not args.output.resolve().is_relative_to(ROOT/'docs'):
            raise ValueError('new repository diagnostic report required')
        with args.output.open('x') as stream:
            json.dump(report,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n')
    elif json.loads(args.check.read_text())!=report:
        raise ValueError('intake metadata report differs')
    print({'status':report['status'],'model_loads':0,'solves':0})
