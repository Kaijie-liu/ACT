"""Exact contract arithmetic, not model certification or execution admission.

No model, dataset, solver or device imports. A norm conversion does not establish
function equivalence, a valid statistical certificate, or source containment.
Local source hashing is optional and never changes the pinned identities.
"""
import argparse
from fractions import Fraction as Q
import hashlib
import json
from math import comb
from pathlib import Path


MOE = Path('/data1/Kane/MOE')
SOURCE_PINS = {
    'baselines/recent_moe_20260921/dual_rs/code/core.py': '4056413e400e5be4443d456cac19986729ce936a1f1c64afd84a4c07126cf8cc',
    'baselines/recent_moe_20260921/dual_rs/code/DRM_sigma_est.py': 'b61c8d4f5012cfa2d5f572b6ffd449e0c890b13e5d2377d745aac7d79da8147a',
    'baselines/recent_moe_20260921/dual_rs/code/DRM_classifier.py': '508e61373ee8109e50ef351ed68089639fdddb6d838a31a3357b1a70d55b98f3',
    'baselines/recent_moe_20260921/dual_rs/code/certify_sigma_est.py': '83741fe4e23c1d43f93882d7066d1d4cac22e73e51dddbf7ef6fd9d6d72e1287',
    'baselines/recent_moe_20260921/dual_rs/code/improved_diffusion/gaussian_diffusion.py': 'a9e59441dbd065e6135f1160327f34d292dd259e231e7fa7cd0f67f49a8cc29d',
    'baselines/recent_moe_20260921/metamoe/src/Vision_Transformer_Pytorch/vision_transformer_moe.py': '560ef6cee9000faf0f1d4579d4e91f7a0439d3cdc3a8722e37b69571f3836c34',
    'baselines/robust_experts_compat_20260921_r3/src/models/nn/moe/routing.py': '2a5158553bf307d69c5e2f8f9d143a85f00f4a2f590e73f84c62a41d58d4720e',
    'baselines/robust_experts_compat_20260921_r3/src/models/nn/moe/gate/topk.py': '9e3e480d7e5ccfec02c19ce8078596122b1fd6b7818b157674bc8c5a144da70c',
    'baselines/robust_experts_compat_20260921_r3/src/models/nn/moe/layer.py': '2988b8ee8a8a5fa0681fcd86c3ac7e2415ee2e4ef0f18345561194004dcad114',
    'baselines/robust_experts_compat_20260921_r3/src/models/nn/moe/resnet_conv_moe.py': 'ca1c17f785495345298a977deea01b9d4383a79c58f7f80dd3853e79c2d39fcc',
    'baselines/recent_moe_20260921/rome/models/lora_moe_mlp.py': 'e5f8921f7cc572949ecfd49d997b43ac8f626186acb86ed1383cd0a717877f66',
    'baselines/recent_moe_20260921/rome/models/lora_moe_linear.py': 'e7a4e6cc14cdbc7e622567674a993ce8504c9a2cab0c158f9502392e5a640598',
    'baselines/recent_moe_20260921/rome/models/vit.py': '45bf6523d8cfff0dfc066b7d472c680e71e83593662cf8934da5cec710c1f748',
    'baseline_runs/recent_moe_20260921/papers/j_tlat.txt': 'bba4cbb7696e5f627c82c59594abfef740cd75cd76f5c3a0f60cff85215748cf',
    'baseline_runs/recent_moe_20260921/papers/feature_noise.txt': '542af38592bc2f74c8cdfe4bc2ac90185e4b39af6c7fae76b7d729dbdeb06f44',
}


def rational(value):
    if type(value) not in (int, Q):
        raise ValueError('explicit rational required; no implicit float conversion')
    return Q(value)


def box_radius_squared(center, lower, upper, scales=None):
    """Maximum ||D(x-center)||_2^2 on a nonempty rational coordinate box.

    D is a specified positive diagonal coordinate change, NOT inferred from a
    dataset name. This does not validate the source of any supplied endpoint.
    """
    if not len(center) or len(center) != len(lower) or len(lower) != len(upper):
        raise ValueError('nonempty matching dimensions required')
    if scales is None:
        scales = [1] * len(center)
    if len(scales) != len(center):
        raise ValueError('scale dimensions')
    result = Q(0)
    for c, lo, hi, s in zip(center, lower, upper, scales):
        c, lo, hi, s = map(rational, (c, lo, hi, s))
        if lo > hi or s <= 0:
            raise ValueError('ordered endpoints and positive coordinate scales required')
        result += (s * max(abs(lo-c), abs(hi-c))) ** 2
    return result


def contained_in_l2(center, lower, upper, radius, scales=None, *, open_ball=False):
    radius = rational(radius)
    if radius < 0 or type(open_ball) is not bool:
        raise ValueError('nonnegative radius and explicit ball boundary')
    squared = box_radius_squared(center, lower, upper, scales)
    return squared < radius**2 if open_ball else squared <= radius**2


def union_failure(alphas):
    """Conditional on valid individual failure bounds, without independence."""
    values = [rational(a) for a in alphas]
    if not values or any(a < 0 or a > 1 for a in values):
        raise ValueError('nonempty probability bounds in [0,1] required')
    return min(Q(1), sum(values, Q(0)))


def selected_mass_bounds(experts, selected, delta):
    """Conditional real softmax/top-k facts; not a floating operator proof."""
    if type(experts) is not int or type(selected) is not int or not 1 <= selected <= experts:
        raise ValueError('1 <= k <= E integer dimensions required')
    delta = rational(delta)
    if delta <= 0:
        raise ValueError('positive author denominator offset required')
    mass = Q(selected, experts)
    return mass, mass/(mass+delta), Q(1)/(1+delta)


def history_count(sites):
    """Unpruned syntactic unordered top-k catalog, not reachable histories."""
    result = 1
    for experts, selected in sites:
        if type(experts) is not int or type(selected) is not int or not 1 <= selected <= experts:
            raise ValueError('invalid site dimensions')
        result *= comb(experts, selected)
    return result


def verify_local_sources(root=MOE):
    for name, expected in SOURCE_PINS.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
            raise ValueError('source identity mismatch: '+name)


def report():
    # Exact dyadic value of the Python literal is explicit, not inferred from
    # its printed decimal. Native tensor conversion may round again.
    delta = Q.from_float(1e-5)
    mass, low, high = selected_mass_bounds(4, 2, delta)
    radius_squared = box_radius_squared([0]*3072, [-Q(2,255)]*3072, [Q(2,255)]*3072)
    return {
        'schema': 'BASELINE_SEMANTIC_ARITHMETIC_V1',
        'reviewed_head': '38478a1379e44775616d7ce6334e5111ef7fb4c2',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_pins_relative_to_MOE': SOURCE_PINS,
        'dual_rs': {
            'conditional_nominal_per_input_failure': str(union_failure([Q(1,2000)]*2)),
            'conditional_nominal_two_input_failure': str(union_failure([Q(1,2000)]*4)),
            'native_probability_tail_arithmetic_checked': False,
            'unclipped_raw_3072_coordinate_box_2_over_255_required_l2_radius_squared': str(radius_squared),
            'pilot_radius_reclassified': False,
            'smoothed_equals_unsmoothed_established': False,
        },
        'robust_experts': {
            'conditional_real_selected_mass_lower': str(mass),
            'python_delta_exact_dyadic': str(delta),
            'conditional_weight_sum_lower': str(low),
            'conditional_weight_sum_upper': str(high),
            'unpruned_trained_k2_histories': history_count([(4,2)]*5),
            'unpruned_untrained_k1_histories': history_count([(4,1)]*5),
            'current_default_history_cap': 4096,
            'all_histories_reachable_established': False,
            'native_floating_definedness_established': False,
        },
        'scope': 'exact contract arithmetic and dated source identities, not a semantic proof checker',
        'model_queries': 0, 'solver_calls': 0, 'gpu_calls': 0,
        'new_execution_admitted': False,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report', type=Path)
    parser.add_argument('--verify-local-sources', action='store_true')
    args = parser.parse_args()
    if args.verify_local_sources:
        verify_local_sources()
    result = report()
    if args.check_report:
        if json.loads(args.check_report.read_bytes()) != result:
            raise ValueError('semantic arithmetic report differs')
        print('PASS: contract arithmetic/identities only; no new model result or admission')
    else:
        print(json.dumps(result, indent=2, sort_keys=True))
