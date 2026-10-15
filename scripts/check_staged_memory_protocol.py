"""Read-only design checks; does not set limits, spawn workers or load models."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD = 'docs/hz_staged_memory_protocol_20261015.json'


def require(condition, label):
    if not condition:
        raise ValueError(label)


def unique_pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate key')
        result[key] = value
    return result


def check():
    value = json.loads((ROOT / RECORD).read_text(), object_pairs_hook=unique_pairs)
    require(value['schema'] == 'STAGED_MEMORY_DESIGN_V1', 'design schema')
    require(value['status'] == 'DESIGN_FROZEN_PENDING_LIMITS_AND_IMPLEMENTATION_APPROVAL', 'design scope')
    for key in ('default_enabled', 'implementation_authorized', 'runtime_controls_authorized',
                'real_native_gpu_authorized', 'full_capacity_admitted'):
        require(value[key] is False, 'nonadmission: ' + key)
    require(value['as_bytes'] is None and value['new_existing_quota'] is None, 'no new limit')
    for name, expected in value['anchors'].items():
        path = ROOT / name
        require(not path.is_symlink() and path.is_file(), 'regular anchor')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'anchor drift: ' + name)
    tree = ast.parse((ROOT / 'scoped_source/hz_request_store.py').read_text())
    policy = next(n.value.args[0] for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'POLICY' for t in n.targets))
    # Closed integer arithmetic only, not eval/import of the implementation.
    def integer(node):
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Pow)):
            left, right = integer(node.left), integer(node.right)
            return left * right if isinstance(node.op, ast.Mult) else left ** right
        raise ValueError('noninteger policy expression')
    require({k.arg: integer(k.value) for k in policy.keywords} == value['unchanged_policy'], 'policy drift')
    require(value['stages'] == ['PRODUCTION', 'PORTABLE', 'RECEIVE'], 'fixed stages')
    require(value['independent_mathematical_passes'] == 2 and value['success_seconds'] == 300, 'checks/time')
    require((value['candidate_width'], value['scoped_check_width'], value['dense_check_width']) == (8, 2, 8), 'widths')
    require(value['two_pass_source_legality_work_floor'] == 2 * (6684672 + 273408 + 3288)
            > value['unchanged_policy']['visits'], 'full source still refuses')
    a = value['address_space_contract']
    require(a['roles'] == ['watchdog', 'controller', 'actor'] and a['max_concurrent_actors'] == 1
            and a['conditional_post_arm_virtual_envelope_multiple'] == 3, 'process topology')
    require(a['same_soft_hard_for_all_roles'] is True and a['controller_group_includes_actor'] is True
            and a['bootstrap_before_limit_trusted'] is True, 'limit/lifetime boundary')
    require(a['rss_bound_proved'] is False and a['full_call_memory_bound'] is None
            and a['success_capacity_bound'] is None, 'no capacity promotion')
    require(value['future_control_scope']['execution_started'] is False
            and value['future_control_scope']['allocation_control_approval'] is None, 'no control execution')
    require(len(value['tail_files']) == len(set(value['tail_files'])) == 6
            and value['tail_file_bytes'] == 65536
            and value['tail_reserve_bytes_within_existing_limit'] == 6 * 65536
            < value['unchanged_policy']['total_bytes'], 'suffix reserve not extra capacity')
    require((ROOT / value['protocol']).is_file(), 'protocol exists')
    return dict(status='PASS_STATIC_DESIGN', anchors=len(value['anchors']),
                policy_entries=len(value['unchanged_policy']),
                runtime_changes=0, worker_runs=0, model_runs=0, gpu_runs=0,
                limits_changed=False, controls_executed=False)


if __name__ == '__main__':
    print(json.dumps(check(), sort_keys=True))
