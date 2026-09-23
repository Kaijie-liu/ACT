"""One bounded, zero-solver provenance replay; no robustness-query rerun."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write
from metamoe_csr_execution import supervise
from metamoe_expert_diagnostic import require_clean
from metamoe_protected_smoke_r1 import validate as validate_source

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT/'configs/recent_moe/metamoe_protected_smoke_r1.json'
DIAG = Path('/data1/Kane/MOE/baseline_runs/metamoe_current_assignment_20260923_r2')
CONFIG = ROOT/'configs/recent_moe/metamoe_assignment_replay_r1.json'
OUTPUT = Path('/data1/Kane/MOE/baseline_runs/metamoe_assignment_replay_20260923_r1')


def assess_point(point, lower, upper, original, represented, rows, thresholds):
    inputs = [point, lower, upper, original, represented, rows, thresholds]
    if (not all(np.isfinite(x).all() for x in inputs) or point.shape != lower.shape or
            point.shape != upper.shape or original.shape != represented.shape or
            original.ndim != 1 or rows.shape != (thresholds.size, original.size)):
        return {'status': 'INVALID_REPLAY_ARRAYS', 'full_model_violation': False}
    if np.any(point < lower) or np.any(point > upper):
        return {'status': 'OUTSIDE_REQUEST', 'full_model_violation': False}
    actual, abstract = rows @ original, rows @ represented
    violated = np.flatnonzero(actual < thresholds).tolist()
    hz_violated = np.flatnonzero(abstract < thresholds).tolist()
    return {'status': 'REPLAYED_VIOLATION' if violated else (
                'HZ_SOURCE_POINT_MISMATCH' if hz_violated else 'NO_VIOLATION_AT_POINT'),
        'full_model_violation': bool(violated), 'source_violated_rows': violated,
        'hz_violated_rows': hz_violated, 'original_margins': actual.tolist(),
        'represented_margins': abstract.tolist(),
        'max_abs_output_difference': float(np.max(np.abs(original-represented), initial=0.)),
        'in_box': True, 'not_a_robustness_certificate': True}


def frozen_config():
    parent = json.loads(PARENT.read_text())
    validate_source(parent)
    deps = [PARENT, DIAG/'launch.json', DIAG/'assignment.json', DIAG/'proposal.npz', DIAG/'summary.json',
        Path(__file__), ROOT/'act/back_end/solver/current_assignment.py',
        ROOT/'tests/test_metamoe_assignment_replay.py', ROOT/'docs/metamoe_assignment_replay_protocol_20260923_r1.md']
    return {'protocol': 'metamoe_one_saved_factor_point_replay_r1',
        'files': {str(p): sha256(p) for p in deps}, 'parent_sha256': sha256(PARENT),
        'output_root': str(OUTPUT), 'seconds': 30., 'group_rss_limit_bytes': parent['group_rss_limit_bytes'],
        'expert': 1, 'request': 'mnist_0', 'python': parent['python']['act'],
        'new_native_queries': 0, 'new_assignment_proposals': 0,
        'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}


def validate(cfg):
    if (cfg['protocol'] != 'metamoe_one_saved_factor_point_replay_r1' or cfg['seconds'] != 30. or
            cfg['output_root'] != str(OUTPUT) or cfg['expert'] != 1 or cfg['request'] != 'mnist_0' or
            cfg['new_native_queries'] != 0 or cfg['new_assignment_proposals'] != 0):
        raise ValueError('bounded replay contract')
    for path, h in cfg['files'].items():
        if sha256(path) != h:
            raise ValueError('replay identity drift: '+path)
    parent = json.loads(PARENT.read_text())
    validate_source(parent)
    if (cfg['parent_sha256'] != sha256(PARENT) or cfg['python'] != parent['python']['act'] or
            cfg['group_rss_limit_bytes'] != parent['group_rss_limit_bytes']):
        raise ValueError('parent identity/resource drift')
    return parent


def worker(cfg):
    began = time.monotonic()
    parent = validate(cfg)
    import torch
    from act.back_end.solver import solver_hz as sh
    from act.back_end.solver.current_assignment import (AssignmentScope, AssignmentProposal,
        check_current_assignment, model_fingerprint)
    from act.back_end.solver.isolated_feasibility import save_npz, csr_arrays
    from act.back_end.moe.route_a import (HybridzTF, _analyze_router, set_transfer_function,
        set_solver_mode, verify_once)
    from act.back_end.moe.hz_routing import guarded_input_domain
    from act.back_end.moe.factory import build_act_moe_program
    from act.back_end.moe.class_separated_top1 import classification_rows
    from act.front_end.specs import OutputSpec, OutKind
    from act.config.config import HybridZConfig
    from act.util.device_manager import initialize_device
    from act.util.stats import VerifyResult, VerifyStatus
    from metamoe_paired_model import load_full
    from metamoe_functional_intake import adapted_class_separated
    torch.set_num_threads(2); torch.set_num_interop_threads(2); torch.manual_seed(100)
    initialize_device('cpu', 'float64')
    request = parent['requests'][0]
    with np.load(request['tensor_file'], allow_pickle=False) as z:
        tensors = {k: torch.from_numpy(z[k].copy()) for k in z.files}
    center, lower, upper = (tensors[k] for k in ('center','lower','upper'))
    original = load_full(parent['repo'], parent['checkpoint'], parent['files'][parent['checkpoint']])
    adapted, _ = adapted_class_separated(original, center)
    surrogate = adapted.reduced_components(center)
    rows = classification_rows(original.total_classes, request['label'])
    thresholds = torch.full((len(rows),), parent['margin'], dtype=torch.float64)
    program = build_act_moe_program(surrogate, center=center, lower=lower, upper=upper,
        output_spec=OutputSpec(kind=OutKind.LINEAR_LE, c=-rows, d=-thresholds))
    tf = HybridzTF(config=HybridZConfig(**parent['hybridz']))
    set_transfer_function(tf); set_solver_mode('hybridz')
    captured = {}
    def capture(self, output_hz, out_spec, *, batch_size, n_out, input_hz=None, input_shape=None, **unused):
        if captured:
            raise ValueError('more than one expert capture')
        captured.update(hz=output_hz, input_hz=input_hz, input_shape=input_shape,
                        frame_match=output_hz.frame_id == input_hz.frame_id)
        return [VerifyResult(VerifyStatus.UNKNOWN, metadata={'reason':'zero_solver_capture_only'})]
    with patch.object(sh, 'milp', side_effect=AssertionError('native queries forbidden')), \
         patch.object(sh.HZSolver, 'evaluate_spec', capture):
        router_hz, input_hz = _analyze_router(program.router, tf)
        branch = guarded_input_domain(input_hz, router_hz, cfg['expert'], 1).hz
        tf.set_entry_hz(branch)
        try:
            verify_once(program.experts[cfg['expert']], model_fn=surrogate.experts[cfg['expert']], timelimit=30.)
        finally:
            tf.clear_entry_hz()
    meta = json.loads((DIAG/'assignment.json').read_text())
    model = sh._lower_hz_milp(captured['hz'])
    info = {'new_native_queries': 0, 'new_assignment_proposals': 0,
            'model_sha256': model_fingerprint(model), 'saved_model_sha256': meta['model_sha256'],
            'source_complete': False, 'full_model_violation': False}
    if not captured['frame_match'] or info['model_sha256'] != meta['model_sha256']:
        info['status'] = 'FRAME_OR_MATRIX_MISMATCH_STOP'
    else:
        with np.load(DIAG/'proposal.npz', allow_pickle=False) as z:
            x = z['point'].copy()
        scope = AssignmentScope(**meta['scope'])
        proposal = AssignmentProposal(scope, meta['model_sha256'], x, meta['free_prefix'], meta['relu_rows'], 0.)
        point, checked = check_current_assignment(model, proposal, scope, began+30.)
        info['fresh_full_matrix_check'] = checked
        if point is None:
            info['status'] = 'PROPOSAL_RECHECK_FAILED'
        else:
            recovered = sh.HZSolver._recover_input(model, point, captured['input_hz'], captured['input_shape'], 0)
            if recovered is None:
                info['status'] = 'INPUT_RECOVERY_FAILED'
            else:
                recovered = recovered.reshape_as(center)
                represented = model.value_center+model.value_matrix @ point
                with torch.no_grad():
                    source, scores = original(recovered)
                    padded = surrogate.experts[cfg['expert']](recovered)
                ih = captured['input_hz']
                folder = OUTPUT/'worker'
                save_npz(folder/'replay.npz', point=recovered.numpy(), source=source.numpy().reshape(-1),
                    represented=represented, padded=padded.numpy().reshape(-1), scores=scores.numpy(),
                    center=center.numpy(), lower=lower.numpy(), upper=upper.numpy(),
                    rows=rows.numpy(), thresholds=thresholds.numpy())
                save_npz(folder/'input_map.npz', center=ih.c,
                    **{'Gc_'+k:v for k,v in csr_arrays(ih.Gc).items()},
                    **{'Gb_'+k:v for k,v in csr_arrays(ih.Gb).items()})
                info.update(assess_point(recovered.numpy(),lower.numpy(),upper.numpy(),source.numpy().reshape(-1),
                    represented,rows.numpy(),thresholds.numpy()),
                    original_prediction=int(source.argmax(1)), original_route=int(scores.argmax(1)),
                    max_abs_recovered_vs_center=float((recovered-center).abs().max()),
                    max_abs_source_vs_padded=float((source-padded).abs().max()),
                    input_n_cont=ih.n_cont, input_n_bin=ih.n_bin,
                    replay_sha256=sha256(folder/'replay.npz'), input_map_sha256=sha256(folder/'input_map.npz'))
    info['worker_through_result_seconds'] = time.monotonic()-began
    write(OUTPUT/'worker/result.json', info)


def run(cfg):
    started = time.monotonic()
    validate(cfg); require_clean()
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write(OUTPUT/'launch.json', {'config_sha256': sha256(CONFIG), 'mode': 'ONE_ZERO_SOLVER_PROVENANCE_REPLAY'})
    receipt = supervise([cfg['python'], str(Path(__file__).resolve()), '--worker'], str(ROOT),
                         OUTPUT/'worker', cfg['seconds'], cfg['group_rss_limit_bytes'])
    validate(cfg)
    result = json.loads((OUTPUT/'worker/result.json').read_text()) if (OUTPUT/'worker/result.json').exists() else None
    write(OUTPUT/'terminal.json', {'outer_status':receipt['status'],
        'status':result['status'] if receipt['status']=='COMPLETED' and result else receipt['status'],
        'result':result, 'receipt':receipt, 'config_sha256':sha256(CONFIG),
        'elapsed_through_terminal_seconds':time.monotonic()-started,
        'historical_result_relabelled':False, 'opens_formal_cohort':False})


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    for flag in ('freeze','run','worker'): g.add_argument('--'+flag,action='store_true')
    a=p.parse_args()
    if a.freeze:
        require_clean()
        if OUTPUT.exists(): raise FileExistsError(OUTPUT)
        write(CONFIG,frozen_config())
        print('FROZEN_NOT_EXECUTED',sha256(CONFIG))
    else:
        cfg=json.loads(CONFIG.read_text())
        worker(cfg) if a.worker else run(cfg)
