"""Frozen full-size convolutional R1 conformance gate (not a SAFE experiment)."""
import argparse
import itertools
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import traceback

from act.pipeline.moe.external_compatibility import ROOT, ENV, TOOL, COMMITS, dump, git, sha

CONFIG = ROOT / 'act/pipeline/moe/configs/conv_family_training_r1.json'
ACT_ENV = '/data1/Kane/miniconda3/envs/act-py312/bin/python'


def worker(kind, root):
    start = time.monotonic()
    result = dict(kind=kind, status='RUNNING', phase='imports', formal_SAFE=False)
    path = root / (kind + '.json')
    def stage(name):
        result['phase'] = name
        dump(path, result)
    try:
        import torch
        from act.back_end.moe.factory import load_output_moe_checkpoint, build_act_moe_program
        from act.back_end.moe.static_pair import StaticSelectedSoftmaxPair
        from act.pipeline.moe.staged_verifier import _model_state_identity, _tensor_identity
        torch.set_num_threads(1)
        model, _ = load_output_moe_checkpoint(root / 'init.pt', map_location='cpu')
        model.double().eval()
        # auto_LiRPA creates intermediate convolution identities using the default
        # dtype. Set it after loading the float32-initialized frozen checkpoint.
        torch.set_default_dtype(torch.float64)
        tensors = torch.load(root / 'input.pt', weights_only=True)
        x, lo, hi = (tensors[k] for k in ('center', 'lower', 'upper'))
        result.update(model_state=_model_state_identity(model),
                      inputs={k: _tensor_identity(v) for k, v in tensors.items()},
                      torch=torch.__version__, python=sys.version, device='cpu', dtype='float64')
        points = [x, lo, hi, torch.where(torch.arange(x.numel()).reshape(x.shape)%2 == 0, lo, hi)]
        if kind == 'act':
            import numpy as np
            from act.util.device_manager import initialize_device
            from act.config.config import HybridZConfig
            from act.front_end.specs import OutputSpec, OutKind
            from act.pipeline.moe.experiment1 import _propagate_component
            from act.back_end.solver.solver_hz import SparseHZono, sparse_hz_fast_bounds
            initialize_device('cpu', 'float64')
            program = build_act_moe_program(model, center=x, lower=lo, upper=hi,
                output_spec=OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=[0]))
            cfg = HybridZConfig(max_input_dim=1, guarded_support_enabled=False,
                               expert_property_solver_backend='scipy')
            result['components'] = []
            for name, net, component in [('router', model.router, program.router),
                    *[(f'expert{i}', net, comp) for i,(net, comp) in enumerate(zip(model.experts,program.experts))]]:
                stage(name)
                tick = time.monotonic()
                hz = _propagate_component(component, hybridz_config=cfg).output_hz
                if not isinstance(hz, SparseHZono) or not hz.exact:
                    raise ValueError('full-shape exact sparse representation was not retained')
                bounds = sparse_hz_fast_bounds(hz)
                for point in points:
                    y = net(point).detach().numpy().reshape(-1)
                    if not (np.all(y >= np.asarray(bounds.lb).reshape(-1)-1e-10) and
                            np.all(y <= np.asarray(bounds.ub).reshape(-1)+1e-10)):
                        raise ValueError('concrete probe outside represented bounds')
                result['components'].append(dict(name=name, seconds=time.monotonic()-tick,
                    output_dimensions=int(hz.c.size), exact=True, sparse=True))
            stage('all_six_static_pairs')
            result['pair_errors'] = []
            for pair in itertools.combinations(range(4), 2):
                adapter = StaticSelectedSoftmaxPair(model, pair)
                errors = []
                for p in points:
                    weights = torch.softmax(model.router(p)[:, list(pair)], dim=1)
                    forced = sum(weights[:,j:j+1]*model.experts[i](p) for j,i in enumerate(pair))
                    errors.append(float((adapter(p)-forced).abs().max().detach()))
                if max(errors) > 1e-10:
                    raise ValueError('static pair expression mismatch')
                result['pair_errors'].append(dict(pair=pair, maximum=max(errors)))
        else:
            sys.path[:0] = [str(TOOL/'auto_LiRPA'), str(TOOL/'complete_verifier')]
            from act.util.typing_compat import install_typing_override
            install_typing_override()
            import auto_LiRPA
            from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
            if not Path(auto_LiRPA.__file__).resolve().is_relative_to(TOOL/'auto_LiRPA'):
                raise ValueError('wrong external import')
            stage('static_pair_0_1')
            adapter = StaticSelectedSoftmaxPair(model, (0,1))
            bounded = BoundedModule(adapter, x, device='cpu', bound_opts={'conv_mode':'matrix'})
            result['maximum_probe_error'] = max(float((bounded(p)-adapter(p)).abs().max().detach()) for p in points)
            if result['maximum_probe_error'] > 1e-10:
                raise ValueError('external lowered conformance mismatch')
            stage('plain_CROWN')
            C = torch.zeros(1,9,10,dtype=torch.float64)
            C[:,:,0] = 1
            for j in range(9): C[0,j,j+1] = -1
            lower, upper = bounded.compute_bounds(x=(BoundedTensor(x,
                PerturbationLpNorm(norm=float('inf'), x_L=lo, x_U=hi)),), C=C, method='CROWN')
            if not torch.isfinite(lower).all() or not torch.isfinite(upper).all() or not (lower<=upper).all():
                raise ValueError('invalid numerical bounds')
            result.update(lower=lower.detach().tolist(), upper=upper.detach().tolist(), nodes=len(bounded._modules))
        result.update(status='CONFORMANCE_PASS', phase='complete')
    except Exception as exc:
        result.update(status='FAILED', message=str(exc), traceback=traceback.format_exc())
    result.update(seconds=time.monotonic()-start, peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    dump(path, result)


def run(root):
    if git(ROOT,'branch','--show-current') != 'feat/moe-route-verification' or git(ROOT,'status','--porcelain'):
        raise ValueError('clean feature branch required')
    if not root.resolve().is_relative_to(ROOT/'data/moe/results'):
        raise ValueError('result root outside project results')
    for p, revision in COMMITS.items():
        if git(p,'rev-parse','HEAD') != revision or git(p,'status','--porcelain'):
            raise ValueError('external source drift')
    import torch
    from dataclasses import asdict
    from act.back_end.moe.conv_factory import ConvOutputMoEConfig, build_conv_output_moe
    torch.set_default_dtype(torch.float32)
    cfg = ConvOutputMoEConfig(**json.loads(CONFIG.read_text())['factory'])
    model = build_conv_output_moe(cfg).eval()
    root.mkdir(parents=True, exist_ok=False)
    torch.save(dict(format='act-output-conv-moe-v1', factory_config=asdict(cfg), state_dict=model.state_dict()),root/'init.pt')
    x = torch.full((1,3,32,32), .5, dtype=torch.float64)
    torch.save(dict(center=x, lower=x-2/255, upper=x+2/255),root/'input.pt')
    dump(root/'launch.json',dict(protocol='CONV_FULLSHAPE_CONFORMANCE_R1', execution_head=git(ROOT,'rev-parse','HEAD'),
         config_sha256=sha(CONFIG), checkpoint_sha256=sha(root/'init.pt'), input_sha256=sha(root/'input.pt'),
         source_hashes={p:sha(ROOT/p) for p in ['act/back_end/moe/conv_factory.py','act/back_end/moe/factory.py',
            'act/back_end/moe/static_pair.py','act/back_end/hybridz_tf/tf_cnn.py','act/back_end/hybridz_tf/tf_mlp.py']},
         commits=COMMITS, timeout_per_backend_seconds=300, probe_tolerance=1e-10,
         scope='synthetic full-shape compatibility, not verified accuracy; no positive-bound gate'))
    for kind, python in [('act',ACT_ENV), ('crown',ENV)]:
        with (root/(kind+'.log')).open('w') as log:
            p = subprocess.Popen([python,'-m',__spec__.name,'--worker',kind,'--output',str(root)],
                cwd=ROOT, start_new_session=True, stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'})
            try: p.wait(timeout=300)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid,signal.SIGKILL); p.wait()
                previous=json.loads((root/(kind+'.json')).read_text()) if (root/(kind+'.json')).exists() else {}
                dump(root/(kind+'.json'),{**previous,'status':'TIMEOUT','formal_SAFE':False})
            if not (root/(kind+'.json')).exists():
                dump(root/(kind+'.json'),dict(status='WORKER_FAILED', returncode=p.returncode, formal_SAFE=False))
    results={k:json.loads((root/(k+'.json')).read_text()) for k in ('act','crown')}
    summary=dict(status='PASS' if all(r['status']=='CONFORMANCE_PASS' for r in results.values()) else 'FAILED',
                 results=results, launch=json.loads((root/'launch.json').read_text()),
                 hashes={p.name:sha(p) for p in root.iterdir() if p.is_file()})
    dump(root/'summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',choices=['act','crown'])
    args=parser.parse_args()
    if args.worker: worker(args.worker,args.output.resolve())
    else: run(args.output.resolve())
