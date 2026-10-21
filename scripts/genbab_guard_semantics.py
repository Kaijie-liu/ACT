"""Two authorized synthetic guard controls, one cumulative 300-second batch.

These distinguish specification/representation effects, not generic limitations
of GenBaB. Old smoke FAIL artifacts are never changed. No trained model or GPU.
"""
import time
ENTRY = time.monotonic()
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
import genbab_cpu_ready as old
from scoped_proof.io import save, load, sha
from scoped_proof.owned_session import execute

MODES = ('input_split_and', 'nonlinear_scalar_max')
PER_MODE_SECONDS = 140


def configuration(mode):
    old.require(mode in MODES, 'fixed two controls')
    cfg = dict(old.CONFIG)
    if mode == 'input_split_and':
        cfg.update({'bab.branching.input_split.enable': True,
                    'bab.branching.method': 'naive',
                    'attack.pgd_order': 'before'})
    return cfg


def values(x):
    f = .5-x*math.tanh(x)
    return f, -.5-x, -.5+x


def worker(run, mode, deadline):
    old.limits(deadline)
    with (run/'native.log').open('xb', buffering=0) as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        save(run/'imports.json', old.probe())
        import torch
        from api import ABCrownSolver, VerificationSpec, default_config
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float64); torch.manual_seed(0)

        class Graph(torch.nn.Module):
            def forward(self, x):
                ea, eb = .5-x, .5+x
                gate = torch.sigmoid(2*x)
                f = eb+gate*(ea-eb)
                if mode == 'input_split_and':
                    return torch.cat((f, x, -x, x*0-.5), dim=1)
                g1, g2 = -.5-x, -.5+x
                maximum = f+torch.relu(g1-f)
                return maximum+torch.relu(g2-maximum)

        lower, upper = torch.tensor([[-1.]]), torch.tensor([[1.]])
        C, rhs = (old.C, old.RHS) if mode == 'input_split_and' else ([[1]], [0])
        spec = VerificationSpec.build_from_input_bounds(lower, upper, [[(C, rhs)]])
        cfg = default_config()
        for key, value in configuration(mode).items():
            old.put_config(cfg, key, value)
        cfg['bab']['timeout'] = max(.001, deadline-time.monotonic()-2)
        model = Graph().eval()
        conformance = []
        for x in (-1., -.5, 0., .5, 1.):
            output = model(torch.tensor([[x]])).detach().tolist()[0]
            f, g1, g2 = values(x)
            expected = [f, x, -x, -.5] if mode == 'input_split_and' else [max(f, g1, g2)]
            old.require(max(abs(a-b) for a,b in zip(output, expected)) < 1e-14, 'fixed graph formula')
            conformance.append(dict(x=x, output=output))
        save(run/'spec.json', dict(mode=mode, domain=[-1.,1.], C=C, rhs=rhs,
            source='F=.5-x*tanh(x); g1=-.5-x; g2=-.5+x',
            unsafe='F<=0 AND g1<=0 AND g2<=0' if mode == 'input_split_and' else 'max(F,g1,g2)<=0',
            strict_positive=True, conformance=conformance, config=old.json_value(cfg), deadline=deadline))
        start = time.monotonic()
        solver = ABCrownSolver(spec=spec, computing_graph=model, config=cfg)
        result = solver.verify(return_reference=True)
        net = getattr(result.reference.get('model'), 'net', None)
        graph = [] if net is None else [dict(name=n.name, kind=type(n).__name__) for n in net.nodes()]
        split_map = {} if net is None else {str(k): [dict(name=a[0].name, kind=type(a[0]).__name__)
            for a in rows] for k, rows in getattr(net, 'split_activations', {}).items()}
        examples = result.stats.get('attack_examples')
        replay = []
        if examples is not None:
            for point in examples.detach().cpu().reshape(-1).tolist():
                f, g1, g2 = values(point)
                replay.append(dict(x=point, atoms=[f,g1,g2], in_domain=-1 <= point <= 1,
                                   violates=(-1 <= point <= 1 and max(f,g1,g2) <= 0)))
        save(run/'outcome.json', dict(status=result.status, success=result.success,
            stats=old.json_value(result.stats), graph=graph, split_map=split_map, replay=replay,
            verifier_seconds=time.monotonic()-start,
            grade='EXTERNAL_NUMERICAL_FILTER_NOT_INDEPENDENT_PROOF',
            terminal_clock=time.monotonic(), real_calls=0))
        sys.stdout.flush(); sys.stderr.flush()
    return 0


def run(root):
    deadline = ENTRY+300
    old.limits(deadline)
    old.require(sys.executable == old.ACT and not root.exists(), 'new authorized control')
    old.require(root.parent == Path('/data1/Kane/MOE/results'), 'control location')
    old.require(old.git(ROOT, 'branch', '--show-current') == 'feat/moe-route-verification'
                and not old.git(ROOT, 'status', '--porcelain'), 'clean frozen code')
    root.mkdir()
    record = dict(execution_head=old.git(ROOT, 'rev-parse', 'HEAD'), script_sha256=sha(__file__),
        modes=list(MODES), entry=ENTRY, deadline=deadline, cases=[], status='COMPLETED',
        old_smoke_rule_correction='R6 satisfied actual nonlinear branching; positive was not an authorized prerequisite',
        unguarded_exact_minimum=.5-math.tanh(1), real_calls=0)
    for mode in MODES:
        folder = root/mode; folder.mkdir(); (folder/'tmp').mkdir(); (folder/'cache').mkdir()
        start = time.monotonic(); end = min(start+PER_MODE_SECONDS, deadline-5)
        env = dict(PATH='/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
            CUDA_VISIBLE_DEVICES='', TMPDIR=str(folder/'tmp'), XDG_CACHE_HOME=str(folder/'cache'))
        receipt = execute([str(old.ENVIRONMENT/'bin/python'), '-I', '-B', __file__, '--worker',
                           str(folder), mode, str(end-5)],
            run_deadline=end-3, cleanup_deadline=end-1, env=env, rss_limit=None,
            development_policy=old.POLICY)
        outcome = load(folder/'outcome.json', limit=old.CAP) if (folder/'outcome.json').exists() else None
        accepted = (receipt['status'] == 'COMPLETED' and receipt['returncode'] == 0
                    and receipt['execution_error'] is None and receipt['cleanup_error'] is None
                    and receipt['cleanup_status'] == 'LEADER_REAPED_NO_LIVE_SESSION'
                    and outcome is not None and outcome['terminal_clock'] < end-5)
        row = dict(mode=mode, session=receipt, outcome=outcome, execution_accepted=accepted,
                   outer_seconds=time.monotonic()-start,
                   files={p.name: sha(p) for p in folder.iterdir() if p.is_file()})
        save(folder/'result.json', row); record['cases'].append(row)
        old.require(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < old.CAP,
                    'cumulative control artifact cap')
        if receipt['cleanup_error'] is not None or receipt['cleanup_status'] != 'LEADER_REAPED_NO_LIVE_SESSION':
            record['status'] = 'STOP_CLEANUP'; break
        if outcome is not None and outcome['status'] in ('unsafe', 'falsified'):
            record['status'] = 'STOP_UNEXPECTED_UNSAFE_REQUIRES_SPEC_REVIEW'; break
        # UNKNOWN is a scientific outcome, not an execution or branch-coverage failure.
        if outcome is None and receipt['status'] == 'ERROR':
            record['status'] = 'TOOLING_ERROR'; break
    record['through_cases_seconds'] = time.monotonic()-ENTRY
    record['bytes_before_summary'] = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    old.require(time.monotonic() < deadline, 'batch reception deadline')
    save(root/'result.json', record)
    print(json.dumps(dict(status=record['status'], cases=[(r['mode'], None if r['outcome'] is None
        else r['outcome']['status']) for r in record['cases']], seconds=time.monotonic()-ENTRY)))
    return 0 if record['status'] == 'COMPLETED' else 1


if __name__ == '__main__':
    if len(sys.argv) == 5 and sys.argv[1] == '--worker':
        raise SystemExit(worker(Path(sys.argv[2]), sys.argv[3], float(sys.argv[4])))
    if len(sys.argv) == 3 and sys.argv[1] == '--run':
        raise SystemExit(run(Path(sys.argv[2])))
    raise SystemExit('explicit --run NEW_DIRECTORY')
