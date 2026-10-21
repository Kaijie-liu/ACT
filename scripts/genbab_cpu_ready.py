"""Repair B2 inventory reception and run one fixed nonlinear guarded-pair smoke.

No installation, source patch or trained model. Existing owned-session supervision
charges import, graph construction, actual BaB, reception and cleanup to 300s.
The historical deployment FAIL is never modified. This is compatibility evidence,
not an ACT comparison or a checked real-network certificate.
"""
import time
ENTRY = time.monotonic()
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import subprocess
import sys

ROOT = Path('/data1/Kane/MOE/ACT')
sys.path.insert(0, str(ROOT))
from scoped_proof.io import load, save, sha
from scoped_proof.owned_session import execute

ACT = '/data1/Kane/miniconda3/envs/act-py312/bin/python'
ENVIRONMENT = Path('/data1/Kane/MOE/envs/genbab-cpu-20261021-v1')
TOOL = Path('/data1/Kane/MOE/baselines/alpha-beta-CROWN')
SOURCES = {str(TOOL): 'e5c7e17bf0488843acb77b7519f59876717a49f4',
           str(TOOL/'auto_LiRPA'): '5a098e8f9fb5786a428a024981d833d303921f2d'}
OLD = Path('/data1/Kane/MOE/results/genbab_cpu_deployment_20261021_r1')
POLICY = 'DEV_MARGIN_20261021_V1'
CAP = 128*2**20
C = [[1, 0, 0, 0], [0, -1, 0, 1], [0, 0, -1, 1]]
RHS = [0, 0, 0]
CONFIG = {
    'general.device': 'cpu', 'general.double_fp': True,
    'general.complete_verifier': 'bab', 'general.enable_incomplete_verification': True,
    'general.precompile_jit': False, 'solver.batch_size': 1,
    'solver.auto_enlarge_batch_size': False, 'solver.bound_prop_method': 'alpha-crown',
    'bab.branching.input_split.enable': False, 'bab.branching.method': 'nonlinear',
    'bab.branching.nonlinear_split.disable': False,
    'bab.branching.nonlinear_split.relu_only': False,
    'bab.branching.nonlinear_split.method': 'bbps',
    'bab.branching.nonlinear_split.branching_point_method': 'uniform',
    'bab.branching.nonlinear_split.num_branches': 2, 'bab.interm_transfer': True,
    'bab.attack.enabled': False, 'attack.pgd_order': 'skip',
    'debug.print_verbose_decisions': True,
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True, timeout=10).strip()


def source_identity():
    for path, revision in SOURCES.items():
        require(git(path, 'rev-parse', 'HEAD') == revision
                and not git(path, 'status', '--porcelain'), 'external verifier source identity')
    return SOURCES


def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def distributions():
    # A list preserves top-level and vendored metadata instead of last-name wins.
    return sorted((dict(name=normalized(d.metadata['Name']), version=d.version,
                        location=str(Path(d.locate_file('')).resolve()),
                        metadata_path=str(d._path))
                   for d in importlib.metadata.distributions()),
                  key=lambda row: (row['name'], row['version'], row['location'], row['metadata_path']))


def check_inventory(rows, site_packages, required):
    for name, version in required.items():
        matches = [r for r in rows if r['name'] == normalized(name)
                   and r['location'] == str(site_packages)]
        require(len(matches) == 1 and matches[0]['version'] == version,
                'top-level installed distribution: '+name)
    require(not any(r['name'].startswith('nvidia-') for r in rows), 'CPU deployment')


def required_packages():
    return dict(line.split('==') for line in
                (ROOT/'configs/genbab_cpu_20261021_requirements.txt').read_text().splitlines()
                if line and not line.startswith('#'))


def put_config(cfg, key, value):
    parts = key.split('.')
    for part in parts[:-1]:
        cfg = cfg[part]
    require(parts[-1] in cfg, 'known pinned configuration key: '+key)
    cfg[parts[-1]] = value


def json_value(value):
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if hasattr(value, 'detach'):
        return json_value(value.detach().cpu().tolist())
    if hasattr(value, 'item'):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Telemetry only; never interpreted as a finite bound.
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('unsupported telemetry type: '+type(value).__name__)


def limits(deadline):
    require(time.monotonic() < deadline, 'shared deadline')
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (CAP, CAP))
    signal.signal(signal.SIGALRM, lambda *_: os._exit(124))
    signal.setitimer(signal.ITIMER_REAL, deadline-time.monotonic())


def probe():
    require(sys.prefix == str(ENVIRONMENT), 'isolated CPU interpreter')
    source_identity()
    sys.path[:0] = [str(TOOL/'auto_LiRPA'), str(TOOL/'complete_verifier')]
    import torch
    import auto_LiRPA
    import api
    import packaging
    import wheel
    require(torch.__version__ == '2.11.0+cpu' and torch.version.cuda is None, 'CPU torch')
    modules = {}
    for name, module in (('torch', torch), ('auto_LiRPA', auto_LiRPA),
                         ('complete_verifier.api', api), ('packaging', packaging), ('wheel', wheel)):
        path = Path(module.__file__).resolve()
        expected = TOOL if name in ('auto_LiRPA', 'complete_verifier.api') else ENVIRONMENT
        require(path.is_relative_to(expected), 'loaded module origin: '+name)
        modules[name] = dict(path=str(path), sha256=sha(path), version=getattr(module, '__version__', None))
    rows = distributions()
    site_packages = ENVIRONMENT/'lib/python3.11/site-packages'
    check_inventory(rows, site_packages, required_packages())
    return dict(status='PASS', modules=modules, distributions=rows, sources=SOURCES,
                python=sys.version, prefix=sys.prefix, torch=torch.__version__, cuda_build=None,
                old_install_reports={n: sha(OLD/n) for n in ('packages.json', 'converter.json')})


def smoke(run, deadline):
    import torch
    from api import ABCrownSolver, VerificationSpec, default_config
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)

    class GuardedPair(torch.nn.Module):
        def forward(self, x):
            ea, eb = .5-x, .5+x
            gate = torch.sigmoid(2*x)
            return torch.cat((eb+gate*(ea-eb), x, -x, x*0-.5), dim=1)

    model = GuardedPair().eval()
    lower, upper = torch.tensor([[-1.]]), torch.tensor([[1.]])
    spec = VerificationSpec.build_from_input_bounds(lower, upper, [[(C, RHS)]])
    cfg = default_config()
    for key, value in CONFIG.items():
        put_config(cfg, key, value)
    cfg['bab']['timeout'] = max(.001, deadline-time.monotonic()-1)
    save(run/'spec.json', dict(source='F=.5-x*tanh(x), r=(x,-x,-.5); graph uses sigmoid and multiplication',
                              lower=lower.tolist(), upper=upper.tolist(), C=C, rhs=RHS,
                              config=json_value(cfg), dtype=str(torch.get_default_dtype()),
                              expected='verified; guarded |x|<=.5 has F>=.25', deadline=deadline))
    started = time.monotonic()
    solver = ABCrownSolver(spec=spec, computing_graph=model, config=cfg)
    result = solver.verify(return_reference=True)
    net = getattr(result.reference.get('model'), 'net', None)
    graph = [] if net is None else [dict(name=n.name, kind=type(n).__name__,
                                        inputs=[i.name for i in n.inputs]) for n in net.nodes()]
    split_activations = {} if net is None else {
        str(k): [dict(name=a[0].name, kind=type(a[0]).__name__, input_index=a[1]) for a in rows]
        for k, rows in getattr(net, 'split_activations', {}).items()}
    record = dict(status=result.status, success=result.success, stats=json_value(result.stats),
                  graph=graph, split_activations=split_activations,
                  verifier_seconds=time.monotonic()-started,
                  evidence_grade='EXTERNAL_NUMERICAL_FILTER_SYNTHETIC_COMPATIBILITY_ONLY',
                  trained_model_calls=0, synthetic_verifier_calls=1, gpu_calls=0)
    save(run/'smoke.json', record)
    return record


def worker(run, deadline):
    limits(deadline)
    # Capture native output too; the existing supervisor's own console stays small.
    with (run/'native.log').open('xb', buffering=0) as log:
        os.dup2(log.fileno(), 1); os.dup2(log.fileno(), 2)
        save(run/'probe.json', probe())
        smoke(run, deadline)
        sys.stdout.flush(); sys.stderr.flush()
    return 0


def accepted_smoke(record, log):
    visited = sum(int(row[2]) for row in record['stats'].get('bab', []))
    nonlinear = any(node['kind'] in ('BoundSigmoid', 'BoundMul') for node in record['graph'])
    has_split_map = any(a['kind'] in ('BoundSigmoid', 'BoundMul')
                        for rows in record['split_activations'].values() for a in rows)
    require(record['status'] == 'verified', 'expected guarded numerical positive')
    require(visited > 0 and nonlinear and has_split_map
            and 'splitting decisions' in log and 'Branching points' in log,
            'actual nonlinear branch execution not covered')
    return visited


def audit(run):
    value = load(run/'result.json', limit=2**20)
    for name, digest in value['files'].items():
        require(sha(run/name) == digest, 'saved record: '+name)
    p = load(run/'probe.json', limit=CAP)
    check_inventory(p['distributions'], ENVIRONMENT/'lib/python3.11/site-packages', required_packages())
    for module in p['modules'].values():
        require(sha(module['path']) == module['sha256'], 'actual loaded source identity')
    source_identity()
    receipt = value['session']
    require(receipt['status'] == 'COMPLETED' and receipt['returncode'] == 0
            and receipt['execution_error'] is None and receipt['cleanup_error'] is None
            and receipt['cleanup_status'] == 'LEADER_REAPED_NO_LIVE_SESSION', 'owned terminal')
    require(value['seconds_before_terminal_export'] < 300, 'same budget')
    visited = accepted_smoke(load(run/'smoke.json', limit=CAP), (run/'native.log').read_text())
    require(sum(p.stat().st_size for p in run.rglob('*') if p.is_file()) <= CAP, 'artifact cap')
    return dict(status='PASS', visited=visited, result_sha256=sha(run/'result.json'))


def main(run):
    deadline = ENTRY+300
    limits(deadline)
    require(sys.executable == ACT, 'ACT controller')
    require(git(ROOT, 'branch', '--show-current') == 'feat/moe-route-verification'
            and not git(ROOT, 'status', '--porcelain'), 'clean execution source')
    source_identity()
    require(run.parent == Path('/data1/Kane/MOE/results') and not run.exists(), 'new result directory')
    run.mkdir(); (run/'tmp').mkdir(); (run/'cache').mkdir()
    env = dict(PATH='/usr/bin:/bin', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1',
               CUDA_VISIBLE_DEVICES='', TMPDIR=str(run/'tmp'), XDG_CACHE_HOME=str(run/'cache'),
               TORCH_HOME=str(run/'cache'))
    value = dict(status='FAIL', execution_head=git(ROOT, 'rev-parse', 'HEAD'),
                 controller_sha256=sha(__file__), sources=SOURCES, entry=ENTRY, deadline=deadline,
                 policy=POLICY, real_request_slots=0, trained_model_calls=0, gpu_calls=0,
                 full_cold_start_seconds=None)
    try:
        value['session'] = execute([str(ENVIRONMENT/'bin/python'), '-I', '-B', __file__,
                                   '--worker', str(run), str(deadline-10)],
                                  run_deadline=deadline-8, cleanup_deadline=deadline-2, env=env,
                                  rss_limit=None, development_policy=POLICY)
        receipt = value['session']
        require(receipt['status'] == 'COMPLETED' and receipt['returncode'] == 0
                and receipt['execution_error'] is None and receipt['cleanup_error'] is None,
                'native import/smoke failed; preserve log')
        value['visited'] = accepted_smoke(load(run/'smoke.json', limit=CAP), (run/'native.log').read_text())
        source_identity()
        value['status'] = 'PASS'
    except Exception as exc:
        value['error'] = repr(exc)
    value['files'] = {p.name: sha(p) for p in sorted(run.iterdir()) if p.is_file()}
    value['seconds_before_terminal_export'] = time.monotonic()-ENTRY
    value['record_bytes_before_terminal_export'] = sum(p.stat().st_size for p in run.rglob('*') if p.is_file())
    require(value['record_bytes_before_terminal_export']+len(json.dumps(value).encode()) <= CAP, 'artifact cap')
    save(run/'result.json', value)
    print(json.dumps(dict(status=value['status'], error=value.get('error'), visited=value.get('visited'),
                         seconds=value['seconds_before_terminal_export'], real_request_slots=0)), flush=True)
    return 0 if value['status'] == 'PASS' else 1


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--worker':
        raise SystemExit(worker(Path(sys.argv[2]), float(sys.argv[3])))
    if len(sys.argv) == 3 and sys.argv[1] == '--execute':
        raise SystemExit(main(Path(sys.argv[2])))
    if len(sys.argv) == 3 and sys.argv[1] == '--audit':
        print(json.dumps(audit(Path(sys.argv[2])))); raise SystemExit(0)
    raise SystemExit('use --execute NEW_DIRECTORY or --audit SAVED_DIRECTORY')
