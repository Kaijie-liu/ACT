"""One S1 adapter control batch under existing owned-session supervision."""
import time
ENTRY = time.monotonic()
import io
import json
import os
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
from scoped_proof.io import save, sha, load
import genbab_cpu_ready as backend

FILES = ('act/pipeline/moe/architecture_evaluation.py', 'act/pipeline/moe/architecture_dense.py',
    'act/back_end/moe/multilayer_external.py', 'scripts/b2_observed_worker.py',
    'scripts/genbab_cpu_ready.py', 'scripts/architecture_s1_control.py',
    'tests/test_architecture_evaluation.py', 'tests/test_multilayer_stage1.py')


def worker(folder, deadline):
    # Native console is not evidence. Keep structured test errors/outcomes.
    fd = os.open(os.devnull, os.O_WRONLY); os.dup2(fd, 1); os.dup2(fd, 2); os.close(fd)
    try:
        backend.limits(deadline)
        environment = backend.probe()
        from act.util.typing_compat import install_typing_override
        install_typing_override()
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1)
        torch.set_default_dtype(torch.float64)
        import unittest
        import test_architecture_evaluation as suite_module
        suite_module.DEADLINE = deadline
        suite = unittest.TestSuite([
            unittest.defaultTestLoader.loadTestsFromModule(suite_module),
            unittest.defaultTestLoader.loadTestsFromName('test_multilayer_stage1')])
        output = io.StringIO()
        result = unittest.TextTestRunner(stream=output, verbosity=2, failfast=True).run(suite)
        save(folder/'outcome.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL',
            tests=result.testsRun, details=output.getvalue(), cases=suite_module.OBSERVED,
            trained_models=0, trained_verification_calls=0, seconds=time.monotonic()-ENTRY,
            backend_source=environment['sources'], torch=environment['torch'],
            source_files={p:sha(ROOT/p) for p in FILES}, deadline=deadline,
            clock=time.monotonic()))
        return 0 if result.wasSuccessful() else 1
    except BaseException as exc:
        save(folder/'error.json', dict(kind=type(exc).__name__, message=str(exc),
            traceback=traceback.format_exc(), seconds=time.monotonic()-ENTRY))
        return 1


def run(folder):
    from scoped_proof.owned_session import execute
    folder.mkdir()
    save(folder/'freeze.json', dict(schema='S1_EVALUATION_CONTROL_V1', seconds=300,
        source_files={p:sha(ROOT/p) for p in FILES},
        synthetic_dense_checkpoint_sha256=sha('/data1/Kane/MOE/results/architecture_s1_control_20261029_r1/native/control.pt'),
        registered_tests=['dense positive both methods/nonzero constants', 'clean wrong full replay',
            'zero unknown', 'two-layer safe/tie all nine histories both methods',
            'two-layer unsafe replay', 'early unknown/missing differential',
            'unsafe tie alternative', 'invalid property/box/deadline',
            'cross environment dense load', 'existing traversal regressions']))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    session = execute([str(backend.ENVIRONMENT/'bin/python'), '-I', '-B', str(Path(__file__).resolve()),
        'worker', str(folder), str(ENTRY+290)], run_deadline=ENTRY+294,
        cleanup_deadline=ENTRY+298, env=env, rss_limit=None, development_policy=backend.POLICY)
    save(folder/'session.json', session)
    outcome = load(folder/'outcome.json') if (folder/'outcome.json').exists() else None
    accepted = (session['status']=='COMPLETED' and session['cleanup_status']=='LEADER_REAPED_NO_LIVE_SESSION'
        and outcome is not None and outcome['status']=='PASS'
        and outcome['clock'] < outcome['deadline'] and time.monotonic() < ENTRY+300)
    record = dict(status='PASS' if accepted else 'FAIL', seconds=time.monotonic()-ENTRY,
        native_status=None if outcome is None else outcome['status'],
        tests=None if outcome is None else outcome['tests'], session_status=session['status'],
        files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},
        source_files={p:sha(ROOT/p) for p in FILES}, trained_calls=0)
    save(folder/'complete.json', record)
    print(json.dumps(record))
    return 0 if accepted else 1


if __name__ == '__main__':
    if sys.argv[1] == 'worker':
        raise SystemExit(worker(Path(sys.argv[2]), float(sys.argv[3])))
    if sys.argv[1] == 'run':
        raise SystemExit(run(Path(sys.argv[2])))
    raise SystemExit('use run NEW_DIRECTORY')
