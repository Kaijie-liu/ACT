"""One frozen synthetic gate-graph batch, using the existing owned supervisor.

No trained checkpoints, data, GPU or real diagnostic calls. The three fixed
external API queries and all controls share the same cumulative300s budget.
"""
import time
ENTRY = time.monotonic()
import argparse
import io
import json
import os
from pathlib import Path
import resource
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts'), str(ROOT/'tests')]
from scoped_proof.io import load, save, sha
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
import genbab_cpu_ready as backend

CODE = ('scripts/gate_decoupling_control.py', 'tests/test_gate_decoupling.py',
    'scripts/gate_decoupling_worker.py', 'scripts/multilayer_stage1.py',
    'scripts/gate_decoupling_stage4.py', 'tests/test_gate_decoupling_stage4.py',
    'act/back_end/moe/gate_decoupling.py', 'act/back_end/moe/multilayer_external.py',
    'act/back_end/moe/multilayer.py', 'act/back_end/moe/model.py', 'act/back_end/moe/schema.py',
    'act/pipeline/moe/multilayer_training.py', 'act/back_end/moe/genbab_pair.py',
    'scripts/b2_observed_worker.py', 'scripts/genbab_cpu_ready.py', 'scripts/b1_observed_batch.py',
    'scoped_proof/io.py', 'scoped_proof/owned_session.py', 'scoped_source/development_work.py')


def freeze(path, prior=None, followup=False):
    retained = None
    if prior is not None:
        old = load(prior)
        assert old['status'] == 'FAIL'
        retained = dict(path=str(prior.resolve()), sha256=sha(prior), seconds=old['seconds'],
            files={str(p.relative_to(prior.parent)): sha(p) for p in prior.parent.rglob('*') if p.is_file()})
    if followup:
        code = (*CODE, 'scripts/gate_decoupling_followup.py', 'tests/test_gate_decoupling_followup.py')
        save(path, dict(schema='gate-decoupling-followup-control-v1', policy=POLICY,
            retained_control=retained, seconds=300, artifacts=128*2**20, as_bytes=8*2**30,
            source_files={p: sha(ROOT/p) for p in code}, native_python=str(backend.ENVIRONMENT/'bin/python'),
            trained_models=0, real_diagnostic_calls=0, solver_calls=3, prefix_crown_calls=2, gpu=False,
            suites=[('test_gate_decoupling_followup.FollowupControls.test_z_native_reconstruction_graph_and_reception'
                     if prior else 'test_gate_decoupling_followup.FollowupControls'),
                'test_gate_decoupling.WorkerControls.test_receiver_source_and_counterfactual_scope',
                'test_gate_decoupling_stage4.BatchControls.test_audit_binds_raw_terminal_and_failed_attempt'],
            cases=['fixed supplementary selection/T1', 'canonical prefix exclusion/ties',
                'source/history/missing/extra/malformed record mutation', 'partial/expired reconstruction',
                'legacy missing unchanged', 'native two-prefix reconstruction + three endpoint API calls',
                'new/legacy reception and original ledger audit regression'],
            acceptance='all new provenance/receipt controls and declared native calls pass; no scientific gain claim'))
        return
    save(path, dict(schema='gate-decoupling-synthetic-v4', policy=POLICY,
        retained_control=retained,
        seconds=300, artifacts=128*2**20, as_bytes=8*2**30,
        source_files={p: sha(ROOT/p) for p in CODE},
        native_python=str(backend.ENVIRONMENT/'bin/python'),
        trained_models=0, real_diagnostic_calls=0, solver_calls=3, gpu=False,
        module='test_gate_decoupling',
        suites=['test_gate_decoupling.SourceControls', 'test_gate_decoupling.WorkerControls',
                'test_gate_decoupling_stage4.BatchControls',
                'test_gate_decoupling.GraphControls'],
        cases=['ordered margins/direction/tie/source mutation',
               'missing prefix/binding/nonfinite/inverted range/alias',
               'inner endpoint unsoundness counterexample',
               'all properties/two endpoints/diagnostic-only/late receipt',
               'L1/L2/L3 joint graph and residual affine suffix',
               'ten classes/nonzero constants/tie/output-level object',
               'midpoint-positive original-negative control',
               'V2 retains inner products/V3 affine-ReLU/no nonlinear suffix',
               'same endpoint deadline/unknown continuation/partial interruption',
               'artifact cap/exclusive publication/late receipt/source/scope mutations',
               'full denominator/Wilson gates/fixed order/call cap/tool-error prefix/failed start accounting',
               'same fixed safe L2 graph:V2_LO,V2_HI,V3 through existing GenBaB API'],
        acceptance='all declared semantic controls and three analytic-positive API calls pass; not performance or real proof'))


def worker(folder):
    request = load(folder/'request.json')
    deadline = request['deadline']
    spec = load(request['manifest'], request['manifest_sha256'])
    assert all(sha(ROOT/p) == h for p, h in spec['source_files'].items())
    backend.limits(deadline)
    from act.util.typing_compat import install_typing_override
    install_typing_override()
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float64)
    os.environ['GATE_CONTROL_DEADLINE'] = str(deadline)
    os.environ['GATE_CONTROL_ROOT'] = str(folder)
    output = io.StringIO()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(name) for name in spec['suites'])
    # Match the real worker: model dumps are not evidence; keep structured API results.
    fd = os.open(os.devnull, os.O_WRONLY); os.dup2(fd, 1); os.close(fd)
    result = unittest.TextTestRunner(stream=output, verbosity=2, failfast=True).run(suite)
    import test_gate_decoupling as tests
    assert time.monotonic() < deadline
    save(folder/'outcome.json', dict(status='PASS' if result.wasSuccessful() else 'FAIL',
        tests=result.testsRun, details=output.getvalue(), api_results=tests.API_RESULTS,
        request_sha256=sha(folder/'request.json'), as_limit=resource.getrlimit(resource.RLIMIT_AS),
        clock=time.monotonic(), real_calls=0))


def run(folder, manifest):
    resource.setrlimit(resource.RLIMIT_AS, (8*2**30, 8*2**30))
    spec = load(manifest)
    assert all(sha(ROOT/p) == h for p, h in spec['source_files'].items())
    assert spec['seconds'] == 300 and spec['policy'] == POLICY
    prior = spec.get('retained_control'); prior_seconds = 0.; prior_bytes = 0
    if prior:
        receipt = load(prior['path'], prior['sha256']); owner = Path(prior['path']).parent
        assert receipt['status'] == 'FAIL' and receipt['seconds'] == prior['seconds']
        assert {str(p.relative_to(owner)): sha(p) for p in owner.rglob('*') if p.is_file()} == prior['files']
        prior_seconds = prior['seconds']
        prior_bytes = sum((owner/p).stat().st_size for p in prior['files'])
    end = ENTRY+300-prior_seconds
    assert end > ENTRY+10 and prior_bytes < spec['artifacts']-65536
    folder.mkdir(); save(folder/'manifest.json', spec)
    assert sha(folder/'manifest.json') == sha(manifest)
    child = folder/'fixed'; child.mkdir()
    save(child/'request.json', dict(manifest=str(manifest.resolve()), manifest_sha256=sha(manifest),
        deadline=end-10, budget_started=ENTRY, retained_control=prior))
    env = dict(PATH='/usr/bin:/bin', CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
        XDG_CACHE_HOME=str(folder/'cache'), TMPDIR=str(folder/'tmp'))
    (folder/'cache').mkdir(); (folder/'tmp').mkdir()
    receipt = execute([spec['native_python'], '-I', '-B', __file__, 'worker', str(child)],
        run_deadline=end-6, cleanup_deadline=end-2, env=env, rss_limit=None, development_policy=POLICY)
    save(folder/'session.json', receipt)
    from b1_observed_batch import session_ok
    outcome = load(child/'outcome.json') if (child/'outcome.json').is_file() else None
    good = bool(session_ok(receipt) and outcome and outcome['status'] == 'PASS'
        and outcome['request_sha256'] == sha(child/'request.json') and outcome['clock'] < end-10
        and len(outcome['api_results']) == spec['solver_calls'] and time.monotonic() < end)
    retained = prior_bytes+sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
    assert retained < spec['artifacts']-65536  # Reserve the final receipt inside the cap.
    save(folder/'complete.json', dict(status='PASS' if good else 'FAIL', seconds=prior_seconds+time.monotonic()-ENTRY,
        manifest_sha256=sha(manifest), session_sha256=sha(folder/'session.json'),
        outcome_sha256=sha(child/'outcome.json') if outcome else None, bytes_before_receipt=retained,
        retained_control=prior, real_calls=0, as_limit=resource.getrlimit(resource.RLIMIT_AS)))
    print(json.dumps(load(folder/'complete.json')), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('freeze', 'run', 'worker'))
    p.add_argument('path', type=Path); p.add_argument('--manifest', type=Path)
    p.add_argument('--prior', type=Path); p.add_argument('--followup', action='store_true'); a = p.parse_args()
    if a.mode == 'freeze': freeze(a.path, a.prior, a.followup)
    elif a.mode == 'worker': worker(a.path)
    else: run(a.path, a.manifest)
