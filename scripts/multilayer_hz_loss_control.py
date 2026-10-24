"""Bounded joint-HZ loss diagnosis; retained separately from frozen requests."""
import time
ENTRY = time.monotonic()
import argparse
import gzip
import json
import os
from pathlib import Path
import resource
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from scoped_proof.io import load, save, sha
from scoped_proof.owned_session import execute
from scoped_source.development_work import POLICY
from genbab_cpu_ready import ACT

FAILED = Path('/data1/Kane/MOE/results/multilayer_stage1_20261023_r6/verify_282')
CODE = ('scripts/multilayer_hz_loss_control.py', 'act/back_end/moe/multilayer.py',
        'act/back_end/hybridz_tf/hybridz_tf.py', 'act/back_end/hybridz_tf/tf_mlp.py',
        'act/back_end/hybridz_tf/tf_transformer.py', 'act/back_end/solver/solver_hz.py',
        'act/pipeline/moe/experiment1.py', 'act/pipeline/verification/torch2act.py')


def diagnose(folder, deadline):
    from multilayer_stage1 import setup, subject, tensor_row
    torch = setup(deadline)
    from act.back_end.moe.multilayer import HistoryPlan, _lower
    from act.back_end.hybridz_tf import HybridzTF
    from act.back_end.analyze import analyze
    from act.back_end.core import Fact, ConSet
    from act.back_end.verifier import (gather_input_spec_layers, seed_from_input_specs,
        add_all_input_specs, get_input_ids, find_entry_layer_id)
    from act.back_end.transfer_functions import set_transfer_function, set_solver_mode
    from act.config.config import HybridZConfig
    from act.util.device_manager import initialize_device
    from act.util.options import PerformanceOptions
    initialize_device('cpu', 'float64')
    PerformanceOptions.set_debug_output_file(os.devnull)
    request = load(FAILED / 'request.json')
    with gzip.open(FAILED / 'outcome.json.gz', 'rt') as stream:
        old = json.load(stream)['value']['result']
    history = tuple(tuple(p) for p in old['records'][-1]['history'])
    model = subject(request['model'])
    tensors = tensor_row(Path(request['selection_root']), request['sample'], torch)
    plan = HistoryPlan(model, tensors['center'], weighted_matmul=True)
    compiled, _ = plan.compile(history)
    net = _lower(compiled, tensors['center'], tensors['lower'], tensors['upper'],
                 plan.output_width + sum(s.experts for s in plan.sites))

    class Trace(HybridzTF):
        def _drop_sparse_hz(self, layer_id, reason):
            event = dict(layer=layer_id, kind=net.by_id[layer_id].kind, reason=reason,
                         traceback=traceback.format_exc(), clock=time.monotonic())
            events.append(event)
            save(folder / 'drop_events.json', events)
            super()._drop_sparse_hz(layer_id, reason)

    events = []
    tf = Trace(config=HybridZConfig(max_input_dim=1))
    set_transfer_function(tf); set_solver_mode('hybridz')
    specs = gather_input_spec_layers(net)
    entry = Fact(bounds=seed_from_input_specs(specs), cons=ConSet())
    add_all_input_specs(entry.cons, get_input_ids(net), specs)
    began = time.monotonic()
    analyze(net, find_entry_layer_id(net), entry)
    from act.back_end.moe.route_a import _component_output_hz
    try:
        hz = _component_output_hz(net, tf)
        retained = dict(retained=True, outputs=hz.n_out, continuous=hz.n_cont, binary=hz.n_bin)
    except RuntimeError as exc:
        retained = dict(retained=False, reason=str(exc))
    return dict(status='PASS', kind='diagnosis_not_verification', history=history,
        checkpoint_sha256=request['model']['checkpoint_sha256'], input=request['sample']['tensors'],
        original_reason=old['reason'], propagation_seconds=time.monotonic()-began,
        result=retained, drops=events, real_full_requests=0,
        scope='one failed history propagation; no new property/feasibility query; no old outcome relabeled')


def worker(folder):
    request = load(folder / 'request.json')
    frozen = load(folder.parent / 'freeze.json', request['freeze_sha256'])
    assert all(sha(ROOT / p) == h for p, h in frozen['sources'].items())
    assert all(sha(FAILED / p) == h for p, h in frozen['failed_files'].items())
    value = diagnose(folder, request['deadline'])
    assert time.monotonic() < request['deadline']
    save(folder / 'outcome.json', value)


def run(folder):
    assert not folder.exists()
    folder.mkdir(); child = folder / 'fixed'; child.mkdir()
    save(folder / 'freeze.json', dict(sources={p:sha(ROOT/p) for p in CODE},
        failed_files={p:sha(FAILED/p) for p in ('request.json','outcome.json.gz','terminal.json')},
        operation='single recorded failing history propagation with drop trace',
        seconds=300, as_bytes=8*2**30, artifacts=128*2**20, threads=1, gpu=False))
    save(child / 'request.json', dict(freeze_sha256=sha(folder/'freeze.json'), deadline=ENTRY+290))
    env = dict(PATH='/usr/bin:/bin', CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONNOUSERSITE='1')
    receipt = execute([ACT,'-I','-B',__file__,'worker',str(child)],run_deadline=ENTRY+294,
        cleanup_deadline=ENTRY+298, env=env, rss_limit=None, development_policy=POLICY)
    save(folder/'session.json',receipt)
    from b1_observed_batch import session_ok
    good = session_ok(receipt) and (child/'outcome.json').exists() and time.monotonic()<ENTRY+300
    assert sum(p.stat().st_size for p in folder.rglob('*') if p.is_file()) < 128*2**20
    save(folder/'complete.json',dict(status='PASS' if good else 'FAIL', seconds=time.monotonic()-ENTRY,
        freeze_sha256=sha(folder/'freeze.json'),session_sha256=sha(folder/'session.json'),
        outcome_sha256=sha(child/'outcome.json') if good else None))
    print(json.dumps(load(folder/'complete.json')),flush=True)


if __name__ == '__main__':
    resource.setrlimit(resource.RLIMIT_AS,(8*2**30,8*2**30))
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker'])
    parser.add_argument('folder',type=Path);args=parser.parse_args()
    if args.mode=='run':run(args.folder)
    else:
        try:worker(args.folder)
        except BaseException as exc:
            save(args.folder/'error.json',dict(error=str(exc),traceback=traceback.format_exc()));raise
