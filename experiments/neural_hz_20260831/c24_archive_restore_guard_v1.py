"""One separate read-only fresh-process guard for the completed C24 archive."""

import hashlib
import json
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
PRIOR = EXP / 'results/c24_dense_closed_20260911_v1'
DIRECTORY = EXP / 'results/c24_archive_restore_guard_20260911_v1'
INPUTS = {'closed_hz.pickle': '8e72ae38cbdafe95f8d8a91928dc2b47a20f4fec5c2d4bd856a377ff267b1be0',
    'closed_proof.json': '99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6',
    'result.json': '0b87f78d5dc73ab6655ac599dbccce8a59fde37a3b5e9e0781f0a59b399890fe',
    'exit.json': '7dbd873f7e34775b5576fb3f0e29ae0549387c22691aa992a5105c68024db8ea'}


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    if DIRECTORY.exists(): raise FileExistsError(DIRECTORY)
    DIRECTORY.mkdir()
    freeze = {'input_sha256': INPUTS, 'guard_source_sha256': _sha256(Path(__file__)),
        'provenance': _provenance(ROOT), 'read_only_completed_archive_guard': True,
        'new_generator_or_solver_authorized': False, 'formal_gain': 0}
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', freeze)
    started, report = time.monotonic(), {'completed': False, 'formal_gain': 0}
    try:
        if any(_sha256(PRIOR / name) != sha for name, sha in INPUTS.items()):
            raise ValueError('completed independent archive manifest drift')
        prior = json.loads((PRIOR / 'result.json').read_text())
        ending = json.loads((PRIOR / 'exit.json').read_text())
        if (not prior['completed'] or ending['worker_exit_code'] != 0 or ending['source_drift']
                or ending['provenance_drift'] or freeze['provenance'] != prior['provenance']):
            raise ValueError('archive is not a completed matching-source qualification')
        if any(_sha256(EXP / name) != sha for name, sha in prior['source_sha256'].items()):
            raise ValueError('complete inherited source manifest changed')
        with (PRIOR / 'closed_hz.pickle').open('rb') as handle:
            saved = pickle.load(handle)
        if saved['schema'] != 'c24_dense_closed_checkpoint_v1':
            raise ValueError('wrong closed archive schema')
        raw = (PRIOR / 'closed_proof.json').read_bytes()
        if saved['proof_bytes'] != raw or saved['closed_proof_sha256'] != INPUTS['closed_proof.json']:
            raise ValueError('checkpoint proof is not the independently anchored artifact')
        restored = restore(saved['fields'], raw, expected_proof_sha256=INPUTS['closed_proof.json'])
        restored.validate()
        if hasattr(restored, 'nodes') or restored.fingerprint() != prior['identity']['closed_identity']:
            raise ValueError('fresh process restored a graph or changed the checked identity')
        overlay = Overlay(restored.owners, saved['phase_events'], restored.report['radix_uid_base'] + 16384)
        overlay.validate()
        event_sha = hashlib.sha256(overlay.events.tobytes()).hexdigest()
        if event_sha != prior['phase_event_audit']['event_words_sha256']:
            raise ValueError('restored sparse events differ from complete phase proof')
        if source_digest(restored.hz) != prior['hz_sha256']:
            raise ValueError('restored HZ differs from full source/native proof')
        report.update(completed=True, closed_identity=restored.fingerprint(),
            restored_HZ_sha256=source_digest(restored.hz), all_frozen_source_hashes_checked=True,
            no_definition_graph_restored=True, source_sharing_and_all_numeric_content_bound=True,
            MAIN_owner_words=len(restored.owners), uid_slab_words=len(restored.uid_slabs),
            sparse_event_words=len(overlay.events), phase_event_sha256=event_sha,
            n_cont=restored.hz.n_cont, n_bin=restored.hz.n_bin, n_eq=restored.hz.n_eq,
            n_ineq=restored.hz.n_ineq, new_generator_executed=False, new_solver_executed=False)
    except Exception as exc:
        report['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
    finally:
        report.update(wall_s=time.monotonic() - started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            input_sha256=INPUTS, source_sha256=freeze['guard_source_sha256'],
            provenance=freeze['provenance'])
        _atomic_exclusive_json(DIRECTORY / 'result.json', report)
        print(json.dumps(report), flush=True)
    if not report['completed']: raise SystemExit(1)


if __name__ == '__main__': main()
