"""Fresh-process read-only restore and complete incidence guard for C25."""

import hashlib
import json
from pathlib import Path
import pickle
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c24_closed_state_v1 import restore
from experiments.neural_hz_20260831.c24_uid_slabs_v1 import closed_uid_tables
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import check_append, incidence_oracle, verify_all_and_discover, BranchPool
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.shadow_worker_dtype_v2 import _provenance
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
PRIOR = EXP / 'results/c25_live_relu_20260911_v1'
DIRECTORY = EXP / 'results/c25_archive_restore_guard_20260911_v1'
INPUTS = {'relu78.pickle': '685e80ba9754fa821d3fa0486309a1572dcffecb6f40c83a67309f1bd3a5b9ba',
    'closed_proof.json': '99a52f7adafb61275690f6993940a1eacb90b71c68b0140da7398d75526c41e6',
    'qualification.json': '88a754ad695a7da0ed2997a1a88542433ecceac9e5d6fa4932424e67a2deb886',
    'exit.json': 'e35d5460ceb2bfbb712ad1f10be1e7bf944511342790e6cc24cdf0d1d219b2ae'}
OLD_LIVE = EXP / 'results/c10_live_relu_20260908_v1/qualification.json'
OLD_LIVE_SHA = '3662f2b3ac45e6c84a6aa4968c8632e6bb658d7b93575812ae4b638e088f0af0'


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    if DIRECTORY.exists(): raise FileExistsError(DIRECTORY)
    DIRECTORY.mkdir()
    freeze = {'input_sha256': INPUTS, 'guard_source_sha256': _sha256(Path(__file__)),
        'old_live_qualification_sha256': OLD_LIVE_SHA,
        'provenance': _provenance(ROOT), 'read_only_completed_archive_guard': True,
        'new_generator_or_solver_authorized': False, 'formal_gain': 0}
    _atomic_exclusive_json(DIRECTORY / 'preregistered.json', freeze)
    started, report = time.monotonic(), {'completed': False, 'formal_gain': 0}
    try:
        if any(_sha256(PRIOR / name) != sha for name, sha in INPUTS.items()) or _sha256(OLD_LIVE) != OLD_LIVE_SHA:
            raise ValueError('completed independent archive manifest drift')
        prior = json.loads((PRIOR / 'qualification.json').read_text())
        ending = json.loads((PRIOR / 'exit.json').read_text())
        if (not prior['passed'] or not prior['physical_decrease'] or not prior['native_ingestion']['passed']
                or ending['worker_exit_code'] != 0 or ending['source_drift'] or ending['provenance_drift']
                or freeze['provenance'] != prior['provenance']):
            raise ValueError('archive is not a completed matching-source live qualification')
        if any(_sha256(EXP / name) != sha for name, sha in prior['source_sha256'].items()):
            raise ValueError('complete inherited source manifest changed')
        old = json.loads(OLD_LIVE.read_text())
        descriptors = lambda r: [(v['layer_for_provenance_only'],v['nnz'],v['sha256']) for v in r['reference']['leaves']]
        if descriptors(prior) != descriptors(old):
            raise ValueError('fresh live leaves differ from original C10 live comparator')
        with (PRIOR / 'relu78.pickle').open('rb') as handle:
            saved = pickle.load(handle)
        if saved['schema'] != 'c25_live_relu_checkpoint_v1' or not saved['whole_live_path_proved']:
            raise ValueError('wrong or incomplete live closed archive schema')
        raw = (PRIOR / 'closed_proof.json').read_bytes()
        if saved['closed_proof_bytes'] != raw or saved['closed_proof_sha256'] != INPUTS['closed_proof.json']:
            raise ValueError('checkpoint proof is not independently anchored')
        closed = restore(saved['closed_fields'], raw, expected_proof_sha256=INPUTS['closed_proof.json'])
        closed.validate()
        if hasattr(closed,'nodes') or closed.fingerprint() != prior['islands'][0]['binding']['closed_identity']:
            raise ValueError('fresh process restored a graph or changed checked identity')
        post, owned = saved['post_relu_hz'], saved['phase_ownership']
        if ('receipt' in owned or owned['post_hz'] is not post or owned['base'] is not closed.owners
                or saved['hz_cache'].get(78) is not post or 78 in saved['expr_cache']
                or source_digest(post) != prior['actual_hz_sha256'] or source_digest(post) != owned['post_sha256']):
            raise ValueError('actual published cache/source/owner identity was not preserved')
        overlay = Overlay(owned['base'],owned['events'],owned['old_uid_ceiling'])
        overlay.validate()
        if hashlib.sha256(overlay.events.tobytes()).hexdigest() != owned['event_sha256']:
            raise ValueError('restored append events changed')
        check_append(closed.hz,post)
        ne,nl = post.n_eq-closed.hz.n_eq,post.n_ineq-closed.hz.n_ineq
        first = closed.report['radix_uid_base']+16384
        if overlay.old_uid_ceiling != first: raise ValueError('changed phase UID boundary')
        pool = WorkPool(256_000_000)
        pool.charge('complete_closed_UID_metadata',32*(len(closed.owners)+post.n_eq+post.n_ineq))
        eq,le = closed_uid_tables(closed)
        eq = np.r_[eq,np.arange(first,first+ne,dtype=np.int64)]
        le = np.r_[le,np.arange(first+ne,first+ne+nl,dtype=np.int64)]
        actual = incidence_oracle(post,eq,le,closed.old_n_cont,closed.logical_n_cont,pool=pool)
        columns, checked = verify_all_and_discover(closed,post,overlay,actual,eq,le,
            whole=pool,branch=BranchPool(pool))
        if not np.array_equal(columns,saved['unit_columns']) or len(columns) != 268:
            raise ValueError('restored actual rows changed complete unit-pair discovery')
        report.update(completed=True, closed_identity=closed.fingerprint(),
            actual_post_HZ_sha256=source_digest(post), all_frozen_source_hashes_checked=True,
            no_definition_graph_restored=True, source_sharing_and_all_numeric_content_bound=True,
            actual_cache_and_owner_aliases_preserved=True, portable_receipts_not_deserialized=True,
            original_C10_live_reference_leaf_hashes_equal=True,
            MAIN_owner_words=len(closed.owners),uid_slab_words=len(closed.uid_slabs),
            sparse_event_words=len(overlay.events),complete_actual_incidence_proof=checked,
            independent_diagnostic_work=pool.used, all_268_unit_columns_rediscovered=True,
            new_generator_executed=False,new_solver_executed=False,new_native_relu_executed=False)
    except Exception as exc:
        report['failure'] = {'type':type(exc).__name__,'reason':str(exc)}
    finally:
        report.update(wall_s=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            input_sha256=INPUTS,source_sha256=freeze['guard_source_sha256'],provenance=freeze['provenance'])
        _atomic_exclusive_json(DIRECTORY/'result.json',report)
        print(json.dumps(report),flush=True)
    if not report['completed']: raise SystemExit(1)


if __name__ == '__main__': main()
