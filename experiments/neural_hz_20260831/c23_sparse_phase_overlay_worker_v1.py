"""One sealed-source complete phase-overlay diagnostic; no fresh NN execution."""

import hashlib
import json
import os
from pathlib import Path
import pickle
import resource
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch

from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import build
from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import (
    BranchPool, check_append, incidence_oracle, verify_all_and_discover)
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c10_fused_emission_v1 import FusedIntegrated
from experiments.neural_hz_20260831.c7_factored_hz_v1 import expression_binding
from experiments.neural_hz_20260831.c10_portable_binding_v1 import identity
from experiments.neural_hz_20260831.c16_box_transfer_v1 import receipt_from_archive
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import row_uid_tables
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c5_functional_transaction_v1 import measured_build
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256

EXP = Path(__file__).resolve().parent
DIRECTORY = EXP / 'results/c23_sparse_phase_overlay_20260911_v1'
FUSED = EXP / 'results/c10_fused_emission_20260908_v1/fused_hz.pickle'
PROOF = FUSED.parent / 'result.json'
PHASE = EXP / 'results/c10_live_relu_20260908_v1/relu78.pickle'
BINDING = PHASE.parent / 'proof_binding.json'
TABLE = EXP / 'results/c14_early_rejection_census_20260910_v1/single_use_factor_table.npz'
INPUT_HASHES = {
    FUSED: '24b22d8a06c0c4d51525dd9d5ee0902941c40f20cc6a711e6231673e3efa9d08',
    PROOF: '315a152e3910b8340f5971be8346434a8d81e5106cd5f67321deea8def29fd5b',
    PHASE: '1c242085191040cdea7db9975aba5a8c2134e4b2776213cfee2c99a7e3ec1c9d',
    BINDING: '7358034b062d0561ea77d08e086aa99e6af421f0e27062605df7d1dd7368723a',
    TABLE: '0208ecdb11a35896c1ecb86a6fabb1c1e9f3c49bf7b47d6248a18d52c6e3328e'}


def array_sha(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    freeze = json.loads((DIRECTORY / 'preregistered.json').read_text())
    record = {'schema': 'c23_sparse_phase_overlay_actual_v1', 'completed': False, 'formal_gain': 0,
        'new_phase_executed': False, 'generator_executed': False, 'unit_splice_executed': False,
        'graph_fields_retired': False, 'solver_executed': False, 'native_ingestion_executed': False,
        'whole_live_path_proved': False, 'source_sha256': freeze['source_sha256'],
        'provenance': freeze['provenance']}
    started = time.monotonic()
    with (DIRECTORY / 'events.jsonl').open('x') as events:
        def emit(name, payload):
            event = {'event': name, **payload, 'worker_elapsed_s': time.monotonic() - started}
            events.write(json.dumps(event) + '\n')
            events.flush()
            print(json.dumps(event), flush=True)
        try:
            if any(_sha256(EXP / n) != sha for n, sha in freeze['source_sha256'].items()):
                raise ValueError('frozen source/artifact drift')
            if any(_sha256(p) != sha for p, sha in INPUT_HASHES.items()):
                raise ValueError('sealed successful source drift')
            with FUSED.open('rb') as handle:
                fused_saved = pickle.load(handle)
            with PHASE.open('rb') as handle:
                phase_saved = pickle.load(handle)
            if (fused_saved['schema'] != 'c10_fused_emission_checkpoint_v1'
                    or phase_saved['schema'] != 'c10_live_relu_checkpoint_v1'):
                raise ValueError('unexpected diagnostic checkpoint schema')
            fields = fused_saved['fields']
            candidate = FusedIntegrated(**fields, origin_binding=expression_binding(fields['expression']))
            candidate.seal = candidate.fingerprint()
            before = identity(candidate)
            receipt = receipt_from_archive(candidate, PROOF.read_bytes(), BINDING.read_bytes(),
                proof_sha256=INPUT_HASHES[PROOF], binding_sha256=INPUT_HASHES[BINDING])
            post = phase_saved['post_relu_hz']
            post_sha = source_digest(post)
            if post_sha != '82df62f1233ca8f34b5163ee3fafa88ae0afc65a9f36fe5b1b7f3b022cca2367':
                raise ValueError('complete post-HZ content drift')
            check_append(candidate.hz, post)
            authentication_s = time.monotonic() - started
            emit('complete_source_proof_and_post_prefix_bound', {'authentication_and_load_s': authentication_s,
                'full_fused_and_phase_dictionaries_retained': True})
            main = candidate.logical_n_cont - candidate.old_n_cont
            width = sum(n['width'] for n in candidate.nodes)
            radix_base = candidate.old_n_eq + len(candidate.ineq_roots) + width
            first = radix_base + 16384
            holder = SimpleNamespace(**vars(candidate))
            holder.report = {**candidate.report, 'radix_uid_base': radix_base}
            pre = candidate.hz
            ne, nl = post.n_eq - pre.n_eq, post.n_ineq - pre.n_ineq
            if first + ne + nl > 2**20:
                raise ValueError('fresh phase UID range exceeds fixed domain')
            whole = WorkPool(256_000_000)
            branch = BranchPool(whole)
            def execute():
                whole.charge('complete_uid_metadata', 32 * (width + main + pre.n_eq + pre.n_ineq))
                eq, le = row_uid_tables(holder)
                base = incidence_oracle(pre, eq, le, candidate.old_n_cont, candidate.logical_n_cont, pool=whole)
                base.flags.writeable = False
                base_sha = array_sha(base)
                # Price materializing only appended row blocks, never a second
                # full owned vector. Full pre/post oracles remain reachable.
                added_nnz = int(post.Ac.nnz - pre.Ac.nnz + post.Auc.nnz - pre.Auc.nnz)
                whole.charge('diagnostic_appended_CSR_slices', 8 * added_nnz + 4 * (ne + nl))
                overlay, report = build(base, [(post.Ac[pre.n_eq:], first), (post.Auc[pre.n_ineq:], first + ne)],
                    old_n_cont=candidate.old_n_cont, old_uid_ceiling=first, pool=branch, enabled=True)
                build_work = branch.used
                build_parts = dict(branch.parts)
                event_sha = array_sha(overlay.events)
                whole.charge('complete_post_UID_arrays', 2 * (len(eq) + len(le) + 2 * (ne + nl)))
                eq = np.r_[eq, np.arange(first, first + ne, dtype=np.int64)]
                le = np.r_[le, np.arange(first + ne, first + ne + nl, dtype=np.int64)]
                actual = incidence_oracle(post, eq, le, candidate.old_n_cont, candidate.logical_n_cont, pool=whole)
                # ALL random-access queries, not a selected positive subset.
                random_work = main * (16 * max(1, len(overlay.events).bit_length()) + 16) + 12 * len(overlay.events)
                if whole.used + random_work > whole.cap or branch.used + random_work > branch.cap:
                    raise MemoryError('full random-access proof cannot fit unchanged caps')
                for index, expected in enumerate(actual):
                    if overlay.query(index, pool=branch) != int(expected):
                        raise ValueError('random-access overlay differs from actual post incidence')
                columns, checked = verify_all_and_discover(candidate, post, overlay, actual, eq, le,
                    whole=whole, branch=branch)
                # Historical admissible rows are loaded ONLY after discovery.
                with np.load(TABLE, allow_pickle=False) as table:
                    expected_columns = table['column'][table['individually_admissible']]
                    if len(expected_columns) != 268 or not np.array_equal(columns, expected_columns):
                        raise ValueError('complete overlay discovery does not recover ALL C15 unit pairs')
                overlay.validate()
                if array_sha(base) != base_sha or array_sha(overlay.events) != event_sha:
                    raise ValueError('base or sparse event owner mutated during complete proof')
                if main != 243162 or ne != 200 or nl != 400 or len(overlay.events) != 200:
                    raise ValueError('complete registered population differs; no partial acceptance')
                report.update(checked, all_MAIN_random_access_queries_checked=main,
                    all_268_C15_columns_match=True, new_EQ_rows=ne, new_INEQ_rows=nl,
                    independent_post_words_sha256=array_sha(actual), original_base_words_sha256=base_sha,
                    event_words_sha256=event_sha, old_uid_ceiling=first,
                    standalone_build_work=build_work, standalone_build_work_parts=build_parts,
                    whole_diagnostic_work=whole.used, whole_diagnostic_work_parts=dict(whole.parts),
                    overlay_branch_work=branch.used, overlay_branch_work_parts=dict(branch.parts),
                    strict_additional_owner_entries_decrease=len(overlay.events) < main,
                    replaced_dense_phase_copy_entries=main, replaced_dense_phase_copy_bytes=8 * main,
                    independent_source_proof_sha256=receipt.proof_sha256,
                    source_portable_identity=before, complete_post_HZ_sha256=post_sha)
                # Dense post oracle/lookup/UID arrays are temporary proof input,
                # as in C21; not hidden returned algorithm state.
                return overlay, columns, report
            (overlay, columns, report), measurement = measured_build(execute)
            if identity(candidate) != before or source_digest(post) != post_sha:
                raise ValueError('diagnostic mutated original HZ/graph/source/lineage')
            report.update(full_fused_and_phase_dictionaries_retained=True, all_sources_unchanged=True,
                authentication_and_load_s=authentication_s, diagnostic_only=True,
                original_graph_fields_retired=False, whole_state_reduction_claimed=False,
                source_authentication_is_not_free_generation_work=True)
            with (DIRECTORY / 'sparse_overlay.npz').open('xb') as handle:
                np.savez(handle, base=overlay.base, events=overlay.events)
                handle.flush(); os.fsync(handle.fileno())
            with (DIRECTORY / 'unit_columns.npz').open('xb') as handle:
                np.savez(handle, columns=columns)
                handle.flush(); os.fsync(handle.fileno())
            _atomic_exclusive_json(DIRECTORY / 'overlay_audit.json', report)
            emit('complete_overlay_proof_passed', {'event_count': len(overlay.events),
                'all_MAIN_checked': main, 'unit_pairs': len(columns), 'measurement': measurement})
            record.update(completed=True, report=report, measurement=measurement)
        except Exception as exc:
            record['failure'] = {'type': type(exc).__name__, 'reason': str(exc)}
        finally:
            record.update(wall_s=time.monotonic() - started, max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            _atomic_exclusive_json(DIRECTORY / 'result.json', record)
            print(json.dumps({'completed': record['completed'], 'failure': record.get('failure'), 'formal_gain': 0}), flush=True)
    if not record['completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
