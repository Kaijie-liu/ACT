"""Authenticated real local-edge reader and full-consumer unavoidable bound."""
from fractions import Fraction as F
import json
from pathlib import Path
import resource
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
from experiments.neural_hz_20260831.c41_observed_checkpoint_gate_v1 import measured
from experiments.neural_hz_20260831.c65_physical_archive_v1 import check_restored, metadata
from experiments.neural_hz_20260831.c62_physical_measure_v1 import numeric_layout
from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
from experiments.neural_hz_20260831.c23_sparse_phase_overlay_v1 import Overlay
from experiments.neural_hz_20260831.c68_local_splice_v1 import compile_journal
from experiments.neural_hz_20260831.c62_local_equations_v1 import SCHEMA
from experiments.neural_hz_20260831.run_tiny143_bn_graph_faithfulness_audit_v2 import _atomic_exclusive_json, _sha256
EXP = Path(__file__).resolve().parent
RUN = EXP / 'results/c68_local_splice_20260913_v1'
SOURCE = EXP / 'results/c67_direct_csr_20260913_v1/actual/physical_hz.pickle'
SOURCE_SHA = '9f52e4a4598ac25e63c945cbe88556dfa09e4aaba688a203841c21a534e6703f'


def main():
    resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
    freeze = json.loads((RUN / 'preregistered.json').read_text())
    if any(_sha256(EXP / name) != sha for name, sha in freeze['source_sha256'].items()):
        raise ValueError('frozen source drift before actual reader')
    pool = WorkPool(256_000_000)
    started = time.monotonic()
    result = dict(completed=False, formal_gain=0, native_or_LIVE_admission=False,
                  actual_native_unit_population_executed=False, original_network_started=False)
    try:
        def check():
            with SOURCE.open('rb') as handle:
                saved, decoder = load(handle, expected_sha256=SOURCE_SHA, pool=pool, enabled=True)
            fields = saved['fields']
            old_layout = numeric_layout(saved, pool)
            pool.charge('c68_full_physical_source_proof_binding', int(old_layout.resident_entries) + 1024)
            proof = check_restored(saved)
            if (proof['physical_identity'] != '261c4bd4af99b859058b83a27d482eeac26dd266dfa15f7a5bce0576c697aeec'
                    or saved['proof_sha256'] != '4c8766034fd094beb90a0297a9f713df486d72e2b445b55e9457c78cad851676'):
                raise ValueError('independent complete source proof identity differs')
            hz = fields['hz']
            j = compile_journal(fields['eq_roots'], fields['eq_scales'], [],
                old_n_cont=fields['old_n_cont'], old_n_eq=fields['old_n_eq'],
                source_n_cont=hz.n_cont, source_schema=SCHEMA, pool=pool, enabled=True)
            pool.charge('c68_diagnostic_full_frame_input_values', 16 * hz.n_cont)
            point = [F(i % 5 - 2, 5) for i in range(hz.n_cont)]
            full = j.reconstruct_fraction(hz, point, pool=pool)
            old, oe = fields['old_n_cont'], fields['old_n_eq']
            pool.charge('c68_independent_original_input_comparison', 8 * old)
            if full[:old] != point[:old]:
                raise ValueError('original input latent coordinates changed')
            checked = 0
            roots, scales = fields['eq_roots'], fields['eq_scales'].view(np.float64)
            pool.charge('c68_independent_all_local_relation_routing', 8 * (len(roots) - oe))
            for at in range(oe, len(roots)):
                if roots[at] >= 0:
                    continue
                pool.charge('c68_independent_exact_local_relation', 128)
                bits = int(roots[at]) + (1 << 64)
                parent, q = bits & ((1 << 32) - 1), (bits >> 32) & 63
                col = old + at - oe
                if F(2) ** q * full[col] != F(float(scales[at])) * full[parent]:
                    raise ValueError('actual full local equation inverse differs')
                checked += 1
            if checked != proof['proof']['local_inverse_equations']:
                raise ValueError('incomplete real inverse population')
            overlay = Overlay(fields['owners'], np.empty(0, np.uint64), fields['report']['radix_uid_base'] + 16384)
            for at, actual in enumerate(j.iter_words(overlay, pool=pool)):
                pool.charge('c68_independent_owner_reader_comparison', 8)
                if actual != int(fields['owners'][at]):
                    raise ValueError('source owner reader changed exact incidence')
            complete = dict(source=saved, journal=vars(j))
            layout = numeric_layout(complete, pool)
            meta = metadata(complete, pool=pool)
            if layout.resident_entries > 64_000_000:
                raise MemoryError('complete source/journal entry cap')
            if j.eq_roots is not fields['eq_roots'] or j.eq_scales is not fields['eq_scales']:
                raise ValueError('complete source maps were copied')
            # This reads the ACTUAL source geometry, never an old unit-count.
            pool.charge('c68_full_consumer_lower_bound', 128)
            source_rows = hz.n_eq + hz.n_ineq
            routing = 12 * source_rows
            actual_generation, conservative_generation = 255937119, 255946971
            bound = dict(all_existing_physical_EQ=hz.n_eq, all_existing_physical_INEQ=hz.n_ineq,
                all_existing_consumer_rows=source_rows, unchanged_header_plus_routing_price=12,
                mandatory_consumer_work_lower_bound=routing,
                combined_actual_generation_plus_lower_bound=actual_generation + routing,
                combined_conservative_generation_plus_lower_bound=conservative_generation + routing,
                unchanged_whole_cap=256_000_000, remaining_actual=256_000_000 - actual_generation,
                remaining_conservative=256_000_000 - conservative_generation,
                complete_population_fits_even_before_other_costs=conservative_generation + routing <= 256_000_000,
                appended_rows_events_writer_journal_and_publication_NOT_counted_in_lower_bound=True,
                not_a_complete_integrated_cost=False)
            # A lower bound can reject, never admit, an unexecuted integration.
            bound['not_a_complete_integrated_cost'] = True
            if bound['complete_population_fits_even_before_other_costs']:
                raise ValueError('lower bound does not decide integration; full derivation still required')
            return dict(source_archive_sha256=SOURCE_SHA, source_physical_identity=proof['physical_identity'],
                decoder=decoder, local_equations_reconstructed=checked, original_input_latents_unchanged=old,
                all_MAIN_owner_words_checked=len(fields['owners']), global_n_cont=hz.n_cont, binaries_retained=hz.n_bin,
                source_maps_shared_unchanged=True, native_unit_journal_entries=0,
                empty_journal_is_reader_test_NOT_actual_native_population=True,
                source_plus_reader_numeric_bytes=layout.resident_bytes, source_plus_reader_entries=layout.resident_entries,
                metadata=meta, consumer_lower_bound=bound, integration_rejected_before_expensive_run=True,
                concrete_network_witness=False)
        data, stats = measured(check, observe=lambda s: result.update(measurement=s))
        result.update(completed=True, data=data, status='SOURCE_READER_PROVED_INTEGRATION_COST_REJECTED')
    except Exception as exc:
        result['failure'] = dict(type=type(exc).__name__, reason=str(exc))
    finally:
        result.update(wall_s=time.monotonic() - started, diagnostic_work=pool.used, work_parts=dict(pool.parts))
        _atomic_exclusive_json(RUN / 'source_result.json', result)
        print(json.dumps(result), flush=True)
    if not result['completed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
