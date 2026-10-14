"""Static adapter/lifetime DESIGN ledger, not a parser or execution supervisor.

Only source files, pinned metadata and integer formulas are read/evaluated.
No checkpoint, dataset, numerical package, worker or solver is loaded.
Parameterized storage bounds do not close Python/native RSS or elapsed time.
"""
import hashlib
import json
from math import prod
from pathlib import Path

from scripts.assess_runtime_lifetime import ENV, ROOT, analyze as prior_analysis

ANCHORS = {
    'docs/hz_runtime_lifetime_20261014.json': '67e30dafc4bd62ef7b425482ff384699ee11105822bac1d9de73c4f1b9425418',
    'scripts/assess_runtime_lifetime.py': 'ca48bca0f4bb3e9603ff984c906d1179ebe29e07c856bba6f90081be466859d4',
    'scoped_source/capture.py': 'd3ca9b562c9840ace096d790d0fd6eb12d7105560a3c38767c85244c65f651e2',
    'scoped_source/wire_archive.py': 'fdf0b0e97522527831a6f03f1c7188403bc8b7b844c3a9df092823e703295abd',
    'scoped_proof/supervisor.py': 'a935cf6bf03d3604e2e2ce581939ca8c9ba120715ef8f289ba3fde59f7a8f14c',
    'scoped_source/endpoint_verify.py': '17298b713a7e3039e8c7fc198904577dd48b46aadc125f95d8e1cb6f6cd31a90',
    'act/pipeline/moe/staged_verifier.py': '8361e88c5f0b609614af954117a457a26287be274fb3539e26fd252cf64fe871',
}
ENV_ANCHORS = {
    'lib/python3.12/site-packages/torch/serialization.py': '7fe6755ba65409b510e75e0435a33f444c893a82ea817ff618a5be5943298a3c',
    'lib/python3.12/site-packages/torch/_utils.py': 'b52db634d2378395685499de1fbb2b3a1bef21bb87677d830792179b492edc75',
    'lib/python3.12/site-packages/torchvision/datasets/cifar.py': '0a2d6786e8e9d1e9bc7db0664d0185bc14fb22c6733e8ee6dde750038b6ed835',
    'lib/python3.12/site-packages/torchvision/transforms/functional.py': '0b2fdf8e6764f4be0113acd95b58d3b14158c13fa2374ed93fbbe1f4b8730ded',
    'lib/python3.12/pickle.py': '7e3447b6d993a6306a7893fd2fba916010ed79f8e09ceb56158a550fa8099a14',
}
PARAMETERS = 6961368
INPUTS = 3072
DATA_ROWS = 10000
DATA_BLOB = DATA_ROWS * INPUTS


def natural(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError('invalid static parameter: '+name)
    return value


def tensor_roster():
    """Shape-only expected factory directory; no model instantiation."""
    result = []
    for prefix, widths in [('router', (3072, 128, 8))]+[
            (f'experts.{i}', (3072, 256, 128, 10)) for i in range(8)]:
        for layer, (left, right) in enumerate(zip(widths, widths[1:])):
            for role, shape in [('weight', [right, left]), ('bias', [right])]:
                result.append(dict(name=f'{prefix}.{2*layer+1}.{role}', shape=shape))
    return sorted(result, key=lambda row: row['name'])


def snapshot_cost(checkpoint_bytes, dataset_bytes, store_bytes):
    """New private snapshots plus SAME final store; no old JSON byte replay.

    Each snapshot has one bounded copy/hash pass, followed by one verification
    scan over that same private object. Parser/gather reads are EXTRA below.
    This is logical transfer, not filesystem blocks, syscalls or elapsed time.
    """
    f = natural(checkpoint_bytes, 'checkpoint_bytes')
    r = natural(dataset_bytes, 'dataset_bytes')
    b = natural(store_bytes, 'store_bytes')
    return dict(snapshot_written_bytes=f+r, snapshot_full_scan_read_bytes=2*(f+r),
        snapshot_hash_bytes=2*(f+r), extra_logical_storage_upper=f+r,
        attempt_store_plus_both_snapshots_upper=b+f+r,
        snapshot_lifetimes_sequential=False, removal_cost_counted=True,
        parser_and_gather_reads_included=False, source_path_immutable_assumed=False,
        private_snapshot_no_writer_required=True, stat_is_immutability_proof=False,
        actual_snapshot_bytes=None, old_archive_cap_increased=False)


def parser_arenas(metadata_bytes):
    """Conditional fixed-record arena design, NOT an implemented parser bound.

    Bytes/string contents remain spans in a private source. IDs never recursively
    expand memo aliases. Each opcode consumes >=1 metadata byte; 4 edges/op is
    a conservative target invariant to establish for the admitted grammar.
    Neither a Python dict nor a byte-derived preallocation is implemented here.
    """
    p = natural(metadata_bytes, 'metadata_bytes')
    if 313*p > 2**64-1:
        raise ValueError('proposed uint64 arena/offset format cannot address this metadata')
    components = dict(node_records=128*p, opcode_records=32*p, stack_and_marks=16*p,
        memo_bindings=16*p, graph_edges=16*4*p, dfs_colors=p, dfs_frames=24*p,
        sort_scratch=32*p)
    return dict(metadata_bytes=p, opcodes_upper=p, nodes_upper=p,
        memo_entries_upper=p, stack_entries_upper=p, graph_edges_upper=4*p,
        typed_components=components, typed_payload_upper=sum(components.values()),
        max_memo_id_is_allocation_size=False, byte_strings_copied=False,
        aliases_expanded=False, arbitrary_reduce_executed=False,
        dictionary_duplicate_keys='reject; never last-wins',
        repeated_memo_put='reject; GET must refer to an earlier PUT',
        cycles='reject with iterative ID-graph walk',
        implementation_correspondence_proved=False, complete_parser_runtime_upper=None)


def capture_counts(checkpoint_bytes, pickle_bytes, dataset_bytes):
    """Finite byte/access schedule under the restricted profile.

    F and R are future frozen *file identities*, P a checked data.pkl span.
    R-B counts metadata bytes only after validating the SINGLE raw uint8 blob.
    Counts are not a total CPU/native/IO operation bound or new resource quota.
    """
    f = natural(checkpoint_bytes, 'checkpoint_bytes')
    p = natural(pickle_bytes, 'pickle_bytes')
    r = natural(dataset_bytes, 'dataset_bytes')
    if p > f or r <= DATA_BLOB or max(f,r) > 2**64-1:
        raise ValueError('invalid file/span shape relationship')
    k = r-DATA_BLOB
    return dict(checkpoint_metadata=parser_arenas(p), input_metadata=parser_arenas(k),
        sequential_parser_payload_upper=max(parser_arenas(p)['typed_payload_upper'],
                                            parser_arenas(k)['typed_payload_upper']),
        checkpoint_directory_entries_upper=52+6,
        tensor_descriptors_upper=52, expected_module_metadata_entries=63,
        descriptor_bytes_upper=256*52,
        source_numeric_output_bytes=8*(PARAMETERS+INPUTS),
        checkpoint_gather_value_reads_upper=2*PARAMETERS,
        checkpoint_gather_stride_terms_upper=4*PARAMETERS,
        checkpoint_gather_requested_bytes_upper=16*PARAMETERS,
        checkpoint_metadata_decode_reads_upper=2*p,
        span_comparison_rereads_included=False,
        metadata_reread_alternative='owned raw P/K bank is additional payload; no free caching',
        # Two independently validated directories; includes malformed overlap
        # prefixes, not only structurally valid nonoverlapping ZIP archives.
        checkpoint_directory_read_bytes_upper=2*((2*(52+6)+5)*f+65557),
        dataset_metadata_decode_reads_upper=2*k,
        selected_input_reads_upper=2*INPUTS,
        labels_validated_upper=2*DATA_ROWS,
        transformed_parameter_values_upper=2*PARAMETERS,
        independently_read_generated_source_bytes=8*(PARAMETERS+INPUTS),
        historical_parameter_hash_payload_bytes=16*PARAMETERS,
        historical_center_hash_payload_bytes=16*INPUTS,
        hash_framing_bytes_additional=True,
        direct_format_hashes='separate prior direct-format ledger; not these historical contexts',
        checkpoint_crc_payload_bytes_upper=f,
        conversion_and_independent_source_binding_passes=2,
        max_tensor_rank=2, requested_read_bytes_are_physical_io=False,
        parser_arenas_overlap=False, parser_release_is_design_condition=True,
        parser_release_requires='bounded independent descriptors; no outstanding borrowed references',
        storage_aliases_or_noncontiguous_views_rejected_by_default=False,
        storage_range_checked_before_access=True, complete_capture_runtime_upper=None)


def input_contract():
    literal_bits = (10**4096-1).bit_length()
    record = 16+2*((literal_bits+7)//8)
    return dict(dataset_rows=DATA_ROWS, raw_data_blob_bytes=DATA_BLOB,
        selected_row_bytes=INPUTS, materialized_center_bytes=8*INPUTS,
        proposed_rn64_table_bytes=256*8, rn64_table_integer_bits_upper=68,
        center_exact_numerator_bits_upper=53, center_exact_denominator_bits_upper=61,
        request_literal_characters=4096, request_integer_bits=8192,
        string_rational_component_bits_upper=literal_bits,
        one_literal_record_bytes_upper=record, four_literal_record_bytes_upper=4*record,
        literal_provenance_text_bytes_upper=4*4096,
        center_plus_literal_unreduced_bits=[max(53+literal_bits,literal_bits+61)+1,61+literal_bits],
        clipping_compare_product_bits_upper=literal_bits+61+literal_bits+1,
        clipping_compare_scope='literal clip versus center plus/minus radius only',
        constructed_endpoints_compare_product_bits_upper=2*(literal_bits+61)+1,
        preserve_original_literal_type_and_value=True,
        float_radius_replaced_with_rational_two_over_255=False,
        center_replaced_with_exact_p_over_255=False,
        frozen_totensor_bit_equivalence_proved=False,
        native_256_value_control_run=False, new_request_bit_gate=None,
        input_scan_free=False, selected_dataset_index=None)


def worker_contract():
    phases = ['SOURCE','PRODUCTION','PORTABLE']
    return dict(parent='stdlib-only, metadata/FDs/counts/hash states; no numerical imports',
        workers=phases, fresh_exec=True, concurrent_worker_upper=1,
        process_descendants='not admitted without separately established ownership/no-fork contract',
        native_threads_upper=None, no_native_fork_proved=False,
        transitions=['INIT','SOURCE','QUIESCE_SOURCE','RECEIVE_SOURCE',
            'PRODUCTION','QUIESCE_PRODUCTION','RECEIVE_CANDIDATES',
            'PORTABLE','QUIESCE_PORTABLE','RECEIVE_RESULT','CLEANUP','PUBLISH_TERMINAL'],
        failure=['STOP_OWNED_WORK','QUIESCE','PRESERVE_PREFIX','NONACCEPTING_TERMINAL'],
        one_absolute_deadline_seconds=300, deadline_reset=False, quota_reset=False,
        production_native_and_exact='sum with overlapping temporaries, never max',
        conditional_process_rss_formula='parent + max(source, native+exact+overlap, portable, reception)',
        phase_nonoverlap_condition='all worker threads/allowed descendants exited; worker-owned FD/mapping references released',
        machine_page_cache_excluded_from_process_rss=True,
        full_worker_lifetime_established=False, peak_process_rss_upper=None,
        os_cleanup_latency_upper=None, cleanup_overrun='count and reject, never backdate acceptance',
        timeout_limit_implies_completion=False, parent_receives_traceback_objects=False,
        error_text='bounded codes and checked spans; not str(exc) followed by slicing',
        missing_partial_or_late_evidence_can_accept=False,
        later_offline_audit_can_complete_online_result=False,
        current_capacity_entrypoint='leader-only observation/wait; unchanged',
        stronger_existing_entries=['scoped_source/wire_archive.py:run_child',
                                  'scoped_proof/supervisor.py:execute'],
        process_group_sample_proves_containment=False, implementation_changed=False)


def verify_files():
    for root, anchors in ((ROOT, ANCHORS), (ENV, ENV_ANCHORS)):
        for name, expected in anchors.items():
            if expected is None or hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise ValueError('changed adapter design anchor: '+name)


def analyze():
    verify_files()
    prior = prior_analysis()
    roster = tensor_roster()
    if len(roster) != 52 or sum(prod(r['shape']) for r in roster) != PARAMETERS:
        raise ValueError('registered tensor roster drift')
    return dict(schema='SOURCE_ADAPTER_LIFETIME_DESIGN_V1', date='2026-10-14',
        based_on_head='f395e899de4bde927ddda775e387b43b2f0681f6',
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        anchors=ANCHORS, environment_anchors=ENV_ANCHORS,
        tensors=roster, tensor_count=len(roster), parameter_values=PARAMETERS,
        binary64_parameters_bytes=8*PARAMETERS,
        source_identity=dict(new_protocol='ACT_MOE_DIRECT_UNIQUE_ROOT_DRAFT_V1',
            old_json_replayed=False, repeated_view_roots_recounted_as_values=False,
            historical_checkpoint_and_state_bindings_exist=True,
            selected_live_checkpoint=None, selected_input=None,
            native_checkpoint_dtype_and_promoted_state_hashes_distinct=True,
            restricted_parser_implemented=False, real_format_compatibility_verified=False,
            permitted_storage_dtypes=['binary32','binary64'],
            tensor_rebuild_metadata='absent/empty or false neg/conj only; empty hooks; no arbitrary tensor BUILD',
            nonfinite_weights='reject', native_program_equivalence_claim=False),
        parameterized_costs=dict(frozen_checkpoint_bytes=None, frozen_data_pickle_bytes=None,
            frozen_dataset_bytes=None, metadata_arena='313*P; 313*K where K=R-30720000',
            capture_peak_named_payload='max(313*P,313*K) + bounded byte banks/descriptors/buffers',
            snapshot_added_storage='F+R', total_store_upper='A_width+F+R',
            snapshot_reads='2*(F+R), excluding parser/gathers', snapshot_hashes='2*(F+R)',
            gather_read_bytes='16*6961368 for two passes, including noncontiguous values',
            random_access_calls_upper='2*6961368; not physical block traffic',
            malformed_inclusive_zip_reads='2*((2*S+5)*F+65557), S<=58, two checks',
            source_bytes_independently_read='8*(6961368+3072)',
            parser_cpu_comparisons='sorting/memo/name comparisons separately charged; no linear-time claim',
            parameterized_bounds_are_not_quotas=True, actual_source_bytes_read=0),
        input=input_contract(), workers=worker_contract(), invariant=prior['invariant'],
        current_policy_unchanged=prior['current_policy_unchanged'],
        closure=dict(logical_source_lifetime_contract='SPECIFIED_NOT_IMPLEMENTED',
            adapter_typed_arenas='CONDITIONAL_FORMULAS_NOT_RUNTIME_PROOF',
            exact_runtime_scratch='OPEN', native_dispatch_temporaries_threads='OPEN',
            source_profile_and_promotion_equivalence='OPEN', worker_containment='OPEN',
            full_runtime_upper=None, full_cost_upper_closed=False),
        production_changed=False, implementation_authorized=False, new_quota=None,
        execution_admitted=False, mathematical_checks_removed=False,
        model_runs=0, dataset_loads=0, checkpoints_loaded=0, gpu_calls=0, solver_calls=0,
        status='RESTRICTED_ADAPTER_AND_LIFETIME_DESIGN_ONLY_NO_COST_ADMISSION')


if __name__ == '__main__':
    print(json.dumps(analyze(), indent=2, sort_keys=True))
