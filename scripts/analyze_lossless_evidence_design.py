"""Static proposed encoding envelope, NOT a codec or execution protocol.

No source arrays, model, numerical library, native kernel or solver is loaded.
Record widths/reserves below are new DESIGN choices, not existing limits.
"""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.analyze_full_resource_ledger import analyze as parent_analysis, ROOT

PARENT = 'scripts/analyze_full_resource_ledger.py'
PARENT_SHA = 'ae9e11ebf873924e48c7ba89bf42b60aa1968cbf991d839f8b97b67e26ec28fc'
MIB = 2**20


def envelope(v, *, copies=1):
    if type(copies) is not int or copies not in (1, 2):
        raise ValueError('explicit single or two complete retained copies')
    if (v['router_widths'] != [3072, 128, 8] or
            v['expert_widths'] != [3072, 256, 128, 10]):
        raise ValueError('registered sequential topology only')
    n, w1 = v['inputs'], v['first_weight_positions']
    f, nodes, local, refs, roots, views = (v[k] for k in
        ('factors_upper', 'nodes_upper', 'local_entries_upper', 'references_upper',
         'roots_upper', 'views'))
    e, m, d = (v[k] for k in ('endpoints_upper', 'pair_dual_rows_upper', 'property_duties'))
    first_rows = w1//n
    # F64 fields represent the same exact dyadic value, NEVER a rounded
    # arbitrary rational. The sequential-output injectivity lemma is needed
    # for references; the existing checker already requires F64 local/root/scale.
    forest = dict(factors=256*f, nodes=64*nodes, source_row_descriptors=64*first_rows,
        remaining_local_entries=16*(local-w1), references=16*refs,
        logical_roots=16*roots, scales=8*f, views=512*views,
        header_roles_source_receipt_reserve=MIB)
    # Both roles remain: proposed is dense F64; zero is an explicit length-
    # and row-bound constant-vector encoding, independently traversed/checked.
    proof = dict(proposed_dual_values=8*e*m, endpoint_headers=512*e,
        manifest_reserve=MIB//4)
    # Checked output bounds are not generally F64. With <=8192-bit numerator
    # and denominator, 2064 bytes covers two length-delimited integer limbs.
    receipts = dict(endpoint_records=e*(1024+3*2064),
        duty_records=d*(128+2064), gate_records=v['pairs']*(64+2*2064),
        manifest_reserve=MIB//4)
    components = dict(source=v['raw_binary64_tensor_bytes']+4*MIB,
        forest=sum(forest.values()), raw_proof=sum(proof.values()),
        exact_receipts_reserve=8*MIB, checker_code_execution_metadata_reserve=8*MIB,
        bounded_logs_failure_progress_terminal_reserve=4*MIB)
    total = copies*sum(components.values())
    return dict(kind='PROPOSED_RECORD_AND_RESERVE_ENVELOPE_NOT_IMPLEMENTED',
        copies=copies, forest_components=forest, proof_components=proof,
        exact_receipt_payload_upper=sum(receipts.values()), exact_receipt_components=receipts,
        components_per_copy=components, total_bytes_upper=total,
        retained_archive_cap_bytes=128*MIB, headroom_bytes=128*MIB-total,
        within_retained_cap=total<=128*MIB,
        conclusion=('DRAFT_SINGLE_COPY_BYTES_FIT_NOT_EXECUTION_ADMISSION' if total<=128*MIB else
                    'THIS_DRAFT_HAS_NO_TWO_COPY_CAPACITY_CONTRACT_NOT_ACTUAL_OVERFLOW'),
        logical_dual_values_checked=2*e*m, candidate_roles=2*e,
        dense_two_role_f64_payload=16*e*m,
        partial_final_layout='same immutable candidate tiles referenced by final manifest; no copied full proof',
        relocation_layout='one canonical package; no assumption of free simultaneous physical copy',
        reserves_are_proposed_not_measured=True, production_caps_changed=False,
        implementation_admitted=False, execution_admitted=False)


def analyze():
    if hashlib.sha256((ROOT/PARENT).read_bytes()).hexdigest() != PARENT_SHA:
        raise ValueError('parent ledger identity changed')
    p = parent_analysis(); v = p['inventory']
    return dict(schema='LOSSLESS_EVIDENCE_DESIGN_STATIC_V1', date='2026-10-14',
        reviewed_head='74c2b609caf14e4ba6c098b0ce95b3a616c731f5',
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        code_and_metadata={**p['code_and_metadata'],PARENT:PARENT_SHA},
        inventory=v, single=envelope(v), two_retained=envelope(v,copies=2),
        exactness_contract=dict(
            supported_source='registered sequential Flatten/Linear/ReLU captured binary64 source',
            output_node_injectivity_required=True,
            references_f64_only_after_independent_sequential_source_check=True,
            source_signed_zero_bits_preserved=True,
            generic_rational_to_float_forbidden=True,
            checked_bounds_encoding='canonical signed integers; unchanged 8192-bit intermediate bound',
            auxiliary_tails_explicit=True, all_logical_roots_counted=True,
            all_explicit_zero_values_checked=True, both_candidates_required=True,
            no_assumed_sparse_proposed_duals=True,
            wire_identity_distinct_from_logical_identity=True),
        native_named_payload=dict(
            scope='named CPU tensor payload only, conditional on pinned upstream dispatch; not peak memory',
            pytorch_git='5811a8d7da873dd699ff6687092c225caffcf1bb',
            installed_version_metadata='2.9.1+cu128; file read only, no torch import',
            one_coo=24*v['local_entries_upper'],
            persistent_coo_and_transpose=48*v['local_entries_upper'],
            coalesce_named_arrays_and_sort_branch=88*v['local_entries_upper'],
            total_native_peak_upper=None, total_rss_upper=None,
            unresolved=['sort/runtime/allocator and binary-build correspondence',
                        'Python/Fraction/container high-water lifetime',
                        'live capture and full end-to-end bound']),
        upstream_observation=dict(
            date='2026-10-14', source_not_executed=True,
            source_prefix='https://raw.githubusercontent.com/pytorch/pytorch/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/',
            source_sha256={
                'sparse/SparseTensor.cpp':'53d7a4b41c7a3111d785b16c2939164aae9b4fd6e4fa6f0f937618e461fed42f',
                'TensorShape.cpp':'6891b1bf07b806dd176eaba637be5d0fbe7b855609ef8594bcace906821e78b8',
                'Sorting.cpp':'e811fde64d20323bd8eda86210040c3ac9d764a9cc62cb4773188d58a96bc101',
                'cpu/SortingKernel.cpp':'75d2bd98662aadaf4a6629d810e1c026f70714b5895a3131ffffa676234083c5'},
            installed_version_file_sha256='3663be2739d58b664874b271ee94290ce4d2c5a8cc17e0b03edd1bbe083481fe',
            binary_correspondence_proven=False,
            offline_analysis_refetches_upstream=False),
        implementation_factors_separate=[
            'lossless wire encoding and independent decode',
            'immutable publication/final-manifest storage layout',
            'checked bounded row consumption instead of full JSON/Fraction materialization',
            'native preparation or kernel changes, if later proposed'],
        legacy_checker_adapter=dict(
            solves_peak_memory=False, extra_source_pass_must_be_charged=True,
            old_full_path_work_upper_reusable_without_recount=False,
            current_logical_byte_and_structure_caps_still_apply=True),
        request_budget_seconds=300, new_work_quota=None, new_absolute_ceiling=None,
        walltime_prediction=None, runtime_policy_changed=False,
        new_source_runs=0, model_loads=0, native_calls=0, solver_calls=0, gpu_calls=0,
        decision='REQUEST_SEPARATE_CODEC_AND_BOUNDED_INTAKE_SCOPE_NO_EXECUTION',
        stop_boundary='do not implement codecs, alter archive policy or retry sources without scoped decision',
        scientific_claim='envelope and representation invariants only; no algorithm, GPU, real proof or external gain')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-report',type=Path)
    args=parser.parse_args(); report=analyze()
    if args.check_report:
        if json.loads(args.check_report.read_bytes()) != report:
            raise ValueError('saved static design differs')
        print('PASS: static proposed envelope; no codec or execution admission')
    else:
        print(json.dumps(report,sort_keys=True,indent=2))
