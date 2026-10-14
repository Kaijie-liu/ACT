"""Static runtime sub-ledger; no numerical libraries, models or native calls.

Magnitude, array-payload, file metadata and inspected binary slices are NOT a
complete allocation/RSS proof. No resource limit or acceptance rule is changed.
"""
import hashlib
import json
from pathlib import Path

from scripts.assess_unique_root_protocol import inventory, analyze as parent_analysis

ROOT = Path(__file__).resolve().parents[1]
ENV = Path('/data1/Kane/miniconda3/envs/act-py312')
ANCHORS = {
    'docs/hz_unique_root_protocol_20261014.json': 'a36b6e393a4d25face25fc7b0ddfe422d54e2adf50035162245da13c20a3293c',
    'scripts/assess_unique_root_protocol.py': 'd4e4f7ab509dfd94e050d672b75358c32186f702ff7228a27d620fd9cd5ddf9f',
    'scoped_source/hz_request_store.py': '7cb2be7647039f59d3c896b7856f634f43be9c532d9d2fd43632be09ab3230e0',
    'scoped_source/scaled_endpoints.py': '93197c1b329e173b25795c48cf117b7113c2b23cc6958438889a7e912d13e8af',
    'scoped_source/check_affine_references.py': 'a0d688ee82bf88c42945edfd3abc9591a46998c01cf236cbf752ad5b549e9c62',
    'scoped_source/tensor_rows.py': '5c3ad6d642868e257ac9dd2edbc5e066b4cd793d4c7567a4ea800fc8e8f97f82',
    'act/back_end/moe/factory.py': 'da472b2bbb3d8f684cb9c9916a735fd7105ceaab925911e1cbc983785a4c7d2d',
    'act/pipeline/moe/train.py': 'bac0e9a50d98f316d577128909ac5c81bf41454debccbc0be54b14a2b1360dd5',
    'act/pipeline/moe/schedule_confirmation_selection.py': '944f2c56dd36201469791ab39c75c40cfdd3abd9569aa69ec0811641bf58c4c7',
    'act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json': 'b4167d8ab501df06570bfb49ad646263962c79736c2252f1398dc0499d3a7ee6',
    'docs/hz_real_intake_20261001_r1.json': '9f5f7f240bbe6757206b52ea2754d30051ebd98d829ab8d6341823f25c87a8ab',
    'scripts/run_scaled_dense_capacity.py': '4b80573abc480c5dde5fc3e9628e95c6284a4ae1d1fa1fa3d4441160227f6181',
}
ENV_ANCHORS = {
    'lib/python3.12/fractions.py': 'd52d1ab9d5be26924e4914a5e40069c992bbfa68f773694c2e5ee8452d0766a0',
    'include/python3.12/cpython/longintrepr.h': 'e2acf37f668089278081539473eeedc8537b848fcd2eb91f53172701c014b9c3',
}
TORCH_COMMIT = '5811a8d7da873dd699ff6687092c225caffcf1bb'
UPSTREAM = {
    'aten/src/ATen/native/sparse/SparseTensorMath.cpp': '13313cff8d04eba91196e309e8b0b937e2f49e9dbdd0d9a275369af3e678bafb',
    'aten/src/ATen/native/CPUBlas.h': '5972e9e6c61b7d317f39bf31d7c41931ced9f408296d41da64431a7f6d8e4154',
    'aten/src/ATen/native/cpu/BlasKernel.cpp': '29e4a32f71c9b92293fa03273eda399766d9c0a4f89968ff3ced127d8b1746fb',
    'aten/src/ATen/native/TensorCompare.cpp': 'f17a64a3de2309f3baea7933d030c18cb7f4f945e3da447eae62477cbfcc3bd1',
}
BINARY = 'lib/python3.12/site-packages/torch/lib/libtorch_cpu.so'
SLICES = (
    ('double_worker', 0x20166e0, 1458, '2753a5d879177e342159316d8089a218dcc68c429985935578992ccf21b7f9da'),
    ('axpy_template', 0x1bdc650, 261, '1e4805d3a0d495ba1b8c68531ed8c8eaf452d94df735d5f508fd58fb252818ed'),
    ('default_axpy', 0x4567b60, 1025, 'f51964e18d0997e7ff0dead9eba437371dbb3e478717bb8b477c5a719f7e1286'),
    ('avx2_axpy', 0x70593b0, 1259, '1f7e6bdcb3e89696acd3fbde69037e4a6a2357f47d52b92b27493dbfa5d7f399'),
)


def magnitude(op, a, b):
    """Unreduced rational expression widths; never truncate to an accept cap."""
    if any(type(v) is not int or v < 1 for v in (*a, *b)) or len(a) != 2 or len(b) != 2:
        raise ValueError('positive numerator/denominator bit upper pairs')
    an, ad = a; bn, bd = b
    if op in ('add', 'sub'): return max(an+bd, bn+ad)+1, ad+bd
    if op == 'mul': return an+bn, ad+bd
    if op == 'div': return an+bd, ad+bn  # Divisor nonzero is a separate check.
    if op == 'compare': return an+bd, bn+ad  # Two integer cross products.
    raise ValueError('unknown expression')


def exact_subledger():
    b = (8192, 8192)
    difference = magnitude('sub', b, b)
    # sum() starts at exact zero; 0+x=x adds no width to the first term.
    three = magnitude('add', magnitude('add', difference, difference), difference)
    four = magnitude('add', difference, three)
    return dict(scope='checked B-bit operands only; not all ingress or C scratch',
        expression_bits=dict(binary_add=magnitude('add', b, b),
            product=magnitude('mul', b, b), fused_multiply_add=magnitude('add', b, magnitude('mul', b, b)),
            relu_four_absolute_differences=four),
        old_decimal_parse_magnitude_bits=dict(candidate_2470_chars=(10**2470-1).bit_length(),
            request_4096_chars=(10**4096-1).bit_length()),
        request_literal_gate='4096 characters, not the candidate 4096-bit gate',
        preserve_request_domain=True, new_request_bit_limit=None,
        hash_cache=dict(entries=16384, key_integer_magnitude_payload_upper=16384*2*1024,
            conditions=['fresh checker process', 'all hashed operands independently bounded by 8192 bits'],
            includes_python_objects_or_allocator=False, clear_cache_implemented=False),
        c_bigint_scratch_upper=None, allocator_retained_upper=None,
        traceback_and_session_lifetime_bound=None)


def native_subledger(width):
    if type(width) is not int or width not in (1, 8):
        raise ValueError('registered widths only')
    v = inventory(); p = v['pairs']; e = v['endpoints_upper']
    n, f, l, r = (v[k] for k in ('nodes_upper', 'factors_upper', 'local_entries_upper', 'references_upper'))
    t = p*((18+width-1)//width)
    return dict(scope='conditional named-array and loop envelope; not measured time/RSS',
        tile_count=t, summed_tile_width=e, adjoints=129*t, forwards=128*t, spmm_calls=257*t,
        coo_axpy_calls_upper=257*t*l, axpy_coordinate_updates_upper=257*e*l,
        logical_coo_index_value_bytes_upper=24*257*t*l,
        cumulative_two_spmm_output_payload_bytes=16*e*(129*f+128*n),
        largest_two_spmm_output_payload_bytes=16*f*width,
        dag_level_select_product_payload_sum_per_call=16*r*width,
        persistent_operator_tensor_payload_upper=48*l+24*r+8*f,
        loop_source='COO nnz loop -> axpy<double> template -> axpy_stub direct loop',
        spmm_recoalesce_each_call=False, inferred_mkl_workspace=False,
        array_cumulative_is_peak=False, logical_reads_are_physical_dram=False,
        runtime_dispatch_proved=False, complete_native_peak_upper=None,
        unresolved=['runtime dispatch registration/overrides', 'index_add/boolean indexing/isfinite/TensorIterator internals',
            'Python construction and old iterate/RHS lifetime overlap', 'allocator/runtime/mapped pages and failure frames'])


def verify_files():
    for root, anchors in ((ROOT, ANCHORS), (ENV, ENV_ANCHORS)):
        for name, expected in anchors.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != expected:
                raise ValueError('changed static anchor: '+name)
    p = ENV/BINARY
    if p.stat().st_size != 436938961: raise ValueError('changed binary size')
    with p.open('rb') as stream:
        for name, offset, length, expected in SLICES:
            stream.seek(offset); raw = stream.read(length)
            if len(raw) != length or hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError('changed inspected binary slice: '+name)


def analyze():
    verify_files(); prior = parent_analysis()
    selection = json.loads((ROOT/'act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json').read_bytes())
    historical = {name:dict(checkpoint_sha256=m['checkpoint_sha256'], model_state=m['model_state'])
                  for name,m in selection['models'].items()}
    return dict(schema='RUNTIME_LIFETIME_STATIC_SUBLEDGER_V1', date='2026-10-14',
        reviewed_head='f88d62478a5ce55b4dd86d87ff1fafc8daf83107',
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        repo_anchors=ANCHORS, local_environment=str(ENV), environment_anchors=ENV_ANCHORS,
        upstream_review=dict(commit=TORCH_COMMIT, file_sha256=UPSTREAM, refetched_by_analysis=False),
        binary_review=dict(path=BINARY, size=436938961,
            slices=[dict(name=n,offset=o,bytes=l,sha256=h) for n,o,l,h in SLICES],
            whole_binary_verified=False, dynamically_executed=False),
        exact=exact_subledger(), native={str(w):native_subledger(w) for w in (1,8)},
        capture=dict(historical_bindings=historical, selected_live_object=None,
            loaded_checkpoints=0, raw_parameters_binary64_bytes=8*inventory()['parameter_values'],
            full_raw_source_binary64_bytes=inventory()['raw_binary64_tensor_bytes'],
            old_loader_weights_only=False, cpu_map_location_excludes_cuda_probe=False,
            old_loader_retains_model_and_payload=True, file_size_bounds_unpickling=False,
            input_semantics='TORCH_DEFAULT_FLOAT64_BEFORE_TOTENSOR',
            complete_capture_upper=None, bounded_adapter_implemented=False),
        supervision=dict(old_rss_scope='leader PID only', old_wait_scope='leader PID only',
            descendants_cannot_outlive_phase_proved=False, full_phase_nonoverlap_proved=False,
            needed='whole-worker lifetime guarantee before using max(phase peaks)',
            existing_supervisor_changed=False),
        invariant=prior['invariant'], current_policy_unchanged=prior['current_policy_unchanged'],
        full_runtime_upper=None, resource_limits_changed=False, new_quota=None,
        production_changed=False, implementation_authorized=False, execution_admitted=False,
        model_runs=0, gpu_calls=0, solver_calls=0, source_builds=0, native_calls=0,
        status='CONCRETE_RUNTIME_GAPS_IDENTIFIED_NO_FULL_COST_ADMISSION')


if __name__ == '__main__':
    print(json.dumps(analyze(), sort_keys=True, indent=2))
