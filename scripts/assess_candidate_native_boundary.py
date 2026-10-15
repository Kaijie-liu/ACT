"""Static candidate-exit and installed-native mapping assessment.

Read text, ELF metadata and prior records only. No ACT runtime, torch import,
native invocation, candidate generation, allocation probe or model execution.
"""
import ast
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV = Path('/data1/Kane/miniconda3/envs/act-py312')
TORCH = ENV / 'lib/python3.12/site-packages/torch'
REPORT = 'docs/hz_candidate_native_boundary_20261016.json'
PARENT = 'docs/hz_native_source_blocks_20261016.json'
NATIVE = 'act/back_end/moe/scaled_forest_endpoints.py'
DIRECT = 'scoped_source/direct_candidates.py'
FILES = (PARENT, 'docs/hz_full_path_decision_20261016.json',
    'docs/hz_runtime_lifetime_20261014.json',
    'docs/hz_model_scaled_resources_20261013.json', NATIVE, DIRECT,
    'scoped_source/bounded_runtime.py', 'scoped_source/scaled_endpoints.py',
    'scoped_source/direct_lifecycle.py', 'scoped_source/direct_method.py',
    'scoped_source/hz_request_store.py', 'scripts/assess_candidate_native_boundary.py')
ENV_FILES = ('version.py', 'include/ATen/Config.h', 'include/ATen/native/CPUBlas.h')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(name):
    return json.loads((ROOT / name).read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expressions(name):
    return {ast.unparse(n) for n in ast.walk(ast.parse((ROOT / name).read_text()))}


def elf_metadata(path):
    """Only the ELF64 header and bounded program-header table are read."""
    size = path.stat().st_size
    with path.open('rb') as stream:
        header = stream.read(64)
        require(len(header) == 64 and header[:7] == b'\x7fELF\x02\x01\x01',
                'installed little-endian ELF64')
        fields = struct.unpack('<16sHHIQQQIHHHHHH', header)
        _, kind, machine, version, _, offset, _, _, eh, stride, count, *_ = fields
        require((kind, machine, version, eh, stride) == (3, 62, 1, 64, 56)
                and 0 < count <= 64 and offset >= 64
                and offset + stride * count <= size, 'bounded ELF program headers')
        stream.seek(offset)
        table = stream.read(stride * count)
    require(len(table) == stride * count, 'complete ELF program headers')
    loads = []
    for index in range(count):
        tag, flags, file_offset, address, _, filesz, memsz, alignment = struct.unpack_from(
            '<IIQQQQQQ', table, index * stride)
        if tag != 1:
            continue
        require(filesz <= memsz and file_offset + filesz <= size and memsz > 0,
                'valid LOAD extent')
        loads.append(dict(index=index, flags=flags, file_offset=file_offset,
                          virtual_address=address, file_bytes=filesz,
                          memory_bytes=memsz, alignment=alignment))
    ordered = sorted(loads, key=lambda r: r['virtual_address'])
    require(ordered and all(a['virtual_address'] + a['memory_bytes'] <= b['virtual_address']
                            for a, b in zip(ordered, ordered[1:])), 'disjoint raw LOAD extents')
    executable = [r['memory_bytes'] for r in loads if r['flags'] & 1]
    require(executable, 'executable LOAD exists')
    return dict(path=str(path.relative_to(ENV)), file_bytes=size,
        header_bytes=64, table_offset=offset, table_bytes=len(table),
        header_and_table_sha256=hashlib.sha256(header + table).hexdigest(),
        load_segments=loads, raw_load_bytes_sum=sum(r['memory_bytes'] for r in loads),
        largest_executable_load_bytes=max(executable),
        whole_binary_hashed=False, library_loaded=False, page_rounding_included=False,
        virtual_mapping_not_rss=True)


def analyze():
    parent = load(PARENT)
    for name, expected in parent['anchors'].items():
        require(sha(ROOT / name) == expected, 'source-block anchor drift: ' + name)
    roster = parent['roster']
    require(roster == dict(pairs=28, duties=252, endpoints=504, role_candidates=1008,
                           tiles=84, candidate_width=8, check_width=2), 'complete roster')
    shape = load('docs/hz_model_scaled_resources_20261013.json')['inventory']
    k, E, T, M = 8, roster['endpoints'], roster['tiles'], shape['pair_dual_rows_upper']
    require(M == 2700, 'registered row bound')
    native, direct = expressions(NATIVE), expressions(DIRECT)
    for term in ("g['best'][:, col].tolist()", 'values.append(str(F(v)))',
                 "dict(query=task.query_id, lower='0')"):
        require(term in native, 'current native text exit changed: ' + term)
    require('F(x) == q' in direct and 'raw == bytes(8)' in direct,
            'losslessness and canonical binary64 acceptance')
    require('remaining=MAX_TOTAL - self.total_bytes' in direct,
            'existing cumulative frame gate')
    runtime = expressions('scoped_source/bounded_runtime.py')
    require('self._sealed = bytes(self._data)' in runtime
            and 'self._data = None' in runtime, 'seal coexistence')

    prefix = struct.calcsize('<8s32s32s32sIIIIII')
    record = struct.calcsize('<32s32sBB6s')
    rational = struct.calcsize('<BBHHH8s') + 2 * (4096 // 8)
    frame = prefix + k * (record + 2 * rational + 16 * M)
    stream = T * prefix + E * (record + 2 * rational + 16 * M)
    require((prefix, record, rational, frame, stream) == (128, 72, 1040, 362944, 22868160),
            'current profile0 exact extent')
    boundary = parent['candidate_boundary']
    require(frame == boundary['binary64_frame_upper']
            and stream == boundary['full_binary64_stream_upper'], 'inherited format accounting')
    cap = boundary['stream_cap']
    best = 8 * k * M
    extraction = dict(row_visits=2 * E, coefficient_visits=E * M, arithmetic=0)
    encoding = dict(row_visits=E + 2 * T,
                    coefficient_visits=4 * E * M + 3 * E + T, arithmetic=0)
    sums = {key: extraction[key] + encoding[key] for key in extraction}

    config = (TORCH / 'include/ATen/Config.h').read_text()
    for flag in ('#define AT_MKL_ENABLED() 1', '#define AT_MKL_SEQUENTIAL() 0',
                 '#define AT_PARALLEL_OPENMP 1'):
        require(flag in config, 'installed native build changed: ' + flag)
    gemm_header = (TORCH / 'include/ATen/native/CPUBlas.h').read_text()
    double_declaration = gemm_header.split('void gemm(', 2)[2].split(';', 1)[0]
    require('double alpha' in double_declaration and 'double *c' in double_declaration
            and 'workspace' not in double_declaration, 'double GEMM API binding')
    installed = ast.parse((TORCH / 'version.py').read_text())
    values = {n.targets[0].id: ast.literal_eval(n.value) for n in installed.body
              if isinstance(n, ast.Assign) and len(n.targets) == 1
              and isinstance(n.targets[0], ast.Name) and n.targets[0].id in ('__version__', 'git_version')}
    elf = elf_metadata(TORCH / 'lib/libtorch_cpu.so')
    proposal = load('docs/hz_full_path_decision_20261016.json')['proposed_as']
    require(not proposal['active'], 'no active AS policy')
    as_proposed = proposal['bytes']
    require(elf['largest_executable_load_bytes'] > as_proposed,
            'installed executable LOAD does not fit small-control AS proposal')

    return dict(schema='CANDIDATE_NATIVE_BOUNDARY_STATIC_DECISION_V1',
        base_head='b4aeffa3e56a54d57091e71d59018d3a2c2ad977',
        anchors={name: sha(ROOT / name) for name in FILES},
        unchanged=parent['unchanged'], roster=roster,
        binary64_exit=dict(implemented=False, profile=0,
            exact_equivalence='Fraction(v) == rational(str(Fraction(v))) for each finite binary64 v',
            negative_zero_normalized=True, nonfinite_rejected=True,
            advice_lower_remains_exact_zero=True, both_roles_and_all_rows_retained=True,
            scalar_bit_bound=1075, stored_bit_cap=4096,
            complete_frame_bytes_upper=frame, frame_cap=4 * 2**20,
            full_stream_upper=stream, stream_cap=cap, excess_over_stream_cap=stream-cap,
            full_stream_guarantee=False, actual_bytes=None,
            consumer_math_changes=False, bitwise_candidate_trajectory_claim=False),
        conditional_candidate_payload=dict(best_dual_tensor_bytes=best,
            new_owner_internal_sink_frame_overlap=2*frame,
            new_owner_internal_sink_plus_best=2*frame+best,
            existing_receive_frame_overlap=3*frame,
            existing_receive_plus_best=3*frame+best,
            requires_no_retained_generator_or_optimizer_frame=True,
            includes_source_native_arena_python_or_allocator=False,
            source_and_native_bank_remain_live=True, full_peak_upper=None),
        current_exit_counted_work=dict(extraction=extraction, encoding_with_allocate_seal=encoding,
            combined=sums, combined_total=sum(sums.values()),
            subsequent_tile_header_row_visits=E,
            includes_math_consumer=False, includes_fraction_string_machine_cost=False,
            new_adapter_counter_upper=None, existing_receive_is_not_zero_cost=True),
        native_environment=dict(version=values, anchors={name:sha(TORCH/name) for name in ENV_FILES},
            elf=elf, proposed_as_bytes=as_proposed,
            executable_load_excess=elf['largest_executable_load_bytes']-as_proposed,
            limitation='256 MiB active before native loading cannot admit this library mapping; not an RSS claim',
            lowering_limit_after_mapping_is_not_prelimit_capacity=True,
            sparse_axpy_analysis_covers_dense_gemm=False,
            header_flags_identify_runtime_backend=False,
            gemm_scratch_upper=None, mapped_runtime_total_upper=None),
        decision='LOSSLESS_EXIT_DESIGN_ONLY_NATIVE_CAPACITY_NOT_CLOSED_NO_STANDALONE_IMPLEMENTATION',
        complete_work_upper=None, full_heap_upper=None, full_rss_upper=None,
        live_source_upper=None, new_resource_policy=None,
        implementation_authorized=False, runtime_changed=False,
        model_runs=0, candidate_calls=0, worker_runs=0, gpu_runs=0,
        full_capacity_admitted=False, scientific_goals_closed=[])


if __name__ == '__main__':
    result = analyze()
    if sys.argv[1:] == ['--check']:
        require(load(REPORT) == result, 'candidate/native boundary record drift')
        print('PASS: static frame/count/ELF metadata only; no runtime or native execution')
    else:
        require(not sys.argv[1:], 'only --check supported')
        print(json.dumps(result, indent=2, sort_keys=True))
