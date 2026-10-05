"""D057 single-use complete mathematics and three-source census; preserve all inherited attempt outcomes."""
import ast
from fractions import Fraction
from itertools import combinations
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import tracemalloc
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d057_triangle_source_census_20260930_v1'
PRIOR = EXP / 'results/d056_phase_triangle_component_20260930_v1'
FAILED_CENSUS = EXP / 'results/d047_multiphase_source_census_20260930_v1'
D056 = HERE.parent / 'd056_phase_triangle_component_20260930'
D054 = HERE.parent / 'd054_shared_phase_cycles_20260930'
D055 = HERE.parent / 'd055_cycle_applicability_audit_20260930'
D038 = HERE.parent / 'd038_descendant_census_20260930'
D017 = HERE.parent / 'd017_applicability_gpu_20260930'
D015 = HERE.parent / 'd015_batch_binding_20260928_v2'
OLDER = EXP / 'results/d015_source_shielding_20260928_v2/preregistered.json'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FREEZE = HERE / 'freeze.json'
NEW_FILES = ('PREREG.md', 'THEORY.md', 'window_relation.py',
             'test_window_relation.py', 'census.py', 'run_census.py')
TEST_NAMES = ('test_complete_window_control', 'test_full_population_and_zero_phases',
              'test_plain_identity_and_intervals', 'test_fail_closed_budget_and_default_off')
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    PRIOR / 'preregistered.json': '648718aa3555ac47b638434a884026464d91c53f4db984e20909d76c39788de3',
    PRIOR / 'inventory.json': '603eb8ca177abb64c89ac76dae99bccc43b93b1001434acc0982eabe6e1ebd75',
    PRIOR / 'exit.json': '5942d9e2b23c20fed67083ed05dfe642b74d97b54a2a9b51ea614ef2426494fd',
    D056 / 'freeze.json': 'a4a04b48897958a9b32ac12af64457f0952745621658854e7e844951aeb2ef79',
    D056 / 'run_math.py': '06b1a1c1a66614a39487973ab250572548665c4a046e5568d82abdca8e3509d6',
    D054 / 'RESEARCH.md': '16eb6daacfdd640fbe647c7d4425f3e6a2d0d1cfb67e341f897f45acb8947066',
    D054 / 'RECORD.md': 'a5ac271b982bc7c0b47eb55c8e4d2c958eb9f3c72d9579eaac0c27ddb2e52239',
    D054 / 'SHA256SUMS': 'e5a517f193852c8b79cb78c0b198cdc2ec0110aa74b98b60460c68dd780cb0bc',
    D055 / 'PLAN.md': '6e6586f27411034173a8be6417b4a13bca6b525173617746ae884a8203fd435d',
    D055 / 'audit.jq': '122d451a9abc8a96563065703839844c6e085e77b85917f990b58b83a1f4f1a7',
    D055 / 'audit_v2.jq': 'c4d858632cc86b7c0dcdd45928b97585a007725f05dfbdb2b4f3bde3af58f44a',
    D055 / 'QUERY_V1_FAILURE.json': 'a383bfbed2ee24c2f93b0f27d8cfbb7394c594b3a4a362c354aa4979a02556f9',
    D055 / 'DIAGNOSTIC.json': '2110b083aeb5059477f5812138951abfb03e0f95c5c857e5afca65f65468513d',
    D055 / 'RESULTS.md': '7a1e0ca6146d97a291d37d6f0c82e70596cb65efc9084a1ed469fdc825bc32ad',
    D055 / 'SHA256SUMS': '1224919a51142f459f2b16dc4bcccb1a25094c9891863f782542852fb6d013c1',
    D038 / 'freeze.json': '8f24ae5d3e7c674a66e80874e6f607733aeff9d46174434a8a68ce723cf8fc73',
    D038 / 'run_reference.py': 'f4abd1608c70aa308a590307b8f3369eff2ab11b217d54be597c99bb77c0efb5',
    D017 / 'gpu_preflight.py': '902437443f6847482d21d0af227b7fc36234868444f11f4dfc580f2abbf89c01',
    D015 / 'run_v2.py': '8ff9d296ce56b8dd9481d0c8367484b8ff37148db5dfa522062136d9d04e12ce',
    OLDER: '4697d2bbc1b86e7732e825e0e8a3b6d9c595fc275516688d746e738d2bba50cd',
}
SCOPE = ('complete inherited mathematics plus all three original first-bank source populations; '
         'D056 success and nested D047 failure remain unchanged; '
         'no native HZ, GPU, complete physical or model-solve qualification')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            value.update(chunk)
    return value.hexdigest()


def read_json(path, cap=8 * 1024**2):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= cap:
        raise ValueError('missing, linked or oversized metadata: ' + str(path))
    return json.loads(path.read_text())


def save(name, value):
    with (RUN / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))


def check_freeze():
    frozen = read_json(FREEZE, RESERVE)
    sources, names = frozen.get('source_sha256'), frozen.get('new_test_names')
    if (frozen.get('schema') != 'd057_frozen_v1'
            or frozen.get('required_tests') != 3797 or frozen.get('required_test_files') != 176
            or type(sources) is not dict or set(sources) != {str(HERE / name) for name in NEW_FILES}
            or type(names) is not list or names != list(TEST_NAMES)):
        raise ValueError('frozen six-file/four-test contract differs')
    for path, digest in sources.items():
        if (type(digest) is not str or len(digest) != 64 or Path(path).is_symlink()
                or sha(path) != digest):
            raise ValueError('new source differs from pre-execution freeze: ' + path)
    return frozen


def check_worker_resources(result, selected):
    if (result.get('source_census_completed') is not True
            or result.get('memory_gate_passed') is not True or result.get('failure')):
        raise ValueError('complete three-source census or measured worker gate failed')
    fields = ('rss_highwater_growth_bytes', 'final_summary_reserve_bytes',
              'traced_peak_bytes', 'tracer_metadata_bytes', 'retained_entries',
              'whole_work_used', 'branch_work_used', 'evidence_work_used', 'preimport_work_used')
    if any(type(result.get(key)) is not int or result[key] < 0 for key in fields):
        raise ValueError('invalid worker resource accounting')
    if (result['final_summary_reserve_bytes'] != RESERVE
            or result['rss_highwater_growth_bytes'] + RESERVE > MEMORY_CAP
            or result['traced_peak_bytes'] + result['tracer_metadata_bytes'] + RESERVE > MEMORY_CAP
            or result['retained_entries'] > 64_000_000
            or result['whole_work_used'] > 256_000_000
            or result['branch_work_used'] > 200_000_000
            or result['evidence_work_used'] > 40_000_000):
        raise ValueError('unchanged worker resource limit exceeded')
    wall = result.get('wall_s')
    if type(wall) not in (int, float) or not math.isfinite(wall) or not 0 <= wall <= 240:
        raise ValueError('worker wall limit exceeded')
    if any(type(result.get(key)) is not int or result[key] != 0 for key in
           ('diagnostic_solver_calls', 'model_forward_calls', 'new_benchmark_solves', 'formal_gain')):
        raise ValueError('source census cannot execute solver/forward or claim solves')
    if any(result.get(key) is not False for key in
           ('native_HZ_admitted', 'actual_phase_column_binding_verified',
            'gpu_computation_completed', 'complete_physical_qualification')):
        raise ValueError('source census exceeded registered qualification scope')
    models = result.get('models')
    if (type(models) is not list or len(models) != 3
            or [item.get('model') for item in models] != [s['model_relative_path'] for s in selected]):
        raise ValueError('complete original three-source order differs')
    total_bytes = total_model_work = total_evidence_work = 0
    model_work = []
    for index, item in enumerate(models):
        for key in ('evidence_bytes', 'evidence_work', 'numerical_model_work', 'retained_entry_upper'):
            if type(item.get(key)) is not int or item[key] < 0:
                raise ValueError('invalid complete model accounting')
        if (item['numerical_model_work'] > 200_000_000
                or item['retained_entry_upper'] > 64_000_000):
            raise ValueError('per-model work or retained entry cap exceeded')
        total_bytes += item['evidence_bytes']
        total_model_work += item['numerical_model_work']
        total_evidence_work += item['evidence_work']
        model_work.append(item['numerical_model_work'])
        if item.get('evidence_file') != 'complete_' + str(index) + '.json':
            raise ValueError('unexpected complete evidence filename')
        path = RUN / item['evidence_file']
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size != item.get('evidence_bytes')
                or sha(path) != item.get('evidence_sha256')):
            raise ValueError('complete evidence identity differs')

    if (total_bytes > result['evidence_work_used'] or total_evidence_work > result['evidence_work_used']
            or result['branch_work_used'] != max(model_work)
            or result['whole_work_used'] != result['preimport_work_used'] + 40_000_000 + RESERVE + total_model_work
            or result.get('address_space_bytes') != AS_CAP
            or result.get('actual_cpu_affinity') != list(os.sched_getaffinity(0))):
        raise ValueError('global evidence/work/CPU accounting differs')


def evidence_fraction(value):
    if (type(value) is not list or len(value) != 2
            or any(type(part) is not int or abs(part).bit_length() > 512 for part in value)
            or value[1] <= 0):
        raise ValueError('invalid finite rational evidence')
    return Fraction(*value)


def evidence_interval(value):
    if type(value) is not list or len(value) != 2:
        raise ValueError('invalid interval evidence')
    lower, upper = map(evidence_fraction, value)
    if lower > upper:
        raise ValueError('reversed interval evidence')
    return lower, upper



COUNTS = ('total_triangles', 'stable_skipped_triangles', 'candidate_triangles',
          'odd_triangles', 'no_odd_triangles', 'pair_proof_count', 'pair_row_count',
          'pair_nnz', 'row_count', 'nnz')


def equal_counts(actual, expected):
    if any(type(actual.get(key)) is not int or actual[key] != value
           for key, value in expected.items()):
        raise ValueError('complete integer population or row counts differ')


def evidence_affine(value, allowed, interval):
    """Check plain original-source forms; never reconstruct a candidate token."""
    if type(value) is not dict:
        raise ValueError('plain source certificate required')
    (evidence_interval if interval else evidence_fraction)(value['bias'])
    terms = value['terms'] if interval else value['source_terms']
    if type(terms) is not list or len(terms) > 65536:
        raise ValueError('bounded canonical source terms required')
    if not interval and value.get('phase_terms') != []:
        raise ValueError('P/N cannot acquire undeclared phase terms')
    previous = -1
    for item in terms:
        if (type(item) is not list or len(item) != (3 if interval else 2)
                or type(item[0]) is not int or item[0] <= previous or item[0] not in allowed):
            raise ValueError('unbound or noncanonical original source')
        coefficient = evidence_interval(item[1:]) if interval else evidence_fraction(item[1])
        if coefficient == ((0, 0) if interval else 0):
            raise ValueError('canonical source form retains zero terms')
        previous = item[0]


def evidence_rows(rows, sources, outputs, multiplier):
    if type(rows) is not list or len(rows) > 4:
        raise ValueError('bounded complete row tuple required')
    nnz = 0
    for row in rows:
        if type(row) is not list or len(row) != 3:
            raise ValueError('plain LE row required')
        values, phases, rhs = row
        evidence_fraction(rhs)
        if type(values) is not list or type(phases) is not list or len(values) + len(phases) > 65536:
            raise ValueError('row support limit differs')
        seen, actual_outputs = set(), {}
        for terms, is_phase in ((values, False), (phases, True)):
            for term in terms:
                if (type(term) is not list or len(term) != 2
                        or type(term[0]) is not list or len(term[0]) != 2
                        or type(term[0][1]) is not int):
                    raise ValueError('typed plain original identity required')
                kind, ordinal = term[0]
                key = (kind, ordinal)
                allowed = outputs if kind in ('phase', 'output') else sources
                if (kind not in (('phase',) if is_phase else ('source', 'output'))
                        or ordinal not in allowed or key in seen):
                    raise ValueError('duplicate or unbound row identity')
                coefficient = evidence_fraction(term[1])
                if coefficient == 0:
                    raise ValueError('sparse row retains a zero coefficient')
                if kind == 'output':
                    actual_outputs[ordinal] = coefficient
                seen.add(key)
        if actual_outputs != {ordinal: Fraction(multiplier) for ordinal in outputs}:
            raise ValueError('row original readout population differs')
        nnz += len(values) + len(phases)
    return nnz


def check_window(relation, source_ordinals, source_bounds, output_ordinals):
    """Check every physical receiver, cached edge and remaining triple once."""
    if relation.get('schema') != 'd057_window_v1' or relation.get('original_bits_deleted') != 0:
        raise ValueError('window schema or original-state scope differs')
    count, width = len(output_ordinals), len(source_ordinals)
    sources = {ordinal for ordinal in source_ordinals if ordinal is not None}
    if (relation.get('source_ordinals') != source_ordinals
            or relation.get('output_ordinals') != output_ordinals
            or len(relation['source_bounds']) != width or len(relation['activation_bounds']) != width
            or len(relation['ordinary_bounds']) != count):
        raise ValueError('complete original window declarations differ')
    for saved, active, expected in zip(relation['source_bounds'], relation['activation_bounds'], source_bounds):
        if evidence_interval(saved) != expected or evidence_interval(active) != tuple(max(0, x) for x in expected):
            raise ValueError('preactivation/activation source box binding differs')
    ordinary = [evidence_interval(value) for value in relation['ordinary_bounds']]
    states = ['A' if lo > 0 else 'I' if hi < 0 else 'O' for lo, hi in ordinary]
    open_channels = sorted((i for i, state in enumerate(states) if state == 'O'),
                           key=lambda i: output_ordinals[i])
    if relation.get('channel_states') != states or relation.get('open_channels') != open_channels:
        raise ValueError('strict stability or zero-touch population differs')
    expected_triangles = math.comb(len(open_channels), 3)
    pairs, triangles = relation['pairs'], relation['triangles']
    expected_pairs = math.comb(len(open_channels), 2) if expected_triangles else 0
    if (type(pairs) is not list or type(triangles) is not list
            or len(pairs) != expected_pairs or len(triangles) != expected_triangles):
        raise ValueError('full cached-edge or unresolved-triple population differs')
    index_by_channels, deltas, receiver_forms = {}, [], {}
    pair_rows = pair_nnz = rows = nnz = odd = 0
    pair_population = combinations(open_channels, 2) if expected_triangles else ()
    for index, (pair, channels) in enumerate(zip(pairs, pair_population)):
        outputs = [output_ordinals[i] for i in channels]
        if pair.get('channels') != list(channels) or pair.get('outputs') != outputs:
            raise ValueError('original cached pair order differs')
        index_by_channels[channels] = index
        for channel, name in zip(channels, ('f', 'g')):
            evidence_affine(pair[name], sources, True)
            if channel in receiver_forms and receiver_forms[channel] != pair[name]:
                raise ValueError('one original receiver has inconsistent affine premises')
            receiver_forms[channel] = pair[name]
        evidence_affine(pair['P'], sources, False)
        evidence_affine(pair['N'], sources, False)
        k0, k1, upper, delta = (evidence_fraction(pair[name])
                                for name in ('kappa0', 'kappa1', 'upper', 'delta'))
        if delta != k1 - k0 or len(pair['rows']) != (1 if delta == 0 else 2):
            raise ValueError('complete pair projection differs')
        deltas.append(delta)
        pair_rows += len(pair['rows'])
        pair_nnz += evidence_rows(pair['rows'], sources, set(outputs), 1)
    for triangle, channels in zip(triangles, combinations(open_channels, 3)):
        outputs = [output_ordinals[i] for i in channels]
        indices = [index_by_channels[(channels[0], channels[1])],
                   index_by_channels[(channels[0], channels[2])],
                   index_by_channels[(channels[1], channels[2])]]
        if (triangle.get('channels') != list(channels) or triangle.get('outputs') != outputs
                or triangle.get('pair_indices') != indices
                or len(triangle['bounds']) != 3
                or [evidence_interval(v) for v in triangle['bounds']] != [ordinary[i] for i in channels]):
            raise ValueError('triangle original bindings or cached edge references differ')
        edge_deltas = [deltas[i] for i in indices]
        is_odd = all(d != 0 for d in edge_deltas) and sum(d < 0 for d in edge_deltas) in (1, 3)
        if (triangle.get('status') != ('odd_cycle' if is_odd else 'no_odd_cycle')
                or (not is_odd and triangle['rows'])
                or (is_odd and not 1 <= len(triangle['rows']) <= 4)):
            raise ValueError('complete signed-triangle status differs')
        row_nnz = evidence_rows(triangle['rows'], sources, set(outputs), 2)
        equal_counts(triangle, dict(row_count=len(triangle['rows']), nnz=row_nnz))
        gap = min(abs(d) for d in edge_deltas) / 2 if is_odd else Fraction(0)
        if evidence_fraction(triangle['cube_midpoint_gap']) != gap:
            raise ValueError('pure cube algebra diagnostic differs')
        rows += len(triangle['rows']); nnz += row_nnz; odd += is_odd
    expected = dict(canonical_slots=width, real_slots=len(sources), padding_slots=width-len(sources),
        receiver_count=count, receiver_canonical_slots=count*width,
        total_triangles=math.comb(count, 3), stable_skipped_triangles=math.comb(count, 3)-expected_triangles,
        candidate_triangles=expected_triangles, odd_triangles=odd,
        no_odd_triangles=expected_triangles-odd, pair_proof_count=len(pairs),
        pair_row_count=pair_rows, pair_nnz=pair_nnz, row_count=rows, nnz=nnz)
    equal_counts(relation, expected)
    for name, cap in (('work_used', 200_000_000), ('transient_numeric_entries', 64_000_000)):
        if type(relation.get(name)) is not int or not 0 <= relation[name] <= cap:
            raise ValueError('window work or temporary storage differs')
    temporary = (65_536 + 512 * width + 48 * len(open_channels) * width
                 + 192 * expected_pairs * (width + 8) + 256 * count)
    if relation['transient_numeric_entries'] != temporary:
        raise ValueError('complete typed edge-cache temporary reserve differs')
    return expected, states


def check_complete_evidence(result, selected, identities):
    """Audit sealed data only; do not rerun compiler, source parser or models."""
    extractor = str(D015 / 'source_binding_v2.py')
    for receipt, source in zip(result['models'], selected):
        data = read_json(RUN / receipt['evidence_file'], 40_000_000)
        ref = data.get('source_packet_ref', {})
        if (data.get('schema') != 'd057_triangle_source_census_v1' or data.get('source') != source
                or data.get('binding_mathematical_only') is not True
                or data.get('actual_phase_column_binding_verified') is not False
                or ref.get('model_sha256') != source['model_sha256']
                or ref.get('extractor_path') != extractor
                or ref.get('extractor_sha256') != identities[extractor]
                or ref.get('input_batch') != 1 or data.get('summary') != receipt.get('summary')):
            raise ValueError('complete source or extractor binding differs')
        shape, frame = data['first_shape'], data['frame_identity']
        if (type(shape) is not list or len(shape) != 4 or shape[0] != 1
                or any(type(n) is not int or not 0 < n <= 65536 for n in shape)
                or type(frame) is not list or len(frame) != 2 or frame[0] != source['model_sha256']
                or type(frame[1]) is not str or not frame[1]):
            raise ValueError('original first-bank shape/frame differs')
        bounds, ordinals, previous = {}, {}, None
        for item in data['source_bounds']:
            coord = tuple(item['coordinate'])
            if (len(coord) != 3 or any(type(v) is not int for v in coord)
                    or any(not 0 <= v < extent for v, extent in zip(coord, shape[1:]))
                    or coord in bounds or (previous is not None and coord <= previous)):
                raise ValueError('source coordinate population differs')
            ordinal = (coord[0]*shape[2]+coord[1])*shape[3]+coord[2]
            if type(item['ordinal']) is not int or item['ordinal'] != ordinal:
                raise ValueError('original source ordinal differs')
            bounds[coord], ordinals[coord], previous = evidence_interval(item['bounds']), ordinal, coord
        expected, specs, receivers = [], {}, 0
        max_slots = 0
        for index, branch in enumerate(data['branches']):
            weight, output = branch['weight_shape'], branch['output_shape']
            strides, dilation, pads = branch['strides'], branch['dilations'], branch['pads']
            if (branch['index'] != index or len(weight) != 4 or len(output) != 4
                    or len(strides) != 2 or len(dilation) != 2 or len(pads) != 4 or branch['group'] != 1
                    or any(type(v) is not int or not 0 < v <= 65536 for v in weight+output+strides+dilation)
                    or any(type(v) is not int or v < 0 for v in pads) or weight[1] != shape[1]):
                raise ValueError('original full-branch geometry differs')
            co, ci, kh, kw = weight
            oh = (shape[2]+pads[0]+pads[2]-dilation[0]*(kh-1)-1)//strides[0]+1
            ow = (shape[3]+pads[1]+pads[3]-dilation[1]*(kw-1)-1)//strides[1]+1
            positions = sorted({(0,0),(0,ow-1),(oh-1,0),(oh-1,ow-1),(oh//2,ow//2)})
            if (output != [1,co,oh,ow] or branch['receiver_count'] != co or ci*kh*kw > 65536
                    or branch.get('positions') != [list(p) for p in positions]
                    or branch.get('admitted') is not (branch['consumer_relu'] is not None)):
                raise ValueError('complete consumer/window declarations differ')
            if branch['consumer_relu'] is None:
                continue
            expected.extend((index, p) for p in positions)
            receivers += co*len(positions)
            max_slots = max(max_slots, ci*kh*kw)
            specs[index] = branch
        if not expected or len(data['windows']) != len(expected):
            raise ValueError('all admitted direct-branch windows required')
        counts = {name: 0 for name in COUNTS}
        canonical = valid = padding = active = inactive = opened = transient = kernel_work = 0
        seen_sources = set()
        for window, (index, position) in zip(data['windows'], expected):
            branch = specs[index]
            if window['branch'] != index or window['position'] != list(position):
                raise ValueError('canonical complete window order differs')
            co, ci, kh, kw = branch['weight_shape']
            sy, sx = branch['strides']; dy, dx = branch['dilations']; top, left, _, _ = branch['pads']
            slots = [(channel, position[0]*sy+ky*dy-top, position[1]*sx+kx*dx-left)
                     for channel in range(ci) for ky in range(kh) for kx in range(kw)]
            slots = [key if 0 <= key[1] < shape[2] and 0 <= key[2] < shape[3] else None for key in slots]
            ids = [ordinals[key] if key is not None else None for key in slots]
            output = branch['output_shape']
            outs = [(channel*output[2]+position[0])*output[3]+position[1] for channel in range(co)]
            identity = dict(model_sha256=source['model_sha256'],
                            source_port=data['original_source_relu']['output'],
                            target_port=branch['consumer_relu']['output'])
            coefficients = dict(source_packet_ref='source_packet_ref', branch_index=index,
                                post_affine_rule='d015_batch_binding_v2.post_affine')
            if (window['source_slots'] != [list(key) if key is not None else None for key in slots]
                    or window['source_ordinals'] != ids or window['output_ordinals'] != outs
                    or window.get('source_identity_ref') != identity
                    or window.get('receiver_coefficients_ref') != coefficients):
                raise ValueError('original source/output/padding identity differs')
            physical_bounds = [bounds[key] if key is not None else (Fraction(0), Fraction(0)) for key in slots]
            relation = window['relation']
            totals, states = check_window(relation, ids, physical_bounds, outs)
            for name in COUNTS:
                counts[name] += totals[name]
            real = [key for key in slots if key is not None]
            seen_sources.update(real)
            canonical += co*len(slots); valid += co*len(real); padding += co*(len(slots)-len(real))
            active += states.count('A'); inactive += states.count('I'); opened += states.count('O')
            transient = max(transient, relation['transient_numeric_entries'])
            kernel_work += relation['work_used']
        summary = data['summary']
        equal_counts(summary, dict(expected_windows=len(expected), completed_windows=len(expected),
            expected_receiver_rows=receivers, completed_receiver_rows=receivers,
            canonical_slots=canonical, valid_slots=valid, padding_slots=padding,
            unique_source_count=len(bounds), strict_crossing_source_count=sum(lo < 0 < hi for lo,hi in bounds.values()),
            strict_active_receiver_rows=active, strict_inactive_receiver_rows=inactive,
            open_receiver_rows=opened, max_slots=max_slots, max_transient_numeric_entries=transient,
            original_bits_deleted=0, **counts))
        if (seen_sources != set(bounds) or summary.get('actual_phase_column_binding_verified') is not False
                or kernel_work > receipt['numerical_model_work']):
            raise ValueError('complete model source/work population differs')


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started, test_started = time.monotonic(), None
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    helper, shared, production, log = None, None, None, None
    identities, inputs = {}, {}
    record = dict(scope=SCOPE, component_tests_passed=False, mathematical_component_gate_passed=False,
        worker_stage_registered=True, worker_launched=False, worker_exit=None,
        worker_wall_cap_s=240, host_observations_within_caps=False,
        complete_physical_qualification=False, candidate_physical_gate_evaluated=False,
        native_HZ_admitted=False, gpu_computation_completed=False,
        source_census_completed=False, source_census_qualified=False,
        actual_phase_column_binding_verified=False, formal_gain=0)

    def emit(value):
        line = json.dumps(value, sort_keys=True, allow_nan=False)
        if log is not None:
            log.write(line + '\n')
            log.flush()
        print(line, flush=True)

    try:
        log = (RUN / 'supervisor.log').open('x')
        limits()
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        sys.dont_write_bytecode = True
        tracemalloc.start()
        if not __debug__ or os.environ.get('PYTHONOPTIMIZE') not in (None, '', '0'):
            raise ValueError('assertions required')
        (RUN / 'tmp').mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1', CUDA_VISIBLE_DEVICES='',
            CUDA_LOG_FILE='stderr', CUDA_CACHE_PATH=str(RUN / 'cuda_cache'),
            TORCH_HOME=str(RUN / 'torch_home'), XDG_CACHE_HOME=str(RUN / 'xdg_cache'),
            TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
            TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        os.environ.update(env)
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        prefreeze = check_freeze()
        identities.update(prefreeze['source_sha256'])
        identities[str(FREEZE)] = sha(FREEZE)
        for path, digest in ANCHORS.items():
            if sha(path) != digest:
                raise ValueError('frozen D056/paper/helper authority drift: ' + str(path))
            identities[str(path)] = digest
        # Authenticated stdlib-only modules: no old main, worker, writer or globals mutation.
        shared = load(D038 / 'run_reference.py', 'd057_d038_readonly_helpers')
        helper = load(D015 / 'run_v2.py', 'd057_d015_readonly_helpers')
        gpu = load(D017 / 'gpu_preflight.py', 'd057_d017_readonly_helpers')
        record['initial_memory'] = shared.memory()
        prior, done = read_json(PRIOR / 'preregistered.json'), read_json(PRIOR / 'exit.json')
        inventory = read_json(PRIOR / 'inventory.json')
        if (done['component_tests_passed'] is not True
                or done['mathematical_component_gate_passed'] is not True
                or done['tests_exit'] != 0 or done['tests_count'] != 3793
                or done['all_stages_passed'] is not True or done.get('failure')
                or done['worker_stage_registered'] is not False
                or done['worker_launched'] is not False or done['worker_exit'] is not None
                or done['source_census_completed'] is not False
                or done['source_census_qualified'] is not False
                or done['host_observations_within_caps'] is not True or done['formal_gain'] != 0
                or done['native_HZ_admitted'] is not False or done['gpu_computation_completed'] is not False
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or prior['required_tests'] != 3793 or prior['required_test_files'] != 175
                or len(prior['tests']) != 175 or len(set(prior['tests'])) != 175
                or len(prior['expected_nodeids']) != 3793
                or len(set(prior['expected_nodeids'])) != 3793
                or inventory['count'] != 3793 or inventory['files'] != 175
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('D056 complete passed mathematical gate differs')
        prior_attempt = prior['prior_attempt']
        if (type(prior_attempt) is not dict or prior_attempt != done['prior_attempt']
                or prior.get('prior_census_failure_preserved') is not True
                or done.get('prior_census_failure_preserved') is not True
                or prior_attempt.get('path') != str(FAILED_CENSUS)
                or prior_attempt.get('tests_count') != 3781 or prior_attempt.get('test_files') != 172
                or prior_attempt.get('mathematical_component_gate_passed') is not True
                or prior_attempt.get('all_stages_passed') is not False
                or prior_attempt.get('source_census_completed') is not False
                or prior_attempt.get('source_census_qualified') is not False
                or prior_attempt.get('worker_exit') != 1 or prior_attempt.get('formal_gain') != 0
                or not prior_attempt.get('failure') or not prior_attempt.get('worker_failure')):
            raise ValueError('nested D047 failed census record differs')
        inherited_gate = dict(path=str(PRIOR), tests_count=3793, test_files=175,
                              mathematical_component_gate_passed=True)
        record.update(prior_attempt=prior_attempt, prior_census_failure_preserved=True,
                      inherited_mathematical_gate=inherited_gate)
        for path, digest in prior['source_sha256'].items():
            helper.bind(identities, path, digest)
        inputs.update(prior['input_sha256'])
        for name, digest in done['artifacts'].items():
            helper.bind(identities, helper.original_path(PRIOR, name), digest)
        if any(path not in identities for path in prior['tests']):
            raise ValueError('inherited test file lacks a frozen identity')
        if (Path(sys.executable).resolve() != PYTHON.resolve()
                or sha(sys.executable) != identities[str(PYTHON.resolve())]):
            raise ValueError('frozen interpreter differs')
        selected = helper.select_sources(identities, inputs)
        if selected != prior['selected_sources']:
            raise ValueError('three original read-only source identities differ')
        decoder = helper.bind_decoder(identities)
        if decoder != read_json(OLDER)['decoder_dependency_files']:
            raise ValueError('frozen decoder dependency population differs')
        dependencies = gpu.gpu_dependencies(helper, identities)
        if dependencies != prior['gpu_dependency_files']:
            raise ValueError('frozen GPU dependency population differs')
        production = helper.provenance()
        if production != prior['provenance'] or production['branch'] != 'redu-hz':
            raise ValueError('production provenance differs')
        if helper.drift(identities) or helper.drift(inputs):
            raise ValueError('pre-run source/input drift')
        test_path = HERE / 'test_window_relation.py'
        tree = ast.parse(test_path.read_text())
        functions = [node for node in tree.body
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                     and node.name.startswith('test_')]
        if (len(functions) != 4 or len({node.name for node in functions}) != 4
                or any(not isinstance(node, ast.FunctionDef) or node.decorator_list
                       or node.args.args or node.args.posonlyargs or node.args.kwonlyargs
                       or node.args.vararg or node.args.kwarg for node in functions)
                or [node.name for node in functions] != prefreeze['new_test_names']):
            raise ValueError('exact four frozen plain top-level new tests required')
        relative = str(test_path.relative_to(ROOT))
        tests = [*prior['tests'], str(test_path)]
        expected = [*inventory['nodeids'], *(relative + '::' + node.name for node in functions)]
        if len(tests) != 176 or len(set(tests)) != 176 or len(expected) != 3797 or len(set(expected)) != 3797:
            raise ValueError('complete 3797/176 population differs')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            selected_sources=selected, decoder_dependency_files=decoder,
            gpu_dependency_files=dependencies, freeze_path=str(FREEZE), freeze_sha256=sha(FREEZE),
            required_tests=3797, required_test_files=176, inherited_tests=3793,
            inherited_test_files=175, new_test_names=prefreeze['new_test_names'],
            prior_attempt=prior_attempt, prior_census_failure_preserved=True,
            inherited_mathematical_gate=inherited_gate,
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, worker_wall_cap_s=240,
            worker_stage_registered=True, expected_models=3, scope=SCOPE, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, whole_work_cap=256_000_000,
            branch_work_cap=200_000_000, evidence_prepaid_work=40_000_000,
            retained_entry_cap=64_000_000, rational_bit_cap=512,
            caches_relocated_to_new_run=True, cuda_visible_devices='',
            complete_physical_qualification=False, candidate_physical_gate_evaluated=False,
            source_census_completed=False, source_census_qualified=False,
            actual_phase_column_binding_verified=False,
            native_HZ_admitted=False, gpu_computation_completed=False, formal_gain=0))
        emit(dict(event='frozen_before_candidate_import', tests=3797, files=176, scope=SCOPE))
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_started = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if (collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3797
                or len({node.split('::', 1)[0] for node in ids}) != 176):
            raise ValueError('exact complete collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=3797, files=176))
        remaining = 60 - (time.monotonic() - test_started)
        if remaining <= 0:
            raise TimeoutError('collection exhausted combined 60-second budget')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=limits)
        record.update(test_wall_s=time.monotonic() - test_started,
                      tests_exit=tested.returncode, tests_count=3797)
        test_started = None
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', '')
                  for case in cases]
        if (tested.returncode or record['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(case.find(key) is not None for case in cases for key in ('failure', 'error', 'skipped'))):
            raise ValueError('complete inherited and mathematical component gate failed')
        record['component_tests_passed'] = True
        emit(dict(event='complete_component_pass', tests=3797, files=176, wall_s=record['test_wall_s']))
        if helper.drift(identities) or helper.drift(inputs) or helper.provenance() != production:
            raise ValueError('source/input/provenance drift before census worker')
        record['worker_launched'] = True
        with (RUN / 'diagnostic.log').open('x') as stream:
            worker = subprocess.run([sys.executable, '-B', str(HERE / 'census.py'), '--enabled'],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=240, preexec_fn=limits)
        record['worker_exit'] = worker.returncode
        record['worker_diagnostic_present'] = (RUN / 'diagnostic.json').is_file()
        if record['worker_diagnostic_present']:
            diagnostic = read_json(RUN / 'diagnostic.json', RESERVE)
            record['worker_diagnostic'] = diagnostic
            record['source_census_completed'] = diagnostic.get('source_census_completed') is True
        else:
            raise ValueError('worker did not preserve its terminal diagnostic')
        if worker.returncode != 0:
            raise ValueError('source census worker failed; partial evidence retained')
        check_worker_resources(diagnostic, selected)
        check_complete_evidence(diagnostic, selected, identities)
        record['worker_evidence_checked'] = True
        emit(dict(event='complete_three_source_census', models=3))
    except BaseException as exc:
        if test_started is not None:
            record['test_wall_s'] = time.monotonic() - test_started
        record['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
        emit(dict(event='failed', failure=record['failure']))
    finally:
        try:
            record['source_drift'] = [p for p, d in identities.items() if sha(p) != d]
            record['input_drift'] = [p for p, d in inputs.items() if sha(p) != d]
            record['provenance_drift'] = (production is not None
                and (helper is None or helper.provenance() != production))
            if record['source_drift'] or record['input_drift'] or record['provenance_drift']:
                raise ValueError('final source/input/provenance drift')
        except BaseException as exc:
            record['component_tests_passed'] = False
            record['final_identity_check_failure'] = str(exc)[:4096]
            record.setdefault('failure', dict(type=type(exc).__name__, reason='final identity check failed'))
        if log is not None:
            log.close()
            log = None
        record['artifacts'] = {}
        try:
            for path in RUN.rglob('*'):
                if path.is_file():
                    record['artifacts'][str(path.relative_to(RUN))] = sha(path)
        except BaseException as exc:
            record['artifact_sealing_failure'] = str(exc)[:4096]
            record.setdefault('failure', dict(type=type(exc).__name__, reason='artifact sealing incomplete'))
        try:
            if not tracemalloc.is_tracing() or shared is None:
                raise ValueError('supervisor host telemetry unavailable')
            _, peak = tracemalloc.get_traced_memory()
            metadata = tracemalloc.get_tracemalloc_memory()
            growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
            record.update(final_memory=shared.memory(), rss_highwater_growth_bytes=growth,
                traced_peak_bytes=peak, tracer_metadata_bytes=metadata, summary_reserve_bytes=RESERVE,
                host_observations_within_caps=(growth + RESERVE <= MEMORY_CAP
                    and peak + metadata + RESERVE <= MEMORY_CAP))
        except BaseException as exc:
            record['host_observations_within_caps'] = False
            record['memory_check_failure'] = str(exc)[:4096]
        if not record['host_observations_within_caps']:
            record.setdefault('failure', dict(type='MemoryError', reason='supervisor host gate failed'))
        record.update(wall_s=time.monotonic() - started,
            memory_scope='supervisor only; pytest has AS/CPU/time limits, not full physical qualification')
        record['mathematical_component_gate_passed'] = (record['component_tests_passed']
            and record['host_observations_within_caps']
            and 'final_identity_check_failure' not in record
            and not record.get('source_drift') and not record.get('input_drift')
            and not record.get('provenance_drift'))
        record['source_census_qualified'] = (record['source_census_completed']
            and record['mathematical_component_gate_passed']
            and record.get('worker_evidence_checked') is True and 'failure' not in record)
        record['all_stages_passed'] = record['source_census_qualified']
        record['supervisor_exit'] = 0 if record['all_stages_passed'] else 1
        save('exit.json', record)
        print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)
    return record['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
