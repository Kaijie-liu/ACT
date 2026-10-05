"""D047 single-use complete tests and three-source multiphase census."""
import ast
from fractions import Fraction
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
RUN = EXP / 'results/d047_multiphase_source_census_20260930_v1'
PRIOR = EXP / 'results/d046_multiphase_envelopes_20260930_v1'
D046 = HERE.parent / 'd046_multiphase_envelopes_20260930'
D038 = HERE.parent / 'd038_descendant_census_20260930'
D017 = HERE.parent / 'd017_applicability_gpu_20260930'
D015 = HERE.parent / 'd015_batch_binding_20260928_v2'
OLDER = EXP / 'results/d015_source_shielding_20260928_v2/preregistered.json'
PYTHON = Path('/data1/Kane/miniconda3/bin/python')
FREEZE = HERE / 'freeze.json'
NEW_FILES = ('PREREG.md', 'THEORY.md', 'seed_relation.py', 'test_seed_relation.py', 'census.py', 'run_census.py')
AS_CAP, MEMORY_CAP, RESERVE = 16 * 1024**3, 1024**3, 65536
ANCHORS = {
    HERE.parent / 'd042_bounded_relational_transfer_20260930/THEORY.md': 'b5d172b2ce7dd60bfc91eebae901824be2bce61a6d9c4d422a5c56496e0f73e7',
    HERE.parent / 'd042_bounded_relational_transfer_20260930/CONTROL.md': '054bf8740b3612fea03a00b66c2b5825fc5e91dd9e12a57dd58226f3314a6566',
    PRIOR / 'preregistered.json': 'a4a5eef3249b0161b0be4322370faba4c42bdfd79d77f4bde3720f5757b99904',
    PRIOR / 'inventory.json': '3e49d547c78940bb6b7428119701e950fc47fb21921eff8f4cd1b5fc16a9d231',
    PRIOR / 'exit.json': '1051d8d61e6d671938b215cfa86df7506ac8161d6bd11a42920a68c385727087',
    D046 / 'freeze.json': 'e0a0569b0854a7c6cfa7b01a76d2e80075a13558ebdd816d2e08fa310cb33d42',
    D046 / 'run_math.py': '83fa2f694a66f1641048ad1e682ddf7e34b7e47fd868a0e7c411fd0020017763',
    D038 / 'freeze.json': '8f24ae5d3e7c674a66e80874e6f607733aeff9d46174434a8a68ce723cf8fc73',
    D038 / 'run_reference.py': 'f4abd1608c70aa308a590307b8f3369eff2ab11b217d54be597c99bb77c0efb5',
    D017 / 'gpu_preflight.py': '902437443f6847482d21d0af227b7fc36234868444f11f4dfc580f2abbf89c01',
    D015 / 'run_v2.py': '8ff9d296ce56b8dd9481d0c8367484b8ff37148db5dfa522062136d9d04e12ce',
    OLDER: '4697d2bbc1b86e7732e825e0e8a3b6d9c595fc275516688d746e738d2bba50cd',
}
SCOPE = ('complete inherited tests plus full three-source first-bank multiphase census; '
         'no native HZ admission, GPU computation or formal verification gain')


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
    if (frozen.get('schema') != 'd047_frozen_v1'
            or frozen.get('required_tests') != 3781 or frozen.get('required_test_files') != 172
            or type(sources) is not dict or set(sources) != {str(HERE / name) for name in NEW_FILES}
            or type(names) is not list or len(names) != 4
            or any(type(name) is not str or not name.startswith('test_') for name in names)
            or len(set(names)) != 4):
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
              'whole_work_used', 'branch_work_used', 'evidence_work_used')
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
    for index, item in enumerate(models):
        if item.get('evidence_file') != 'complete_' + str(index) + '.json':
            raise ValueError('unexpected complete evidence filename')
        path = RUN / item['evidence_file']
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size != item.get('evidence_bytes')
                or sha(path) != item.get('evidence_sha256')):
            raise ValueError('complete evidence identity differs')


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


def evidence_form(value, allowed):
    if type(value) is not list or len(value) != 2 or type(value[1]) is not list:
        raise ValueError('invalid sparse phase form evidence')
    bias = evidence_fraction(value[0])
    previous, terms = -1, []
    for item in value[1]:
        if (type(item) is not list or len(item) != 2
                or type(item[0]) is not int or item[0] <= previous or item[0] not in allowed):
            raise ValueError('noncanonical or unbound original phase')
        coefficient = evidence_fraction(item[1])
        if coefficient == 0:
            raise ValueError('sparse phase form retained a zero coefficient')
        terms.append((item[0], coefficient))
        previous = item[0]
    return bias, terms


def check_complete_evidence(result, selected, identities):
    """Independently check the sealed geometry, full population and phase IDs."""
    seed_names = ('delta_lower', 'delta_upper', 'w_lower', 'w_upper')
    after_names = ('difference_lower', 'difference_upper', 'companion_lower', 'companion_upper')
    extractor = str(D015 / 'source_binding_v2.py')
    data = None
    for index, (receipt, source) in enumerate(zip(result['models'], selected)):
        data = None
        data = read_json(RUN / receipt['evidence_file'], 40_000_000)
        ref = data.get('source_packet_ref', {})
        if (data.get('schema') != 'd047_multiphase_source_census_v1'
                or data.get('source') != source
                or ref.get('model_sha256') != source['model_sha256']
                or ref.get('extractor_path') != extractor
                or ref.get('extractor_sha256') != identities[extractor]
                or data.get('summary') != receipt.get('summary')):
            raise ValueError('complete source or extractor binding differs')
        shape = data.get('first_shape')
        if (type(shape) is not list or len(shape) != 4 or shape[0] != 1
                or any(type(n) is not int or not 0 < n <= 65536 for n in shape)):
            raise ValueError('invalid original first-bank shape')
        frame = data.get('frame_identity')
        if (type(frame) is not list or len(frame) != 2
                or frame[0] != source['model_sha256'] or type(frame[1]) is not str or not frame[1]):
            raise ValueError('original frame identity differs')
        bounds, ordinals = {}, {}
        previous = None
        for item in data['source_bounds']:
            coord = tuple(item['coordinate'])
            if (len(coord) != 3 or any(type(v) is not int for v in coord)
                    or any(not 0 <= v < extent for v, extent in zip(coord, shape[1:]))
                    or coord in bounds or (previous is not None and coord <= previous)):
                raise ValueError('source coordinate population is not canonical')
            ordinal = (coord[0] * shape[2] + coord[1]) * shape[3] + coord[2]
            if item['ordinal'] != ordinal:
                raise ValueError('original phase coordinate ordinal differs')
            bounds[coord] = evidence_interval(item['bounds'])
            ordinals[coord] = ordinal
            previous = coord
        expected, specs = [], {}
        expected_rows = expected_pairs = 0
        for branch_index, branch in enumerate(data['branches']):
            if branch.get('index') != branch_index:
                raise ValueError('raw branch order differs')
            if branch['consumer_relu'] is None:
                continue
            weight, output = branch['weight_shape'], branch['output_shape']
            strides, dilation, pads = branch['strides'], branch['dilations'], branch['pads']
            if (len(weight) != 4 or len(output) != 4 or len(strides) != 2
                    or len(dilation) != 2 or len(pads) != 4 or branch['group'] != 1
                    or any(type(v) is not int or v <= 0 for v in weight + output + strides + dilation)
                    or any(type(v) is not int or v < 0 for v in pads)
                    or weight[1] != shape[1]):
                raise ValueError('unsupported or mismatched original convolution geometry')
            co, ci, kh, kw = weight
            oh = (shape[2] + pads[0] + pads[2] - dilation[0] * (kh - 1) - 1) // strides[0] + 1
            ow = (shape[3] + pads[1] + pads[3] - dilation[1] * (kw - 1) - 1) // strides[1] + 1
            pairs = [[channel, min(channel + 1, co - 1)] for channel in range(0, co, 2)]
            if (output != [1, co, oh, ow] or branch['receiver_count'] != co
                    or branch['channel_pairs'] != pairs or ci * kh * kw > 65536):
                raise ValueError('complete original consumer population differs')
            positions = sorted({(0, 0), (0, ow - 1), (oh - 1, 0),
                                (oh - 1, ow - 1), (oh // 2, ow // 2)})
            expected.extend((branch_index, point) for point in positions)
            expected_rows += co * len(positions)
            expected_pairs += len(pairs) * len(positions)
            specs[branch_index] = branch
        windows = data['windows']
        if not expected or len(windows) != len(expected):
            raise ValueError('complete direct-branch window population differs')
        seen_sources, canonical, valid, padding, nonconstant, nnz = set(), 0, 0, 0, 0, 0
        hinge_total = last_strict_total = ordinary_crossing = pair_slots = 0
        support_totals = {name: 0 for name in after_names}
        for window, (branch_index, position) in zip(windows, expected):
            if window['branch'] != branch_index or window['position'] != list(position):
                raise ValueError('original window ordering differs')
            branch = specs[branch_index]
            co, ci, kh, kw = branch['weight_shape']
            sy, sx = branch['strides']; dy, dx = branch['dilations']
            top, left, _, _ = branch['pads']
            slots = []
            for channel in range(ci):
                for ky in range(kh):
                    for kx in range(kw):
                        y, x = position[0] * sy + ky * dy - top, position[1] * sx + kx * dx - left
                        slots.append((channel, y, x) if 0 <= y < shape[2] and 0 <= x < shape[3] else None)
            if window['source_slots'] != [list(key) if key is not None else None for key in slots]:
                raise ValueError('full canonical padding/source geometry differs')
            if window['source_ordinals'] != [ordinals[key] if key is not None else None for key in slots]:
                raise ValueError('canonical original phase identity differs')
            real = [key for key in slots if key is not None]
            seen_sources.update(real)
            canonical += co * len(slots); valid += co * len(real); padding += co * (len(slots) - len(real))
            allowed = {ordinals[key] for key in real if bounds[key][0] < 0 < bounds[key][1]}
            pairs = window['pairs']
            if [item['channels'] for item in pairs] != branch['channel_pairs']:
                raise ValueError('fixed complete consumer pairs differ')
            for item in pairs:
                relation = item['relation']
                if (relation.get('binding_mathematical_only') is not True
                        or relation.get('actual_phase_column_binding_verified') is not False
                        or relation.get('same_consumer_certified') is not (item['channels'][0] == item['channels'][1])):
                    raise ValueError('source relation exceeded its binding scope')
                pair_slots += len(slots)
                for name in seed_names:
                    _, terms = evidence_form(relation[name], allowed)
                    if relation['supports'][name] != len(terms):
                        raise ValueError('seed support counter differs')
                pair_support = pair_nonconstant = pair_hinge = pair_strict = 0
                for j, name in enumerate(after_names):
                    _, terms = evidence_form(relation[name], allowed)
                    if relation['supports'][name] != len(terms):
                        raise ValueError('output support counter differs')
                    support_totals[name] += len(terms)
                    nonconstant += bool(terms)
                    nnz += len(terms)
                    pair_support += len(terms)
                    pair_nonconstant += bool(terms)
                    hinge, strict = relation['hinge_crossing'][name], relation['last_anchor_strict'][name]
                    gap = evidence_fraction(relation['last_anchor_gap'][name])
                    if type(hinge) is not bool or type(strict) is not bool or gap < 0 or strict != (gap > 0):
                        raise ValueError('invalid conditional diagnostic count')
                    pair_hinge += hinge; pair_strict += strict
                if (relation['nonconstant_relations'] != pair_nonconstant
                        or relation['hinge_crossing_relations'] != pair_hinge
                        or relation['last_anchor_strict_relations'] != pair_strict
                        or relation['potential_coordinate_nnz'] != 6 + pair_support):
                    raise ValueError('relation summary differs from its complete forms')
                hinge_total += pair_hinge; last_strict_total += pair_strict
                # Four readouts: (r-t), -(r-t), t, -t; self-pairs may be cheaper.
                nnz += 6
                lh, uh = evidence_interval(relation['ordinary_h'])
                lw, uw = evidence_interval(relation['ordinary_w'])
                ordinary_crossing += lh < 0 < uh
                if item['channels'][0] != item['channels'][1]:
                    ordinary_crossing += lw < 0 < uw
                evidence_interval(relation['plain_shared_difference'])
        summary = data['summary']
        must_equal = dict(expected_windows=len(expected), completed_windows=len(expected),
            expected_receiver_rows=expected_rows, completed_receiver_rows=expected_rows,
            expected_pairs=expected_pairs, completed_pairs=expected_pairs,
            canonical_slots=canonical, valid_slots=valid, padding_slots=padding,
            unique_source_count=len(bounds), strict_crossing_source_count=sum(l < 0 < u for l, u in bounds.values()),
            nonconstant_relations=nonconstant, support_totals=support_totals,
            pair_canonical_slots=pair_slots, hinge_crossing_relations=hinge_total,
            last_anchor_strict_relations=last_strict_total, ordinary_crossing_receiver_rows=ordinary_crossing,
            physical_coordinate_nnz_upper=nnz)
        if seen_sources != set(bounds) or any(summary.get(key) != value for key, value in must_equal.items()):
            raise ValueError('complete evidence totals differ from all sealed records')


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
                raise ValueError('frozen D046/helper authority drift: ' + str(path))
            identities[str(path)] = digest
        # Authenticated stdlib-only modules: no old main, worker, writer or globals mutation.
        shared = load(D038 / 'run_reference.py', 'd047_d038_readonly_helpers')
        helper = load(D015 / 'run_v2.py', 'd047_d015_readonly_helpers')
        gpu = load(D017 / 'gpu_preflight.py', 'd047_d017_readonly_helpers')
        record['initial_memory'] = shared.memory()
        prior, done = read_json(PRIOR / 'preregistered.json'), read_json(PRIOR / 'exit.json')
        inventory = read_json(PRIOR / 'inventory.json')
        if (done['component_tests_passed'] is not True or done['tests_exit'] != 0
                or done['tests_count'] != 3777 or done['all_stages_passed'] is not True
                or done['host_observations_within_caps'] is not True or done['formal_gain'] != 0
                or done['source_drift'] or done['input_drift'] or done['provenance_drift']
                or prior['required_tests'] != 3777 or prior['required_test_files'] != 171
                or len(prior['tests']) != 171 or len(set(prior['tests'])) != 171
                or len(prior['expected_nodeids']) != 3777
                or len(set(prior['expected_nodeids'])) != 3777
                or inventory['count'] != 3777 or inventory['files'] != 171
                or sorted(inventory['nodeids']) != sorted(prior['expected_nodeids'])):
            raise ValueError('D046 passed complete component population differs')
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
        test_path = HERE / 'test_seed_relation.py'
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
        if len(tests) != 172 or len(set(tests)) != 172 or len(expected) != 3781 or len(set(expected)) != 3781:
            raise ValueError('complete 3781/172 population differs')
        save('preregistered.json', dict(source_sha256=identities, input_sha256=inputs,
            provenance=production, tests=tests, expected_nodeids=expected,
            selected_sources=selected, decoder_dependency_files=decoder,
            gpu_dependency_files=dependencies, freeze_path=str(FREEZE), freeze_sha256=sha(FREEZE),
            required_tests=3781, required_test_files=172, inherited_tests=3777,
            inherited_test_files=171, new_test_names=prefreeze['new_test_names'],
            cpu_affinity=list(os.sched_getaffinity(0)), address_space_bytes=AS_CAP,
            tests_combined_wall_cap_s=60, worker_wall_cap_s=240,
            worker_stage_registered=True, scope=SCOPE, host_memory_cap_bytes=MEMORY_CAP,
            summary_reserve_bytes=RESERVE, whole_work_cap=256_000_000,
            branch_work_cap=200_000_000, evidence_prepaid_work=40_000_000,
            retained_entry_cap=64_000_000, rational_bit_cap=512,
            caches_relocated_to_new_run=True, cuda_visible_devices='',
            complete_physical_qualification=False, candidate_physical_gate_evaluated=False,
            expected_models=3, source_census_qualified=False,
            native_HZ_admitted=False, gpu_computation_completed=False, formal_gain=0))
        emit(dict(event='frozen_before_candidate_import', tests=3781, files=172, scope=SCOPE))
        command = [sys.executable, '-B', '-m', 'pytest', '-q', '--tb=short',
                   '-p', 'no:cacheprovider', *tests]
        test_started = time.monotonic()
        with (RUN / 'collection.log').open('x') as stream:
            collected = subprocess.run([*command, '--collect-only'], cwd=ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, timeout=60, preexec_fn=limits)
        ids = [line for line in (RUN / 'collection.log').read_text().splitlines()
               if line.startswith(('experiments/', 'act/')) and '::' in line]
        if (collected.returncode or sorted(ids) != sorted(expected) or len(set(ids)) != 3781
                or len({node.split('::', 1)[0] for node in ids}) != 172):
            raise ValueError('exact complete collection inventory differs')
        save('inventory.json', dict(nodeids=ids, count=3781, files=172))
        remaining = 60 - (time.monotonic() - test_started)
        if remaining <= 0:
            raise TimeoutError('collection exhausted combined 60-second budget')
        with (RUN / 'tests.log').open('x') as stream:
            tested = subprocess.run([*command, '--junitxml=' + str(RUN / 'tests.xml')],
                cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                timeout=remaining, preexec_fn=limits)
        record.update(test_wall_s=time.monotonic() - test_started,
                      tests_exit=tested.returncode, tests_count=3781)
        test_started = None
        cases = ET.parse(RUN / 'tests.xml').findall('.//testcase')
        actual = [case.get('classname', '').replace('.', '/') + '.py::' + case.get('name', '')
                  for case in cases]
        if (tested.returncode or record['test_wall_s'] > 60 or sorted(actual) != sorted(expected)
                or any(case.find(key) is not None for case in cases for key in ('failure', 'error', 'skipped'))):
            raise ValueError('complete inherited and mathematical component gate failed')
        record['component_tests_passed'] = True
        emit(dict(event='complete_component_pass', tests=3781, files=172, wall_s=record['test_wall_s']))
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
