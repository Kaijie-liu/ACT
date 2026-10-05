"""Once-only metadata admission audit; never import or execute a candidate."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time
import tracemalloc


HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d210_full_prefix_admission_20261005_v1'
PRIOR = EXP / 'results/d209_owned_slack_closure_20261005_v1'
STRUCTURE = EXP / 'results/d179_preterminal_domain_20261004_v1'
STRUCTURE_INPUTS = HERE.parent / 'd179_preterminal_domain_20261004/inputs.json'
FREEZE = HERE / 'freeze.json'
SCHEMA = 'd210_full_prefix_admission_v1'
FILES = ('PREREG.md', 'THEORY.md', 'run_audit.py')
AS_CAP, WALL_CAP, ENTRIES_CAP = 16 * 1024**3, 60, 64_000_000
MEMORY_CAP, RESERVE = 1024**3, 65536
EXPECTED_DIMENSIONS = ((3072, 65536, 65536), (3072, 14400, 8192), (9408, 46656, 25088))
ANCHORS = {
    PRIOR / 'preregistered.json': '06bf3b6dfce47c0d83571f459e3269acc064001e1dfa2b252979e2185ac846b6',
    PRIOR / 'exit.json': 'd2220587a746d95f3e1db62a7d215aafb9384c24a78f83fe83787da24f706f2e',
    STRUCTURE_INPUTS: 'ecc18b0f01c6ec08fd765274e138a96939405681d27d8578dfd04b76f375eef3',
    STRUCTURE / 'result.json': 'ef8598fba10579778bd0a2668e24d45b9cd8249ccc58ac07de26d72d09caa3ae',
    STRUCTURE / 'model_0.json': '5c1fbd033e91c5fe044a409a782f7f7aa39bc98b47a42719e615b95cc6733da1',
    STRUCTURE / 'model_1.json': 'f93c16ae170e62d688a76a4f630dfadfb8dcaf75a394b0e59eaf87390405aadb',
    STRUCTURE / 'model_2.json': 'd6e049295ba60748fee67a9bf9a8a7bc7716d07964ccff413aac2cda3132f4e7',
    HERE.parent / 'd209_owned_slack_closure_20261005/fiber.py':
        'ef8bafaa09512d36c268a72caf8725af5325c73edcc9b721b291ad6210d04430',
    HERE.parent / 'd207_owned_source_phase_20261005/fiber.py':
        '1e33d626749abecba4236a932151edd3411ccfacc1418049c5bd1fd8c6d5f1ef',
}
# Preserve the original D209 production digest algorithm without importing it.
PRODUCTION = (
    'act/back_end/solver/neural_hz.py', 'act/back_end/solver/solver_hz.py',
    'act/back_end/hybridz_tf/hybridz_tf.py', 'act/back_end/hybridz_tf/tf_mlp.py',
    'act/back_end/hybridz_tf/tf_cnn.py', 'act/back_end/hybridz_tf/exact_linear_op.py',
    'act/back_end/verifier.py', 'act/config/config.py',
    'experiments/neural_hz_20260831/shadow_worker.py',
    'experiments/neural_hz_20260831/shadow_worker_dtype_v2.py',
    'act/pipeline/verification/torch2act.py',
    'act/pipeline/verification/batchnorm_graph.py',
    'act/front_end/vnnlib_loader/onnx_converter.py',
)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def check(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError('complete metadata audit exceeded 60 seconds')


def alarm(_signum, _frame):
    raise TimeoutError('complete metadata audit exceeded 60 seconds')


def identity_map(value, count=None):
    require(type(value) is dict and (count is None or len(value) == count),
            'identity population differs')
    for name, digest in value.items():
        require(type(name) is str and Path(name).is_absolute()
                and type(digest) is str and len(digest) == 64
                and all(c in '0123456789abcdef' for c in digest), 'malformed identity')
    return value


def merge(target, incoming):
    for name, digest in identity_map(incoming).items():
        require(name not in target or target[name] == digest, 'conflicting identity: ' + name)
        target[name] = digest


def stream_hash(path, deadline, stats, phase, accumulator=None):
    """Only one 1 MiB byte block at a time, including raw ONNX files."""
    check(deadline)
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), 'missing or linked source: ' + str(path))
    record = stats.setdefault(phase, dict(files_started=0, files_completed=0,
                                        bytes_read=0, wall_s=0.0))
    record['files_started'] += 1
    started, digest = time.monotonic(), hashlib.sha256()
    try:
        with path.open('rb') as stream:
            while True:
                check(deadline)
                block = stream.read(1024**2)
                if not block:
                    break
                record['bytes_read'] += len(block)
                digest.update(block)
                if accumulator is not None:
                    accumulator.update(block)
        record['files_completed'] += 1
        return digest.hexdigest()
    finally:
        record['wall_s'] += time.monotonic() - started


def verify(mapping, deadline, stats, phase, drift):
    for name, digest in mapping.items():
        if stream_hash(name, deadline, stats, phase) != digest:
            drift.append(name)
            raise ValueError('identity drift: ' + name)


def read_json(path, deadline, stats):
    check(deadline)
    path = Path(path)
    require(not path.is_symlink() and path.is_file()
            and 0 < path.stat().st_size <= 8 * 1024**2, 'invalid metadata JSON: ' + str(path))
    raw = path.read_bytes()
    stats['json_files_read'] += 1
    stats['json_bytes_read'] += len(raw)
    check(deadline)
    return json.loads(raw)


def save(name, value):
    require(name in ('preregistered.json', 'admission.json', 'exit.json'), 'unregistered output')
    require(RUN.is_dir() and not RUN.is_symlink(), 'exclusive output directory changed')
    with (RUN / name).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')


def provenance(deadline, stats, phase):
    def git(*args):
        check(deadline)
        return subprocess.check_output(['git', *args], cwd=ROOT,
            env=dict(os.environ, GIT_OPTIONAL_LOCKS='0'),
            timeout=max(0.001, min(3.0, deadline - time.monotonic())))
    combined = hashlib.sha256()
    for name in PRODUCTION:
        combined.update(name.encode())
        stream_hash(ROOT / name, deadline, stats, phase, combined)
    return dict(branch=git('branch', '--show-current').decode().strip(),
        commit=git('rev-parse', 'HEAD').decode().strip(),
        candidate_sha256=combined.hexdigest(),
        tracked_diff_sha256=hashlib.sha256(git('diff', '--binary', 'HEAD', '--')).hexdigest())


def dimensions(value):
    require(type(value) is list and len(value) == 4 and value[0] == 1
            and all(type(v) is int and v > 0 for v in value), 'complete batch-one shape required')
    return math.prod(value[1:])


def inspect(model, source, receipt):
    require(model.get('schema') == 'd179_preterminal_domain_v1'
            and model.get('structure_complete') is True and model.get('metadata_complete') is True
            and model.get('structural_shape_binding_only') is True
            and model.get('sample_batch_binding') == 1 and model.get('source') == source
            and receipt.get('source') == source and receipt.get('structure_complete') is True
            and model.get('coefficient_values_read') is False
            and model.get('native_domain_state_bound') is False, 'D179 structural qualification differs')
    nodes = model.get('nodes')
    require(type(nodes) is list and len(nodes) >= 6, 'complete graph node metadata missing')
    require(all(type(node) is dict and node.get('index') == i for i, node in enumerate(nodes)),
            'graph node index population differs')
    prefix = nodes[:6]
    require([n.get('op') for n in prefix] == ['Conv', 'BatchNormalization', 'Relu',
            'Conv', 'BatchNormalization', 'Relu'], 'first complete two-bank chain differs')
    for i, node in enumerate(prefix):
        require(type(node.get('inputs')) is list and node['inputs']
                and type(node.get('output')) is str and node['output'], 'malformed prefix ports')
        dimensions(node.get('output_shape'))
        if i:
            require(node['inputs'][0] == prefix[i - 1]['output'], 'first-bank path is not consecutive')
    require(prefix[0]['inputs'][0] == 'modelInput', 'prefix does not begin at the original input')
    require(prefix[0]['output_shape'] == prefix[1]['output_shape'] == prefix[2]['output_shape']
            and prefix[3]['output_shape'] == prefix[4]['output_shape'] == prefix[5]['output_shape'],
            'BN/ReLU shape identity differs')
    d, m, n = (dimensions(model['input_shape']), dimensions(prefix[2]['output_shape']),
               dimensions(prefix[5]['output_shape']))
    port = prefix[2]['output']
    uses = [[node['index'], slot] for node in nodes
            for slot, value in enumerate(node.get('inputs', [])) if value == port]
    require(type(model.get('consumers')) is dict and model['consumers'].get(port) == uses
            and [3, 0] in uses, 'complete first-bank consumer population differs')
    consumer_nodes = [nodes[index] for index in sorted({index for index, _ in uses})]
    lower = 9 * d * m
    require(lower > ENTRIES_CAP, 'registered complete first-bank exclusion was not proved')
    return dict(source=source, input_shape=model['input_shape'], first_six_nodes=prefix,
        first_bank_port=port, first_bank_consumers=uses, first_bank_consumer_nodes=consumer_nodes,
        d=d, m=m, n=n, identity_entry_lower_bound=lower, entries_cap=ENTRIES_CAP,
        first_unavoidable_gate=ENTRIES_CAP // (9 * d) + 1,
        gate_indexing='one-based; actual fail-closed may occur earlier',
        lower_bound_terms=dict(readout_prefix_copies_per_gate=1, support_directions=2,
                               proof_prefix_copies_per_direction=4),
        candidate_admitted=False, candidate_executed=False, coefficient_values_read=False,
        native_domain_state_bound=False, all_first_bank_consumers_retained_in_metadata=True,
        claim_scope='frozen D209 cumulative logical-entry lower bound, not resident memory or measured runtime')


def main():
    require(sys.argv[1:] == ['--enabled'], 'explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started = time.monotonic()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    deadline = started + WALL_CAP
    stats = dict(json_files_read=0, json_bytes_read=0)
    sources, inputs, prior, before = {}, {}, None, None
    outputs = []
    result = dict(schema=SCHEMA, metadata_only=True, metadata_audit_passed=False,
        candidate_admitted=False, candidate_executed=False, candidate_execution_registered=False,
        mathematical_tests_executed=False, mathematical_component_gate_passed=False,
        source_component_qualified=False, source_census_qualified=False,
        actual_model_binding_qualified=False, actual_phase_column_binding_verified=False,
        native_HZ_admitted=False, gpu_computation_completed=False,
        complete_physical_qualification=False, solver_executed=False, model_forward_executed=False,
        source_precheck_complete=False, source_postcheck_complete=False,
        input_precheck_complete=False, input_postcheck_complete=False,
        source_drift=[], input_drift=[], provenance_drift=None,
        formal_gain=0, new_benchmark_solves=0, source_count=0, input_count=0,
        inherited_tests=4100, inherited_test_files=216, new_tests=0,
        address_space_bytes=AS_CAP, total_wall_cap_s=WALL_CAP,
        timing_scope='metadata, streaming identity checks and git only; no candidate timing',
        memory_scope='supervisor RSS/trace only; no candidate or full physical qualification')
    try:
        signal.signal(signal.SIGALRM, alarm)
        signal.setitimer(signal.ITIMER_REAL, WALL_CAP)
        tracemalloc.start()
        resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
        require(0 in os.sched_getaffinity(0), 'required CPU 0 is unavailable')
        os.sched_setaffinity(0, {0})
        result['cpu_affinity'] = list(os.sched_getaffinity(0))
        sys.dont_write_bytecode = True
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        frozen = read_json(FREEZE, deadline, stats)
        new_sources = identity_map(frozen.get('source_sha256'), 3)
        anchor_sources = {str(path): digest for path, digest in ANCHORS.items()}
        require(frozen.get('schema') == SCHEMA and frozen.get('metadata_only') is True
                and frozen.get('candidate_execution_registered') is False
                and set(new_sources) == {str(HERE / name) for name in FILES}
                and frozen.get('anchor_sha256') == anchor_sources, 'frozen D210 contract differs')
        merge(sources, new_sources)
        merge(sources, anchor_sources)
        sources[str(FREEZE)] = stream_hash(FREEZE, deadline, stats, 'freeze_identity')
        verify(sources, deadline, stats, 'anchor_precheck', result['source_drift'])
        prior = read_json(PRIOR / 'preregistered.json', deadline, stats)
        done = read_json(PRIOR / 'exit.json', deadline, stats)
        require(prior.get('schema') == 'd209_owned_slack_closure_v1'
                and prior.get('required_tests') == 4100 and prior.get('required_test_files') == 216
                and prior.get('cpu_affinity') == [0]
                and done.get('schema') == 'd209_owned_slack_closure_v1'
                and done.get('tests_count') == 4100 and done.get('test_files') == 216
                and done.get('tests_exit') == 0 and done.get('supervisor_exit') == 0
                and 0 <= done.get('test_wall_s', 61) <= 60 and 'failure' not in done
                and all(done.get(k) is True for k in ('component_tests_passed',
                    'mathematical_component_gate_passed', 'host_observations_within_caps',
                    'inventory_validated_before_execution', 'all_registered_stages_passed'))
                and done.get('formal_gain') == 0 and done.get('new_benchmark_solves') == 0
                and done.get('source_drift') == [] and done.get('input_drift') == []
                and done.get('provenance_drift') is False, 'D209 complete receipt differs')
        require(all(done.get(k) is False for k in ('worker_launched', 'worker_stage_registered',
                    'source_component_qualified', 'actual_model_binding_qualified',
                    'actual_phase_column_binding_verified', 'native_HZ_admitted',
                    'gpu_computation_completed', 'complete_physical_qualification')),
                'D209 qualification boundary differs')
        merge(sources, identity_map(prior.get('source_sha256'), 7425))
        merge(inputs, identity_map(prior.get('input_sha256'), 14))
        artifacts = done.get('artifacts')
        require(type(artifacts) is dict and 'inventory.json' in artifacts, 'D209 artifacts missing')
        for name, digest in artifacts.items():
            path = PRIOR / name
            require(type(name) is str and path.resolve() == path and path.is_relative_to(PRIOR),
                    'historical artifact path escapes D209')
            merge(sources, {str(path): digest})
        result.update(source_count=len(sources), input_count=len(inputs),
                      inherited_source_count=7425, inherited_input_count=14)
        verify(sources, deadline, stats, 'source_precheck', result['source_drift'])
        result['source_precheck_complete'] = True
        verify(inputs, deadline, stats, 'input_precheck', result['input_drift'])
        result['input_precheck_complete'] = True
        require(Path(sys.executable).resolve() == Path('/data1/Kane/miniconda3/bin/python').resolve()
                and str(Path(sys.executable).resolve()) in sources, 'unregistered interpreter')
        inventory = read_json(PRIOR / 'inventory.json', deadline, stats)
        tests, nodeids = prior.get('tests'), prior.get('expected_nodeids')
        require(type(tests) is list and len(tests) == len(set(tests)) == 216
                and type(nodeids) is list and len(nodeids) == len(set(nodeids)) == 4100
                and all(path in sources for path in tests)
                and inventory.get('nodeids') == nodeids and inventory.get('count') == 4100
                and inventory.get('files') == 216 and inventory.get('validated_before_execution') is True
                and inventory.get('manifest_sha256') == ANCHORS[PRIOR / 'preregistered.json'],
                'inherited ordered mathematical population differs')
        structure_inputs = read_json(STRUCTURE_INPUTS, deadline, stats)
        structure_receipt = read_json(STRUCTURE / 'result.json', deadline, stats)
        require(structure_inputs.get('schema') == 'd179_preterminal_domain_v1'
                and len(structure_inputs.get('models', [])) == 3
                and structure_receipt.get('diagnostic_complete') is True
                and structure_receipt.get('exit_status') == 0
                and all(structure_receipt.get(k) is True for k in ('source_precheck_complete',
                    'source_postcheck_complete', 'input_precheck_complete', 'input_postcheck_complete'))
                and len(structure_receipt.get('models', [])) == 3, 'D179 complete receipt differs')
        before = provenance(deadline, stats, 'production_precheck')
        result['provenance_before'] = before
        require({k: before[k] for k in ('branch', 'commit', 'candidate_sha256')} == prior['provenance']
                and {k: before[k] for k in ('branch', 'commit', 'tracked_diff_sha256')}
                    == structure_receipt['provenance_before'] == structure_receipt['provenance_after'],
                'production provenance differs')
        save('preregistered.json', dict(schema=SCHEMA, source_sha256=sources,
            input_sha256=inputs, inherited_source_count=7425, inherited_input_count=14,
            inherited_tests=tests, expected_nodeids=nodeids, required_tests=4100,
            required_test_files=216, tests_executed=False, new_tests=0,
            inherited_mathematical_receipt=dict(path=str(PRIOR),
                manifest_sha256=ANCHORS[PRIOR / 'preregistered.json'],
                exit_sha256=ANCHORS[PRIOR / 'exit.json'], mathematical_component_gate_passed=True,
                qualification_transferred=False), source_models=structure_inputs['models'],
            property_selection='none; no new spec or instance is selected',
            provenance=before, cpu_affinity=[0], address_space_bytes=AS_CAP,
            total_wall_cap_s=WALL_CAP, metadata_only=True, candidate_execution_registered=False,
            candidate_admitted=False, formal_gain=0))
        outputs.append('preregistered.json')
        records = []
        for index, source in enumerate(structure_inputs['models']):
            check(deadline)
            path = STRUCTURE / ('model_%d.json' % index)
            receipt = structure_receipt['models'][index]
            require(receipt.get('index') == index and receipt.get('artifact') == path.name
                    and receipt.get('sha256') == ANCHORS[path]
                    and inputs.get(source['model_path']) == source['model_sha256'],
                    'complete authenticated model population differs')
            record = inspect(read_json(path, deadline, stats), source, receipt)
            require((record['d'], record['m'], record['n']) == EXPECTED_DIMENSIONS[index],
                    'preregistered complete bank dimensions differ')
            records.append(dict(index=index, structure_artifact=str(path),
                                structure_sha256=ANCHORS[path], **record))
        save('admission.json', dict(schema=SCHEMA, models=records, complete_model_population=3,
            entries_cap=ENTRIES_CAP, candidate_admitted=False, formal_gain=0,
            mathematical_claim='complete first-bank admission fails for frozen D209 logical ledger',
            current_implementation_only=True, neural_hz_impossibility_claimed=False,
            model_or_candidate_executed=False, original_parameters_only_stream_hashed=True))
        outputs.append('admission.json')
        result['metadata_computation_complete'] = True
    except BaseException as exc:
        result['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
    finally:
        try:
            require(prior is not None and before is not None, 'complete inherited admission was not reached')
            verify(sources, deadline, stats, 'source_postcheck', result['source_drift'])
            result['source_postcheck_complete'] = True
            verify(inputs, deadline, stats, 'input_postcheck', result['input_drift'])
            result['input_postcheck_complete'] = True
            after = provenance(deadline, stats, 'production_postcheck')
            result['provenance_after'] = after
            result['provenance_drift'] = after != before
            require(not result['provenance_drift'] and list(os.sched_getaffinity(0)) == [0],
                    'post-audit provenance or CPU affinity drift')
            result['artifacts'] = {name: stream_hash(RUN / name, deadline, stats, 'output_seal')
                                   for name in outputs}
            check(deadline)
        except BaseException as exc:
            result['postcheck_failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
        signal.setitimer(signal.ITIMER_REAL, 0)
        rss_growth = max(0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
        _, trace_peak = tracemalloc.get_traced_memory()
        trace_metadata = tracemalloc.get_tracemalloc_memory()
        host_ok = (rss_growth + RESERVE <= MEMORY_CAP
                   and trace_peak + trace_metadata + RESERVE <= MEMORY_CAP)
        result.update(rss_highwater_growth_bytes=rss_growth, traced_peak_bytes=trace_peak,
            tracer_metadata_bytes=trace_metadata, summary_reserve_bytes=RESERVE,
            host_memory_cap_bytes=MEMORY_CAP, host_observations_within_caps=host_ok)
        passed = (result.get('metadata_computation_complete') is True
                  and result['source_postcheck_complete'] and result['input_postcheck_complete']
                  and result['provenance_drift'] is False
                  and host_ok and time.monotonic() < deadline
                  and 'failure' not in result and 'postcheck_failure' not in result)
        result.update(metadata_audit_passed=passed, all_registered_stages_passed=passed,
            supervisor_exit=0 if passed else 1, streaming_hash_statistics=stats,
            wall_s=time.monotonic() - started,
            observed_process_max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
        save('exit.json', result)
        print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return result['supervisor_exit']


if __name__ == '__main__':
    raise SystemExit(main())
