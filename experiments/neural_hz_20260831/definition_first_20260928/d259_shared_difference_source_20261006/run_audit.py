"""One complete three-source audit, only after this version's full math gate."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time
import tracemalloc
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
MATH = EXP / 'results/d259_shared_difference_source_20261006_v1'
RUN = EXP / 'results/d259_shared_difference_source_20261006_sources_v1'
SCHEMA = 'd259_shared_difference_source_audit_v1'
OLD_PATH = HERE.parent / 'd241_residual_source_binding_20261005/run_audit.py'
OLD_SHA = '233595d46cccccb10302a10a39355a4e24b97bebc19e5527f1662fb3a9e37da8'
GEOMETRY = OLD_PATH.parent / 'source_audit.py'
REFERENCE = EXP / 'results/d241_residual_source_binding_20261005_v1'
REFERENCES = tuple(REFERENCE / ('source_' + str(i) + '.json') for i in range(3))
REFERENCE_SHA = ('9494cb57122809d61f62a7c75f7ee4533c8b41e9683cda86a668321fbef20dc5',
    '5374d47e99db2a34e2099603b469dcb50d828050e536900e1d832267bf8222e0',
    'ccac731af636a70c1b6210fb8ea6bdd6108fd72ab0949547263fee1bd48b7bed')
FALSE_FLAGS = ('native_HZ_admitted', 'actual_phase_column_binding_verified',
    'actual_model_verification_qualified', 'gpu_computation_completed',
    'complete_physical_qualification', 'new_domain_qualified', 'new_capability_qualified',
    'capability_improvement_claimed', 'solver_executed', 'qg_installed',
    'mathematical_tests_executed', 'domain_definition_changed')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stop(signum, frame):
    raise TimeoutError('complete source audit terminated: ' + str(signum))


def bootstrap():
    require(not OLD_PATH.is_symlink() and OLD_PATH.is_file()
            and OLD_PATH.stat().st_size <= 131072, 'historical harness identity')
    raw = OLD_PATH.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == OLD_SHA, 'historical harness hash differs')
    spec = importlib.util.spec_from_file_location('_d259_authenticated_source_harness', OLD_PATH)
    old = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = old
    spec.loader.exec_module(old)
    old.RUN = RUN  # This private imported harness writes only the fresh destination.
    return old, len(raw)


def receipt(old, meter):
    paths = [MATH / name for name in ('preregistered.json', 'inventory.json', 'exit.json')]
    hashes = {str(path): old.digest(path, meter) for path in paths}
    prior, inventory, done = [old.json_read(path, hashes[str(path)], meter) for path in paths]
    require(prior.get('schema') == 'd259_shared_difference_source_v1'
        and prior.get('required_tests') == 4277 and prior.get('required_test_files') == 229
        and prior.get('inherited_tests') == 4269 and prior.get('inherited_test_files') == 228
        and prior.get('cpu_affinity') == [0]
        and len(prior.get('expected_nodeids', ())) == 4277
        and len(set(prior['expected_nodeids'])) == 4277
        and len(prior.get('tests', ())) == len(set(prior['tests'])) == 229,
        'complete ordered D259 mathematics registration differs')
    require(all(done.get(k) is True for k in ('component_tests_passed',
        'mathematical_component_gate_passed', 'shared_difference_bounds_math_passed',
        'all_registered_stages_passed', 'host_observations_within_caps',
        'inventory_validated_before_execution'))
        and done.get('tests_exit') == done.get('supervisor_exit') == 0
        and done.get('tests_count') == 4277 and done.get('test_files') == 229
        and 0 <= done.get('test_wall_s', 61) <= 60 and 'failure' not in done
        and done.get('source_drift') == done.get('input_drift') == []
        and done.get('provenance_drift') is False
        and done.get('formal_gain') == done.get('independent_e0_gain') == 0,
        'this candidate has no successful complete mathematical receipt')
    require(inventory.get('count') == 4277 and inventory.get('files') == 229
        and inventory.get('nodeids') == prior['expected_nodeids']
        and inventory.get('manifest_sha256') == hashes[str(paths[0])]
        and inventory.get('validated_before_execution') is True,
        'mathematical inventory differs')
    sources = dict(old.identities(prior['source_sha256']))
    inputs = dict(old.identities(prior['input_sha256'], 14))
    for path, digest in hashes.items():
        old.merge(sources, path, digest)
    for name, digest in done['artifacts'].items():
        path = MATH / name
        require(path.resolve() == path and path.is_relative_to(MATH), 'math artifact escapes RUN')
        old.merge(sources, path, digest)
    xml = old.read_bytes(MATH / 'tests.xml', sources[str(MATH / 'tests.xml')], old.JSON_BYTES_CAP, meter)
    meter.charge(4 * len(xml), entries=len(xml))
    cases = ET.fromstring(xml).findall('.//testcase')
    actual = [c.get('classname', '').replace('.', '/') + '.py::' + c.get('name', '') for c in cases]
    require(actual == prior['expected_nodeids'] and not any(c.find(k) is not None
        for c in cases for k in ('failure', 'error', 'skipped')), 'math JUnit differs')
    frozen = old.json_read(HERE / 'freeze.json', sources[str(HERE / 'freeze.json')], meter)
    require(frozen.get('schema') == prior['schema']
        and frozen.get('required_tests') == 4277 and frozen.get('required_test_files') == 229
        and all(sources.get(path) == digest for path, digest in frozen['source_sha256'].items())
        and str(HERE / 'run_audit.py') in frozen['source_sha256'], 'source audit was not frozen with math')
    old.merge(sources, OLD_PATH, OLD_SHA)
    for path, digest in zip(REFERENCES, REFERENCE_SHA):
        old.merge(sources, path, digest)
    selected = prior.get('selected_sources')
    require(type(selected) is list and len(selected) == 3
        and len({s['model_path'] for s in selected}) == 3, 'complete three-source population differs')
    for source, structure in zip(selected, old.STRUCTURES):
        require(inputs.get(source['model_path']) == source['model_sha256']
            and inputs.get(source['spec_path']) == source['spec_sha256']
            and sources.get(source['manifest_path']) == source['manifest_sha256']
            and str(structure) in sources, 'original source identity missing')
    return prior, sources, inputs, selected


def main():
    require(sys.argv[1:] == ['--enabled'], 'explicit --enabled required')
    RUN.mkdir(exist_ok=False)
    started = time.monotonic()
    rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    old = meter = None
    status, model_start = 1, None
    sources, inputs, artifacts = {}, {}, {}
    report = dict(schema=SCHEMA, diagnostic_complete=False, models=[], formal_gain=0,
        independent_e0_gain=0, new_benchmark_solves=0, source_postcheck_complete=False,
        input_postcheck_complete=False, mathematical_receipt_validated=False,
        registered_pair_population=97532, registered_geometry_classes=2853)
    report.update({k: False for k in FALSE_FLAGS})
    try:
        signal.signal(signal.SIGALRM, stop)
        signal.signal(signal.SIGTERM, stop)
        signal.alarm(240)
        resource.setrlimit(resource.RLIMIT_AS, (16 * 1024**3, 16 * 1024**3))
        require(0 in os.sched_getaffinity(0), 'CPU 0 is unavailable')
        os.sched_setaffinity(0, {0})
        sys.dont_write_bytecode = True
        os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', CUDA_VISIBLE_DEVICES='',
            CUDA_CACHE_PATH=str(RUN / 'cuda_cache'), TORCH_HOME=str(RUN / 'torch_home'),
            XDG_CACHE_HOME=str(RUN / 'xdg_cache'), TRITON_CACHE_DIR=str(RUN / 'triton_cache'),
            TORCHINDUCTOR_CACHE_DIR=str(RUN / 'inductor_cache'), TMPDIR=str(RUN / 'tmp'))
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ[name] = '1'
        (RUN / 'tmp').mkdir()
        tracemalloc.start()
        require(__debug__ and os.environ.get('PYTHONOPTIMIZE') in (None, '', '0'), 'assertions required')
        old, boot_bytes = bootstrap()

        class Meter(old.Meter):
            def charge(self, amount=None, entries=0, *, work=None):
                require((amount is None) != (work is None), 'exactly one work charge required')
                super().charge(work if amount is None else amount, entries)

        meter = Meter()
        meter.started, meter.rss0 = started, rss0
        meter.charge(4096 + boot_bytes)
        prior, sources, inputs, selected = receipt(old, meter)
        old.verify_all(sources, meter)
        old.verify_all(inputs, meter)
        before = old.provenance(meter)
        require({key: before[key] for key in ('branch', 'commit', 'candidate_sha256')}
            == prior['provenance'], 'math/source production provenance differs')
        require(Path(sys.executable).resolve() == old.PYTHON.resolve()
            and str(old.PYTHON.resolve()) in sources, 'authenticated interpreter required')
        sys.path.insert(0, str(ROOT))
        k = old.load_checked(old.KERNEL, '_d259_source_kernel', sources, meter)
        meter.attach(k)
        evidence = old.load_checked(old.EVIDENCE, '_d259_source_evidence', sources, meter)
        meter.evidence = evidence.Meter(limit=old.EVIDENCE_CAP)
        helper = old.load_checked(old.HELPER, '_d259_source_helpers', sources, meter)
        base = old.load_checked(old.READER, '_d259_metadata_reader', sources, meter)
        geometry = old.load_checked(GEOMETRY, '_d259_source_geometry', sources, meter)
        bounds = old.load_checked(HERE / 'source_bounds.py', '_d259_source_bounds', sources, meter)
        observer = old.load_checked(HERE / 'source_observer.py', '_d259_source_observer', sources, meter)
        onnx_spec = importlib.util.find_spec('onnx')
        require(onnx_spec is not None and onnx_spec.origin
            and str(Path(onnx_spec.origin).resolve()) in sources, 'ONNX provenance')
        import onnx
        require(Path(onnx.__file__).resolve() == Path(onnx_spec.origin).resolve(), 'ONNX import identity')
        report.update(mathematical_receipt_validated=True, source_precheck_complete=True,
            input_precheck_complete=True, provenance_before=before, source_count=len(sources),
            input_count=len(inputs), cpu_affinity=[0], cuda_visible_devices='')
        registration = dict(schema=SCHEMA, source_sha256=sources, input_sha256=inputs,
            selected_sources=selected, provenance=before, mathematical_run=str(MATH),
            mathematical_manifest_sha256=sources[str(MATH / 'preregistered.json')],
            mathematical_exit_sha256=sources[str(MATH / 'exit.json')],
            mathematical_tests=4277, mathematical_files=229, qualification_transferred=False,
            registered_pair_population=97532, registered_geometry_classes=2853,
            wall_cap_s=240, whole_work_cap=256000000, model_work_cap=200000000,
            evidence_work_cap=40000000, entries_cap=64000000, rational_bits=512,
            address_space_bytes=16*1024**3, host_memory_cap_bytes=1024**3,
            cpu_affinity=[0], formal_gain=0, independent_e0_gain=0)
        written = evidence.write_evidence(RUN / 'preregistered.json', registration, meter.evidence, {})
        artifacts['preregistered.json'] = written['sha256']
        del prior, registration
        for index, (source, structure_path) in enumerate(zip(selected, old.STRUCTURES)):
            meter.check()
            model_start = meter.used
            meter.budget.limit = min(old.WORK_CAP, model_start + old.MODEL_WORK_CAP)
            item = dict(index=index, complete=False, source=source)
            report['models'].append(item)
            structure = old.json_read(structure_path, sources[str(structure_path)], meter)
            reference = old.json_read(REFERENCES[index], REFERENCE_SHA[index], meter)
            raw = old.read_bytes(source['model_path'], source['model_sha256'], old.MODEL_BYTES_CAP, meter)
            spec = old.read_bytes(source['spec_path'], source['spec_sha256'], old.JSON_BYTES_CAP, meter)
            roots, record = observer.extract(raw, spec, source, structure, meter, k, helper, base,
                reference=reference, reference_identity=dict(path=str(REFERENCES[index]),
                    sha256=REFERENCE_SHA[index]), geometry=geometry, bounds=bounds, enabled=True)
            summary = observer.validate_record(record, source, structure, meter)
            temporary_upper = summary['temp_numeric_entries_upper']
            require(type(temporary_upper) is int and temporary_upper >= 0, 'temporary entry ledger missing')
            held = evidence.bounded_ledger((roots, report, sources, inputs), meter.evidence)
            entries = meter.metadata_entries + held['retained_entries'] + temporary_upper
            require(entries <= old.ENTRY_CAP, 'complete numeric-entry observation cap')
            meter.peak_entries = max(meter.peak_entries, entries)
            name = 'source_' + str(index) + '.json'
            partial = RUN / (name + '.partial')
            written = evidence.write_evidence(partial, record, meter.evidence,
                {id(raw): source['model_sha256'], id(spec): source['spec_sha256']})
            require(written['bytes'] <= old.JSON_BYTES_CAP and not (RUN / name).exists(),
                    'source output byte or exclusive-publication cap')
            partial.rename(RUN / name)
            artifacts[name] = written['sha256']
            item.update(complete=True, summary=summary, evidence_file=name,
                evidence_sha256=written['sha256'], evidence_bytes=written['bytes'],
                model_work=meter.used-model_start, retained_ledger=held, entries_upper=entries)
            meter.model_work = max(meter.model_work, item['model_work'])
            meter.budget.limit = old.WORK_CAP
            model_start = None
            del roots, record, reference, structure, raw, spec, summary
            meter.check()
        require(len(report['models']) == 3 and all(x['complete'] for x in report['models']),
                'incomplete source population')
        summaries = [item['summary'] for item in report['models']]
        require(sum(s['registered_pair_population'] for s in summaries) == 97532
            and sum(s['pair_class_records'] for s in summaries) == 2853
            and sum(s['initializer_count'] for s in summaries) == 70
            and sum(s['packed_scalar_count'] for s in summaries) == 875072,
            'complete three-source aggregate population differs')
        report['classification_totals'] = {
            name: sum(s[name] for s in summaries) for name in
            ('registered_pair_population', 'pair_class_records', 'classified_population',
             'excluded_population', 'not_excluded_population', 'ineligible_population')}
        old.verify_all(sources, meter)
        old.verify_all(inputs, meter)
        for name, digest in artifacts.items():
            require(old.digest(RUN / name, meter) == digest, 'new artifact identity drift')
        report.update(source_postcheck_complete=True, input_postcheck_complete=True,
                      provenance_after=old.provenance(meter))
        require(report['provenance_after'] == before and list(os.sched_getaffinity(0)) == [0],
                'postcheck provenance/CPU drift')
        meter.check()
        report['diagnostic_complete'] = True
        status = 0
    except BaseException as exc:
        report['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:4096])
    finally:
        signal.alarm(0)
        if meter is not None:
            if model_start is not None:
                meter.model_work = max(meter.model_work, meter.used-model_start)
            observations = meter.snapshot()
            within = (observations['wall_s'] < 240
                and observations['rss_highwater_growth_bytes']+65536 <= 1024**3
                and observations['traced_peak_bytes']+observations['tracer_metadata_bytes']+65536 <= 1024**3
                and meter.used <= 256000000 and meter.model_work <= 200000000
                and meter.metadata_entries <= 64000000 and meter.peak_entries <= 64000000
                and (meter.evidence is None or meter.evidence.used <= 40000000))
            report.update(host_observations=observations, host_observations_within_caps=within)
            if not within:
                report.setdefault('failure', dict(type='ResourceError', reason='final resource gate'))
        else:
            report['host_observations_within_caps'] = False
        if 'failure' in report or not report['host_observations_within_caps']:
            status, report['diagnostic_complete'] = 1, False
        report.update(exit_status=status, artifact_sha256=dict(artifacts))
        reserve = [65536]
        def save(name, obj):
            raw = (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
            require(len(raw) <= reserve[0], 'shared terminal reserve exhausted')
            reserve[0] -= len(raw)
            with (RUN / name).open('xb') as stream:
                stream.write(raw)
                stream.flush()
            return hashlib.sha256(raw).hexdigest()
        artifacts['report.json'] = save('report.json', report)
        final_wall = time.monotonic()-started
        if final_wall >= 240:
            status = 1
        terminal = dict(schema=SCHEMA, status=status, diagnostic_complete=status == 0,
            wall_s=final_wall, artifact_sha256=artifacts, formal_gain=0, independent_e0_gain=0,
            new_benchmark_solves=0, report_complete_before_final_write=report['diagnostic_complete'])
        terminal.update({k: False for k in FALSE_FLAGS})
        save('exit.json', terminal)
        print(json.dumps(dict(status=status, result_path=str(RUN / 'report.json'),
            diagnostic_complete=status == 0, failure=report.get('failure')), sort_keys=True), flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
