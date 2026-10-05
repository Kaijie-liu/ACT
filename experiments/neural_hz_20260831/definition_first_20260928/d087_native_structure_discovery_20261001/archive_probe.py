"""One-shot trusted historical native-prefix discovery; never a model solve.

The independently anchored protocol-5 archive is read ONLY by the old C41
authenticated loader: one prehash pass, then one decode-and-hash pass. There
is no third archive reread. C41 is not a safe unpickler for untrusted data.
The full saved net, all caches, bounds and metadata remain live and are charged
through a closed header walk plus the existing numeric-owner visitor. No model,
LP/MILP, GPU operation, changed old main, or partial group application occurs.
"""
from collections import OrderedDict
from dataclasses import asdict, fields, is_dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time
import tracemalloc

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d087_native_structure_discovery_20261001_v1'
OUT = RUN / 'archive_probe'
OLD = EXP / 'results/c5_corrected_prefix_20260905_v2'
ARCHIVE = OLD / 'layer09.pickle'
ARCHIVE_BYTES = 68_718_806
ARCHIVE_SHA = '3a4d6a3432f76a43db332356319c74c412fbad04487b05a763eca44ede15524c'
ANCHORS = {
    OLD / 'layer09.snapshot.json': 'b15de137a35a2e4cbb0e696eb180fadeb391c7396bda14cdf7f5a24b44becc2f',
    OLD / 'exit.json': 'c5880ede838161a4c4a290fb4eb743be59a1ca1dcda6249d72f502aaec34bb33',
    OLD / 'preregistered.json': '7246a23285a27eb690e54571764dffbf5580774efe305705f752b250843d2afa',
    EXP / 'c41_owned_pickle_decode_v1.py': '85412990c8becf4a168098ced52199db2907118c3be22db155f58a19ebd0e417',
    EXP / 'c14_early_rejection_census_v1.py': 'f75c0543aedd2e0443de5e6014cd6566f75c14d0a6df0666f83b518514b83bac',
    EXP / 'c23_phase_overlay_audit_v1.py': '33b8f31418198efe3b35dd81dee043b7be8763e3b3511531519c4e40443ab9c9',
    EXP / 'c5_live_roots_v2.py': '6d82522bb1cdf366d10891c885387345ed72db348e668d9613fa82174f206979',
    EXP / 'c5_partial_csr_owner_ledger_v3.py': '043017915c743f0ff843c56e8266d19c20e279a7923af4d402785872de67df5c',
    EXP / 'c5_explicit_schema_ledger_v2.py': '2f87b1bac7798553ff378a0f65eec74becd1ed4f3272445bac55a92bc8bfe8b7',
    EXP / 's0_c2_whole_state_ledger_prototype.py': 'd9fd6948c0d8e7fc4e5399846ec650d0b9b28f3044ca97dbee242d4002098c2e',
    HERE.parent / 'd025_interval_capacity_20260930/evidence.py': 'b38dc733a3868c1a7621c164c490199f885efe008c951755254552cb14a5f696',
}
CAP, BRANCH_CAP, EVIDENCE_CAP = 256_000_000, 200_000_000, 40_000_000
MEMORY_CAP, AS_CAP, RESERVE = 1024**3, 16*1024**3, 65536
NEW_FILES = ('CONTRACT.md', 'PREREG.md', 'native_discovery.py',
             'test_native_discovery.py', 'archive_probe.py', 'run_math.py',
             'collection_contract.py')


class Bootstrap:
    """Charge actual pre-import file reads before authenticated project imports."""
    def __init__(self):
        self.used, self.parts = 0, {}

    def charge(self, name, amount):
        if type(amount) is not int or amount < 0 or self.used + amount > CAP:
            raise MemoryError('whole budget exhausted before bootstrap operation')
        self.used += amount
        self.parts[name] = self.parts.get(name, 0) + amount


def regular(path, cap=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or linked identity: ' + str(path))
    size = path.stat().st_size
    if cap is not None and not 0 < size <= cap:
        raise ValueError('oversized or empty metadata')
    return size


def sha(path, pool, prepaid=False):
    size = regular(path)
    if not prepaid:
        pool.charge('identity_file_bytes', size)
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    if Path(path).stat().st_size != size:
        raise ValueError('identity size changed during read')
    return digest.hexdigest()


def read_json(path, pool, cap=8*1024**2):
    size = regular(path, cap)
    pool.charge('metadata_parse_bytes_and_headers', 4096 + 8*size)
    return json.loads(Path(path).read_text())


def current_rss():
    with Path('/proc/self/status').open() as stream:
        for line in stream:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])*1024
    raise ValueError('own process RSS unavailable')


def memory_record(initial):
    current, peak = tracemalloc.get_traced_memory()
    metadata = tracemalloc.get_tracemalloc_memory()
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    growth = max(0, rss-initial)
    return dict(initial_rss_bytes=initial, current_rss_bytes=current_rss(),
                peak_rss_bytes=rss, rss_growth_bytes=growth,
                tracemalloc_current_bytes=current, tracemalloc_peak_bytes=peak,
                tracemalloc_metadata_bytes=metadata, summary_reserve_bytes=RESERVE,
                memory_gate_passed=(rss+RESERVE <= MEMORY_CAP and
                    growth+RESERVE <= MEMORY_CAP and peak+metadata+RESERVE <= MEMORY_CAP))


def full_roots(roots, pool, modules):
    """Closed metadata walk; existing owner visitor performs no value hashing."""
    np, torch, sp, known, partial, nb, nd = modules
    seen, active, numeric = set(), set(), {}
    shallow = 0
    hz_fields = frozenset(('c','Gc','Gb','Ac','Ab','b','Auc','Aub','ub','frame_id','exact'))

    class MeteredVisitor(partial.PartialCSRVisitor):
        def visit(self, value, role):
            pool.charge('numeric_owner_header', 128 + 8*len(self._storage_records))
            return super().visit(value, role)

        def _visit_numpy(self, value, role, *, entry_semantics):
            # Existing span checks can compare against previously seen owners.
            pool.charge('numeric_owner_span_header', 128 + 8*len(self._storage_records))
            return super()._visit_numpy(value, role, entry_semantics=entry_semantics)

        def _visit_torch(self, value, role):
            pool.charge('torch_storage_header', 128)
            if value.device.type != 'cpu':
                raise ValueError('archived tensor is not CPU resident')
            return super()._visit_torch(value, role)

    visitor = MeteredVisitor()

    def visit(value, depth=0):
        nonlocal shallow
        pool.charge('closed_root_reference', 8)
        if depth > 128:
            raise ValueError('closed root depth exceeded')
        if id(value) in active:
            raise ValueError('closed root cycle')
        if id(value) in seen:
            return
        pool.charge('closed_root_header', 16)
        seen.add(id(value))
        shallow += sys.getsizeof(value)
        kind = type(value)
        if value is nd._SEAL:
            return  # One frozen module-owned, fieldless capability object.
        if value is None or kind in (str, int, float, bool, bytes, Fraction):
            if kind is Fraction and max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 512:
                raise ValueError('metadata Fraction exceeds 512 bits')
            return
        if isinstance(value, (np.generic, torch.dtype, torch.device)):
            return
        if kind is np.ndarray or isinstance(value, torch.Tensor) or sp.isspmatrix_csr(value):
            role = 'root_' + str(len(numeric))
            numeric[role] = value
            visitor.visit(value, role)
            return
        if kind is nb.SparseHZono:
            if frozenset(vars(value)) != hz_fields:
                raise ValueError('unknown native HZ metadata fields')
            role = 'root_' + str(len(numeric))
            numeric[role] = value
            visitor.visit(value, role)
            # Numeric visitor covers all arrays; retain scalar identity headers.
            visit(value.frame_id, depth+1)
            visit(value.exact, depth+1)
            return
        active.add(id(value))
        try:
            if kind in known.KNOWN_FIELDS:
                payload = vars(value)
                if frozenset(payload) != known.KNOWN_FIELDS[kind]:
                    raise ValueError('unknown fields in archived ' + kind.__name__)
                pool.charge('closed_schema_fields', 8*len(payload))
                for item in payload.values():
                    visit(item, depth+1)
            elif kind is known.Bounds:
                if frozenset(vars(value)) != {'lb', 'ub'}:
                    raise ValueError('unknown Bounds fields')
                visit(value.lb, depth+1)
                visit(value.ub, depth+1)
            elif kind in (dict, OrderedDict):
                pool.charge('closed_container_fields', 8*len(value))
                if kind is OrderedDict:
                    attributes = vars(value)
                    if set(attributes)-{'_metadata'}:
                        raise ValueError('unknown OrderedDict fields')
                    visit(attributes, depth+1)
                for key, item in value.items():
                    visit(key, depth+1)
                    visit(item, depth+1)
            elif kind in (list, tuple, set, frozenset):
                pool.charge('closed_container_fields', 8*len(value))
                for item in value:
                    visit(item, depth+1)
            elif is_dataclass(value) and kind.__module__ in (
                    nb.__name__, nd.__name__, nd.nt.__name__):
                names = tuple(field.name for field in fields(value))
                if frozenset(vars(value)) != frozenset(names):
                    raise ValueError('candidate record has undeclared fields')
                pool.charge('candidate_record_fields', 8*len(names))
                for name in names:
                    visit(getattr(value, name), depth+1)
            else:
                raise ValueError('unsupported full-root type: ' + kind.__name__)
        finally:
            active.remove(id(value))

    visit(roots)
    pool.charge('owner_ledger_export', 256 + 128*len(visitor._storage_records))
    ledger = visitor.ledger()
    if ledger.resident_entries > 64_000_000:
        raise MemoryError('complete retained numeric entry cap exceeded')
    return dict(numeric_bytes=ledger.resident_bytes, numeric_entries=ledger.resident_entries,
                numeric_storages=ledger.numeric_storage_count,
                python_shallow_bytes=shallow, unique_objects=len(seen),
                complete_payload_walk=True, value_bytes_rehashed=False,
                numeric_rule_id=ledger.rule_id)


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    if not RUN.is_dir():
        raise ValueError('the single-use mathematical run must exist first')
    OUT.mkdir(exist_ok=False)
    started, initial = time.monotonic(), current_rss()
    tracemalloc.start()
    sys.dont_write_bytecode = True
    boot, whole, branch, meter = Bootstrap(), None, None, None
    code_ids, identities, post_reserved = {}, {}, False
    manifest = saved = discovered = applied = evidence = transfer_evidence = None
    result = dict(schema='d087_archived_native_probe_v1', archive=str(ARCHIVE),
        archive_expected_sha256=ARCHIVE_SHA, archive_expected_bytes=ARCHIVE_BYTES,
        archive_loaded=False, archive_census_completed=False, transformed=False,
        all_groups_applied=False, source_census_qualified=False,
        actual_model_binding_qualified=False, actual_phase_column_binding_verified=False,
        native_HZ_admitted=False, gpu_computation_completed=False,
        complete_physical_qualification=False, formal_gain=0,
        source_drift=[], identities_unchecked=[], historical_prefix_timeout=True)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
        os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
        if os.environ.get('LD_PRELOAD'):
            raise ValueError('unexpected LD_PRELOAD')
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            os.environ[name] = '1'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for name in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH',
                     'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
            path = OUT / name.lower()
            path.mkdir()
            os.environ[name] = str(path)
        def alarm(signum, frame):
            raise TimeoutError('archive worker 240 second cap')
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(240)
        boot.charge('global_evidence_and_final_summary_reserve', EVIDENCE_CAP+RESERVE)
        manifest = read_json(RUN/'preregistered.json', boot)
        done = read_json(RUN/'exit.json', boot)
        frozen = read_json(HERE/'freeze.json', boot, RESERVE)
        sources = frozen.get('source_sha256')
        if (frozen.get('schema') != 'd087_native_structure_discovery_v1'
                or type(sources) is not dict or set(sources) != {str(HERE/n) for n in NEW_FILES}
                or done.get('all_stages_passed') is not True or done.get('supervisor_exit') != 0
                or done.get('tests_count') != 3825 or done.get('test_files') != 183
                or manifest.get('required_tests') != 3825 or manifest.get('required_test_files') != 183):
            raise ValueError('the complete frozen mathematical gate has not passed')
        for path, digest in sources.items():
            if manifest['source_sha256'].get(path) != digest:
                raise ValueError('new mathematical source binding differs')
        # All local Python sources are authenticated before any project import.
        code_ids = {p:d for p,d in manifest['source_sha256'].items()
                    if Path(p).is_relative_to(ROOT) and Path(p).suffix == '.py'}
        identities = {**code_ids, **sources, **{str(p):d for p,d in ANCHORS.items()}}
        identities[str(HERE/'freeze.json')] = manifest['freeze_sha256']
        for path, digest in identities.items():
            if path in manifest['source_sha256'] and manifest['source_sha256'][path] != digest:
                raise ValueError('conflicting historical identity')
        # Reserve BOTH checks before the first; finally cannot spend an unheld budget.
        boot.charge('local_source_identity_before_after', 2*sum(regular(p) for p in identities))
        post_reserved = True
        for path, digest in identities.items():
            if sha(path, boot, prepaid=True) != digest:
                raise ValueError('source identity mismatch: '+path)
        prior = read_json(OLD/'preregistered.json', boot)
        terminal = read_json(OLD/'exit.json', boot)
        snapshot = read_json(OLD/'layer09.snapshot.json', boot)
        if (snapshot.get('pickle_bytes') != ARCHIVE_BYTES or snapshot.get('pickle_sha256') != ARCHIVE_SHA
                or snapshot.get('hz_layers') != [0,1,5,9] or snapshot.get('expr_layers') != []
                or terminal.get('timeout') is not True or terminal['artifacts'].get('layer09.pickle') != ARCHIVE_SHA
                or prior.get('candidate_enabled') is not False
                or snapshot.get('provenance') != prior.get('provenance')
                or regular(ARCHIVE) != ARCHIVE_BYTES):
            raise ValueError('fixed historical prefix contract differs')
        result['historical_prefix_record'] = dict(timeout=True, wall_s=terminal['wall_s'],
            provenance=prior['provenance'], target=prior['target'], source=str(OLD))
        for name, digest in prior['sources'].items():
            path = str(ROOT/name)
            if manifest['source_sha256'].get(path) != digest:
                raise ValueError('historical source identity not inherited: '+path)
            if path not in identities:
                boot.charge('additional_historical_identity_before_after', 2*regular(path))
                identities[path] = digest
                if sha(path, boot, prepaid=True) != digest:
                    raise ValueError('historical source identity mismatch: '+path)
        sys.path.insert(0, str(ROOT))
        import numpy as np
        import scipy.sparse as sp
        import torch
        from experiments.neural_hz_20260831.c14_early_rejection_census_v1 import WorkPool
        from experiments.neural_hz_20260831.c23_phase_overlay_audit_v1 import BranchPool
        from experiments.neural_hz_20260831.c41_owned_pickle_decode_v1 import load
        from experiments.neural_hz_20260831 import c5_live_roots_v2 as known
        from experiments.neural_hz_20260831 import c5_partial_csr_owner_ledger_v3 as partial
        from experiments.neural_hz_20260831.definition_first_20260928.d064_native_predicate_binding_20261001 import native_binding as nb
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import evidence as ev
        from experiments.neural_hz_20260831.definition_first_20260928.d087_native_structure_discovery_20261001 import native_discovery as nd
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        whole = WorkPool(CAP)
        for name, amount in boot.parts.items():
            whole.charge(name, amount)
        branch, meter = BranchPool(whole, BRANCH_CAP), ev.Meter(EVIDENCE_CAP)
        for module in tuple(sys.modules.values()):
            path = getattr(module, '__file__', None)
            if path and Path(path).is_relative_to(ROOT) and Path(path).suffix == '.py' and str(Path(path).resolve()) not in identities:
                raise ValueError('unbound project import: '+str(path))
        if not memory_record(initial)['memory_gate_passed']:
            raise MemoryError('startup host memory gate failed')
        branch.charge('archive_prehashed_and_decode_hashed_bytes', 2*ARCHIVE_BYTES)
        with ARCHIVE.open('rb') as stream:
            saved, decoder = load(stream, expected_sha256=ARCHIVE_SHA, pool=branch, enabled=True)
        result.update(archive_loaded=True, authenticated_decoder=decoder)
        if (type(saved) is not dict or saved.get('schema') != 'c5_incremental_snapshot_v2'
                or saved.get('layer') != 9 or sorted(saved.get('hz_cache', {})) != [0,1,5,9]
                or saved.get('expr_cache') != {} or saved.get('provenance') != prior['provenance']):
            raise ValueError('complete native prefix population differs')
        modules = (np, torch, sp, known, partial, nb, nd)
        result['loaded_roots'] = full_roots((saved, manifest, frozen, prior, terminal, snapshot, decoder), branch, modules)
        hz = saved['hz_cache'][9]
        if (type(hz) is not nb.SparseHZono or any(saved['hz_cache'][i].frame_id != hz.frame_id for i in (0,1,5))):
            raise ValueError('archived input/parent/child frames differ')
        discovered = nd.discover(hz, pool=branch, enabled=True)
        result.update(archive_census_completed=True, discovery_summary=discovered.summary)
        population = len(discovered.graphs) + len(discovered.groups)
        branch.charge('discovery_complete_evidence_population', 64*population)
        population += sum(2+len(consumers) for _,consumers in discovered.groups)
        branch.charge('discovery_complete_evidence_construction', 256*population)
        evidence = dict(schema='d087_complete_native_discovery_v1', archive=str(ARCHIVE),
            archive_sha256=ARCHIVE_SHA, archive_bytes=ARCHIVE_BYTES,
            analyzed_layer=9, full_hz_cache_layers=[0,1,5,9],
            graphs=tuple(asdict(g) for g in discovered.graphs),
            groups=tuple(dict(parents=tuple(asdict(g) for g in parents),
                consumers=tuple((asdict(g), a, b) for g,a,b in consumers))
                for parents,consumers in discovered.groups), summary=discovered.summary)
        # This is complete census evidence, even if the later ALL-group prepay fails.
        receipt = ev.write_evidence(OUT/'discovery.json.partial', evidence, meter, {})
        (OUT/'discovery.json.partial').rename(OUT/'discovery.json')
        result['discovery_evidence'] = dict(file='discovery.json', **receipt)
        current_roots = full_roots((saved, manifest, frozen, prior, terminal, snapshot,
                                   decoder, discovered, evidence), branch, modules)
        result['pre_application_roots'] = current_roots
        branch.charge('whole_bank_physical_reserve_check', 128)
        reserve = discovered.summary['apply_numeric_entries_upper']
        if type(reserve) is not int or reserve < 0:
            raise ValueError('invalid full-bank temporary entry reserve')
        plain = ev.bounded_ledger(evidence, meter)
        result['evidence_ledger'] = plain
        if current_roots['numeric_entries'] + plain['retained_entries'] + reserve > 64_000_000:
            raise MemoryError('complete roots plus full-bank application entry reserve exceed64M')
        if not memory_record(initial)['memory_gate_passed']:
            raise MemoryError('pre-application host memory gate failed')
        applied = nd.apply_groups(hz, discovered, pool=branch, enabled=True)
        result.update(transformed=True, all_groups_applied=True, transfer_summary=applied.summary)
        branch.charge('complete_receipt_population', 64*len(applied.receipts))
        occurrences = sum(len(receipt.exact_rows) for receipt in applied.receipts)
        branch.charge('complete_receipt_row_metadata', 64*occurrences)
        occurrences += sum(len(row.continuous)+len(row.binary)
            for receipt in applied.receipts for row,_ in receipt.exact_rows)
        branch.charge('complete_receipt_evidence_construction', 128*occurrences)
        transfer_evidence = dict(schema='d087_complete_group_transfer_v1',
            archive_sha256=ARCHIVE_SHA, summary=applied.summary,
            receipts=tuple(dict(old_n_cont=item.old_n_cont,
                shared_columns=item.shared_columns, residual_bindings=item.residual_bindings,
                exact_rows=tuple((asdict(row), rhs) for row,rhs in item.exact_rows),
                row_errors=item.row_errors, installed_rhs=item.installed_rhs,
                extra_residual_count=item.extra_residual_count,
                physical_bytes=item.physical_bytes, nnz=item.nnz) for item in applied.receipts))
        receipt = ev.write_evidence(OUT/'transfer.json.partial', transfer_evidence, meter, {})
        (OUT/'transfer.json.partial').rename(OUT/'transfer.json')
        result['transfer_evidence'] = dict(file='transfer.json', **receipt)
        result['final_roots'] = full_roots((saved, manifest, frozen, prior, terminal, snapshot,
            decoder, discovered, applied, evidence, transfer_evidence), branch, modules)
        result['all_requested_stages_completed'] = True
    except Exception as exc:
        result['failure'] = dict(type=type(exc).__name__, reason=str(exc)[:2048])
    finally:
        if post_reserved:
            for path, digest in identities.items():
                try:
                    if sha(path, boot, prepaid=True) != digest:
                        result['source_drift'].append(path)
                except Exception:
                    result['identities_unchecked'].append(path)
        result.update(whole_work_used=whole.used if whole is not None else boot.used,
            branch_work_used=branch.used if branch is not None else 0,
            evidence_work_used=meter.used if meter is not None else 0,
            worker_wall_s=time.monotonic()-started, archive_rehash_after_decode=False,
            archive_authentication_completed=result['archive_loaded'],
            archive_planned_read_passes=2, complete_model_replay=False)
        try:
            result.update(memory_record(initial))
        except Exception as exc:
            result.update(memory_gate_passed=False, memory_error=str(exc)[:512])
        result['archive_probe_qualified'] = bool(result.get('all_requested_stages_completed')
            and result.get('memory_gate_passed') and not result['source_drift']
            and not result['identities_unchecked'] and 'failure' not in result
            and result['worker_wall_s'] <= 240)
        encoded = json.dumps(result, sort_keys=True, indent=2, allow_nan=False)
        if len(encoded.encode()) > RESERVE:
            result['archive_probe_qualified'] = False
            encoded = json.dumps(dict(schema=result['schema'], archive_probe_qualified=False,
                failure={'type':'SummaryReserveExceeded','reason':'summary exceeds65536'},
                formal_gain=0, worker_wall_s=result['worker_wall_s']))
        with (OUT/'worker.json').open('x') as stream:
            stream.write(encoded+'\n')
        signal.alarm(0)
    if not result['archive_probe_qualified']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
