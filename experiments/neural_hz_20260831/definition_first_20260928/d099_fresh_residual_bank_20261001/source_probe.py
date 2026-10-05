"""One fresh Tiny residual prefix and complete, closed D098 bank diagnostic.

No old entry point, pickle, solver or post-bank production propagation is used.
The frozen C5 materializer is explicit source-construction support, only around
INPUT -> ACT20. Its original per-island product pools are NOT the new whole
256M/branch200M pools. Production arithmetic/temporary qualification remains
false. New identity, current-root, bank and evidence operations share one pool.
"""
from collections import OrderedDict, defaultdict
from dataclasses import fields, is_dataclass
from enum import Enum
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
import types
import weakref

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent
ROOT = EXP.parent.parent
RUN = EXP / 'results/d099_fresh_residual_bank_20261001_v1'
OUT = RUN / 'source_probe'
MODEL = Path('/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/tinyimagenet_2024/onnx/TinyImageNet_resnet_medium.onnx')
SPEC = EXP / 'vnnlib_v2_full_v1/tinyimagenet_2024/vnnlib/TinyImageNet_resnet_medium_prop_idx_3553_sidx_3392_eps_0.0039.vnnlib'
BN_ANCHOR = EXP / 'evidence/tiny_iid143_bn_graph_faithfulness_audit_v2.json'
C5_ANCHOR = EXP / 'evidence/c5_live_transaction_20260905_v2.json'
CLOSURE = HERE.parent / 'd090_bound_native_discovery_20261001/project_import_closure.json'
FIXED = {
    str(MODEL): '234b04b151d640f8fc859fab00729448ba533d8feb3679427cbadb94467ec776',
    str(SPEC): 'd105a0c7ca711eb46b9f20ab772a0564444569d06f6206bca0e95d5a19990cca',
    str(BN_ANCHOR): '31ad2f6b0d5a30cff226340d554ff9fa92883c894a3b9f1c0e02dcfd4f8f2c2a',
    str(C5_ANCHOR): '5b2af877318515dd7daa06eb0c79181d13c3dd71a351c6cb4b2cbeddc91fde94',
    str(CLOSURE): 'c340e32221b429a6dfc0a49ffdcce93f1baf4a84cfc1967079da5cb99cd6f03f',
}
FILES = ('CONTRACT.md', 'PREREG.md', 'source_probe.py', 'run_source.py',
         'run_math.py', 'collection_contract.py')
CAP, BRANCH_CAP, EVIDENCE_CAP = 256_000_000, 200_000_000, 40_000_000
ENTRY_CAP, AS_CAP, MEMORY_CAP, RESERVE = 64_000_000, 16*1024**3, 1024**3, 65536


class Pool:
    """Same hard whole/branch ceilings; failed requests are retained as evidence."""
    def __init__(self, cap=CAP, parent=None):
        if type(cap) is not int or cap < 0 or cap > (BRANCH_CAP if parent else CAP):
            raise ValueError('invalid or increased work ceiling')
        self.cap, self.parent, self.used, self.parts = cap, parent, 0, {}
        self.failure = None

    def charge(self, name, amount):
        if type(amount) is not int or amount < 0:
            raise ValueError('invalid work charge')
        if amount > self.cap-self.used:
            self.failure = dict(name=str(name), used=self.used, requested=amount, cap=self.cap)
            raise MemoryError('hard work budget exhausted before '+str(name))
        if self.parent is not None:
            self.parent.charge(name, amount)
        self.used += amount
        self.parts[name] = self.parts.get(name, 0)+amount


def regular(path, cap=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError('missing or linked identity: '+str(path))
    size = path.stat().st_size
    if cap is not None and not 0 < size <= cap:
        raise ValueError('file exceeds declared size guard')
    return size


def sha(path, pool, *, prepaid=False):
    size = regular(path)
    if not prepaid:
        pool.charge('identity_file_bytes', size)
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        remaining = size
        while remaining:
            block = stream.read(min(1024**2, remaining))
            if not block:
                raise ValueError('identity truncated during read')
            digest.update(block)
            remaining -= len(block)
        if stream.read(1):
            raise ValueError('identity grew during read')
    return digest.hexdigest()


def read_json(path, pool, cap=8*1024**2):
    size = regular(path, cap)
    pool.charge('metadata_read_parse', 4096+8*size)
    with Path(path).open('rb') as stream:
        raw = stream.read(size+1)
    if len(raw) != size:
        raise ValueError('metadata size changed')
    return json.loads(raw)


def rss():
    with Path('/proc/self/status').open() as stream:
        for line in stream:
            if line.startswith('VmRSS:'):
                return int(line.split()[1])*1024
    raise ValueError('own RSS unavailable')


def memory(initial):
    current, peak = tracemalloc.get_traced_memory()
    metadata = tracemalloc.get_tracemalloc_memory()
    high = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    growth = max(0, high-initial)
    return dict(initial_rss_bytes=initial, current_rss_bytes=rss(), peak_rss_bytes=high,
        rss_growth_bytes=growth, tracemalloc_current_bytes=current,
        tracemalloc_peak_bytes=peak, tracemalloc_metadata_bytes=metadata,
        memory_gate_passed=(high+RESERVE <= MEMORY_CAP and growth+RESERVE <= MEMORY_CAP
                            and peak+metadata+RESERVE <= MEMORY_CAP))


def closed_roots(roots, pool, modules):
    """Complete current task roots, with explicit FX C++-Node fields.

    Primitive scalars are conservatively counted by occurrence, not retained
    in an enormous identity set. Containers/owners are identity-deduplicated.
    Cycles of the declared FX/Module graph are permitted and fully visited.
    This does not prove all construction temporaries or library-global heaps.
    """
    np, torch, sp, known, partial, nb, bank, tf_type, converter_type, cert = modules
    from torch.fx import graph as fxg
    from torch.fx.node import Node
    from torch.fx.immutable_collections import immutable_dict, immutable_list
    from act.back_end.solver.solver_hz import HZono
    schemas = {**known.KNOWN_FIELDS,
        nb.SparseHZono: frozenset(('c','Gc','Gb','Ac','Ab','b','Auc','Aub','ub','frame_id','exact')),
        HZono: frozenset(('c','Gc','Gb','Ac','Ab','b','eq_mask','col_ids','bcol_ids')),
        known.Bounds: frozenset(('lb','ub')),
        known.SparseHZAffineExpr: frozenset(('terms','bias','n_out','frame_id')),
        known.SparseHZAffineTerm: frozenset(('source','operators')),
        known.CSRLinearOp: frozenset(('_matrix','_content_key')),
        known.DiagonalLinearOp: frozenset(('_diagonal','_content_key')),
        known.ImplicitConv2DOp: frozenset(('_kernel','_input_shape','_stride','_padding',
            '_dilation','_groups','_output_shape','_row_mask','_logical_expanded_nnz','_content_key')),
        fxg._Namespace: frozenset(('_obj_to_name','_used_names','_base_count')),
        fxg.CodeGen: frozenset(('_body_transformer','_func_name')),
        fxg._FindNodesLookupTable: frozenset(('table',)),
    }
    node_fields = ('graph','name','op','target','type','_sort_key','_args','_kwargs',
                   '_erased','_prev','_next','_input_nodes','users','_repr_fn','meta')
    graph_fields = frozenset(('_root','_used_names','_insert','_len','_graph_namespace',
        '_owning_module','_tracer_cls','_tracer_extras','_codegen','_co_fields','_find_nodes_lookup_table'))
    seen, numeric = set(), {}
    shallow = scalars = references = 0
    record_modules = {nb.__name__, bank.__name__, bank.nt.__name__, cert.__name__,
        'experiments.neural_hz_20260831.definition_first_20260928.d092_native_observation_relay_20261001.native_observation_relay'}

    class Visitor(partial.PartialCSRVisitor):
        def visit(self, value, role):
            pool.charge('numeric_owner_header', 128+8*len(self._storage_records))
            return super().visit(value, role)

        def _visit_numpy(self, value, role, *, entry_semantics):
            pool.charge('numeric_owner_span_header', 128+8*len(self._storage_records))
            return super()._visit_numpy(value, role, entry_semantics=entry_semantics)

        def _visit_torch(self, value, role):
            pool.charge('torch_owner_header', 128)
            if value.device.type != 'cpu':
                raise ValueError('non-CPU task root')
            return super()._visit_torch(value, role)

    visitor = Visitor()

    def visit(value, depth=0):
        nonlocal shallow, scalars, references
        pool.charge('closed_root_reference', 8)
        references += 1
        if depth > 192:
            raise ValueError('current root nesting exceeds bound')
        kind = type(value)
        if value is None or kind in (str, bytes):
            return
        if kind in (int, float, bool, Fraction) or isinstance(value, np.generic):
            if kind is Fraction:
                pool.charge('closed_fraction_words', 16)
                if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > 512:
                    raise ValueError('root rational exceeds512bits')
                scalars += 3
            elif kind is int:
                if abs(value).bit_length() > 512:
                    raise ValueError('root integer exceeds512bits')
                scalars += 1
            elif isinstance(value, np.generic) and (value.dtype.hasobject or value.dtype.kind not in 'biuf'):
                raise ValueError('unknown numeric scalar')
            else:
                scalars += 1
            if scalars+RESERVE > ENTRY_CAP:
                raise MemoryError('current scalar occurrence limit')
            return
        if id(value) in seen:
            return
        pool.charge('closed_root_object', 16)
        seen.add(id(value))
        shallow += sys.getsizeof(value)
        if kind in (torch.dtype, torch.device, torch.layout, torch.memory_format, np.dtype):
            return
        if isinstance(value, Enum):
            visit(value.value, depth+1)
            return
        if isinstance(value, type):
            if not (value.__module__.startswith(('torch', 'onnx2torch', 'act.', 'builtins', 'typing'))):
                raise ValueError('unregistered type metadata')
            return  # Library classes/globals are not instance-owned numeric roots.
        if kind is weakref.WeakValueDictionary:
            return  # Weak arena membership does not own its referents.
        if kind is weakref.ReferenceType:
            return
        if kind is types.MethodType:
            visit(value.__self__, depth+1)
            visit(value.__func__, depth+1)
            return
        if kind in (types.FunctionType, types.BuiltinFunctionType):
            if kind is types.FunctionType:
                visit(vars(value), depth+1)
                visit(value.__defaults__, depth+1)
                visit(value.__kwdefaults__, depth+1)
                for cell in value.__closure__ or ():
                    visit(cell.cell_contents, depth+1)
            owner = getattr(value, '__self__', None)
            if owner is not None and not isinstance(owner, types.ModuleType):
                visit(owner, depth+1)
            return
        if kind is np.ndarray or kind in (torch.Tensor, torch.nn.Parameter) or sp.isspmatrix_csr(value):
            if kind in (torch.Tensor, torch.nn.Parameter):
                if vars(value) or value.grad_fn is not None or value.grad is not None:
                    raise ValueError('uncovered tensor attributes or autograd roots')
            if sp.isspmatrix_csr(value):
                payload = vars(value)
                required = {'_shape','indptr','indices','data'}
                allowed = required | {'maxprint','_has_sorted_indices','_has_canonical_format'}
                if not required <= set(payload) <= allowed:
                    raise ValueError('unknown CSR fields')
                for name in allowed-{'indptr','indices','data'}:
                    if name in payload:
                        visit(payload[name], depth+1)
            role = 'numeric_'+str(len(numeric))
            numeric[role] = value
            visitor.visit(value, role)
            return
        if kind in (dict, OrderedDict, defaultdict, types.MappingProxyType, immutable_dict):
            pool.charge('closed_mapping_fields', 8*len(value))
            if kind is defaultdict:
                visit(value.default_factory, depth+1)
            if kind is OrderedDict:
                attributes = vars(value)
                if set(attributes)-{'_metadata'}:
                    raise ValueError('unregistered OrderedDict fields')
                visit(attributes, depth+1)
            for key, item in value.items():
                visit(key, depth+1)
                visit(item, depth+1)
            return
        if kind in (list, tuple, set, frozenset, torch.Size, immutable_list):
            pool.charge('closed_sequence_fields', 8*len(value))
            for item in value:
                visit(item, depth+1)
            return
        if kind is slice:
            visit((value.start,value.stop,value.step), depth+1)
            return
        if kind is Node:
            pool.charge('closed_fx_node_fields', 8*(len(node_fields)+len(vars(value))))
            visit(vars(value), depth+1)
            for name in node_fields:
                visit(getattr(value, name), depth+1)
            return
        if kind is fxg.Graph:
            if frozenset(vars(value)) != graph_fields:
                raise ValueError('unknown FX Graph fields')
            visit(vars(value), depth+1)
            return
        if kind in schemas:
            if frozenset(vars(value)) != schemas[kind]:
                raise ValueError('unknown declared fields: '+kind.__name__)
            visit(vars(value), depth+1)
            return
        if kind in (tf_type, converter_type) or isinstance(value, torch.nn.Module):
            # Full instance dictionaries, including non-registered tensors;
            # never a state_dict-only substitute for live model roots.
            visit(vars(value), depth+1)
            return
        if is_dataclass(value) and kind.__module__ in record_modules:
            names = tuple(field.name for field in fields(value))
            if frozenset(vars(value)) != frozenset(names):
                raise ValueError('undeclared candidate record fields')
            visit(vars(value), depth+1)
            return
        raise ValueError('unsupported current root: '+kind.__module__+'.'+kind.__name__)

    try:
        visit(roots)
        pool.charge('complete_owner_ledger_export', 256+128*len(visitor._storage_records))
        ledger = visitor.ledger()
        header_reserve = 4*len(seen)+16*len(visitor._storage_records)+8*len(numeric)
        retained = ledger.resident_entries+scalars
        if retained+header_reserve+RESERVE > ENTRY_CAP:
            raise MemoryError('complete current roots exceed64M')
        return dict(numeric_bytes=ledger.resident_bytes, numeric_entries=ledger.resident_entries,
            scalar_occurrences_upper=scalars, retained_entries_upper=retained,
            walk_header_entries_upper=header_reserve, numeric_storages=ledger.numeric_storage_count,
            python_shallow_bytes=shallow, unique_objects=len(seen), reference_visits=references,
            complete_current_root_walk=True, construction_temporaries_qualified=False,
            numeric_rule_id=ledger.rule_id)
    finally:
        visit = None


def plain(value, pool):
    """Paid evidence conversion; no HZ or dense array is serialized here."""
    pool.charge('plain_evidence_reference', 16)
    kind = type(value)
    if value is None or kind in (str, bool, int, float, Fraction):
        return value
    if kind in (dict, types.MappingProxyType):
        pool.charge('plain_evidence_mapping', 16*len(value))
        return {key: plain(item, pool) for key,item in value.items()}
    if kind in (tuple, list):
        pool.charge('plain_evidence_sequence', 8*len(value))
        return tuple(plain(item, pool) for item in value)
    if is_dataclass(value):
        members = fields(value)
        pool.charge('plain_evidence_record', 32*len(members))
        if any(item.name == 'hz' for item in members):
            raise ValueError('do not serialize an intermediate HZ')
        return {item.name: plain(getattr(value,item.name),pool) for item in members}
    raise ValueError('unsupported evidence type')


def check_prefix(old, new, pool, np):
    """Check unchanged original readouts/predicates without sparse subtraction."""
    if (old.frame_id != new.frame_id or old.exact != new.exact or old.n_bin != new.n_bin
            or old.n_cont > new.n_cont or old.n_eq != new.n_eq):
        raise ValueError('native original frame/columns/rows changed')
    for name in ('c','b','ub','Gc','Gb','Ac','Ab','Auc','Aub'):
        a,b = getattr(old,name),getattr(new,name)
        if name in ('c','b','ub'):
            arrays = ((a,b[:a.size]),)
        else:
            if a.shape[0] > b.shape[0] or a.shape[1] > b.shape[1]:
                raise ValueError('native matrix shrank')
            arrays = ((a.indptr,b.indptr[:a.indptr.size]),
                      (a.indices,b.indices[:a.indices.size]),(a.data,b.data[:a.data.size]))
        for left,right in arrays:
            pool.charge('native_prefix_equality', 4*int(left.size)+32)
            if left.dtype != right.dtype or left.shape != right.shape:
                raise ValueError('native original array metadata changed')
            for offset in range(0,int(left.size),4096):
                if not np.array_equal(left[offset:offset+4096],right[offset:offset+4096]):
                    raise ValueError('native original values changed')


def main():
    if sys.argv[1:] != ['--enabled']:
        raise ValueError('explicit --enabled required')
    if not RUN.is_dir():
        raise ValueError('complete single-use mathematical run required')
    OUT.mkdir(exist_ok=False)
    started, initial = time.monotonic(), rss()
    tracemalloc.start()
    sys.dont_write_bytecode = True
    whole, branch, evidence_meter = Pool(), None, None
    identities, post_reserved = {}, False
    manifest = frozen = done = closure = bn_anchor = c5_anchor = None
    model = wrapped = converter = net = tf = queries = labeled = bounds = None
    before, after, events, raw_inputs = {}, {}, [], {}
    applied = payload = modules = graph_certificate = None
    result = dict(schema='d099_fresh_residual_bank_probe_v1', stopped_at_layer=None,
        prefix_completed=False, bank_application_completed=False, all_groups_applied=False,
        execution_completed=False, native_closed_bank_diagnostic_completed=False,
        production_scalar_work_qualified=False, complete_physical_qualification=False,
        actual_model_binding_qualified=False, source_census_qualified=False,
        complete_input_decoder_qualified=False, gpu_computation_completed=False,
        formal_gain=0, source_drift=[], identities_unchecked=[], stage='bootstrap')
    try:
        resource.setrlimit(resource.RLIMIT_AS, (AS_CAP, AS_CAP))
        os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
        if os.environ.get('LD_PRELOAD'):
            raise ValueError('unexpected LD_PRELOAD')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                     'NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
            os.environ[name] = '1'
        for name in ('TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH',
                     'TRITON_CACHE_DIR','TORCHINDUCTOR_CACHE_DIR'):
            path = OUT/name.lower()
            path.mkdir()
            os.environ[name] = str(path)
        def timeout(signum,frame):
            raise TimeoutError('fresh prefix internal235s deadline')
        signal.signal(signal.SIGALRM,timeout)
        signal.alarm(235)
        whole.charge('evidence_and_terminal_reserve',EVIDENCE_CAP+RESERVE)
        manifest = read_json(RUN/'preregistered.json',whole)
        done = read_json(RUN/'exit.json',whole)
        if sha(RUN/'preregistered.json',whole) != done.get('artifacts',{}).get('preregistered.json'):
            raise ValueError('mathematical manifest is not sealed')
        frozen = read_json(HERE/'freeze.json',whole,RESERVE)
        sources = frozen.get('source_sha256')
        if (frozen.get('schema') != 'd099_fresh_residual_bank_v1'
                or type(sources) is not dict or set(sources) != {str(HERE/name) for name in FILES}
                or done.get('all_stages_passed') is not True or done.get('supervisor_exit') != 0
                or done.get('tests_count') != 3845 or done.get('test_files') != 188
                or manifest.get('required_tests') != 3845 or manifest.get('required_test_files') != 188):
            raise ValueError('complete3845/188 mathematical gate required')
        for path,digest in sources.items():
            if manifest['source_sha256'].get(path) != digest:
                raise ValueError('new frozen source differs')
        identities = {p:d for p,d in manifest['source_sha256'].items()
                      if Path(p).is_relative_to(ROOT) and Path(p).suffix == '.py'}
        closure = read_json(CLOSURE,whole,RESERVE)
        if (closure.get('schema') != 'd090_project_import_closure_v1'
                or closure.get('project_root') != str(ROOT) or closure.get('file_count') != 116):
            raise ValueError('fixed complete ACT source inventory differs')
        for path,item in closure['files'].items():
            if path in identities and identities[path] != item['sha256']:
                raise ValueError('ACT closure identity conflict')
            identities[path] = item['sha256']
        identities.update(sources)
        identities[str(HERE/'freeze.json')] = manifest['freeze_sha256']
        for path,digest in FIXED.items():
            recorded = manifest['source_sha256'].get(path,manifest['input_sha256'].get(path))
            if recorded != digest:
                raise ValueError('fixed true-source binding absent or different: '+path)
            if path != str(C5_ANCHOR):
                identities[path] = digest
        whole.charge('identity_inventory_headers',128*(len(identities)+1))
        whole.charge('all_local_identity_pre_post',2*sum(regular(path) for path in identities))
        post_reserved = True
        for path,digest in identities.items():
            if sha(path,whole,prepaid=True) != digest:
                raise ValueError('source identity mismatch: '+path)
        bn_anchor = read_json(BN_ANCHOR,whole)
        # The194MB historical record is already byte-authenticated before
        # and after by this run's mathematical/source supervisors. It is NOT
        # numerical input here, and this worker does not reparse or rehash it.
        # These two facts were statically checked in that exact frozen record.
        c5_anchor = dict(sha256=FIXED[str(C5_ANCHOR)],
            live_registered_gates_passed=True,input_roots_unchanged=True,
            historical_payload_reread_by_worker=False,
            authority='sealed_mathematical_manifest_and_exact_fixed_record')
        if (bn_anchor['target']['model'] != str(MODEL)
                or bn_anchor['target']['converted_spec'] != str(SPEC)
                or not c5_anchor.get('live_registered_gates_passed')
                or not c5_anchor.get('input_roots_unchanged')):
            raise ValueError('frozen corrected-graph/C5 source support not admitted')
        sys.path.insert(0,str(ROOT))
        import numpy as np
        import scipy.sparse as sp
        import torch
        from act.back_end.core import Bounds
        from act.back_end.hybridz_tf.hybridz_tf import HybridzTF
        from act.back_end.hybridz_tf import tf_cnn
        from act.front_end.spec_creator_base import LabeledInputTensor
        from act.front_end.verifiable_model import InputLayer,InputSpecLayer,OutputSpecLayer,VerifiableModel
        from act.front_end.vnnlib_loader.onnx_converter import convert_onnx_to_pytorch,get_onnx_input_shape
        from act.front_end.vnnlib_loader.vnnlib_parser import parse_vnnlib_queries
        from act.pipeline.verification.torch2act import TorchToACT
        from experiments.neural_hz_20260831 import c5_live_roots_v2 as known
        from experiments.neural_hz_20260831 import c5_partial_csr_owner_ledger_v3 as partial
        from experiments.neural_hz_20260831 import bn_graph_faithfulness_certificate_prototype as cert
        from experiments.neural_hz_20260831.c5_runtime_materializer_v2 import installed
        from experiments.neural_hz_20260831.definition_first_20260928.d090_bound_native_discovery_20261001 import archive_probe as old_probe
        from experiments.neural_hz_20260831.definition_first_20260928.d025_interval_capacity_20260930 import evidence as ev
        from experiments.neural_hz_20260831.definition_first_20260928.d098_native_relation_bank_20261001 import native_relation_bank as bank
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        branch, evidence_meter = Pool(BRANCH_CAP,whole), ev.Meter(EVIDENCE_CAP)
        old_probe.project_closure(closure,branch)
        old_probe.check_project_imports(identities,branch)
        modules = (np,torch,sp,known,partial,bank.nb,bank,HybridzTF,TorchToACT,cert)
        if not memory(initial)['memory_gate_passed']:
            raise MemoryError('startup host observation gate failed')
        for path in (MODEL,SPEC):
            size = regular(path,64*1024**2)
            branch.charge('retained_raw_source_read_and_hash',2*size)
            with path.open('rb') as stream:
                raw = stream.read(size+1)
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != FIXED[str(path)]:
                raise ValueError('fresh raw source changed')
            raw_inputs[str(path)] = raw
        result['stage'] = 'fresh_model_conversion'
        with torch.no_grad():
            shape = tuple(get_onnx_input_shape(MODEL))
            if shape != (1,3,56,56):
                raise ValueError('registered input shape changed')
            model = convert_onnx_to_pytorch(MODEL).eval()
            branch.charge('converted_model_parameter_headers',128*(len(tuple(model.parameters()))+len(tuple(model.buffers()))+1))
            dtypes = {value.dtype for value in (*model.parameters(),*model.buffers()) if value.is_floating_point()}
            if dtypes != {torch.float64}:
                raise ValueError('converted original model dtype changed')
            labeled = LabeledInputTensor(torch.zeros(shape,dtype=torch.float64),torch.tensor([0]))
            queries = parse_vnnlib_queries(SPEC,labeled_tensor=labeled)
            if len(queries) != 1:
                raise ValueError('one complete registered input/output query required')
            input_spec,output_spec = queries[0]
            input_spec.lb = input_spec.lb.to(dtype=torch.float64)
            input_spec.ub = input_spec.ub.to(dtype=torch.float64)
            wrapped = VerifiableModel(input_layer=InputLayer(labeled_input=labeled,shape=shape,dtype=torch.float64),
                input_spec=InputSpecLayer(input_spec),model=model,output_spec=OutputSpecLayer(output_spec))
            converter = TorchToACT(wrapped,repair_batchnorm_producer_graph=True)
            net = converter.run()
            branch.charge('graph_audit_headers',128*(len(net.layers)+1))
            occurrences = sum(len(layer.in_vars)+len(layer.out_vars)
                +len(layer.params.get('x_vars',()))+len(layer.params.get('y_vars',())) for layer in net.layers)
            # Covers validation, producer maps and JSON graph identity. This
            # is new binding work, not a scalar-work claim for TorchToACT.
            branch.charge('complete_corrected_graph_audit',64*(occurrences+len(net.layers)+1))
            graph_certificate = cert.audit_graph_faithfulness(net.layers,net.preds,net.succs)
            if (not graph_certificate.accepted or graph_certificate.graph_sha256
                    != bn_anchor['candidate_certificate']['graph_sha256']):
                raise ValueError('fresh corrected graph identity differs')
            if [(net.by_id[i].kind,net.preds[i]) for i in (5,9,16,20)] != [
                    ('RELU',[4]),('RELU',[8]),('ADD',[12,15]),('RELU',[19])]:
                raise ValueError('complete registered residual block topology differs')
            tf = HybridzTF()
            tf._SPARSE_MAX_AFFINE_CELLS = ENTRY_CAP
            tf._rq4_original_relu = False
            # Match the frozen shadow arm. Constructor GC remains its old
            # default; do not silently enable GC by changing a config object.
            for name in ('affine_nnz_guard','relu_nnz_guard','conv_csr_builder',
                         'deferred_relu_materialization','lazy_affine_dag','frontier_image_rebase',
                         'phase_separated_relu','phase_selective_materialization','implicit_conv_dag'):
                setattr(tf,'_neural_hz_sparse_'+name,True)
            tf._neural_hz_compact_relu = False
            bounds = Bounds(input_spec.lb.detach().clone(),input_spec.ub.detach().clone())
            result['stage'] = 'fresh_corrected_c5_prefix'
            saved_functions = (tf_cnn._lazy_materialize,tf_cnn._try_deferred_expr_conv_relu)
            with (OUT/'events.jsonl').open('x') as stream:
                def event(item):
                    branch.charge('source_event_header',128)
                    item = plain(item,branch)
                    evidence_meter.charge(128)
                    encoded = json.dumps(item,sort_keys=True,allow_nan=False)
                    evidence_meter.charge(len(encoded)+1)
                    stream.write(encoded+'\n')
                    stream.flush()
                    events.append(item)
                with installed(enabled=True,emit=event):
                    for layer in net.layers:
                        if layer.id > 20:
                            break
                        event(dict(event='layer_start',layer=layer.id,kind=layer.kind,
                                   elapsed_s=time.monotonic()-started))
                        predecessors = net.preds.get(layer.id,[])
                        incoming = bounds if layer.id == 0 or not predecessors else after[predecessors[0]].bounds
                        after[layer.id] = tf.apply(layer,incoming,net,before,after)
                        event(dict(event='layer_end',layer=layer.id,kind=layer.kind,
                                   elapsed_s=time.monotonic()-started))
                        if not memory(initial)['memory_gate_passed']:
                            raise MemoryError('prefix host observation gate failed')
                if saved_functions != (tf_cnn._lazy_materialize,tf_cnn._try_deferred_expr_conv_relu):
                    raise ValueError('C5 source support did not restore production functions')
            if set(after) != set(range(21)) or type(tf.get_sparse_hz(20)) is not bank.nb.SparseHZono:
                raise ValueError('fresh complete current HZ20 unavailable')
            hz = tf.get_sparse_hz(20)
            result.update(prefix_completed=True,stopped_at_layer=20,stage='current_roots',
                corrected_graph_sha256=graph_certificate.graph_sha256,
                original_hz=dict(n_out=hz.n_out,n_cont=hz.n_cont,n_bin=hz.n_bin,
                    n_eq=hz.n_eq,n_ineq=hz.n_ineq,frame_id=hz.frame_id,exact=hz.exact),
                c5_original_island_whole_cap=CAP,c5_original_island_branch_cap=BRANCH_CAP,
                c5_production_support_restored=True)
            roots = (model,wrapped,converter,net,tf,before,after,queries,labeled,bounds,raw_inputs,
                     manifest,frozen,done,closure,bn_anchor,c5_anchor,events,graph_certificate,result)
            result['pre_bank_roots'] = closed_roots(roots,branch,modules)
            old_probe.check_project_imports(identities,branch)
            remaining = min(branch.cap-branch.used,whole.cap-whole.used)
            branch.charge('complete_bank_entry_reserve_preflight',128)
            # Each allocation population is prepaid in D098/D097: at most16
            # new numeric occurrences per128 scalar/container work units.
            # Include fixed row/header bookkeeping independently of that sum.
            reserve = remaining//8 + 32*(hz.n_cont+hz.n_bin+hz.n_eq+hz.n_ineq+65536+1)
            held = result['pre_bank_roots']['retained_entries_upper']+result['pre_bank_roots']['walk_header_entries_upper']
            result['bank_preflight'] = dict(remaining_work=remaining,held_entries_upper=held,
                                           new_numeric_entries_upper=reserve)
            if held+reserve+RESERVE > ENTRY_CAP:
                raise MemoryError('complete current roots plus bank temporary reserve exceed64M')
            result['stage'] = 'complete_native_bank'
            applied = bank.append_native_observation_bank(hz,pool=branch,enabled=True)
            check_prefix(hz,applied.hz,branch,np)
            result.update(bank_application_completed=True,all_groups_applied=True,
                bank_summary=plain(applied.bank.summary,branch),application_summary=plain(applied.summary,branch))
            result['stage'] = 'complete_evidence'
            payload = dict(schema='d099_complete_native_bank_v1',source=dict(model=str(MODEL),
                model_sha256=FIXED[str(MODEL)],spec=str(SPEC),spec_sha256=FIXED[str(SPEC)],
                corrected_graph_sha256=graph_certificate.graph_sha256,stopped_at_layer=20),
                bank=plain(applied.bank,branch),receipts=plain(applied.receipts,branch),
                summary=plain(applied.summary,branch),formal_gain=0)
            result['final_roots'] = closed_roots((roots,applied,payload),branch,modules)
            receipt = ev.write_evidence(OUT/'complete.json.partial',payload,evidence_meter,{})
            result['evidence'] = dict(file='complete.json',**receipt)
            old_probe.check_project_imports(identities,branch)
            result['execution_completed'] = True
    except Exception as exc:
        result['failure'] = dict(type=type(exc).__name__,reason=str(exc)[:2048])
    finally:
        # External supervisor remains at240s. The235s alarm remains active
        # through bounded, prepaid post-identities, not an unbounded cleanup.
        if post_reserved:
            for path,digest in identities.items():
                try:
                    if sha(path,whole,prepaid=True) != digest:
                        result['source_drift'].append(path)
                except Exception:
                    result['identities_unchecked'].append(path)
        result.update(whole_work_used=whole.used,branch_work_used=branch.used if branch else 0,
            evidence_work_used=evidence_meter.used if evidence_meter else 0,
            whole_work_parts=whole.parts,branch_work_parts=branch.parts if branch else {},
            whole_failed_charge=whole.failure,branch_failed_charge=branch.failure if branch else None,
            worker_wall_s=time.monotonic()-started,production_scalar_work_qualified=False,
            complete_physical_qualification=False,actual_model_binding_qualified=False,
            source_census_qualified=False)
        try:
            result.update(memory(initial))
        except Exception as exc:
            result.update(memory_gate_passed=False,memory_error=str(exc)[:512])
        passed = bool(result['execution_completed'] and result['prefix_completed']
            and result['all_groups_applied'] and result.get('memory_gate_passed')
            and not result['source_drift'] and not result['identities_unchecked']
            and 'failure' not in result and result['worker_wall_s'] <= 240)
        result['native_closed_bank_diagnostic_completed'] = passed
        if passed:
            (OUT/'complete.json.partial').rename(OUT/'complete.json')
        signal.alarm(0)
        encoded = json.dumps(result,sort_keys=True,indent=2,allow_nan=False)
        if len(encoded.encode()) > RESERVE:
            passed = False
            encoded = json.dumps(dict(schema=result['schema'],native_closed_bank_diagnostic_completed=False,
                execution_completed=False,failure='terminal summary exceeded65536bytes',formal_gain=0))
        with (OUT/'worker.json').open('x') as stream:
            stream.write(encoded+'\n')
            stream.flush()
            os.fsync(stream.fileno())
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
