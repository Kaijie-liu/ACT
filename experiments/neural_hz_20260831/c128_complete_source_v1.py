"""Full old/new ordinary nonconvex source equality, not a graph-only sample.

Independent builds retain every source equation, graph map, owner and inverse.
The only allowed scalar differences are explicitly listed support-work totals.
All numeric arrays and full point populations are returned for complete saving.
"""
from fractions import Fraction as F
import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf.exact_linear_op import ImplicitConv2DOp
from experiments.neural_hz_20260831.c120_complete_source_fixture_v1 import expression, _binding
from experiments.neural_hz_20260831.c97_birth_emission_v1 import lift as old_lift
from experiments.neural_hz_20260831.c128_birth_emission_v1 import lift as new_lift
from experiments.neural_hz_20260831.c17_ownership_audit_v1 import actual_words
from experiments.neural_hz_20260831.c62_local_equations_v1 import reconstruct, SCHEMA

MODES = ('dense', 'masked', 'heterogeneous', 'noop')
SOURCE_RESERVES = dict(dense=32000000, masked=32000000, heterogeneous=32000000, noop=4000000)
CATEGORY_CAPS = dict(source=200000000, setup=2000000, comparison=10000000,
                     proof=6000000, evidence=20000000, ledger=16000000)
COMPLETE_WORK_UPPER = sum(CATEGORY_CAPS.values())
MATRICES = ('Gc','Gb','Ac','Ab','Auc','Aub')
VECTORS = ('c','b','ub')
FIELD_ARRAYS = ('keep','eq_roots','eq_scales','ineq_roots','ineq_scales',
                'def_rows','owners','uid_slabs','radix_gauges')
FIELD_SCALARS = ('old_n_cont','old_n_bin','old_n_eq','logical_n_cont')
WORK_FIELDS = {'support_work','total_work_upper','largest_branch_work_upper',
               'affine_work_upper','whole_base_work','branch_base_work'}


def fixture(mode, pool):
    if mode not in MODES:
        raise ValueError('unregistered complete source fixture')
    pool.charge('c128_complete_original_fixture_creation', 131072)
    expr = expression(c=16,k=32,h=6)
    if mode != 'dense':
        if mode == 'masked':
            removed = np.indices((16,6,6)).sum(axis=0).reshape(-1)%4 == 0
        else:
            removed = np.ones((16,6,6),bool)
            if mode == 'heterogeneous':
                removed[:12] = False
                removed[:,0,0] = False
            else:
                for channel in range(16):
                    removed[channel,2+channel%2,2+(channel//2)%2] = False
            removed = removed.reshape(-1)
        source = expr.terms[0].source
        for name in ('Gc','Gb'):
            matrix = getattr(source,name)
            matrix.data[removed] = 0
            matrix.eliminate_zeros()
        source.c[removed] = 0
    return expr, np.ones(expr.n_out,bool)


def source_arrays(expr):
    arrays = dict(expression_bias=expr.bias)
    # These fixed fixtures have one complete shared source and one Conv; all
    # later CSR operators are retained as well, not just their graph support.
    if len(expr.terms) != 1:
        raise ValueError('complete fixed single-term source expected')
    source = expr.terms[0].source
    for name in VECTORS:
        arrays['source_'+name] = getattr(source,name)
    for name in MATRICES:
        matrix = getattr(source,name)
        for part in ('data','indices','indptr'):
            arrays['source_'+name+'_'+part] = getattr(matrix,part)
    for index,op in enumerate(expr.terms[0].operators):
        if type(op) is ImplicitConv2DOp:
            arrays['operator_'+str(index)+'_kernel'] = op._kernel
            if op._row_mask is not None:
                arrays['operator_'+str(index)+'_row_mask'] = op._row_mask
        elif type(op) is sp.csr_matrix:
            for part in ('data','indices','indptr'):
                arrays['operator_'+str(index)+'_'+part] = getattr(op,part)
        else:
            raise ValueError('complete source operator inventory differs')
    return arrays


def source_header(expr, origin_binding):
    source = expr.terms[0].source
    operators = []
    for op in expr.terms[0].operators:
        header = dict(shape=list(op.shape),type=type(op).__name__)
        if type(op) is ImplicitConv2DOp:
            header.update(input_shape=list(op.input_shape),output_shape=list(op.output_shape),
                stride=list(op._stride),padding=list(op._padding),dilation=list(op._dilation),
                groups=op._groups,kernel_shape=list(op._kernel.shape),
                output_row_mask_present=op._row_mask is not None)
        operators.append(header)
    return dict(frame_id=expr.frame_id,n_out=expr.n_out,term_count=1,source_count=1,
        source=dict(frame_id=source.frame_id,exact=source.exact,n_cont=source.n_cont,
            n_bin=source.n_bin,n_eq=source.n_eq,n_ineq=source.n_ineq,n_out=source.n_out,
            matrix_shapes={name:list(getattr(source,name).shape) for name in MATRICES}),
        operators=operators,origin_binding=origin_binding,
        binding_object_ids_are_process_local_not_portable_authority=True)


def packet(state):
    if set(state) != {'schema','lineage_schema','fields','construction','independent_qualification_pending'}:
        raise ValueError('unknown complete generated state field omitted from comparison')
    fields, construction = state['fields'],state['construction']
    if set(fields) != {'expression','hz','report',*FIELD_ARRAYS,*FIELD_SCALARS}:
        raise ValueError('unknown complete source field omitted from comparison')
    if set(construction) != {'nodes','root','origin_binding','eq_uids','ineq_uids'}:
        raise ValueError('unknown complete construction field omitted from comparison')
    hz = fields['hz']
    if set(vars(hz)) != {*VECTORS,*MATRICES,'frame_id','exact'}:
        raise ValueError('unknown HZ semantic field omitted from complete source inventory')
    arrays = {name:fields[name] for name in FIELD_ARRAYS}
    shapes = {}
    for name in VECTORS:
        arrays['hz_'+name] = getattr(hz,name)
    for name in MATRICES:
        matrix = getattr(hz,name)
        if type(matrix) is not sp.csr_matrix:
            raise ValueError('complete canonical source matrices required')
        shapes[name] = list(matrix.shape)
        for part in ('data','indices','indptr'):
            arrays['hz_'+name+'_'+part] = getattr(matrix,part)
    nodes = []
    for index,node in enumerate(construction['nodes']):
        expected = {'kind','width','parents','support','needed','support_work','slots','exponents'}
        expected |= {'source'} if node['kind']=='source' else {'op'} if node['kind']=='op' else set()
        if set(node) != expected:
            raise ValueError('unknown complete graph node field omitted from comparison')
        for name in ('support','needed','slots','exponents'):
            arrays['node_'+str(index)+'_'+name] = node[name]
        nodes.append(dict(kind=node['kind'],width=node['width'],parents=list(node['parents']),
                          support_work=node['support_work']))
    arrays.update(eq_uids=construction['eq_uids'],ineq_uids=construction['ineq_uids'])
    if any(type(a) is not np.ndarray or a.dtype.kind not in 'biuf' for a in arrays.values()):
        raise ValueError('complete source numeric inventory must be primitive arrays')
    meta = dict(schema=state['schema'],lineage_schema=state['lineage_schema'],
        independent_qualification_pending=state['independent_qualification_pending'],
        field_scalars={name:fields[name] for name in FIELD_SCALARS},
        hz=dict(frame_id=hz.frame_id,exact=hz.exact,n_cont=hz.n_cont,n_bin=hz.n_bin,
                n_eq=hz.n_eq,n_ineq=hz.n_ineq,n_out=hz.n_out,matrix_shapes=shapes),
        root=construction['root'],nodes=nodes,report=fields['report'],
        origin_binding=construction['origin_binding'])
    return meta, arrays


def semantic_metadata(value):
    # Work totals may differ ONLY in these exact report locations. There is no
    # recursive name-based omission of arbitrary semantic or ownership fields.
    report = {key:value for key,value in value['report'].items() if key not in WORK_FIELDS}
    report['node_counts'] = [{k:v for k,v in count.items() if k != 'support_work'}
                             for count in report['node_counts']]
    nodes = [{k:v for k,v in node.items() if k != 'support_work'} for node in value['nodes']]
    return dict(value,report=report,nodes=nodes)


def complete_source(mode, budgets, held, emit, *, retain):
    expr,keep = fixture(mode,budgets['setup'])
    # Fixed reserve precedes all packet/source/graph header walks, manifest
    # assembly and scalar metadata equality; array-dependent scans are paid
    # separately once their complete header inventory is available.
    budgets['comparison'].charge('c128_complete_packet_source_and_archive_headers',16384)
    before = _binding(expr,budgets['setup'])
    reserve = SOURCE_RESERVES[mode]
    built,metadata,arrays,artifacts = {},{},{},{}
    case = dict(expression=expr,keep=keep,builds=built,packets=arrays,metadata=metadata,
                artifacts=artifacts)
    held[mode] = case
    case['source_arrays'] = source_arrays(expr)
    case['source_header'] = source_header(expr,before)
    artifacts['source'] = retain(mode,'source',case['source_arrays'],case['source_header'])
    for name,builder in (('old',old_lift),('new',new_lift)):
        budgets['source'].charge('c128_complete_'+name+'_source_reservation',reserve)
        built[name] = builder(expr,keep,enabled=True,max_work=reserve,max_branch_work=reserve)
        metadata[name],arrays[name] = packet(built[name])
        # Retain the complete returned state BEFORE any cross-build/source
        # comparison can fail, and before beginning the other construction.
        artifacts[name] = retain(mode,name,arrays[name],metadata[name])
        if _binding(expr,budgets['setup']) != before:
            raise ValueError('original source/expression changed during complete source build')
        emit(dict(event='complete_source_built',mode=mode,version=name,
                  actual_upper=built[name]['fields']['report']['total_work_upper']))
    old,new = metadata['old'],metadata['new']
    entries = sum(int(a.size) for packet_arrays in arrays.values() for a in packet_arrays.values())
    budgets['comparison'].charge('c128_all_source_graph_inverse_arrays_and_scalar_comparison',
                                 8*entries)
    if any(not getattr(state['fields']['hz'],name).has_canonical_format
           for state in built.values() for name in MATRICES):
        raise ValueError('complete source sparse matrices must remain canonical')
    if set(arrays['old']) != set(arrays['new']) or semantic_metadata(old) != semantic_metadata(new):
        raise ValueError('full source graph/frame/predicate/UID metadata differs')
    for name,a in arrays['old'].items():
        b = arrays['new'][name]
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes(order='C') != b.tobytes(order='C'):
            raise ValueError('complete original source numeric field differs: '+name)
    if (old['hz']['n_bin'] != 1 or old['hz']['n_ineq'] != 1
        or old['hz']['frame_id'] != expr.frame_id or not old['hz']['exact']
        or old['lineage_schema'] != SCHEMA):
        raise ValueError('original nonconvex/shared-frame/exact-inverse semantics lost')
    for left,right in zip(built['old']['construction']['nodes'],built['new']['construction']['nodes'],strict=True):
        if left.get('source') is not right.get('source') or left.get('op') is not right.get('op'):
            raise ValueError('original graph source/operator sharing identity differs')
    if built['old']['construction']['origin_binding'] != built['new']['construction']['origin_binding']:
        raise ValueError('original source binding differs')
    owners,points = {},{}
    case.update(owner_oracles=owners,point_evidence=points)
    for name,state in built.items():
        fields,construction = state['fields'],state['construction']
        hz = fields['hz']
        budgets['proof'].charge('c128_complete_independent_sparse_owner_and_inverse',
            1024+8*(hz.Ac.nnz+hz.Auc.nnz)+32*(hz.n_eq+hz.n_ineq)+64*hz.n_cont)
        expected = actual_words(hz,fields['old_n_cont'],fields['logical_n_cont'],
                                construction['eq_uids'],construction['ineq_uids'])
        if not np.array_equal(expected,fields['owners']):
            raise ValueError('complete sparse incidence owner oracle differs')
        owners[name] = expected
        point = [F(i%5-2,8) for i in range(hz.n_cont)]
        recovered = reconstruct(point,fields['eq_roots'],fields['eq_scales'],
            old_n_cont=fields['old_n_cont'],old_n_eq=fields['old_n_eq'],n_cont=hz.n_cont,schema=SCHEMA)
        points[name] = dict(original=[(v.numerator,v.denominator) for v in point],
                           recovered=[(v.numerator,v.denominator) for v in recovered])
    if points['old'] != points['new']:
        raise ValueError('complete exact inverse point population differs')
    point_counts = {name:{key:len(values) for key,values in populations.items()}
                    for name,populations in points.items()}
    owner_metadata = dict(full_sparse_owner_oracles_equal=True,
        complete_inverse_arrays_and_points_equal=True,
        complete_owner_counts={name:int(values.size) for name,values in owners.items()},
        full_point_counts=point_counts,points_are_not_network_witnesses=True,
        complete_case_qualification_pending=True)
    # The callback also saves the full separate point JSON here. Later cost or
    # reporting failures cannot erase already completed owner/inverse evidence.
    artifacts['owners'] = retain(mode,'owners',owners,owner_metadata,points=points)
    costs = {name:built[name]['fields']['report'] for name in built}
    for name,report in costs.items():
        if (report['total_work_upper'] > reserve or report['largest_branch_work_upper'] > reserve
            or report['source_first_native_or_LIVE_admission']):
            raise ValueError('complete ordinary source cost or admission scope differs')
    if costs['old']['total_work_upper']-costs['old']['whole_base_work'] != costs['new']['total_work_upper']-costs['new']['whole_base_work']:
        raise ValueError('unchanged coupled row/quotient work differs')
    report = dict(mode=mode,complete_source_arrays_bitwise_equal=True,
        complete_semantic_metadata_equal=True,complete_original_source_sharing_equal=True,
        full_sparse_owner_oracles_equal=True,complete_inverse_arrays_and_points_equal=True,
        arrays_per_version=len(arrays['old']),entries_both_versions=entries,
        nonconvex_binary_factors=1,inequalities=1,complete_source_metadata=metadata,
        complete_original_source_header=case['source_header'],artifacts=artifacts,
        original_source_unchanged=True,source_reservation_each=reserve,
        support_work={name:costs[name]['support_work'] for name in costs},
        whole_work={name:costs[name]['total_work_upper'] for name in costs},
        branch_work={name:costs[name]['largest_branch_work_upper'] for name in costs},
        full_point_counts=point_counts,
        points_are_not_network_witnesses=True,source_or_LIVE_admitted=False,formal_gain=0)
    return report,case
