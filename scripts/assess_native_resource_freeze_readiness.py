"""Read-only V2 freeze-readiness arithmetic, not execution or capacity approval.

Parses source AST and constructs small filename/JSON templates only. No project
runtime imports, source factories, model arrays, numerical kernels or actors.
The independent source checks remain mandatory; their costs are not optimized
away by this analysis. Unknown slot/heap/return demands remain explicit nulls.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = 'd6455b676f6b139d110e9a3abdf1f4cd2d5bc7ad'
REPORT = 'docs/hz_native_resource_freeze_readiness_20261018.json'
DOC = 'docs/hz_native_resource_freeze_readiness_20261018.md'
IMPLEMENTATION = 'docs/hz_native_resource_implementation_20261018.json'
STORAGE = 'docs/hz_native_storage_lifetimes_20261018.json'
FULL = 'docs/hz_native_full_shape_decision_20261018.json'
M, L = 2**20, 65536
H = '0'*64


def require(ok, why):
    if not ok:
        raise ValueError(why)


def raw(name): return (ROOT/name).read_bytes()
def sha(name): return hashlib.sha256(raw(name)).hexdigest()
def load(name): return json.loads(raw(name))
def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def assignment(name, variable):
    return next(n.value for n in ast.parse(raw(name)).body if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == variable for t in n.targets))


def code_files():
    a = assignment('scoped_source/direct_package.py', 'CODE_FILES')
    gen = a.left.args[0]
    require(ast.unparse(gen.elt) == "'scoped_source/' + name + '.py'", 'closed code names')
    base = tuple('scoped_source/'+n+'.py' for n in ast.literal_eval(gen.generators[0].iter))
    base += ast.literal_eval(a.right)
    f = 'scoped_source/direct_lifecycle_package.py'
    base += ast.literal_eval(assignment(f, 'CODE_FILES').right)
    shared = ast.literal_eval(assignment(f, 'SHARED_CODE'))
    paged = ast.literal_eval(assignment(f, 'PAGED_CODE'))
    expr = assignment('scoped_source/native_package.py', 'CODE_FILES')
    require(ast.unparse(expr).startswith('tuple(sorted(set(BASE_CODE + SHARED_CODE + PAGED_CODE + '),
            'closed native roster')
    return tuple(sorted(set(base+shared+paged+ast.literal_eval(expr.args[0].args[0].args[0].right))))


def package_names(ns, nf, nt, nr, code):
    # A one-chunk-per-tensor naming template suffices for length algebra only.
    # This is NOT a generated source, manifest or a claim about real chunking.
    names = ['source/manifest.json'] + [f'source/t/{i:05d}-00000.bin' for i in range(ns-1)]
    names += [f'forest/{i:04d}.bin' for i in range(nf)] + ['forest/index.json']
    names += [f'tiles/{i:04d}.bin' for i in range(nt)]
    names += [f'result/{i:04d}.json' for i in range(nr)] + ['result/index.json']
    names += ['code/'+n for n in code] + ['contract.json','manifest.json','bundle.json','bindings.json']
    return ['artifacts/package/'+n for n in names]


def bundle_bytes(names, sizes):
    prefix = 'artifacts/package/'
    entries = [dict(file=n.removeprefix(prefix), bytes=sizes[n], sha256=H)
               for n in sorted(names) if n != prefix+'bundle.json']
    value = dict(schema='NATIVE_SHARED_BUNDLE_V1',manifest='manifest.json',members=entries)
    return len(compact(value))


def analyze():
    implementation, storage, full = load(IMPLEMENTATION), load(STORAGE), load(FULL)
    for name, expected in implementation['anchors'].items():
        require(sha(name) == expected, 'frozen V2 implementation drift: '+name)
    for name, expected in full['source_bindings'].items():
        require(sha(name) == expected, 'full-shape premise drift: '+name)
    require(load('docs/hz_native_resource_execution_approvals.json') ==
            {'schema':'NATIVE_RESOURCE_APPROVALS_V2','envelopes':[]}, 'no new execution approval')
    old = code_files()
    require(list(old) == storage['code_files'], 'unchanged old 32-file roster')
    aux = ast.literal_eval(assignment('scoped_source/native_resource_portable.py','AUXILIARY_CODE'))
    require(len(old) == 32 and len(aux) == 4 and not set(old).intersection(aux), 'new auxiliary files')
    added_sizes = {n:len(raw(n)) for n in aux}
    require(sum(added_sizes.values()) == 43921, 'current auxiliary byte identity')
    refs = [dict(file='code/'+n, bytes=added_sizes[n], sha256=H) for n in aux]
    # E/I placeholders measure only fixed punctuation/fields; never executable.
    capsule = dict(schema='NATIVE_RESOURCE_BINDING_V2',envelope={},invocation={},
        bindings={k:H for k in ('envelope','registry','common','layout')},invocation_hash=H,
        auxiliary=refs,source_scope='DECLARED_SOURCE_NOT_CHECKPOINT_OR_NATIVE_FLOAT_PROOF')
    capsule_constant = len(compact(capsule))-4
    require(capsule_constant == 1129, 'capsule constant excluding encoded E and I')
    added = ['artifacts/package/code/'+n for n in aux]+['artifacts/package/bindings.json']
    require(sum(map(len,added)) == 270, 'additional root-relative names')

    # Exact schema differentials with varied positive sizes; no evidence files.
    for ns,nf,nt,nr in ((1,1,1,1),(16,2,4,3),(96,107,84,32)):
        names = package_names(ns,nf,nt,nr,old+aux)
        np, lp = len(names), sum(map(len,names))
        require(np == ns+nf+nt+nr+42 and lp == 2185+42*ns+33*nf+32*nt+34*nr,
                'package roster/length formula')
        sizes = {n:(7,83,1024,98765)[i%4] for i,n in enumerate(names)}
        for n in aux: sizes['artifacts/package/code/'+n] = added_sizes[n]
        sizes['artifacts/package/bindings.json'] = len(compact(capsule))
        digits = sum(len(str(v)) for n,v in sizes.items() if not n.endswith('/bundle.json'))
        require(bundle_bytes(names,sizes) == lp+79*np+digits-33, 'exact bundle algebra')
        old_names = [n for n in names if n not in added]
        delta = bundle_bytes(names,sizes)-bundle_bytes(old_names,sizes)
        require(delta == 684+len(str(sizes['artifacts/package/bindings.json'])), 'bundle delta')

    stages = ('production','replay','reception')
    meta = ['envelope.json','invocation.json','controller.log']
    meta += [s+suffix for s in stages for suffix in ('.json','.log','.progress')]
    meta += ['controller.json','terminal.json','caller.json']
    require(len(meta) == len(set(meta)) == 15, 'V2 root roster')
    census_constants = {}
    for phase,excluded in (
        ('reception',('reception.json','controller.json','terminal.json','caller.json')),
        ('controller',('controller.json','terminal.json','caller.json')),
        ('terminal',('terminal.json','caller.json')),
        ('caller',('caller.json',)),('return',())):
        present = [n for n in meta if n not in excluded]
        census_constants[phase] = 40+2*sum(map(len,present))+83*len(present)
    require(list(census_constants.values()) == [1267,1378,1491,1600,1705], 'census stage deltas')

    # Independent dimensions only; never evaluate tensor values or call factories.
    def tensor_bytes(dims, center):
        return 8*(center+sum(a*b+b for graph in dims for a,b in zip(graph,graph[1:])))
    template_bytes = tensor_bytes(((3,2),(3,4,2),(3,2)),3)
    dimensions = dict(mixed=((2,3),(2,5,2),(2,5,2),(2,5,2)),
        clipped=((2,3,3,3),(2,2,2,2,2),(2,2,2,2,2),(2,2,2,2,2)),
        guarded=((1,3),(1,5),(1,5),(1,5)))
    mixed_bytes = tensor_bytes(dimensions['mixed'],2)
    clipped_bytes = tensor_bytes(dimensions['clipped'],2)
    require((template_bytes,mixed_bytes,clipped_bytes) == (360,736,856), 'factory dimension-only payload')
    sources = {case:dict(files=2+2*sum(len(g)-1 for g in dims),
        binary_bytes=tensor_bytes(dims,1 if case == 'guarded' else 2),manifest_bytes=None)
        for case,dims in dimensions.items()}
    require([(v['files'],v['binary_bytes']) for v in sources.values()] == [(16,736),(32,856),(10,296)],
            'source files differ from retained template files')
    generated = {}
    for case,prefix in (('mixed','generated/'),('clipped','generated/template/')):
        names = [prefix+'manifest.json']+[prefix+f't/{i:05d}-00000.bin' for i in range(9)]
        generated[case] = dict(files=len(names),root_name_bytes=sum(map(len,names)),
                              binary_bytes=template_bytes,manifest_bytes=None)
    generated['guarded'] = dict(files=0,root_name_bytes=0,binary_bytes=0,manifest_bytes=0)
    require(generated['mixed']['root_name_bytes'] == 266 and generated['clipped']['root_name_bytes'] == 356,
            'retained template names')

    p,a = full['shape']['parameter_values'],full['shape']['affine_outputs']
    w = full['work']
    require(w['first_snapshot_mandatory'] == 2*p+a and w['current_complete_path_loop_floor'] == 12*p
            and w['two_parameterwise_validation_floor'] == 2*p, 'source traversal premises')
    without_one_bank = 12*p-(2*p+a)
    require(without_one_bank == 69610392 and 2*p > 8000000, 'bank reuse cannot cure Work admission')
    extra = (DOC,IMPLEMENTATION,STORAGE,FULL,
        'docs/hz_native_joint_contract_20261018.json',
        'scoped_source/source_dag_controls.py','scoped_source/affine_reference_controls.py',
        'scoped_source/tensor_row_controls.py','scoped_source/shared_archive.py',
        'scoped_source/direct_supervised.py','scoped_source/forest_endpoint_controls.py',
        'scoped_source/shared_source.py','scoped_source/shared_build.py',
        'scoped_source/check_shared_source.py','scripts/assess_native_resource_freeze_readiness.py')
    anchors = {n:sha(n) for n in sorted(set(extra)|set(implementation['anchors'])|set(aux))}
    return dict(schema='NATIVE_RESOURCE_FREEZE_READINESS_STATIC_V2',base_head=BASE,
        status='NOT_READY_FOR_CONTROL_FREEZE_OR_FULL_SHAPE_ADMISSION',anchors=anchors,
        archive=dict(metadata_root_files=meta,metadata_reserve=8*M,
            cap_sum='5*min(M,m)+6*min(L,m)+f+t+c+e',required_metadata_encoded_lower=M,
            maximum_terminal_caller_envelope_sum_if_f_equals_M=8*M-6*M-6*L,
            all_report_caps_at_M_without_envelope=8*M+6*L,
            all_report_caps_at_M_excess_before_envelope=6*L,
            new_code_bytes=added_sizes,new_files=5,new_name_bytes=270,
            package_members='Ns+Nf+Nt+Nr+42',
            package_root_name_bytes='2185+42*Ns+33*Nf+32*Nt+34*Nr',
            capsule_bytes='e+i+1129',bundle_bytes='LP+79*nP+D_nonbundle-33',
            package_increment='Kcap+44605+digits(Kcap)',
            package_increment_upper_if_capsule_at_most_1MiB='e+i+45741',
            source_static_dimensions=sources,generated_templates=generated,
            root_members='nP+15+nG',root_bytes='BP+G+actual_metadata_bytes',
            census_constant=census_constants,census_variable='2*(LP+LG)+83*(nP+nG)',
            census_scope='complete source transfer and successful root publication order; not failure path names',
            four_disk_censuses_upper='5736+8*(LP+LG)+332*(nP+nG)',
            archive_default_reserve=8192,
            success_capacity='BP<=128MiB-OB-8MiB-8192; nP<=4096-ON-16',
            failure_scope='listed owned writer prefixes only; no arbitrary external/native file guarantee',
            failure_upper='BP_max+G_max+R15; files<=nP_max+nG_max+15',
            envelope_actual_bytes=None,invocation_actual_bytes=None,complete_archive_upper=None,
            terminal_failure_schema_upper=None,caller_failure_schema_upper=None,
            return_anchor_persisted=False,return_anchor_bytes=None),
        own_memory=dict(
            joint_slot='3*(session+metadata+source+observer+observer_next+phase)+terminal+caller',
            external_join='J_slot+B_outside<=joint_peak<=capacity',
            outside=['caller registry/intake/launcher',
                'controller and current actor retained bootstrap raw E/raw I/decoded I',
                'returned object and persistent return anchor save overlap'],
            replay_distinct_profile_trees=5,
            encoded_floor_is_complete_demand=False,
            phase_requires_sum_of_retained_and_temporary_copies=True,
            concrete_slots=None,outside_upper=None,complete_self_owned_heap_upper=None,
            AS_or_RSS_is_not_self_owned_capacity=True),
        full_shape=dict(parameter_values=p,affine_rows=a,first_snapshot_floor=2*p+a,
            current_loop_floor=12*p,ideal_one_bank_removal_remaining_floor=without_one_bank,
            conditional_two_parameterwise_checks_floor=2*p,work_cap=8000000,
            membership_lower=126632,membership_cap=65536,
            conditional_floor_not_algorithm_universality=True,admitted=False,
            source_bank_reuse_is_not_G1_algorithm_or_precision_gain=True),
        decision=dict(continue_action='one object/slot/phase and return-anchor contract using this interface',
            stop=['repeated old synthetic controls','new wrapper interface without evidence',
                  'old-cap full-shape attempts','bank-cache microbenchmarks as G1 evidence'],
            frozen_pair_or_execution_plan=False,numerical_resource_policy_changed=False,
            native_calls=0,control_attempts=0,real_models=0,gpu_runs=0,
            algorithm_gain=False,scientific_upgrade=False,goals='G1-G6 OPEN'))


if __name__ == '__main__':
    result = analyze()
    if sys.argv[1:] == ['--check']:
        require(result == load(REPORT), 'freeze-readiness record drift')
        print('PASS: V2 static roster/length/Work arithmetic; NOT a runtime or capacity approval')
    else:
        require(not sys.argv[1:], 'only --check is supported')
        print(json.dumps(result,sort_keys=True,indent=2))
