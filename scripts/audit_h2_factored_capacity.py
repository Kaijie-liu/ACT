"""Source/metadata-only H2 capacity ledger, NOT a real-run admission command.

No ACT, tensor, checkpoint, dataset, native solver or proof constructor imports.
Existing limits are read from source AST; no arbitrary code is evaluated.
Counts assume the registered dense recipe, not a reloaded native topology.
"""
import argparse
import ast
import hashlib
import json
from math import comb, prod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SELECTION = 'act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json'
TRAINING = 'act/pipeline/moe/configs/experiment1_multiseed_training_r1.json'
CONFIG = 'configs/h2_factored_supervision_20260930.json'
IO = 'scoped_source/factored_io.py'
SOURCE = 'scoped_source/factored_source.py'
VERIFY = 'scoped_source/factored_verify.py'
SUPERVISOR = 'scoped_source/factored_supervised.py'
FILES = (SELECTION, TRAINING, CONFIG, IO, SOURCE, VERIFY, SUPERVISOR,
    'act/back_end/moe/factory.py', 'scoped_source/capture.py',
    'scoped_source/sparse_intake.py', 'scoped_source/graph.py',
    'scoped_source/factored_worker.py', 'scoped_source/factored_ir.py',
    'scoped_source/sparse_ir.py', 'scoped_source/factored_build.py',
    'scoped_source/factored_check.py', 'scoped_source/factored_portable.py',
    'scoped_source/endpoint_source_build.py', 'scoped_source/endpoint_source_check.py',
    'scoped_source/endpoint_check.py',
    'act/back_end/solver/sparse_lp_certificate.py', 'scoped_source/sparse_check.py',
    'upstream_source/checker.py', 'router_source/checker.py',
    'source_enclosure/format.py', 'scoped_proof/io.py', 'scoped_proof/supervisor.py',
    'scripts/audit_h2_factored_capacity.py')


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def integer(node):
    if isinstance(node, ast.Constant) and type(node.value) is int and 0 <= node.value <= 2**40:
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Mult, ast.Pow)):
        a, b = integer(node.left), integer(node.right)
        if isinstance(node.op, ast.Pow) and b > 40:
            raise ValueError('unbounded exponent')
        value = a*b if isinstance(node.op, ast.Mult) else a**b
        if value <= 2**40:
            return value
    raise ValueError('nonliteral or unbounded integer policy')


def definition(text, name):
    nodes = ast.parse(text).body
    for part in name.split('.'):
        found = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == part]
        if len(found) != 1:
            raise ValueError('missing/ambiguous definition: '+name)
        node = found[0]; nodes = node.body
    return node


def assignments(text, names):
    result = {}
    for name in names:
        found = [n.value for n in ast.parse(text).body if isinstance(n, ast.Assign)
                 and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and n.targets[0].id == name]
        if len(found) != 1:
            raise ValueError('missing/ambiguous constant: '+name)
        result[name] = integer(found[0])
    return result


def default(text, function, name):
    args = definition(text, function).args
    values = dict(zip([a.arg for a in args.args][-len(args.defaults):], args.defaults)) if args.defaults else {}
    values.update(zip([a.arg for a in args.kwonlyargs], args.kw_defaults))
    if name not in values or values[name] is None:
        raise ValueError('missing default: '+name)
    return values[name]


def require_expression(text, function, expression):
    """Fail closed if a reviewed call/guard changes; not a Python proof engine."""
    needle = ast.dump(ast.parse(expression, mode='eval').body)
    if needle not in {ast.dump(n) for n in ast.walk(definition(text, function))}:
        raise ValueError('reviewed execution contract changed: '+function+' / '+expression)


def policy(raw):
    texts = {k: v.decode() for k, v in raw.items()}
    core = assignments(texts[IO], ('MEMBER_LIMIT', 'HEADER_LIMIT', 'BLOCK_LIMIT', 'TOTAL_LIMIT'))
    source = assignments(texts[SOURCE], ('SOURCE_TOTAL_LIMIT', 'TENSOR_ELEMENTS'))
    portable = assignments(texts[VERIFY], ('MEMBER_LIMIT', 'ENVELOPE_LIMIT', 'TOTAL_LIMIT'))
    cfg = json.loads(raw[CONFIG])
    if (cfg['source_chunk_bytes'] != 64 or cfg['real_requests'] != 0 or len(cfg['cases']) != 5
            or cfg['maximum_budget_seconds'] != 300):
        raise ValueError('frozen synthetic protocol changed')
    if not (core['MEMBER_LIMIT'] == portable['MEMBER_LIMIT'] and
            core['HEADER_LIMIT'] == portable['ENVELOPE_LIMIT'] and core['TOTAL_LIMIT'] == portable['TOTAL_LIMIT']):
        raise ValueError('mismatched core/portable policy; requires new review')
    budget_default = default(texts[SUPERVISOR], 'supervise', 'budget')
    if (not isinstance(budget_default, ast.Constant) or type(budget_default.value) not in (int, float)
            or budget_default.value != cfg['maximum_budget_seconds']):
        raise ValueError('supervisor budget default changed')
    node = default(texts[SOURCE], 'pack', 'chunk_bytes')
    if not isinstance(node, ast.Name) or node.id != 'BLOCK_LIMIT':
        raise ValueError('pack default changed')
    for function in ('read', 'load', 'write', 'referenced'):
        node = default(texts[IO], function, 'limit')
        if not isinstance(node, ast.Name) or node.id != 'MEMBER_LIMIT':
            raise ValueError('member default changed')
    node = default(texts[IO], 'inventory', 'total_limit')
    if not isinstance(node, ast.Name) or node.id != 'TOTAL_LIMIT':
        raise ValueError('inventory default changed')
    contracts = (
        (SOURCE, 'pack', '8 <= chunk_bytes <= BLOCK_LIMIT'),
        (SOURCE, 'pack', "write(root,'manifest.json',manifest,HEADER_LIMIT)"),
        (SOURCE, 'Source.tensor_shape', "math.prod(shape) > TENSOR_ELEMENTS"),
        (SOURCE, 'Source.__init__', 'inventory(root,self.files,SOURCE_TOTAL_LIMIT)'),
        (SOURCE, 'Source.__init__', "load(root,'manifest.json',HEADER_LIMIT)"),
        (SOURCE, 'Source.__init__', "2 <= r['experts'] <= 64"),
        (SOURCE, 'Source.__init__', "2 <= r['classes'] <= 1024"),
        (SUPERVISOR, 'supervise', '0 < budget <= 300'),
        (SUPERVISOR, 'supervise', 'uuid.uuid4().hex'),
        (SUPERVISOR, 'supervise', 'rss_limit <= 0'),
        (SUPERVISOR, 'receive', "load(tree/'manifest.json',limit=HEADER_LIMIT)"),
        (SUPERVISOR, 'receive', "load(tree/'source/manifest.json',limit=HEADER_LIMIT)"),
        (SUPERVISOR, 'receive', "load(root/'check.stdout',checker_stdout_sha256,limit=8*2**20)"),
        (SUPERVISOR, 'receive', "load(root/'built.json',limit=2**20)"),
        (SUPERVISOR, 'receive', "load(root/'model_intake.json',limit=2**20)"),
        (SUPERVISOR, 'supervise', "load(root/'accepted.json', limit=8*2**20)"),
        ('scoped_source/factored_worker.py', 'run', "pack(doc,tree/'source',lambda:tick(deadline),protocol()['source_chunk_bytes'])"),
        ('scoped_source/factored_build.py', 'build', "write(root,'manifest.json',manifest,HEADER_LIMIT)"),
        ('scoped_source/factored_check.py', 'check', "load(root,'manifest.json',HEADER_LIMIT)"),
        ('scoped_source/factored_check.py', 'check', 'inventory(root,expected)'),
        ('scoped_source/factored_portable.py', 'publish', 'len(raw)>ENVELOPE_LIMIT'),
        ('scoped_source/factored_portable.py', 'publish', 'total+len(raw)>TOTAL_LIMIT'),
        ('scoped_source/factored_portable.py', 'publish', '0<size<=MEMBER_LIMIT'),
        ('scoped_source/factored_verify.py', 'envelope', "member(root,'manifest.json',ENVELOPE_LIMIT)"),
        ('scoped_source/factored_verify.py', 'envelope', 'total>TOTAL_LIMIT'),
        ('scoped_source/factored_verify.py', 'envelope', "0<ref['bytes']<=MEMBER_LIMIT"),
        ('act/back_end/moe/factory.py', '_mlp', 'nn.Flatten(start_dim=1)'),
        ('act/back_end/moe/factory.py', '_mlp', 'layers.append(nn.Linear(in_features,out_features))'),
        ('act/back_end/moe/factory.py', '_mlp', 'index+1<len(widths)-1'),
        ('act/back_end/moe/factory.py', '_mlp', 'layers.append(nn.ReLU())'),
    )
    for file, function, expression in contracts:
        require_expression(texts[file], function, expression)
    code = [n.value for n in ast.parse(texts[VERIFY]).body if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'CODE' for t in n.targets)]
    if len(code) != 1:
        raise ValueError('portable code inventory')
    code = ast.literal_eval(code[0])
    if type(code) is not tuple or len(set(code)) != len(code) or any(n not in raw for n in code):
        raise ValueError('unbound/duplicate portable code')
    return {'core_bytes': core, 'source': source, 'portable_bytes': portable,
        'existing_pack_default_bytes': core['BLOCK_LIMIT'], 'control_chunk_bytes': cfg['source_chunk_bytes'],
        'supervisor': {'maximum_seconds': cfg['maximum_budget_seconds'],
                       'default_rss_bytes': integer(default(texts[SUPERVISOR], 'supervise', 'rss_limit')),
                       'rss_default_is_not_a_global_hard_maximum': True,
                       'built_and_capture_receipt_bytes': 2**20, 'stdout_and_accepted_bytes': 8*2**20,
                       'ordinary_synthetic_seconds': cfg['ordinary_control_seconds'],
                       'fixed_synthetic_cases_only': len(cfg['cases']), 'real_intake_supported_by_protocol': False},
        'dimension_contract': {'experts': [2, 64], 'classes': [2, 1024]},
        'checked_callsite_contracts': len(contracts)}, code


def shapes_from_recipe(cfg):
    if (cfg['dataset'] != 'CIFAR10' or cfg['gate'] != 'selected_softmax' or cfg['top_k'] != 2
            or type(cfg['num_experts']) is not int or not 2 <= cfg['num_experts'] <= 64):
        raise ValueError('registered family changed')
    inputs, classes, experts = 3072, 10, cfg['num_experts']
    rw = [inputs, *cfg['router_hidden'], experts]; ew = [inputs, *cfg['expert_hidden'], classes]
    if any(type(v) is not int or not 0 < v <= 3072 for v in rw+ew):
        raise ValueError('unsupported metadata dimensions')
    shapes = {'@center': [1, 3, 32, 32]}
    nodes = [f'input/{i}' for i in range(inputs)]
    for prefix, graph, widths in [('router', 'router', rw)]+[
            (f'experts.{i}', f'expert{i}', ew) for i in range(experts)]:
        for j, (a, b) in enumerate(zip(widths, widths[1:])):
            layer = 2*j+1
            shapes[f'{prefix}.{layer}.weight'] = [b, a]; shapes[f'{prefix}.{layer}.bias'] = [b]
            nodes.extend(f'{graph}/layer/{layer}/value/{i}' for i in range(b))
            if j < len(widths)-2:
                nodes.extend(f'{graph}/layer/{layer+1}/value/{i}' for i in range(b))
    r_aff, e_aff = sum(rw[1:]), sum(ew[1:])
    r_relu, e_relu = sum(rw[1:-1]), sum(ew[1:-1])
    affine_entries = sum(a*b for a,b in zip(rw,rw[1:]))+r_aff+2*(sum(a*b for a,b in zip(ew,ew[1:]))+e_aff)
    return shapes, nodes, {'input_dimensions': inputs, 'experts': experts, 'classes': classes,
        'router_widths': rw, 'expert_widths': ew, 'all_unordered_pairs': comb(experts, 2),
        'properties_per_pair': classes-1, 'duties': comb(experts,2)*(classes-1),
        'maximum_endpoint_LPs': 2*comb(experts,2)*(classes-1),
        'source_nodes': len(nodes), 'pair_variables': inputs+r_aff+r_relu+2*(e_aff+e_relu),
        'pair_affine_equalities': r_aff+2*e_aff, 'pair_relu_nodes': r_relu+2*e_relu,
        'pair_guard_rows': 2*(experts-2), 'pair_affine_entries_if_dense': affine_entries,
        'pair_relu_inequalities_if_all_unstable': 3*(r_relu+2*e_relu),
        'pair_relu_entries_if_all_unstable': 5*(r_relu+2*e_relu),
        'scope': 'recipe-derived counts; dense affine/all-unstable counts are scenarios, not actual nnz or memory'}


def chunk_summary(shapes, size, limits):
    if type(size) is not int or not 8 <= size <= limits['core_bytes']['BLOCK_LIMIT'] or size % 8:
        raise ValueError('unsupported aligned chunk size')
    counts = [(8*prod(shape)+size-1)//size for shape in shapes.values()]
    total = sum(8*prod(shape) for shape in shapes.values())
    # Every file name is at least this long; all counts/offsets use at least 1 digit.
    ref_minimum = len(compact({'file':'t/00000-00000.bin','bytes':8,'offset':0,'sha256':'0'*64}))
    lower = sum(counts)*ref_minimum
    tensor_ok = max(map(prod, shapes.values())) <= limits['source']['TENSOR_ELEMENTS']
    return {'chunk_bytes': size, 'chunks': sum(counts), 'raw_tensor_bytes_with_center': total,
        'largest_tensor_elements': max(map(prod, shapes.values())), 'tensor_element_necessary_test_passed': tensor_ok,
        'minimum_bytes_per_chunk_reference': ref_minimum, 'source_header_ref_bytes_lower_bound': lower,
        'source_header_definitely_rejected': lower > limits['core_bytes']['HEADER_LIMIT'],
        'source_raw_bytes_definitely_rejected': total > limits['source']['SOURCE_TOTAL_LIMIT'],
        'actual_source_total_bytes': None, 'actual_peak_rss_bytes': None, 'actual_request_seconds': None,
        'full_path_admitted': False}


def manifest_structure(shapes, nodes, pair_count, size, limits, code):
    """Only small default-policy metadata; never allocate the 870k-ref control header."""
    if chunk_summary(shapes, size, limits)['chunks'] > 10000:
        raise ValueError('metadata construction cap; use arithmetic rejection')
    descriptors = []; chunk_files = []
    for i, (name, shape) in enumerate(sorted(shapes.items())):
        raw = 8*prod(shape); chunks = []
        for j, offset in enumerate(range(0, raw, size)):
            file = f't/{i:05d}-{j:05d}.bin'; chunk_files.append('proof/source/'+file)
            chunks.append({'file':file,'bytes':min(size,raw-offset),'sha256':'0'*64,'offset':offset})
        descriptors.append({'name':name,'dtype':'torch.float64','shape':shape,'byte_order':'little','chunks':chunks})
    members = {'proof/manifest.json', 'proof/source/manifest.json', 'verify.py'} | {'code/'+n for n in code}
    members.update(chunk_files)
    members.update(f'proof/b/{i:06d}.json' for i in range(len(nodes)))
    members.update(f'proof/p/{i:06d}.json' for i in range(pair_count))
    refs = [{'name':name,'file':f'b/{i:06d}.json','bytes':limits['core_bytes']['MEMBER_LIMIT'],
             'sha256':'0'*64} for i,name in enumerate(nodes)]
    envelope = {'schema':'HF_PORTABLE_V1','source_manifest_sha256':'0'*64,'proof_manifest_sha256':'0'*64,
        'mode':'endpoints','invocation':'0'*32,'files':{
            n:{'bytes':limits['portable_bytes']['MEMBER_LIMIT'],'sha256':'0'*64} for n in sorted(members)}}
    size_upper = len(compact(envelope))
    return {'tensor_descriptor_list_bytes':len(compact(descriptors)),
        'descriptor_list_is_not_entire_source_manifest':True,
        'source_node_reference_list_bytes_upper_bound':len(compact(refs)),
        'node_references_are_not_entire_inner_proof_manifest':True,
        'portable_members_excluding_outer_manifest':len(members),
        'portable_envelope_bytes_upper_bound':size_upper,
        'portable_envelope_necessary_test_passed':size_upper <= limits['portable_bytes']['ENVELOPE_LIMIT'],
        'assumptions':['current fixed names/CODE', '32-hex supervisor invocation',
                       'members must individually satisfy existing member limit', 'canonical compact JSON'],
        'actual_source_header_bytes':None,'actual_inner_proof_header_bytes':None,
        'actual_node_member_bytes':None,'actual_pair_member_bytes':None,'actual_bundle_bytes':None}


# These are observations of source lifetimes, not additive RAM estimates.
RESIDENCY = (
    ('capture', 'scoped_source/capture.py', 'capture',
     'Live model plus all base64 tensors; per tensor raw/encoding buffers and float/Fraction lists. Contiguous detach need not copy.'),
    ('capture_validation', 'scoped_source/graph.py', 'validate',
     'Whole declaration and live model; parameter identity decoding plus operator() full affine row dictionaries.'),
    ('pack', SOURCE, 'pack',
     'Whole declaration plus metadata, current decoded tensor and chunk slice; worker deletes model before pack and doc after pack.'),
    ('tensor_parser', SOURCE, 'Source.tensor',
     'Chunk bytes are streamed, but Fraction numbers accumulates a whole tensor; nodes holds a whole affine layer.'),
    ('pair_assembly', 'scoped_source/factored_ir.py', 'assemble',
     'All named rows/bounds plus CSR containers during lp_record; immutable strings may be shared, not necessarily duplicated bytes.'),
    ('proposal_copy', 'scoped_source/factored_build.py', 'build',
     'Global node bounds/refs, one pair base, objectives, deepcopy(lp) containers and returned candidate; immutable values may be shared.'),
    ('mccormick', 'scoped_source/endpoint_source_build.py', 'mc_lp',
     'Base CSR plus exact row dictionaries plus reconstructed augmented constraints/CSR.'),
    ('native_adapter', 'act/back_end/solver/sparse_lp_certificate.py', 'propose',
     'list(rows()) materializes all rational rows before float CSR. HiGHS/SciPy internal workspace unknown.'),
    ('exact_candidate_evaluation', 'act/back_end/solver/sparse_lp_certificate.py', 'evaluate',
     'Row iteration with residual/dual vectors; propose calls evaluation then check evaluates again.'),
    ('independent_bound_check', 'scoped_source/sparse_check.py', 'check_bound',
     'csr() returns full rational row dictionaries per endpoint. Previous A rows can coexist while the full E row dictionaries are constructed.'),
    ('identity_serialization', 'source_enclosure/format.py', 'identity',
     'compact() materializes canonical JSON string and bytes before hashing; not streaming LP hashing.'),
    ('portable_publication', 'scoped_source/factored_portable.py', 'publish',
     'Full file descriptor map, sorted paths and envelope JSON; save serializes envelope again.'),
    ('portable_verification', VERIFY, 'verify',
     'Envelope byte hashing streams twice, but descriptors and inner exact check remain resident.'),
    ('reception', SUPERVISOR, 'receive',
     'Source/proof headers and full bounded checker stdout; per-pair records, without rebuilding LPs.'),
)


def derive(root=ROOT):
    # Only source and registered JSON metadata. Checkpoint paths are never opened.
    raw = {name:(root/name).read_bytes() for name in FILES}
    limits, code = policy(raw)
    shapes, nodes, counts = shapes_from_recipe(json.loads(raw[TRAINING])['training'])
    models = json.loads(raw[SELECTION])['models']
    if set(models) != {'seed0','seed1','seed2'}:
        raise ValueError('registered model set changed')
    parameters = sum(prod(shape) for name,shape in shapes.items() if name != '@center')
    for m in models.values():
        if (m['dataset'] != 'CIFAR10' or m['model_state']['parameter_count'] != parameters
                or m['model_state']['tensor_count'] != len(shapes)-1):
            raise ValueError('model metadata versus recipe mismatch')
    control = chunk_summary(shapes, limits['control_chunk_bytes'], limits)
    existing_default = chunk_summary(shapes, limits['existing_pack_default_bytes'], limits)
    structure = manifest_structure(shapes, nodes, counts['all_unordered_pairs'],
                                   limits['existing_pack_default_bytes'], limits, code)
    residency = []
    for phase,file,function,description in RESIDENCY:
        node = definition(raw[file].decode(), function)
        residency.append({'phase':phase,'file':file,'function':function,
            'first_line':node.lineno,'last_line':node.end_lineno,'observation':description,'measured_peak_bytes':None})
    return {'schema':'H2_FACTORED_STATIC_CAPACITY_V1','status':'NO_REAL_FREEZE_CAPACITY_NOT_ADMITTED',
        'source_bindings':{n:hashlib.sha256(v).hexdigest() for n,v in raw.items()},
        'scope':'registered dense recipe only; seed0 count match is not native topology revalidation; no source values read',
        'models':{name:{'checkpoint_sha256':m['checkpoint_sha256'],'model_state':m['model_state'],
                       'checkpoint_opened':False,'native_topology_revalidated':False} for name,m in models.items()},
        'parameters':parameters,'parameter_tensors':len(shapes)-1,'tensor_shapes_including_center':shapes,
        'policy':limits,'recipe_counts':counts,'frozen_64_byte_control_policy':control,
        'existing_1_mib_default_policy':existing_default,'default_manifest_structure':structure,
        'residency_by_stage':residency,
        'remaining_unknowns':['actual rational digit lengths, weights nnz and stable/unstable ReLU rows',
            'complete source and proof headers, node/pair members, total bundle bytes',
            'process Python heap and native workspace peak; sampled RSS is not an upper bound',
            'full capture, construction, proposal, checking, reception and publication within 300 seconds'],
        'phase_accounting':'sequential producer/check/receive peaks are not summed; include parent plus active children; serialized size is not heap size',
        'next_gate':'No real freeze and no further static-ledger expansion. Separately test row-wise consumption of native CSR validation and independent exact bound checks, preserving every structural check, identity and residual. This removes a known temporary full-matrix copy, not a measured dominant bottleneck. Differential/mutation/deadline controls precede a bounded full-path capacity control; existing limits and the 64-byte synthetic protocol stay unchanged.',
        'conv_status':'NOT_ADMITTED_UNSUPPORTED_CONV_POOL_CAPTURE',
        'new_model_loads':0,'new_input_selections':0,'new_solves':0,'real_requests_started':0,
        'mathematical_certificate':False,'performance_measurement':False,'full_path_admitted':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--output', type=Path); group.add_argument('--check', type=Path)
    args = parser.parse_args(); result = derive()
    if args.output:
        if not args.output.resolve().is_relative_to(Path('/data1/Kane/MOE')):
            raise ValueError('output outside project')
        with args.output.open('x') as f:
            json.dump(result, f, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    elif json.loads(args.check.read_text()) != result:
        raise ValueError('capacity archive differs')
    print(json.dumps({'status':result['status'],'control_header_rejected':result['frozen_64_byte_control_policy']['source_header_definitely_rejected'],
        'default_envelope_test':result['default_manifest_structure']['portable_envelope_necessary_test_passed'],
        'full_path_admitted':False,'new_solves':0}))


if __name__ == '__main__':
    main()
