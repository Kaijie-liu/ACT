"""Static stopping decision for the unchanged shared-source resource contract.

Only saved dimension inventories, hashes and Python syntax are read. No source
object, candidate, numerical package, subprocess worker or device is created.
Lower bounds prove current-contract refusal; oversized uppers do not prove
actual archive overflow. Neither is a successful-completion resource contract.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]
PARENT = 'docs/hz_shared_source_cost_20261015.json'
SHAPES = ('docs/hz_full_resource_ledger_20261014.json',
          'docs/hz_pair_scoped_capacity_20261015.json')
BASE = 'a210f8224ad0c0472fe2151c881b8b5e3b3a2fe2'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def load(name):
    return json.loads((ROOT / name).read_bytes())


def validate_parent(parent):
    require(parent['schema'] == 'SHARED_SOURCE_COST_V1'
            and parent['decision'] == 'CONTROLS_PASS_FULL_SHAPE_REFUSED', 'parent decision')
    for name, expected in parent['anchors'].items():
        require(sha(name) == expected, 'bound implementation drift: ' + name)


def assignment(path, name):
    for node in ast.parse((ROOT / path).read_bytes()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name
                                                for t in node.targets):
            return node.value
    raise ValueError('missing literal: ' + name)


def integer(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return node.value
    if isinstance(node, ast.BinOp):
        a, b = integer(node.left), integer(node.right)
        if isinstance(node.op, ast.Add): return a + b
        if isinstance(node.op, ast.Sub): return a - b
        if isinstance(node.op, ast.Mult): return a * b
        if isinstance(node.op, ast.Pow) and 0 <= b <= 32: return a ** b
    raise ValueError('not a bounded integer expression')


def structure(path, name):
    node = assignment(path, name)
    require(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name) and node.func.value.id == 'struct'
            and node.func.attr == 'Struct' and len(node.args) == 1, 'literal Struct')
    return struct.calcsize(ast.literal_eval(node.args[0]))


def policy():
    outer = assignment('scoped_source/hz_request_store.py', 'POLICY')
    require(isinstance(outer, ast.Call) and isinstance(outer.func, ast.Name)
            and outer.func.id == 'MappingProxyType' and len(outer.args) == 1, 'policy wrapper')
    inner = outer.args[0]
    require(isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
            and inner.func.id == 'dict' and not inner.args, 'policy literal')
    return {k.arg: integer(k.value) for k in inner.keywords}


def positive(*values):
    require(all(type(v) is int and v > 0 for v in values), 'positive integer counts')


def classify(cap, *, lower=None, upper=None):
    positive(cap)
    for value in (lower, upper):
        require(value is None or type(value) is int and value >= 0, 'bound domain')
    require(lower is None or upper is None or lower <= upper, 'bound order')
    if lower is not None and lower > cap: return 'CURRENT_CONTRACT_REFUSAL'
    if upper is not None and upper <= cap: return 'UPPER_FITS_THIS_COMPONENT_ONLY'
    if upper is not None: return 'UPPER_EXCEEDS_CAP_ACTUAL_UNKNOWN'
    return 'BOUND_OPEN'


def semantic_floor(v, endpoints):
    positive(v['parameter_values'], v['affine_outputs'], v['inputs'], endpoints)
    weights = v['parameter_values'] - v['affine_outputs']
    require(weights == v['first_weight_positions'] + v['later_weight_positions']
            and weights > 0, 'weights exclude biases')
    # Two fresh equation checks. Every position is visited, including zero.
    source = 2 * weights
    # Per present endpoint: merge + proposed norm + zero norm; each has one
    # coefficient visit and one exact add per input factor. All map accesses,
    # private factors, pullbacks, scale-group sums and other steps are free here.
    consumers = 2 * 6 * v['inputs'] * endpoints
    return dict(source_checks=2,all_affine_weight_positions=weights,
                source_coefficient_floor=source,complete_consumers=2,
                endpoints=endpoints,consumer_access_free_floor=consumers,
                joint_floor=source + consumers,
                assumptions=['two fresh full source equation checks',
                    'all endpoint candidate pairs present',
                    'unchanged full-factor merge and both norms',
                    'unchanged logical Work units'],
                not_counted=['producer','Snapshot','source parsing/norms',
                    'buffer/map access','private factors','pullbacks','candidate generation',
                    'dual rows','serialization','IO/hash/reception/cleanup'])


def frame_sizes(rows, widths, profile, stored, prefix, record):
    positive(rows, stored, prefix, record)
    require(type(profile) is int and profile in (0, 1), 'candidate profile')
    require(type(widths) is list and widths and all(type(k) is int and 1 <= k <= 8
                                                   for k in widths), 'candidate widths')
    scalar = 8 if profile == 0 else stored
    stride = record + 2 * stored + 2 * rows * scalar
    return [prefix + k * stride for k in widths]


def analyze():
    parent = load(PARENT); validate_parent(parent)
    v, second = (load(name)['inventory'] for name in SHAPES)
    require(v == second, 'two frozen shape inventories agree')
    require((v['inputs'], v['experts'], v['classes'], v['property_duties'],
             v['endpoints_min'], v['endpoints_upper']) == (3072,8,10,252,252,504),
            'registered conservative full topology')
    p = policy(); cp = 'scoped_source/direct_candidates.py'
    prefix, record = structure(cp, 'PREFIX'), structure(cp, 'RECORD')
    stored = structure('scoped_source/bounded_runtime.py', 'HEADER') + 2*((p['stored_bits']+7)//8)
    tile_cap, stream_cap = (integer(assignment(cp,k)) for k in ('MAX_TILE','MAX_TOTAL'))
    stream = {}
    widths = [8,8,2]
    require(v['pairs']*sum(widths) == v['endpoints_upper']
            and v['pairs']*len(widths) == v['pair_local_tiles_upper'], 'complete candidate roster')
    for name, profile in (('binary64',0),('general_rational',1)):
        sizes = frame_sizes(v['pair_dual_rows_upper'],widths,profile,stored,prefix,record)
        stream[name] = dict(per_pair_tile_upper=sizes,tiles=v['pair_local_tiles_upper'],
            stream_upper=v['pairs']*sum(sizes),largest_tile_upper=max(sizes),
            one_endpoint_upper=frame_sizes(v['pair_dual_rows_upper'],[1],profile,stored,prefix,record)[0],
            tile_assessment=classify(tile_cap,upper=max(sizes)),
            stream_assessment=classify(stream_cap,upper=v['pairs']*sum(sizes)),
            actual_bytes=None,upper_excess_proves_actual_overflow=False)
    fpath='scoped_source/direct_source_format.py'
    fields={k:structure(fpath,k) for k in ('PREFIX','NODE','ENTRY','ROOT','INDEX','SCALAR')}
    # Input basis, at most two affine compensation tails, and the conservative
    # twelve local entries over the four ReLU rows. Not an actual-nnz subtraction.
    explicit=v['inputs']+2*v['affine_outputs']+12*v['relu_units']
    require(explicit==v['local_entries_upper']-v['first_weight_positions'], 'explicit local envelope')
    parts=dict(prefix=fields['PREFIX'],header=p['header_bytes'],
        nodes=fields['NODE']*v['nodes_upper'],
        explicit_locals=fields['ENTRY']*explicit,
        references=fields['ENTRY']*v['references_upper'],
        unique_roots=fields['ROOT']*v['nodes_upper'],
        stored_memberships=fields['INDEX']*v['roots_upper'],
        scales=fields['SCALAR']*v['factors_upper'])
    payload=sum(parts.values()); page=integer(assignment('scoped_source/shared_format.py','PAGE'))
    pages=(payload+page-1)//page
    anchors=dict(parent['anchors'])
    for name in (PARENT,'scripts/assess_shared_source_decision.py','scripts/test_shared_source_decision.py'):
        anchors[name]=sha(name)
    source_min=semantic_floor(v,v['endpoints_min'])
    source_max=semantic_floor(v,v['endpoints_upper'])
    return dict(schema='SHARED_SOURCE_CAPACITY_DECISION_V1',implementation_commit=BASE,anchors=anchors,
        decision='STOP_REPRESENTATION_ONLY_FUSION_AS_8M_ADMISSION_STRATEGY',inventory=v,
        semantic_floor_min_endpoint_roster=source_min,semantic_floor_max_endpoint_roster=source_max,
        source_floor_assessment=classify(p['visits'],lower=source_min['source_coefficient_floor']),
        membership=dict(mandatory=v['mandatory_roots'],pair_output_aliases=v['pair_output_roots'],
            cap=p['rows'],assessment=classify(p['rows'],lower=v['mandatory_roots']),
            changing_expanded_to_unique_count_is_new_resource_policy=True),
        candidate_layout=dict(prefix=prefix,record=record,stored_rational=stored,
            both_roles_full_vectors=True,widths=widths,tile_cap=tile_cap,stream_cap=stream_cap),
        candidate_stream=stream,
        forest_envelope=dict(parts=parts,explicit_local_entries_upper=explicit,
            payload_upper=payload,pages_upper=pages,source_tensor_bytes=v['raw_binary64_tensor_bytes'],
            source_plus_forest_upper=v['raw_binary64_tensor_bytes']+payload,
            applies_only_if_header_and_scalar_encoding_admitted=True,
            includes=['nodes','explicit tails','references','unique roots','ordered sequences','scales'],
            excludes=['page index','source header','candidate stream','code','manifests',
                      'results','logs','receipts','Python objects','arithmetic temporaries'],
            total_archive_upper=None,full_rss_upper=None),
        other_structural_upper_only=dict(
            logical_locals=classify(p['nnz'],upper=v['local_entries_upper']),
            references=classify(65536,upper=v['references_upper']),
            active_global_factors=classify(p['active'],upper=v['factors_upper'])),
        full_work_upper=None,full_storage_upper=None,full_rss_upper=None,new_quota=None,
        actual_counts=None,full_capacity_admitted=False,runtime_changed=False,
        source_builds=0,candidate_builds=0,model_runs=0,gpu_runs=0,solver_calls=0,
        unchanged=dict(work=p['visits'],expanded_memberships=p['rows'],seconds=300,
            stored_bits=p['stored_bits'],work_bits=p['work_bits'],acceptance='strict > 1/10000000'),
        next_decision='Versioned complete resource/rechecking contract before more runtime work; '
            'no implicit change of logical counting, no quota from these partial bounds')


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--check-report',type=Path)
    args=ap.parse_args(); result=analyze()
    if args.check_report:
        require(json.loads(args.check_report.read_bytes())==result, 'static decision drift')
        print('PASS: current-contract refusal; no execution or quota change')
    else:
        print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__': main()
