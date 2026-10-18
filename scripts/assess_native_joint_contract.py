"""Read-only static phase/binding ledger; never imports the native pipeline.

Recomputes small integer/schema facts only. This is not a memory experiment,
an executable resource profile, or a proof of full-shape completion.
"""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'docs/hz_native_joint_contract_20261018.json'
PREVIOUS = 'docs/hz_native_generation_contract_20261018.json'
BASE = '8caba399a5494108c5bc606c7966244a936a3cee'
EXTRA = ('scoped_source/shared_source.py', 'scoped_source/native_source.py',
    'scoped_source/native_protocol.py', 'scoped_source/native_supervised.py',
    'scoped_source/native_production.py', 'scoped_source/native_package.py',
    'scoped_source/check_shared_source.py', 'scoped_source/direct_candidates.py',
    'scoped_source/tensor_rows.py', 'scoped_source/result_stream.py',
    'scoped_source/wire_archive.py', 'scoped_proof/owned_bounded.py',
    'scripts/assess_native_joint_contract.py')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def digits(bits):
    return (bits + 29) // 30


def capacity(p):
    # Conditional returned-object envelope for the enumerated fixed call chain.
    # p includes operand/intermediate/token provenance, NOT just final bit_length.
    return 2 * digits(p + 2) + 4


def long_bytes(c):
    return max(32, 24 + 4 * max(1, c))


def fraction_bytes(p):
    return 48 + 2 * long_bytes(capacity(p))


def tuple_envelope(n):
    return 2 * (40 + 8 * (n + n // 4 + 16))


def analyze():
    require(sys.implementation.name == 'cpython' and sys.version_info[:3] == (3, 12, 12)
            and sys.maxsize == 2**63-1 and sys.int_info.bits_per_digit == 30
            and sys.int_info.sizeof_digit == 4,
            'fixed inspected interpreter only')
    previous = json.loads((ROOT / PREVIOUS).read_bytes())
    for path, digest in previous['anchors'].items():
        require(sha(path) == digest, 'inherited implementation drift: ' + path)
    require(previous['policy']['work'] == 8000000 and
            previous['policy']['artifact_bytes'] == 2**27, 'original caps')
    env = previous['python_large_object_components']['environment_anchors']
    for path, digest in env.items():
        require(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest,
                'inherited environment drift: ' + path)
    frac = Path(sys.prefix) / 'lib/python3.12/fractions.py'
    tree = ast.parse(frac.read_text())
    h = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_hash_algorithm')
    decor = h.decorator_list[0]
    require(isinstance(decor, ast.Call) and isinstance(decor.func, ast.Attribute)
            and decor.func.attr == 'lru_cache' and len(decor.keywords) == 1
            and decor.keywords[0].arg == 'maxsize'
            and ast.dump(decor.keywords[0].value) == ast.dump(ast.parse('1 << 14', mode='eval').body),
            'Fraction hash cache identity')
    extra_env = ('include/python3.12/internal/pycore_object.h',
                 'include/python3.12/cpython/longintrepr.h',
                 'lib/python3.12/_pylong.py')
    env = dict(env, **{str(Path(sys.prefix) / p): hashlib.sha256(
        (Path(sys.prefix) / p).read_bytes()).hexdigest() for p in extra_env})
    B = 8192
    fragments = {'two_checked_operands': 2*B+1, 'node_norm_three_operands': 3*B+1,
        'input_midpoint': 2*17460+2, 'relu_four_error_terms': 4*(B+1075+1)+3,
        'result_decimal_token': 4*4936}
    integer = {name: dict(provenance_bits=p, returned_digits=capacity(p),
        returned_long_requested_bytes=long_bytes(capacity(p)),
        fraction_with_two_returned_longs=fraction_bytes(p)) for name, p in fragments.items()}
    require(integer['relu_four_error_terms'] == dict(provenance_bits=37075,
        returned_digits=2476, returned_long_requested_bytes=9928,
        fraction_with_two_returned_longs=19904), 'reviewed arithmetic')
    n = 3072
    inputs = tuple_envelope(n) + n*tuple_envelope(2) + 2*n*fraction_bytes(17460)
    metadata = 7*2**20 + 6*65536
    logs = 2**20 + 3*65536
    # Read-only structural anchors: no arbitrary profile, forest or source objects.
    snippets = {
        'scoped_source/shared_source.py': 'F(lo.numerator,lo.denominator)',
        'scoped_source/native_source.py': "copy = TensorRows(root/'source'",
        'scoped_source/native_protocol.py': 'raw = compact(obj); require(len(raw) <= cap',
        'scoped_source/native_supervised.py': 'message=str(exc)[:400]',
        'scoped_proof/owned_bounded.py': "for path in Path('/proc').iterdir():",
        'scoped_source/wire_archive.py': 'len(self.attempted)<POLICY[\'members\']',
    }
    for path, snippet in snippets.items():
        require(snippet in (ROOT/path).read_text(), 'reviewed site drift: ' + path)
    phases = {
        'intake_second_reader': ['first TensorRows resident', 'second TensorRows initialization',
            'pinned raw manifest', 'refs/inodes/archive/census', 'two decoded requests at equality'],
        'source_build': ['one Snapshot packed bank', 'builder forest/norms/scales',
            'up to two affine rows during advance, including list/tuple reference overlap',
            'pending layer/ReLU lists', 'duplicate input endpoints only in input stage'],
        'source_check': ['published packet', 'one new Snapshot packed bank',
            'checker forest and local/global norms', 'visible factor sets',
            'up to two source rows during advance', 'duplicate input endpoints during initial input check'],
        'property_plan': ['checked forest plus retained first-layer packed rows',
            'allocated RootTable', 'plan/tasks', 'generic pullback temporaries'],
        'native_construct': ['checked forest/plan/RootTable', 'arm-specific constructor objects',
            'arm-specific tensor payload', 'native/library scratch (hard-fail boundary)'],
        'native_exact_consume': ['checked forest/plan', 'native operator and best dual',
            'candidate/frame ownership', 'pair-scoped exact arena and temporaries', 'endpoint receipts/values'],
        'production_seal': ['checked forest/plan/receipts', 'new TensorRows/source recheck',
            'archive/result metadata', 'Fraction hash-cache integer references'],
        'replay_seal': ['checked forest/plan/receipts', 'new TensorRows/source recheck',
            'last raw frame', 'archive/result metadata', 'Fraction hash-cache integer references'],
        'reception': ['package/member metadata', 'result page aggregation',
            'census/suffix records; no fourth source lowering'],
    }
    bindings = {name: dict(value=None, executable=False, need=need) for name, need in (
        ('frozen_metadata', 'exact profile/root/id/version/margin bytes and identities; no request computation moved offline'),
        ('source_layout', 'raw manifest hash/length, tensor/chunk roster, rank and trust encoding length'),
        ('partial_roster', 'success and partial file/directory/temporary roster plus ownership; not success schema only'),
        ('observations', 'pre-construction bounds for live/zombie/proc/error fields, or explicit incomplete rejection'),
        ('joint_heap', 'finish self-owned object-count joins; returned-object envelopes are not a whole heap'),
    )}
    return dict(schema='NATIVE_JOINT_STATIC_BOUNDARY_V1', based_on=BASE,
        anchors={p:sha(p) for p in sorted(set(previous['anchors']) | set(EXTRA) | {PREVIOUS})},
        environment_anchors=env,
        scope='static requested objects, phase joins and missing bindings; not allocation/RSS measurement or admission',
        integer=dict(fragments=integer, formula='D(b)=ceil(b/30); C(p)=2D(p+2)+4; J(c)=max(32,24+4max(1,c)); F(p)=48+2J(C(p))',
            gate_bits=B, universal_python_int_bound=False, internal_recursive_scratch_included=False,
            external_preallocated_integer_aliases_in_scope=False),
        source=dict(tensors=53, layers=52, chunks='K', shape_tuple_elements='rank+78',
            TensorRows_attributes=13, bounded_raw_manifest_bytes=4*2**20,
            input_tuple_requested_component=inputs, duplicate_input_stage_component=2*inputs,
            source_row_max_coefficients=3072, source_row_with_bias_fraction_count=3073,
            advancing_row_fraction_count_upper=2*(3072+1),
            tree_accounting='sum LP(array counts)+DP(object counts)+U(string codepoints)+Adec(integer token lengths)+24*float count',
            decoder_and_encoder_internal_workspace_included=False),
        fraction_hash_cache=dict(max_entries=16384, normal_generated_entry_upper=min(16384,n+4),
            normal_integer_digit_upper=37, normal_integer_bytes_each=long_bytes(37),
            normal_key_integer_payload_upper=min(16384,n+4)*2*long_bytes(37),
            retained='integer references, not Fraction objects; survives forest/session cleanup',
            malformed_or_prewarmed_scope='16384 entry count alone is not an integer payload bound; separate provenance/state binding'),
        phases=phases,
        joint_formula='max_t(caller(t)+controller(t)+active_actor(t)); actors sequential, caller/controller coexist',
        current_rejection_and_retention_limits=dict(retained_metadata_and_logs=metadata,
            log_subset=logs, log_subset_already_in_metadata=True,
            outside_caller_receipt_included=False,
            archive_attempts=4096, relative_path_characters=200,
            own_directory_bound_per_archive=4096*100,
            source_inventory_levels=2, quota_increase=False),
        failure=dict(retained_error_message_400_chars_json_bytes=2+12*400,
            retained_error_message_500_chars_json_bytes=2+12*500,
            str_before_slice_unbounded_by_retained_cap=True,
            live_roster_must_be_counted=True, zombie_roster_must_be_counted=True,
            sampled_stop_not_observed_rss_upper=True, partial_result_and_error_may_coexist=True,
            outer_caller_inherits_actor_AS=False),
        bindings=bindings,
        proposed_interface=['immutable pre-encoded frozen envelope',
            'schema-aware metadata reservation before construction/encoding',
            'bounded diagnostic and process observation; incomplete means refusal',
            'partial terminal with unchanged complete acceptance and caller cost'],
        permissions=dict(runtime_change=False,new_quota=None,seconds=300,work=8000000,
            artifact_bytes=2**27,native_calls=0,model_runs=0,gpu_runs=0,execution_approved=False),
        complete_self_owned_heap_upper=None, complete_archive_failure_upper=None,
        decision='STOP_ISOLATED_COMPONENT_CONTROLS_PROPOSE_ONE_VERSIONED_OWN_RESOURCE_INTERFACE_NO_ADMISSION')


if __name__ == '__main__':
    result = analyze()
    if sys.argv[1:] == ['--check']:
        require(result == json.loads((ROOT/REPORT).read_bytes()), 'joint-contract record drift')
        print('PASS: static phase/binding record; no production import or native execution')
    else:
        require(not sys.argv[1:], 'only --check is supported')
        print(json.dumps(result, sort_keys=True, indent=2))
