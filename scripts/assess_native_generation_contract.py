"""Static generation bounds, without production imports or source construction.

Lengths are calculated from small schema templates and integer/polynomial
arithmetic. Unknown frozen metadata and OS fields remain explicit parameters.
This is not a resource policy, execution profile, heap/RSS or time guarantee.
"""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = 'docs/hz_native_generation_contract_20261018.json'
BASE = '0f2ef5a4b49681f23fc7ddcdc042668940787933'
PREVIOUS = 'docs/hz_native_storage_lifetimes_20261018.json'
EXTRA = ('scoped_source/source_dag.py', 'scoped_source/affine_references.py',
    'scoped_source/scaled_references.py', 'scoped_source/factored_source.py',
    'scoped_source/factored_io.py', 'scoped_source/hz_request_store.py',
    'scoped_source/native_method.py', 'scoped_source/check_source_dag.py',
    'scoped_source/forest_endpoints.py', 'scripts/assess_native_generation_contract.py')


def require(ok, why):
    if not ok:
        raise ValueError(why)


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


class Length:
    """Polynomial in independently bound encoded lengths, not JSON content."""
    def __init__(self, n=0, **terms):
        self.terms = {'constant': n, **terms}
    def __add__(self, other):
        if isinstance(other, int):
            other = Length(other)
        out = dict(self.terms)
        for k, v in other.terms.items():
            out[k] = out.get(k, 0) + v
        return Length(**{k: v for k, v in out.items() if v})
    __radd__ = __add__
    def __mul__(self, n):
        require(type(n) is int, 'linear symbolic length multiplier')
        return Length(**{k: v*n for k, v in self.terms.items() if v*n})
    __rmul__ = __mul__
    def __sub__(self, other):
        return self + (other * -1 if isinstance(other, Length) else -other)
    def evaluate(self, **values):
        return sum(v if k == 'constant' else v*values[k] for k, v in self.terms.items())
    def record(self):
        return {k: v for k, v in sorted(self.terms.items()) if v}


def size(x):
    if isinstance(x, Length):
        return x
    if isinstance(x, dict):
        return sum((size(k)+1+size(v) for k, v in x.items()), Length(2+max(0, len(x)-1)))
    if isinstance(x, (tuple, list)):
        return sum((size(v) for v in x), Length(2+max(0, len(x)-1)))
    return Length(len(json.dumps(x, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()))


def scalar(x):
    return size(x).evaluate()


def native_python_large_objects():
    # These are requested/logical named buffers under the inspected CPython ABI,
    # not all objects, allocated integer capacities, allocator overhead or RSS.
    abi = dict(pointer=struct.calcsize('P'),list=sys.getsizeof([]),tuple=sys.getsizeof(()),
        float=sys.getsizeof(0.),index=sys.getsizeof(1000),fraction=sys.getsizeof(Fraction(2,3)))
    require(sys.implementation.name=='cpython' and sys.version_info[:3]==(3,12,12)
        and abi==dict(pointer=8,list=56,tuple=40,float=24,index=28,fraction=48), 'inspected ABI only')
    def pow2(n): return 1 << (max(1,n)-1).bit_length()
    def lp(n): return 56+16*((n+n//8+6)&~3 if n else 0)
    def dp(n):
        t=pow2(max(8,3*n)); w=1 if t<256 else 2 if t<65536 else 4 if t<2**32 else 8
        return 64+2*(32+w*t+24*(2*t//3))
    def sp(n): return 216+32*pow2(max(8,4*n+1))
    def tp(n): return 2*(40+8*(n+n//4+16))
    N,F,R,D=19496,32048,277280,64
    def common(X,Y):
        return (3*lp(X)+lp(N)+lp(F)+28*Y+56*N+24*(X+F)+dp(D)
            +152*D+18*R+88*R+lp(R)+lp(D)+tp(D)+64*D)
    B,S,n,s=9,2176,3072,256
    extra=(dp(B)+B*dp(s)+56*(B+S)+lp(s)+lp(n)+24*n+2*sp(s)+28*s+lp(B)+40+8*B+56*B)
    coo,block=common(6732720,6684672),common(48048,0)+extra
    require((coo,block,extra)==(749521272,41675712,729672),'named Python construct arithmetic')
    files=('include/python3.12/cpython/longintrepr.h','include/python3.12/internal/pycore_dict.h',
           'lib/python3.12/fractions.py')
    return dict(abi=abi,version=sys.version,environment_anchors={str(Path(sys.prefix)/p):
        hashlib.sha256((Path(sys.prefix)/p).read_bytes()).hexdigest() for p in files},
        named_constructor_python_coo=coo,named_constructor_python_blocks=block,
        row_block_extra=extra,scope='listed large containers/elements only; summed resize envelopes, not full heap or RSS',
        normalized_integer_size_is_not_allocation_size=True,
        full_self_owned_heap_upper=None,allocator_and_bigint_internal_workspace_upper=None,
        source_urls={name:'https://raw.githubusercontent.com/python/cpython/v3.12.12/Objects/'+name+'.c'
                     for name in ('listobject','dictobject','setobject','longobject','abstract')})


def schemas(arm):
    H = '0'*64
    D, BB, F = Length(d=1), Length(b=1), Length(24)
    w = {k: D for k in ('row_visits', 'coefficient_visits', 'arithmetic')}
    ref = lambda path: dict(file=path, bytes=Length(9), sha256=H)
    rr, br = ref('result/index.json'), ref('bundle.json')
    summary = dict(status='CHECKED_POSITIVE_CAPTURED_REAL_SOURCE', required=252,
        positive=252, required_endpoints=504, checked_endpoints=504, missing_endpoints=504,
        native_float_proof=False, checkpoint_correspondence_checked=False)
    request = dict(experts=8, classes=10, label=9)
    context = {k: H for k in ('source','forest','forest_index','contract','method','invocation','roster')}
    context['schema'] = 'SHARED_RESULT_CONTEXT_V1'
    owner = dict(live_bytes=Length(9), peak_bytes=Length(9), allocated_bytes=BB,
        copied_bytes=BB, released_bytes=BB, allocations=D, cleanup_seconds=F)
    driver = {k: F for k in ('init_seconds','prepare_seconds','encode_seconds','receive_seconds',
        'publish_seconds','check_seconds','terminal_seconds','release_seconds')}
    driver.update(publication_attempted_bytes=Length(9), acknowledged_bytes=Length(9))
    construction = dict(native_seconds=0., native_coordinates=0, packed_source_bytes=Length(9),
        forest_bytes=Length(9), pages=Length(4))
    limits = dict(address_space_bytes=8589934592, file_soft_bytes=Length(9),
        file_hard_bytes=Length(9), per_process=True, aggregate_rss_is_sampled=True, bootstrap_trusted=True)
    setup = dict(seconds=F, torch_version=Length(V=1), device='cpu', threads=1,
        environment_scope='BOUND_PRIMARY_FILES_NOT_FULL_OS_ATTESTATION')
    op = dict(algorithm='CPU_SCALED_REFERENCE_PROJECTED_SUBGRADIENT_128_V3', device='cpu',
        dtype='float64', nodes=Length(5), factors=Length(5), local_nnz=Length(7),
        references=Length(6), levels=Length(1), expanded_constraint_entries=0,
        representation='native_coo_reference')
    if arm == 'blocks':
        op.update(representation='native_source_row_blocks', local_nnz=Length(5),
            source_blocks=Length(1), source_rows=Length(4), source_values=Length(7), source_bank_payload_bytes=Length(8))
    else:
        require(arm == 'coo', 'two unchanged native representations')
    def actor(stage, output, native_setup=None):
        return dict(schema='NATIVE_ACTOR_RECEIPT_V1', stage=stage, status='COMPLETE', error=None,
            invocation=H, previous=H, initial_work=w, work=w, delta_work=w,
            begin=F, end=F, deadline=F, output=output, limits=limits, native_setup=native_setup,
            progress_bytes=Length(5), actor_seconds=F)
    prod = dict(result=summary, result_ref=rr, package=br, source=H, request=request,
        source_scope='DECLARED_BINARY64_ARTIFACT_NOT_CHECKPOINT_CORRESPONDENCE', source_acquire_seconds=F,
        construction=construction, snapshots=dict(native=dict(setup_calls=1, propose_calls=Length(2),
        close_calls=1, operator=op), driver=driver, owner=owner), phases=Length(Phi=1), initial_work=w, work=w, seconds=F)
    rep = dict(result=summary, result_ref=rr, context=context, package_sha256=H, package_bytes=Length(9),
        source=H, request=request, owner=owner, driver_cost=driver, candidate_bytes=Length(9), work=w, seconds=F)
    rec = dict(result=summary, result_ref=rr, package_sha256=H, storage_before_final_metadata=Length(C0=1),
        source=H, declared_source_only=True, checkpoint_correspondence_checked=False, native_float_proof=False)
    stages = [dict(stage=s, begin=F, end=F, seconds=F, log_bytes=Length(5), returncode=0,
        remaining_live_actor_processes=[]) for s in ('production','replay','reception')]
    ctrl = dict(schema='NATIVE_CONTROLLER_RECEIPT_V1', status='COMPLETE', invocation=H,
        last_receipt=H, result=summary, work=w, phases=stages, limits=dict(limits, file_soft_bytes=Length(7)),
        storage_before_two_terminals=Length(C1=1), started=F, end=F, deadline=F, received_all_obligations=True)
    owned = dict(schema='OWNED_BOUNDED_CPU_V1', status='COMPLETED', error=None, pid=Length(p=1),
        returncode=0, seconds=F, execution_seconds=F, cleanup_seconds=F,
        cleanup_status='LEADER_REAPED_NO_LIVE_GROUP', remaining_group=dict(live=[], zombies=Length(Z=1), rss=Length(rho=1)),
        descendant_on_leader_exit=False, exit_observed_at=F, sampled_peak_rss=Length(10),
        run_deadline=F, cleanup_deadline=F, escaped_descendants_or_driver_cleanup=False)
    term = dict(schema='NATIVE_SUPERVISED_TERMINAL_V1', invocation=H, status='COMPLETE', result=summary,
        error=None, owned_execution=owned, cumulative_work=w, work_complete=True, storage=Length(C2=1),
        storage_complete=False, storage_scope='SUPERVISED_CENSUS_PLUS_QUIESCENT_SUFFIX_BEFORE_SELF',
        started=F, observed_at=F, seconds=F, source_scope='DECLARED_SOURCE_NOT_CHECKPOINT_OR_NATIVE_FLOAT_PROOF', online_acceptance=False)
    sr = dict(bytes=Length(9), sha256=H)
    ret = dict(term, storage=Length(C3=1), storage_complete=True, terminal_ref=sr,
        storage_scope='SUPERVISED_CENSUS_PLUS_QUIESCENT_FINAL_SUFFIX', online_acceptance=True,
        caller_seconds=F, attempted_terminal_ref=sr, overrun_seconds=F)
    caller = dict(id=Length(Iid=1), kind='normal', profile=H, result=ret, error=None,
        seconds_before_receipt=F, request_started=F)
    result = dict(production=size(actor('production', prod, setup)), replay=size(actor('replay', rep)),
        reception=size(actor('reception', rec)), controller=size(ctrl), terminal=size(term),
        returned=size(ret), caller=size(caller))
    for k, c in dict(production=2923+(100 if arm == 'blocks' else 0), replay=2714,
                    reception=1467, controller=1433, terminal=1353, returned=1662, caller=1878).items():
        require(result[k].terms.get('constant') == c, 'schema constant: '+k)
    return result


def analyze():
    prev = json.loads((ROOT/PREVIOUS).read_bytes())
    for p, h in prev['anchors'].items():
        require(sha(p) == h, 'inherited implementation drift: '+p)
    code_bytes = sum((ROOT/p).stat().st_size for p in prev['code_files'])
    require(code_bytes == 274543, 'bound packaged code sizes')
    inv = prev['inventory']; N, F, H = inv['nodes_upper'], inv['factors_upper'], inv['roots_upper']
    policy = dict(factors=32768,active=16384,views=128,depth=64,view_rows=16384,rows=65536,
        nnz=1048576,members=4096,page_rows=64,header_bytes=8*2**20,member_bytes=2**20,
        total_bytes=128*2**20,stored_bits=4096,work_bits=8192,visits=8000000)
    names = ['@center']
    for graph, indices in [('router',(1,3))]+[(f'experts.{i}',(1,3,5)) for i in range(8)]:
        names += [f'{graph}.{i}.{kind}' for i in indices for kind in ('weight','bias')]
    require(len(names)==53 and sum(map(len,names))==879, 'registered tensor names')
    h = '0'*64
    receipt = dict(schema='CAPTURED_TENSOR_ROWS_RECEIPT_V1', source_sha256=h,request_sha256=h,
        tensor_hashes={}, source_bytes=59909824,statistics=dict(chunk_reads=33288,raw_bytes=55715520,
        values=6964440,affine_rows=3288,max_chunk_bytes=1048576,max_row_values=3072),
        scope='captured binary64 parameters only; not HZ lowering or model safety')
    roles = dict(input=0,router=4,common=5,experts=[81]*8,pairs=[[7,7,81]]*28,prefix=None)
    meta = dict(schema='SHARED_SOURCE_REFERENCE_V1',source=h,request=h,frame=1,cut=None,policy=policy,
        factors=[],views=[],roles=roles,source_receipt=receipt,aliases=[],sequences=[])
    f_len = scalar(['expert7/4/lift/output/255','continuous',-1,81])
    v_len = scalar(dict(name='expert7/5',scope=-1,parent=None,op='Flatten',factors=[32048,32048],output=245,eq=245,ine=245))
    a_len, s_len = scalar([19495,'expert7',1,255,3072]), scalar([136232,3072])
    require((scalar(policy),scalar(receipt),scalar(meta),f_len,v_len,a_len,s_len)==(253,471,1371,48,116,28,13), 'header templates')
    receipt_upper = scalar(receipt)-2+(1+sum(map(len,names))+70*len(names))
    header = scalar(meta)-8-scalar(receipt)+receipt_upper+4+(f_len+1)*F+(v_len+1)*82+(a_len+1)*2176+(s_len+1)*246
    forest = prev['source_payload']['forest_binary_payload_upper']+header
    pages = (forest+65535)//65536
    require((header,forest,pages)==(1652449,6981605,107),'derived header demand, not acceptance cap')
    K = Length(K=1); nP=K+261; LP=42*K+9264
    censuses = [40+2*(LP+delta)+83*(nP+count) for count,delta in ((10,144),(11,158),(12,173),(13,186))]
    E, D, Nt, Nr, M = 504,252,84,32,2700
    candidate_bytes=128*Nt+E*(2152+16*M)
    result_bytes=145*Nr+4989*D
    work=max(json.loads((ROOT/'docs/hz_native_counted_envelope_20261018.json').read_bytes())['arms'][a]['declared_complete_traversal_counted_upper']
             for a in ('native_coo_reference','native_source_row_blocks'))
    d,b=len(str(work)),len(str((2**27)*work))
    Phi=1+(Nt+15)*(140+3*d); G=(Nt+20)*(137+3*d)
    arms={}
    for arm in ('coo','blocks'):
        records=schemas(arm)
        outer=sum((records[k] for k in ('production','replay','reception','controller','terminal')), Length(362,Pprofile=1,Qroot=1,G=1,Logs=1))
        pkg=(55715520+Length(Jmanifest=1)+forest+67+110*pages+candidate_bytes+code_bytes+452+Length(Qmargin=1)
            +5563+(4 if arm=='blocks' else 0)+120*Nt+LP+88*nP-42+result_bytes+951+140*Nr)
        def substitute(poly):
            out=Length()
            values=dict(d=Length(d),b=Length(b),Phi=Length(Phi),G=Length(G),**{f'C{i}': c for i,c in enumerate(censuses)})
            for k,v in poly.terms.items():
                out += v if k=='constant' else values.get(k,Length(**{k:1}))*v
            return out
        request=substitute(pkg+outer); joint=request+substitute(records['caller'])
        arms[arm]=dict(actor_schema_lengths={k:v.record() for k,v in records.items()},
            package_length=pkg.record(),outer_length=outer.record(),request_length=request.record(),joint_length=joint.record())
    require(candidate_bytes==22868160,'all candidate roles retained')
    literal_bits=4*4096; input_bits=literal_bits+1075+1
    # No network/source objects: lexical/schema algebra only.
    # Source JSON is decoded/re-encoded for identity, not required canonical on disk.
    # For valid finite JSON: strings expand <=6x, float tokens >=3 chars map to
    # <=24 chars, integers/punctuation do not expand. Hence C <=8*H.
    source_parse_text=17*(4*2**20)
    manifest_cap=4*2**20
    chunk_upper=(manifest_cap-20433-4)//126
    rank_upper=(manifest_cap-20433-126*95)//4
    require((chunk_upper,rank_upper)==(33125,1040475),'raw manifest necessary limits')
    # Equality validation permits bool/float spelling in request-side shape,
    # model_state counts and Flatten dimensions. Do not silently restrict those
    # already accepted representations to the factory's integer-only spelling.
    model_delta=scalar(dict(tensor_count=52.,parameter_count=6961368.))-scalar(dict(tensor_count=52,parameter_count=6961368))
    flatten_delta=9*(scalar([True,-1.])-scalar([1,-1]))
    require((model_delta,flatten_delta)==(4,45),'equal numeric metadata widths')
    heap=native_python_large_objects()
    return dict(schema='NATIVE_GENERATION_STATIC_CONTRACT_V1',based_on=BASE,
        anchors={p:sha(p) for p in sorted(set(prev['anchors'])|set(EXTRA)|{PREVIOUS})},
        scope='conditional registered declared-source complete traversal; not admission, full heap, RSS, model correspondence or performance',
        source=dict(tensors=53,tensor_names_length_sum=879,network_and_center_payload=55715520,
            manifest_existing_read_cap=manifest_cap,source_total_upper=59909824,full_chunk_minimum=95,
            manifest_only_chunk_upper=chunk_upper,manifest_only_rank_upper=rank_upper,
            chunk_symbols='K actual source chunks, Ns=K+1; not assumed fully packed',
            manifest_raw_lower='20433+4*rank+126*K',
            manifest_canonical_exact='20427+tensor_shape_json_length+request_shape_json_length+four_domain_values_json_length+124*K+sum_chunk_size_digits+sum_chunk_offset_digits+model_count_encoding_delta+flatten_encoding_delta+optional(9+trust_json_length)',
            manifest_canonical_upper='36876+7*rank+138*K+optional(9+trust_json_length)',
            equality_validated_metadata=dict(model_count_encoding_delta_upper=model_delta,
                flatten_encoding_delta_upper=flatten_delta,tensor_shape_upper='2*rank+4',
                request_shape_upper='5*rank+4',strict_integer_only_not_assumed=True),
            canonical_to_raw_multiplier_upper=8,parse_named_text_overlap_upper=source_parse_text,
            source_decode_object_tree_included=False,
            center_rank='unbound by n=3072 alone; manifest grammar gives rank<=1040475, not assumed rank 4'),
        forest=dict(factor_record=f_len,view_record=v_len,alias_record=a_len,sequence_span=s_len,
            source_receipt=receipt_upper,header=header,binary=5329156,packet=forest,pages=pages),
        package=dict(code_files=32,code_bytes=code_bytes,candidate_bytes=candidate_bytes,
            result_page_bytes=result_bytes,result_index_bytes=951+140*Nr,contract='452+Qmargin <=4550',
            files=nP.record(),relative_name_sum=LP.record(),request_file_count=(nP+13).record(),joint_file_count=(nP+14).record(),
            census_lengths=[c.record() for c in censuses],
            admission='nP <=4096-O-16 for production archive; O=outside source members. Also every individual cap applies.'),
        arms=arms,conditional_work_width=dict(W=work,d=d,b=b,Phi=Phi,G=G,admitted=False),
        input_arithmetic=dict(literal_bits=literal_bits,input_endpoint_bits=input_bits,
            two_operand_intermediate_bits=2*input_bits+1,unchanged_work_acceptance_bits=8192,
            scope='pre-Work lexical/rational quantities only; no acceptance weakening'),
        python_large_object_components=heap,
        unresolved_parameters=['immutable profile/root/source layout and rank bindings','normal run id and Torch version encoding',
            'cleanup zombie roster Z, PID digits p, post-cleanup rss digits rho','native log demand versus separate hard rejection',
            'complete self-owned Python object/temporary heap and interpreter allocation internals'],
        policy=dict(changed=False,new_quota=None,seconds=300,work=8000000,artifact_bytes=2**27,execution_approved=False),
        source_builds=0,native_calls=0,model_runs=0,gpu_runs=0,
        decision='FOREST_HEADER_AND_SUCCESS_ARCHIVE_GENERATION_PARAMETERIZED_NO_FULL_CAPACITY_ADMISSION')


if __name__=='__main__':
    result=analyze()
    if sys.argv[1:]==['--check']:
        require(result==json.loads((ROOT/REPORT).read_bytes()),'generation-contract record drift')
        print('PASS: schema lengths, header generation and original bindings; no native execution')
    else:
        require(not sys.argv[1:],'only --check is supported')
        print(json.dumps(result,sort_keys=True,indent=2))
