"""Conditional archive demand and actual lifetimes; static, standard library only.

No tensors, forests, candidates, workers or numerical libraries are constructed.
Size is an encoded-length symbol, NOT a generated source object. The canonical
captured-source layout below is an assumption, not a new runtime admission rule.
"""
import argparse
import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path

from scripts import assess_shared_execution_envelope as previous

prior = previous.prior
ROOT = prior.ROOT
PARENT = 'docs/hz_shared_execution_envelope_20261015.json'
BASE = 'a3681b44f3bad58e52cb64e4f6b881c4e93afecc'
H = '0' * 64


@dataclass(frozen=True)
class Size:
    bytes: int

    def __post_init__(self):
        prior.require(type(self.bytes) is int and self.bytes >= 0, 'encoded length')


def length(value):
    """Compact ASCII JSON length algebra; never expands a symbolic array."""
    if type(value) is Size:
        return value.bytes
    if type(value) is dict:
        return 2 + sum(length(k) + 1 + length(v) for k, v in value.items()) + max(0, len(value)-1)
    if type(value) in (list, tuple):
        return 2 + sum(map(length, value)) + max(0, len(value)-1)
    return len(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode())


def array(count, item):
    prior.require(type(count) is int and count >= 0, 'array count')
    return Size(2 + count*length(item) + max(0, count-1))


def source_layout(v, *, full_chunks=True, fixed_trust=True):
    prior.require(full_chunks is True and fixed_trust is True,
                  'canonical full-chunk/fixed-trust layout required; generic source bound remains open')
    block = prior.integer(prior.assignment('scoped_source/factored_io.py', 'BLOCK_LIMIT'))
    # Only 52 shape tuples; no parameter arrays or source documents are made.
    shapes = []
    for widths, repeat in ((v['router_widths'], 1), (v['expert_widths'], v['experts'])):
        for _ in range(repeat):
            for a, b in zip(widths, widths[1:]):
                shapes.extend(((b, a), (b,)))
    prior.require(len(shapes) == 52, 'registered parameter tensor roster')
    sizes = [8*math.prod(s) for s in shapes] + [8*v['inputs']]
    chunks = sum((n+block-1)//block for n in sizes)
    prior.require(chunks == 95, 'full chunks per tensor, not whole-bank chunking')
    # Two four-digit dimensions dominate the registered matrix and center shape
    # strings. This is a field-length bound, not a hypothetical larger tensor.
    state = array(52, dict(name='experts.7.1.weight', dtype='torch.float64', shape=[3072,3072], sha256=H))
    descriptor = length(dict(name='experts.7.1.weight', dtype='torch.float64', shape=[3072,3072],
                             byte_order='little', chunks=[]))
    chunk = length(dict(file='t/00052-00005.bin', bytes=block, sha256=H, offset=5*block))
    tensor_array = 2 + 53*descriptor + chunks*(chunk+1) - 53 + 52
    linear = dict(kind='Linear', index=5, training=False, weight_name='experts.7.5.weight',
                  bias_name='experts.7.5.bias', weight=dict(tensor_ref='experts.7.5.weight'),
                  bias=dict(tensor_ref='experts.7.5.bias'))
    graph_array = (2+9*length(dict(name='expert7',layers=[]))+8 +
                   26*length(linear)+17*length(dict(kind='ReLU',index=4,training=False,inplace=False))+
                   9*length(dict(kind='Flatten',index=0,training=False,dimensions=[1,-1]))+52-9)
    request = dict(experts=8, classes=10, label=9, top_k=2, gate='SELECTED_SOFTMAX',
        tie_policy='ANY_LEGAL_TOPK', training=False,
        center=dict(dtype='torch.float64',shape=[1,3,32,32],sha256=H), radius='2/255', margin='0',
        clip=['0','1'],model_state=dict(sha256=H,tensor_count=52,parameter_count=v['parameter_values']))
    declaration = dict(schema='SCOPED_DECLARED_TOP2_V1', request=request,
        center=dict(tensor_ref='@center'), state_inventory=state, networks=Size(graph_array),
        trust='declared real graph; graph/program correspondence and native floating execution are separate')
    fixed = length(dict(schema='H2_CHUNKED_SOURCE_V1',declaration=declaration,tensors=Size(tensor_array)))
    header = fixed + 4*4096 - 8  # four numeric-literal strings; keep their quotes
    tensor_bytes = sum(sizes)
    return dict(tensors=53,parameter_tensors=52,chunks=chunks,files=chunks+1,
        tensor_bytes=tensor_bytes,manifest_fixed_upper=fixed,manifest_domain_upper=header,
        stored_source_upper=tensor_bytes+header,
        components=dict(state_inventory=length(state),tensor_inventory=tensor_array,networks=graph_array,
                        state_record=length(dict(name='experts.7.1.weight',dtype='torch.float64',shape=[3072,3072],sha256=H)),
                        empty_tensor=descriptor,chunk_record=chunk,linear_record=length(linear)),
        conditions=['registered 3072/8/10 topology; center shape [1,3,32,32]',
                    'capture.py fixed ASCII trust text and no additional provenance',
                    'pack() default 1 MiB full chunks per tensor, final remainder',
                    'four domain literals each <=4096 ASCII numeric characters'],
        generic_parser_layout_upper=None, current_supervisor_full_source_intake=False)


def forest_layout(v, source):
    tensor_hashes = Size(2+53*(length({'experts.7.1.weight':H})-2)+52)
    receipt = dict(schema='CAPTURED_TENSOR_ROWS_RECEIPT_V1',source_sha256=H,request_sha256=H,
        tensor_hashes=tensor_hashes,source_bytes=source['stored_source_upper'],
        statistics=dict(chunk_reads=source['chunks'],raw_bytes=source['tensor_bytes'],
            values=source['tensor_bytes']//8,affine_rows=v['affine_outputs'],max_chunk_bytes=2**20,max_row_values=3072),
        scope='captured binary64 parameters only; not HZ lowering or model safety')
    factors = array(v['factors_upper'], ['x'*25,'continuous',-1,81])
    views = array(v['views'],dict(name='expert7/5',scope=-1,parent=None,op='Flatten',
                                  factors=[32048,32048],output=245,eq=245,ine=245))
    aliases = array(2176,[19495,'expert7',1,255,3072])
    sequences = array(3*v['views'],[v['roots_upper'],prior.policy()['view_rows']])
    roles = dict(input=81,router=81,common=81,experts=[81]*8,pairs=array(v['pairs'],[7,7,81]),prefix=None)
    header = length(dict(schema='SHARED_SOURCE_REFERENCE_V1',source=H,request=H,frame=1,cut=None,
        policy=prior.policy(),factors=factors,views=views,roles=roles,source_receipt=receipt,
        aliases=aliases,sequences=sequences))
    explicit = v['inputs']+2*v['affine_outputs']+12*v['relu_units']
    n,f,r,h = (v[k] for k in ('nodes_upper','factors_upper','references_upper','roots_upper'))
    path = 'scoped_source/direct_source_format.py'
    binary = (prior.structure(path,'PREFIX') + n*(prior.structure(path,'NODE')+prior.structure(path,'ROOT'))+
              (explicit+r)*prior.structure(path,'ENTRY') + h*prior.structure(path,'INDEX')+f*prior.structure(path,'SCALAR'))
    payload = header+binary
    pages = (payload+65535)//65536
    index = length(dict(schema='SHARED_SOURCE_REFERENCE_V1',bytes=payload,
                        pages=array(pages,dict(file='9999.bin',bytes=65536,sha256=H))))
    return dict(header_upper=header,binary_upper=binary,payload_upper=payload,pages=pages,
        index_upper=index,files=pages+1,stored_upper=payload+index,
        components=dict(factors=length(factors),views=length(views),aliases=length(aliases),
                        sequences=length(sequences),roles=length(roles),source_receipt=length(receipt)),
        memberships_remain_expanded=True)


def lifetime(v, source, forest):
    n,f,views,w = v['nodes_upper'],v['factors_upper'],v['views'],3072
    return dict(packed_all_payload=8*v['parameter_values'],
        packed_first_payload=8*(v['first_weight_positions']+2176),
        later_packed_rows_release_after_check=True,
        builder_and_checker_graphs_coexist=False,
        production=dict(packet_payload_retained=forest['payload_upper'],
                        packet_released_before_consumer=False),
        replay=dict(packet_released_before_consumer=True,
                    previous_and_new_caller_frames_can_overlap=True,frame_copies_at_seal=3),
        scratch=dict(plan_fraction_entries_upper=n+2*f,visible_memberships_upper=views*f,
            checker_norm_logical_entries_upper=2*n,row_fraction_values_upper=2*(w+1),
            row_logical_reference_entries_upper=3*w,chunk_buffers_upper=2*2**20,row_fill_payload_upper=2*8*(w+1),
            counts_are_not_allocated_container_capacities=True),
        outside_owned_arena=['graph/source Python containers and numeric objects','ordinary dict plan pullback',
            'checker visibility sets and norm lists','caller frame/candidate lists',
            'result/receipt/hash inventories and JSON encoding temporaries','allocator retained memory'],
        supervisor=dict(sampled_rss_not_os_limit=True,nominal_poll_seconds=.01,
            parent_receiving_sampled=False,worker_file_size_limit_not_heap_limit=True),
        full_live_heap_upper=None,full_rss_upper=None,
        missing=['fixed Python/allocator object-size contract and temporaries',
            'memory enforcement across worker, descendants and parent reception',
            'charged source capture/native proposal interfaces'],actual_peak=None)


def code_inventory():
    """Read only the closed literal file lists, never import packaged code."""
    node = prior.assignment('scoped_source/direct_package.py', 'CODE_FILES')
    prior.require(isinstance(node,ast.BinOp) and isinstance(node.op,ast.Add), 'code list syntax')
    gen = node.left.args[0]
    prior.require(isinstance(gen,ast.GeneratorExp) and len(gen.generators)==1
        and isinstance(gen.generators[0].target,ast.Name) and gen.generators[0].target.id=='name'
        and not gen.generators[0].ifs and not gen.generators[0].is_async,
        'closed checker file generator')
    prior.require(ast.unparse(gen.elt)=="'scoped_source/' + name + '.py'", 'checker filename expression')
    v1 = tuple('scoped_source/'+x+'.py' for x in ast.literal_eval(gen.generators[0].iter)) + ast.literal_eval(node.right)
    life = prior.assignment('scoped_source/direct_lifecycle_package.py','CODE_FILES')
    prior.require(isinstance(life,ast.BinOp) and isinstance(life.op,ast.Add) and isinstance(life.left,ast.Name)
                  and life.left.id=='V1_CODE', 'lifecycle code extension')
    code = v1 + ast.literal_eval(life.right)
    shared = ast.literal_eval(prior.assignment('scoped_source/direct_lifecycle_package.py','SHARED_CODE'))
    def literal(n):
        if isinstance(n,ast.Name):
            return {'CODE_FILES':code,'SHARED_CODE':shared,
                'PROTOCOL_FILE':ast.literal_eval(prior.assignment('scoped_source/direct_supervised.py','PROTOCOL_FILE')),
                'SHARED_PROTOCOL_FILE':ast.literal_eval(prior.assignment('scoped_source/direct_supervised.py','SHARED_PROTOCOL_FILE'))}[n.id]
        if isinstance(n,ast.Tuple): return tuple(literal(x) for x in n.elts)
        if isinstance(n,ast.Constant): return n.value
        if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Add): return literal(n.left)+literal(n.right)
        if (isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('tuple','sorted','set')
                and len(n.args)==1 and not n.keywords):
            return {'tuple':tuple,'sorted':sorted,'set':set}[n.func.id](literal(n.args[0]))
        raise ValueError('closed code inventory expression')
    implementations = literal(prior.assignment('scoped_source/direct_supervised.py','FILES'))
    files = code+shared
    prior.require(len(files)==26 and len(implementations)==44 and len(set(files))==26,
                  'checker/implementation roster')
    sizes = {p:(ROOT/p).stat().st_size for p in files}
    prior.require(all(0<n<=65536 and len(p)<=200 and p.isascii() for p,n in sizes.items()), 'code member bounds')
    return dict(files=sizes,bytes=sum(sizes.values()),count=len(files),bindings=list(implementations))


def metadata(v, source, forest, code, counter_digits):
    """Encode the whole successful metadata tree as lengths, not report caps.

    Count widths are conditional on the complete loop envelope. Failure strings,
    arbitrary caller root names and provenance are not bounded by this schema.
    """
    prior.require(counter_digits in (7,12), 'bound Work counter widths')
    S = lambda n: Size(n+2)
    I, F, B, HASH = Size(20),Size(24),Size(5),S(64)
    W = {k:Size(counter_digits) for k in previous.COUNTERS}
    consumer = dict(kind='pair_scoped',check_width=2)  # dominates dense metadata
    method = dict(schema='SHARED_LIFECYCLE_METHOD_V1',protocol=H,
        producer=dict(kind='FIXED_SCALAR_STUB_V1',profile=1),candidate_width=8,
        production=consumer,replay=consumer)
    ref = dict(file=S(200),bytes=I,sha256=HASH)
    q = v['pair_local_tiles_upper']; ns=source['files']; p=forest['pages']; k=code['count']
    artifact_files = ns+p+q+k+6
    bundle_members = artifact_files-3
    phases = array(18+2*q,dict(name=S(22),complete=B,seconds=F,work=W))
    progress = (32+4*q)*(length(dict(name=S(22),event=S(5),work=W,at=F))+1)
    source_stats = dict.fromkeys(('nodes','local_nnz','references','factors','views','forest_bytes',
        'unique_roots','memberships','stored_memberships','sequence_count','physical_bytes',
        'packed_source_bytes','alias_rows','explicit_locals','aliased_locals'),I)
    transfer = dict(bytes=I,members=I,seconds=F,copies=1,source=HASH)
    owner = dict.fromkeys(('live_bytes','peak_bytes','allocated_bytes','copied_bytes','released_bytes','allocations'),I)
    owner['cleanup_seconds']=F
    cost = dict.fromkeys(('init_seconds','prepare_seconds','encode_seconds','receive_seconds',
        'publish_seconds','check_seconds','terminal_seconds','release_seconds'),F)
    cost.update(publication_attempted_bytes=I,acknowledged_bytes=I)
    snapshots = dict(source_transfer=transfer,source_stats=source_stats,consumer=consumer,
        lifecycle=dict(setup_calls=1,propose_calls=Size(2),close_calls=1,root_terms=I),owner=owner,driver_cost=cost)
    construction = dict(native_seconds=F,native_coordinates=I,packed_source_bytes=I,forest_bytes=I,pages=I)
    event = dict(file=S(200),bytes=I,seconds=F,bytes_fully_written=B,timely_published=B,complete=B)
    events = Size(length(array(artifact_files,event))+ns*(length(dict(ownership_transfer=True))-2+1))
    ledger = dict(logical_bytes=I,limit=I,reserved_terminal=I,written=I,attempted=I,events=events,
        elapsed_seconds=F,elapsed_after_deadline=F)
    contract = dict(schema='SCALED_FOREST_ENDPOINT_CONTRACT_V3',forest=HASH,source=HASH,request=HASH,
        factors=HASH,classes=10,label=9,margin=S(4096),gate='SELECTED_SOFTMAX',top_k=2,tie_policy='ANY_LEGAL_TOPK')
    manifest = dict(schema='DIRECT_LIFECYCLE_MANIFEST_V1',source=dict(directory='source',sha256=HASH),
        forest=ref,contract=ref,tiles=array(q,ref),method=method,invocation=HASH,code=array(k,ref))
    bundle = dict(schema='DIRECT_LIFECYCLE_BUNDLE_V1',manifest='manifest.json',members=array(bundle_members,ref))
    R = Size(previous.metadata(8192,v['property_duties'],q,counter_digits)['result_upper'])
    replay = dict(result=R,owner=owner,driver_cost=cost,candidate_bytes=I,consumer=consumer,package_sha256=HASH,
        package_bytes=I,method=method,invocation=HASH,work=W,seconds=F)
    inner = dict(status='PENDING_RECEPTION',method=method,invocation=HASH,source=HASH,result=ref,package=ref,work=W)
    candidate = dict(status='COMPLETE',result=R,result_ref=ref,terminal_ref=ref,package=ref,replay=replay,
        construction=construction,method=method,invocation=HASH,source=HASH,phases=phases,snapshots=snapshots,
        work=W,initial_work=W,deadline=F,seconds=F,archive=ledger,isolated=True,site_disabled=True,
        fixture_seconds=F,numerical_imports=[])
    impl = Size(2+len(code['bindings'])*(200+3+length(HASH))+len(code['bindings'])-1)
    spec = dict(schema='DIRECT_SUPERVISED_STUB_V1',case='guarded',method=method,fault='')
    invocation = dict(schema='DIRECT_SUPERVISED_INVOCATION_V1',id=HASH,spec=spec,start=F,deadline=F,
        run_deadline=F,cleanup_deadline=F,initial_work=W,implementations=impl)
    child = dict(schema='OWNED_BOUNDED_CPU_V1',status='COMPLETED',error=None,pid=I,returncode=0,
        seconds=F,execution_seconds=F,cleanup_seconds=F,cleanup_status='LEADER_REAPED_NO_LIVE_GROUP',
        remaining_group=dict(live=[],zombies=[],rss=0),descendant_on_leader_exit=B,exit_observed_at=F,
        sampled_peak_rss=I,run_deadline=F,cleanup_deadline=F,escaped_descendants_or_driver_cleanup=B)
    accepted = dict(result=R,package_sha256=HASH,source=HASH,method=method,invocation=HASH,package_bytes=I,
        worker_work=W,snapshots=snapshots,phases=phases,worker_seconds=F,fixture_seconds=F)
    receipt = dict(schema='DIRECT_SUPERVISED_RECEIPT_V1',status='COMPLETE',invocation=HASH,invocation_sha256=HASH,
        method=method,candidate_sha256=HASH,accepted=accepted,error=None,child=child,work=W,work_complete=True,
        seconds_before_publication=F,budget=F)
    members = dict(contract=length(contract),manifest=length(manifest),bundle=length(bundle),result=length(R),
        inner_terminal=length(inner),candidate_report=length(candidate),invocation=length(invocation),
        parent_receipt=length(receipt),progress=progress,worker_log=0)
    return dict(counter_digits=counter_digits,nonnegative_count_digits=20,finite_float_chars=24,
        report_result_copies=dict(candidate=2,standalone_result=1,parent_receipt=1),
        members_upper=members,total_upper=sum(members.values()),artifact_files=artifact_files,
        whole_request_files=artifact_files+5,bundle_members=bundle_members,
        components=dict(method=length(method),work=length(W),phases=length(phases),snapshots=length(snapshots),
                        archive_ledger=length(ledger),replay=length(replay),child=length(child)),
        over_cap={name:prior.classify(2**20,upper=members[name])
                  for name in ('result','candidate_report','parent_receipt')},
        failure_generation_upper=None,actual_generated_bytes=None,
        excludes=['caller separately saving returned call/census','later offline relocation or audit artifacts',
                  'arbitrary provenance, OS error strings or native proposal reports'],
        conditions=['complete no-fault fixed-stub successful metadata shape',
                    'ASCII relative archive paths <=200 characters; fixed trust and topology',
                    'nonnegative counters fit uint64; counts derived separately, not quotas',
                    'no descendants; successful cleanup leaves empty live/zombie lists'])


def analyze():
    saved = prior.load(PARENT)
    prior.require(saved==previous.analyze(), 'bound execution envelope drift')
    v = saved['inventory']; source=source_layout(v); forest=forest_layout(v,source); code=code_inventory()
    # No numeric field is promoted to a resource limit. Explicit upper estimates
    # justify the 20-digit *representation* of cumulative owner statistics.
    work = max(a['postcapture_counted_work_upper'] for a in saved['consumers'].values())
    candidates = prior.load(previous.PARENT)['candidate_stream']
    # RootTable <=3V+1; dense per endpoint <=7 allocations, scoped <=6.
    # Descriptor/pullback/dual work plus at most four frame allocations per
    # tile are dominated by 32E. Four per tile additionally cover scope maps.
    allocation_count = 3*v['views']+1+32*v['endpoints_upper']+4*v['pair_local_tiles_upper']
    allocation_payload = max(4*2**20,2065*(v['nodes_upper']+v['factors_upper']))
    # Include the general-rational demand even though its largest tile would be
    # rejected by the current cap. A cap is not a full-format demand bound.
    copy_payload = max(4*2**20,*(a['largest_tile_upper'] for a in candidates.values()))
    copied = work*copy_payload
    allocated = allocation_count*allocation_payload
    prior.require(max(copied,allocated,work)<2**64, '20-digit nonnegative counter representation')
    metas={name:metadata(v,source,forest,code,d) for name,d in (('current_counters',7),('conditional_complete_loops',12))}
    whole = {}
    for profile, data in candidates.items():
        physical = source['stored_source_upper']+forest['stored_upper']+data['stream_upper']+code['bytes']
        whole[profile]=dict(source_and_forest=source['stored_source_upper']+forest['stored_upper'],
            candidate_stream=data['stream_upper'],code=code['bytes'],
            success_metadata=metas['conditional_complete_loops']['total_upper'],
            request_upper=physical+metas['conditional_complete_loops']['total_upper'],
            total_storage_assessment=prior.classify(prior.policy()['total_bytes'],
                upper=physical+metas['conditional_complete_loops']['total_upper']),
            actual_bytes=None,full_capacity_admitted=False)
    anchors=dict(saved['anchors'])
    for name in set(code['bindings']) | {PARENT,'scoped_source/capture.py',
            'scripts/assess_shared_archive_contract.py','scripts/test_shared_archive_contract.py'}:
        anchors[name]=prior.sha(name)
    return dict(schema='SHARED_ARCHIVE_LIFETIME_STATIC_V1',based_on=BASE,anchors=anchors,
        source_layout=source,forest_layout=forest,code_inventory=code,metadata=metas,
        counter_representation=dict(work_upper=work,allocation_count_upper=allocation_count,
            one_allocation_payload_upper=allocation_payload,allocated_bytes_upper=allocated,
            one_copy_payload_upper=copy_payload,copied_bytes_upper=copied,
            digits=20,not_a_new_quota=True),
        conditional_success_archive=whole,lifetime=lifetime(v,source,forest),
        current_refusal=saved['current_refusal'],unchanged=saved['unchanged'],
        open_contract=dict(generic_source_layout_upper=None,full_rss_upper=None,
            all_failure_metadata_generation_upper=None,capture_native_work_upper=None,completion_seconds_upper=None),
        decision='CONDITIONAL_ARCHIVE_DEMAND_CLOSED_RESOURCE_ADMISSION_STILL_REFUSED',
        next_decision='explicit representation/ownership and whole-process enforcement contract; no repeated full-shape retries',
        runtime_changed=False,new_quota=None,full_capacity_admitted=False,
        source_builds=0,candidate_builds=0,model_runs=0,gpu_runs=0,solver_calls=0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--check-report',type=Path)
    args=p.parse_args(); result=analyze()
    if args.check_report:
        prior.require(json.loads(args.check_report.read_bytes())==result,'archive contract drift')
        print('PASS: conditional archive/lifetime ledger; unchanged limits refuse; no runtime execution')
    else:
        print(json.dumps(result,sort_keys=True,indent=2))


if __name__=='__main__': main()
