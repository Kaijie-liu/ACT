"""Source-bound integer accounting, not model construction or execution.

Owned payload upper and unavoidable Work lower are different statements. This
report never converts either into a new quota, RSS bound or runtime prediction.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
ANCHORS={
    'scoped_source/pair_workspace.py':'18df56b27f952189c52e268450c0e7dd53e58da5e4f2060a4eb098b0278a2829',
    'scoped_source/bounded_runtime.py':'c9c1d295e6a23bb92042f1060f99a5ee14e2e448d46e3f35af739663e750f7af',
    'scoped_source/direct_candidates.py':'cb9b791b9a1b828c9d4fd6b351361c6fe59072b40caf0c91e55beefeaf2665a4',
    'scoped_source/scaled_endpoints.py':'48cf0be6fd9fe1776089f982a69d4100fdbd5867e207579286422bfbc45ad044',
    'scoped_source/check_source_dag.py':'58e5e2158843631c9e6ed324cfeeb2eb38e51df1205d2e14d1ed8ac8ce634458',
    'scoped_source/check_scaled_references.py':'244dc7ee5cf7fe46ab7097b2494dfec64d818e926b4726eb19d59a53c5074f9e',
    'scoped_source/wire_forest.py':'981385019f93f97d41c02eab245cc21e8c430ccdfc8e0f853d5ffacb6981a3b3',
    'scoped_source/forest_endpoints.py':'86396e9c662ba8c91f6ac3921dc7f410d9b977937df38a9fb782f82991c50a4c',
    'scoped_source/hz_request_store.py':'7cb2be7647039f59d3c896b7856f634f43be9c532d9d2fd43632be09ab3230e0',
    'docs/hz_full_resource_ledger_20261014.json':'04c51e4765ada9445dcdd80f63f745459366154037443a68dd565c093acbe9d2',
    'docs/hz_pair_scoped_consumer_controls_20261014.json':'9324e21aef911b2e1e7633e8348e9cd0e230f8399358ed8fde8d740badfb29d2',
}
ARENA=128*2**20
FRAME=4*2**20
CANDIDATES=16*2**20-8192
WORK=8000000


def verify_files():
    for name,h in ANCHORS.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:
            raise ValueError('pair-scoped capacity anchor drift: '+name)


def struct_size(path,name):
    for n in ast.parse((ROOT/path).read_bytes()).body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):
            c=n.value
            if (isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute)
                and isinstance(c.func.value,ast.Name) and c.func.value.id=='struct'
                and c.func.attr=='Struct' and len(c.args)==1):
                return struct.calcsize(ast.literal_eval(c.args[0]))
    raise ValueError('literal struct absent: '+name)


def layout():
    header=struct_size('scoped_source/bounded_runtime.py','HEADER')
    return dict(slot=1+header+2*1024,scalar=header+2*1024,stored=header+2*512,
                root=struct_size('scoped_source/bounded_runtime.py','ROOT'),
                index=struct_size('scoped_source/pair_workspace.py','U32'),
                prefix=struct_size('scoped_source/direct_candidates.py','PREFIX'),
                record=struct_size('scoped_source/direct_candidates.py','RECORD'))


def positive(*values):
    if any(type(v) is not int or v<=0 for v in values): raise ValueError('positive integer counts')


def shape(v):
    a=sum(v['router_widths'][1:])+2*sum(v['expert_widths'][1:])
    u=sum(v['router_widths'][1:-1])+2*sum(v['expert_widths'][1:-1])
    return dict(pair_nodes=v['inputs']+a+4*u+2*(v['experts']-2),
                pair_factors=v['inputs']+2*a+5*u,
                tighter_global_factors=v['inputs']+2*v['affine_outputs']+5*v['relu_units'])


def frame_size(rows,k,profile):
    positive(rows,k)
    if k>8 or type(profile) is not int or profile not in (0,1): raise ValueError('frame profile/width')
    l=layout(); scalar=8 if profile==0 else l['stored']
    return l['prefix']+k*(l['record']+2*l['stored']+2*rows*scalar)


def candidate_stream(v,profile):
    widths=[8,8,2]
    sizes=[frame_size(v['pair_dual_rows_upper'],k,profile) for k in widths]
    return dict(profile=profile,per_pair_widths=widths,per_pair_frame_upper=sizes,
                tiles=v['pairs']*len(widths),endpoints=v['endpoints_upper'],roles=v['candidates_upper'],
                largest_frame_upper=max(sizes),stream_upper=v['pairs']*sum(sizes),
                frame_upper_fits=max(sizes)<=FRAME,stream_upper_fits=v['pairs']*sum(sizes)<=CANDIDATES,
                actual_bytes=None,upper_overflow_proves_actual_overflow=False)


def arena(v,scope,*,global_factors=None,frame_override=None):
    """Actual owned-buffer lifetime, with conservative dimension substitution.

    No Python banks, source/native allocations or caller retained copies count
    as owned arena bytes. Events contain only integer sizes, never buffers.
    """
    l=layout(); n=v['nodes_upper']; f=v['factors_upper'] if global_factors is None else global_factors
    p,q=scope['pair_nodes'],scope['pair_factors']; h=v['roots_upper']; positive(n,f,p,q,h)
    if p>n or q>f: raise ValueError('scope cannot exceed global dimensions')
    roots=l['root']*n+4*h; maps=4*(n+f+p+q)
    desc=4*(4+l['scalar']); a=p*l['slot']; b=q*l['slot']
    t=frame_size(v['pair_dual_rows_upper'],8,0) if frame_override is None else frame_override
    positive(t)
    if t>FRAME: raise ValueError('no admitted frame above fixed cap')
    events=[]; live=roots
    def event(name,delta):
        nonlocal live
        live+=delta
        if live<0: raise ValueError('negative lifetime')
        events.append(dict(event=name,delta=delta,owned_upper=live))
    event('root index with all memberships',0); event('scope forward/reverse maps',maps)
    for j in range(8):
        event(f'descriptor {j} fill',desc); event(f'descriptor {j} seal overlap',desc)
        event(f'descriptor {j} release fill',-desc)
    event('frame fill or intake copy',t); event('frame seal overlap',t); event('frame release fill',-t)
    for j in range(2):
        event(f'objective {j} node accumulator',a); event(f'objective {j} factor output',b)
        event(f'objective {j} consume nodes',-a)
    event('both dual node accumulators',2*a)
    for j in range(2): event(f'dual {j} factor output without node copy',b)
    event('consume dual nodes',-2*a)
    event('release complete microblock factor maps',-4*b)
    # Subsequent check blocks have the same peak, never retained old maps.
    event('clear descriptors and scope',-8*desc-maps); event('release candidate',-t)
    if live!=roots: raise ValueError('tile release imbalance')
    peak=max(e['owned_upper'] for e in events)
    return dict(global_nodes=n,global_factors=f,pair_nodes=p,pair_factors=q,
                roots=roots,maps=maps,sparse_roots=8*desc,frame=t,
                vector_peak=2*a+4*b,owned_upper=peak,arena_headroom=ARENA-peak,
                owned_upper_fits=peak<=ARENA,block_count_upper=1+3*v['views']+4+8+1+6,
                events=events,actual_peak=None,full_peak_rss_upper=None)


def minimum_support_work(inputs,endpoints,scoped):
    """Every input factor exists/in-scope even if its coefficient is zero.

    Counts intentionally omit known-present payload reads and all other work.
    Full-present endpoint evidence is necessary for complete positive closure.
    """
    positive(inputs,endpoints)
    if type(scoped) is not bool: raise ValueError('explicit consumer selection')
    get=(2+1) if scoped else 1  # scope fwd/reverse + exact-map presence
    put=(2+1+1) if scoped else 1  # scope fwd/reverse + exact + data write
    merge=1+get+get+1+put  # loop visit, two gets, checked add, set
    norm=get+1+1  # get, loop visit, checked group accumulation
    unit=merge+2*norm
    return dict(shared_input_factors=inputs,required_endpoints_min=endpoints,
                get_floor=get,set_floor=put,merge_floor=merge,one_norm_floor=norm,
                per_input_endpoint_floor=unit,work_lower=unit*inputs*endpoints,
                exceeds_work_cap=unit*inputs*endpoints>WORK,
                requires_all_endpoint_candidate_pairs_present=True,
                omitted=['source','planning','scope construction','pullbacks','native candidates',
                         'dual rows','private factors','payload reads','scale-group final sum','publication'])


def analyze():
    verify_files(); v=json.loads((ROOT/'docs/hz_full_resource_ledger_20261014.json').read_bytes())['inventory']
    if (v['inputs'],v['experts'],v['classes'],v['endpoints_min'],v['endpoints_upper'])!=(3072,8,10,252,504):
        raise ValueError('registered topology binding')
    s=shape(v)
    return dict(schema='PAIR_SCOPED_IMPLEMENTATION_CAPACITY_STATIC_V1',date='2026-10-15',
        based_on='d2c7a6f68184c7773ba3f2496db03398a9156ff4',anchors=ANCHORS,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),layout=layout(),inventory=v,scope=s,
        conservative_global_arena=arena(v,s),tighter_global_arena=arena(v,s,global_factors=s['tighter_global_factors']),
        any_admitted_frame_arena=arena(v,s,frame_override=FRAME),
        native_stream=candidate_stream(v,0),general_stream=candidate_stream(v,1),
        support_work_floor={name:minimum_support_work(v['inputs'],v['endpoints_min'],flag)
                            for name,flag in (('dense_reference',False),('pair_scoped',True))},
        scope_rebuilds_upper=v['pair_local_tiles_upper'],old_pair_once_count=v['pairs'],
        upstream=dict(memberships_min=v['mandatory_roots'],membership_cap=65536,
                      first_weight_visit_lower=2*v['first_weight_positions']),
        unchanged_caps=dict(arena=ARENA,frame=FRAME,candidates=CANDIDATES,work=WORK,seconds=300),
        decision='STOP_CURRENT_FULL_SHAPE_EXECUTION_INDEPENDENT_CONSUMER_WORK_REFUSAL',
        complete_work_upper=None,complete_rss_upper=None,new_quota=None,complete_cost_closed=False,
        execution_admitted=False,production_changed=False,source_builds=0,model_runs=0,gpu_runs=0,
        solver_calls=0,large_allocations=0,
        next='Separate complete source/evidence representation and common resource-policy decision; no quota from a lower bound')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-report',type=Path)
    args=parser.parse_args(); report=analyze()
    if args.check_report:
        if json.loads(args.check_report.read_bytes())!=report: raise ValueError('pair capacity report drift')
        print('PASS: owned payload upper; independent Work refusal; no runtime admission')
    else: print(json.dumps(report,sort_keys=True,indent=2))
