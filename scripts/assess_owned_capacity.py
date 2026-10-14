"""Static refusal analysis of the implemented optional workspace and frames.

Only pinned source text and frozen topology metadata are read. No production
imports, large arrays, tensors, source construction or candidate generation.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import struct

ROOT=Path(__file__).resolve().parents[1]
ANCHORS={
    'scoped_source/bounded_runtime.py':'8f4867b885c890890b097cb87e0cf8f84fe60e3b7c6b366c929b3a5839c9de2b',
    'scoped_source/direct_candidates.py':'3cbebd0a894b4db304c6d71d353c5b9d23ce7dbebc7367475fdeeb5a84ae9150',
    'scoped_source/scaled_endpoints.py':'e8bd15830187dfe376231fc44fc780e09dc632eade411b6327b34ffffd9276ff',
    'scoped_source/forest_endpoints.py':'86396e9c662ba8c91f6ac3921dc7f410d9b977937df38a9fb782f82991c50a4c',
    'scoped_source/check_source_dag.py':'58e5e2158843631c9e6ed324cfeeb2eb38e51df1205d2e14d1ed8ac8ce634458',
    'scoped_source/scaled_references.py':'534761cad44cbdf62ed31ee4d999fce0a5bd51aed4461b98d8f986b75d50d75d',
    'scoped_source/affine_references.py':'0f0d10033640210e6ce79642fa340a61c6e4293cb76779e4851a1f0f5cf582d8',
    'scoped_source/source_dag.py':'38863bafe654242077732bac9d9862d928ceb62eb5a77dcc44991cf2fab934e0',
    'scoped_source/hz_request_store.py':'7cb2be7647039f59d3c896b7856f634f43be9c532d9d2fd43632be09ab3230e0',
    'scoped_source/wire_forest.py':'981385019f93f97d41c02eab245cc21e8c430ccdfc8e0f853d5ffacb6981a3b3',
    'docs/hz_full_resource_ledger_20261014.json':'04c51e4765ada9445dcdd80f63f745459366154037443a68dd565c093acbe9d2',
    'docs/hz_bounded_runtime_controls_20261014.json':'f5432a1e22c765c772b349e7fafdeaedfeab25054dbf0a2b63a6bb78eb9bd641',
}
ARENA=128*2**20
TILE=4*2**20
TOTAL=16*2**20-8192


def verify_files():
    for name,expected in ANCHORS.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('implemented capacity anchor drift: '+name)


def layout():
    def size(path,name):
        tree=ast.parse((ROOT/path).read_bytes())
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
                call=node.value
                if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                        and isinstance(call.func.value,ast.Name) and call.func.value.id=='struct'
                        and call.func.attr=='Struct' and len(call.args)==1):
                    return struct.calcsize(ast.literal_eval(call.args[0]))
        raise ValueError('fixed struct declaration absent')
    header=size('scoped_source/bounded_runtime.py','HEADER')
    return dict(scalar_header=header,working_rational_slot=1+header+2*1024,
                stored_rational=header+2*512,root=size('scoped_source/bounded_runtime.py','ROOT'),
                prefix=size('scoped_source/direct_candidates.py','PREFIX'),
                record=size('scoped_source/direct_candidates.py','RECORD'))


def positive(*values):
    if any(type(v) is not int or v<=0 for v in values): raise ValueError('positive bounded counts')


def frame_bytes(m,k,profile):
    positive(m,k)
    if k>8 or type(profile) is not int or profile not in (0,1): raise ValueError('tile/profile scope')
    d=layout(); scalar=8 if profile==0 else d['stored_rational']
    return d['prefix']+k*(d['record']+2*d['stored_rational']+2*m*scalar)


def candidate_envelope(v,width,profile):
    if type(width) is not int or width not in (1,8): raise ValueError('registered widths 1/8')
    per_pair=2*(v['classes']-1)
    chunks=[min(width,per_pair-i) for i in range(0,per_pair,width)]
    sizes=[frame_bytes(v['pair_dual_rows_upper'],k,profile) for k in chunks]
    total=v['pairs']*sum(sizes)
    return dict(per_pair_tile_widths=chunks,tiles=v['pairs']*len(chunks),
                endpoints=v['endpoints_upper'],roles=v['candidates_upper'],
                largest_tile=max(sizes),all_frames=total,
                tile_upper_fits=max(sizes)<=TILE,total_upper_fits=total<=TOTAL,
                actual_bytes=None,upper_overflow_proves_actual_overflow=False)


def workspace(n,f,h,m,k,profile=0):
    """Counterfactual allocation transcript in integer bytes, not allocations.

    Counts may be substituted with proved lower/upper bounds by the caller.
    Mathematically continuing after the first refused event describes requested
    capacity only; the real Owner refuses before making that allocation.
    Consumption assumes at least one candidate is present; preparation does not.
    """
    positive(n,f,h,m,k)
    if k>8: raise ValueError('bounded tile width')
    d=layout(); root=d['root']*n+4*h
    a=d['working_rational_slot']*n; b=d['working_rational_slot']*f
    events=[]; live=root
    def event(label,delta):
        nonlocal live
        live+=delta
        if live<0: raise ValueError('invalid ownership transcript')
        events.append(dict(event=label,delta=delta,requested_live_bytes=live,exceeds_arena=live>ARENA))
    event('root index and all ordered memberships',0)
    for i in range(k):
        event(f'prepare {i+1} original objective alpha',a)
        event(f'prepare {i+1} pullback alpha copy',a)
        event(f'prepare {i+1} objective factor vector',b)
        event(f'prepare {i+1} release pullback alpha',-a)
    prepare_peak=max(e['requested_live_bytes'] for e in events); prepared=live
    t=frame_bytes(m,k,profile)
    event('candidate fill',t); event('candidate seal immutable copy',t); event('release fill buffer',-t)
    event('consume original dual alpha vectors',k*a)
    event('consume pullback alpha copies',k*a)
    event('consume dual factor vectors',k*b)
    event('release dual pullback alpha copies',-k*a)
    event('clear original objective and dual vectors',-k*(2*a+2*b))
    event('release candidate bytes',-t)
    if live!=root: raise ValueError('tile cleanup fails to retain only roots')
    refusal=next((e for e in events if e['exceeds_arena']),None)
    return dict(nodes=n,factors=f,memberships=h,pair_rows=m,width=k,
                root_index_bytes=root,node_map_bytes=a,factor_map_bytes=b,
                prepare_peak=prepare_peak,prepared=prepared,frame_bytes=t,
                seal_peak=prepared+2*t,consume_peak=root+k*(3*a+2*b)+t,
                requested_peak=max(e['requested_live_bytes'] for e in events),
                first_refusal=refusal,events=events,actual_peak=None,
                full_process_rss_upper=None,post_refusal_events_are_counterfactual=True,
                consumption_assumes_present_candidate=True)


def analyze():
    verify_files(); v=json.loads((ROOT/'docs/hz_full_resource_ledger_20261014.json').read_bytes())['inventory']
    if (v['inputs'],v['experts'],v['classes'],v['property_duties'],v['endpoints_upper'])!=(3072,8,10,252,504):
        raise ValueError('frozen registered topology')
    lower=dict(n=v['inputs']+v['affine_outputs']+v['guards'],f=v['inputs'],
               h=v['mandatory_roots'],m=2*(v['experts']-2))
    upper=dict(n=v['nodes_upper'],f=v['factors_upper'],h=v['roots_upper'],m=v['pair_dual_rows_upper'])
    methods={}
    for width in (1,8):
        lo=workspace(**lower,k=width); hi=workspace(**upper,k=width)
        methods[str(width)]=dict(native_frames=candidate_envelope(v,width,0),
            generic_frames=candidate_envelope(v,width,1),
            owned_lower=lo,owned_conditional_upper=hi,
            arena_refusal_proved_by_lower=lo['requested_peak']>ARENA,
            largest_minimum_pair_tile=min(width,v['classes']-1),
            full_request_work_upper=None,full_peak_rss_upper=None)
    return dict(schema='IMPLEMENTED_OWNED_CAPACITY_STATIC_V1',date='2026-10-14',
        based_on='a0c955cb1fcc3ad6494297c4e79c93ad128d96c9',anchors=ANCHORS,
        analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        layout=layout(),inventory=v,lower_counts=lower,upper_counts=upper,methods=methods,
        unchanged_caps=dict(arena=ARENA,tile=TILE,candidate_total=TOTAL,work=8000000,seconds=300),
        upstream=dict(mandatory_memberships=v['mandatory_roots'],membership_cap=65536,
            pair_aliases_alone=v['pair_output_roots'],necessary_refusal=True,
            builder_plus_checker_first_weight_visit_lower=2*v['first_weight_positions'],
            new_index_is_after_old_source_check=True),
        decision='STOP_FULL_SHAPE_CAPACITY_RETRY_CURRENT_DENSE_WORKSPACE',
        reason='upstream mandatory-root refusal; independently width8 owned preparation lower exceeds arena',
        cost_closed=False,new_quota=None,execution_admitted=False,production_changed=False,
        real_model_runs=0,source_builds=0,solver_calls=0,gpu_calls=0,large_allocations=0,
        unresolved=['direct source construction and source checks without old root/logical format',
                    'working-vector scope, sparse objective roots and simultaneous copies',
                    'complete candidate format under native/generic profiles',
                    'arithmetic scratch, Python metadata, native coexistence and full lifetime',
                    'complete archive, hard supervision and both-method cumulative costs'],
        next='separate representation/lifetime decision, not quota arithmetic or another full-size run')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--check-report',type=Path); args=p.parse_args()
    result=analyze()
    if args.check_report:
        if json.loads(args.check_report.read_bytes())!=result: raise ValueError('static capacity report drift')
        print('PASS: implemented byte/lifetime refusal; full cost not closed; no execution')
    else: print(json.dumps(result,sort_keys=True,indent=2))
