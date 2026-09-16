"""Saved-clock decomposition and static-path workload: no solver or proof replay."""
from collections import Counter, defaultdict
from pathlib import Path
import statistics

from portable_proof.runtime import digest
from single_check_portable.execution import ROOT, read, save_new
from upstream_portable.study import verify, OUTPUT

DEST=ROOT/'docs/upstream_generation_cost_v1.json'
ARCHIVE=ROOT/'docs/upstream_portable_v1_execution_results.json'
SOURCES=('moe_evidence/generate.py','evidence_handoff/proposal.py',
 'act/back_end/solver/lp_certificate.py','act/back_end/solver/sparse_lp_certificate.py',
 'act/back_end/solver/check_hz_lp_export.py','act/back_end/solver/rational_mccormick.py',
 'act/back_end/solver/check_rational_mccormick.py','scripts/optional_evidence_dev_contract.py',
 'exact_matrix_cache/checker.py')


def kind(key):
    if key.endswith('_weighted'):return 'weighted'
    if '_gate_' in key:return 'router_range'
    if key.endswith(('_lo','_hi')) and '_p' in key:return 'difference_range'
    raise ValueError('unknown query kind')


def windows(stage,queries):
    begin,end=stage['start_seconds'],stage['end_seconds']
    if stage['state']!='COMPLETED' or not begin<=end or abs(end-begin-stage['elapsed_seconds'])>1e-7:
        raise ValueError('invalid proposal stage')
    prev=begin;segments=[];totals=defaultdict(float);counts=Counter();seen=set()
    for i,q in enumerate(queries):
        if q['key'] in seen:raise ValueError('duplicate query')
        seen.add(q['key']);k=kind(q['key'])
        if 'seconds' not in q or q['status']=='PENDING':raise ValueError('censored query needs separate accounting')
        enter=q['entered_seconds'];finish=enter+q['seconds']
        if q['seconds']<0 or not prev<=enter<=finish<=end:raise ValueError('overlap or out-of-phase query')
        segments.append({'before_query':q['key'],'previous_query':queries[i-1]['key'] if i else None,
                         'seconds':enter-prev,'kind':'prefix' if not i else 'inter_query'})
        counts[k]+=1;totals[k]+=q['seconds'];prev=finish
    segments.append({'previous_query':queries[-1]['key'] if queries else None,
                     'before_query':None,'seconds':end-prev,'kind':'suffix'})
    logged=sum(totals.values());outside=sum(s['seconds'] for s in segments)
    if abs(logged+outside-stage['elapsed_seconds'])>1e-7:raise ValueError('cost identity does not close')
    return {'phase_seconds':stage['elapsed_seconds'],'query_window_seconds':logged,
            'outside_query_windows_seconds':outside,'outside_fraction':outside/stage['elapsed_seconds'],
            'query_counts_by_kind':dict(counts),'query_seconds_by_kind':dict(totals),
            'prefix_seconds':segments[0]['seconds'],'suffix_seconds':segments[-1]['seconds'],
            'inter_query_seconds':sum(s['seconds'] for s in segments if s['kind']=='inter_query'),
            'segments':segments,'native_solver_seconds':None,'exact_check_seconds':None,
            'construction_seconds':None,'serialization_seconds':None}


def readiness(manifest):
    by=defaultdict(list)
    for row in manifest['obligations']:
        if row['kind']!='residual':continue
        pair=row['pair'];prefix='s'+'_'.join(map(str,pair));i=row['property_index']
        names=[prefix+'_gate_lo',prefix+'_gate_hi',prefix+f'_p{i}_lo',prefix+f'_p{i}_hi']
        available=all(manifest['supports'][k]['status']=='PROPOSED' and
                      manifest['supports'][k]['certificate'] is not None for k in names)
        by[prefix].append({'property_index':i,'four_range_certificates_present':available,
                          'weighted_certificate_present':row.get('weighted_status')=='PROPOSED' and row.get('certificate') is not None})
    return dict(by)


def analyze():
    verify();archive=read(ARCHIVE);results=[]
    for name,sha in archive['artifact_sha256'].items():
        if digest((ROOT/name).read_bytes())!=sha:raise ValueError('archived artifact changed: '+name)
    for r in archive['rows']:
        base=OUTPUT/r['job_id'];src=base/'source';m=read(src/'manifest.json')
        queries=read(src/'query_log.json');stage=read(base/'propose_stage.json')
        data=windows(stage,queries);ready=readiness(m)
        support_queries=[q for q in queries if kind(q['key'])!='weighted']
        weighted_queries=[q for q in queries if kind(q['key'])=='weighted']
        files=list(src.glob('*_weighted.export.json'))
        generated_names={p.name for p in files};requested_names={q['key']+'.export.json' for q in weighted_queries}
        # Lower bounds on actual reads/writes from completed code paths, not disk bandwidth measurements.
        support_read_bytes=sum((src/m['supports'][q['key']]['export']['file']).stat().st_size for q in support_queries)
        joint_reads=Counter()
        for q in weighted_queries:
            pair_key=q['key'].split('_p')[0];joint_reads[m['contexts'][pair_key]['joint_source']['file']]+=1
        known_joint_bytes=sum((src/name).stat().st_size*n for name,n in joint_reads.items())
        unique_joint_bytes=sum((src/name).stat().st_size for name in joint_reads)
        queried_export_bytes=sum((src/name).stat().st_size for name in requested_names)
        # Every returned PROPOSED sparse proposal evaluates twice internally; post-check once.
        proposed=sum(q['status']=='PROPOSED' for q in queries)
        if proposed!=len(queries):raise ValueError('static call count needs failure-specific branch')
        data.update(job_id=r['job_id'],dataset_index=r['dataset_index'],arm=r['arm'],status=r['status'],
            query_status_counts=dict(Counter(q['status'] for q in queries)),
            grant_seconds_min=min(q['granted_seconds'] for q in queries),
            grant_seconds_max=max(q['granted_seconds'] for q in queries),
            source_support_status=dict(Counter(v['status'] for v in m['supports'].values())),
            handoff_reason=read(src/'handoff.json')['reason'],
            available_range_obligations=sum(v['four_range_certificates_present'] for group in ready.values() for v in group),
            weighted_certificate_obligations=sum(v['weighted_certificate_present'] for group in ready.values() for v in group),
            residual_obligations=sum(len(group) for group in ready.values()),readiness=ready,
            static_completed_path_counts={'export_structure_checks':2*len(support_queries),
                 'weighted_construction_checks':2*len(weighted_queries),
                 'weighted_builds_at_least':len(files),
                 'exact_sparse_dual_evaluations':3*proposed,
                 'query_log_writes':2*len(queries),'completed_certificate_writes':proposed,
                 'completed_manifest_writes':len(queries)},
            logical_bytes={'support_export_reads_at_least':support_read_bytes,
                 'joint_source_reads_for_entered_weighted':known_joint_bytes,
                 'unique_joint_sources_for_entered_weighted':unique_joint_bytes,
                 'queried_weighted_export_writes':queried_export_bytes,
                 'all_weighted_exports_on_disk':sum(p.stat().st_size for p in files),
                 'all_certificate_files_on_disk':sum(p.stat().st_size for p in src.glob('*.certificate.json'))},
            weighted_exports_without_entered_query=sorted(generated_names-requested_names),
            observed_file_sizes={p.name:p.stat().st_size for p in files},
            phase_sha256=digest((base/'propose_stage.json').read_bytes()),
            manifest_sha256=digest((src/'manifest.json').read_bytes()),
            query_log_sha256=digest((src/'query_log.json').read_bytes()))
        results.append(data)
    phase=sum(r['phase_seconds'] for r in results);inside=sum(r['query_window_seconds'] for r in results)
    return {'schema':'UPSTREAM_GENERATION_COST_V1','status':'PASS','rows':results,
        'aggregate':{'phase_seconds':phase,'query_window_seconds':inside,
                     'outside_query_windows_seconds':phase-inside,'outside_fraction':(phase-inside)/phase},
        'timing_limits':'query windows include preparation, exact arithmetic and I/O; gaps do not separate checking/construction/serialization; no native timing recorded',
        'static_count_limits':'counts inferred for completed successful paths, not measured profiler counts; unentered final work may add checks/builds',
        'byte_limits':'logical file sizes/repeated-read lower bounds, not physical I/O or measured serialization seconds',
        'parent_archive_sha256':digest(ARCHIVE.read_bytes()),
        'reviewed_source_sha256':{n:digest((ROOT/n).read_bytes()) for n in SOURCES},
        'new_model_calls':0,'new_solver_calls':0,'proof_checker_replays':0,'original_budget_changed':False}


if __name__=='__main__':
    import json
    if DEST.exists():raise FileExistsError('immutable analysis; no overwrite')
    result=analyze();save_new(DEST,result)
    print(json.dumps({'aggregate':result['aggregate'],'rows':[{k:r[k] for k in (
        'job_id','phase_seconds','query_window_seconds','outside_query_windows_seconds',
        'available_range_obligations','weighted_certificate_obligations','weighted_exports_without_entered_query','logical_bytes')}
        for r in result['rows']]},indent=2))
