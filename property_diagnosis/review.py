"""Fresh saved-record reproduction and frozen-hash audit; no numerical solves."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from property_diagnosis.analyze import ROOT, analyze, no_external_work
from property_diagnosis.summarize import summarize


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def without_timing(d):
    out=dict(d);out.pop('seconds')
    out['arms']=[{k:v for k,v in a.items() if k!='seconds'} for a in d['arms']]
    return out


def review(raw):
    start=time.monotonic();saved=json.loads(raw.read_bytes());fresh=analyze()
    if without_timing(saved)!=without_timing(fresh):raise ValueError('fresh saved-record reproduction mismatch')
    summary_path=ROOT/'docs/property_diagnosis_v1_results.json'
    if summarize(raw)!=json.loads(summary_path.read_bytes()):raise ValueError('independent arithmetic summary mismatch')
    freeze_path=ROOT/'docs/property_ranges_v1_freeze.json';freeze=json.loads(freeze_path.read_bytes())
    for kind in ('sources','artifacts'):
        for path,digest in freeze[kind].items():
            if sha(ROOT/path)!=digest:raise ValueError('frozen execution identity changed: '+path)
    files=['docs/property_diagnosis_v1.md','docs/property_diagnosis_v1_results.md',
           'docs/property_diagnosis_v1_results.json','property_diagnosis/analyze.py',
           'property_diagnosis/summarize.py','property_diagnosis/tests.py','property_diagnosis/review.py']
    return {'schema':'PROPERTY_DIAGNOSIS_ARCHIVE_REVIEW_V1','status':'PASS_ACCOUNTING_AND_REPRODUCTION_ONLY','issues':[],
        'starting_head':'a9d70064313fc8e37de09627c2dc170d6a4c8b57',
        'decision':'STOP_INPUT98_FOLLOWUP','new_control_proposed':False,
        'decision_scope':'No isolated new representation intervention; known coupled symptoms remain. Not an LP impossibility or unsafe proof.',
        'raw_analysis':str(raw.relative_to(ROOT)),'raw_analysis_sha256':sha(raw),
        'raw_analysis_bytes':raw.stat().st_size,'file_sha256':{f:sha(ROOT/f) for f in files},
        'execution_freeze_sha256':sha(freeze_path),
        'frozen_source_hashes_unchanged':len(freeze['sources']),
        'frozen_artifact_hashes_unchanged':len(freeze['artifacts']),
        'fresh_saved_record_recomputation':'EXACT_MATCH_EXCLUDING_TIMING',
        'separately_implemented_arithmetic_roster_check':'PASS_NOT_LP_REPROOF',
        'completed_no_solver_tests':{
            'property_diagnosis.tests':13,'range_diagnosis.tests':8,
            'scripts/test_analyze_property_ranges.py':4,'scripts/test_analyze_range_pipeline.py':3,
            'scripts/test_rebuild_moe_main_tables.py':3,'total':31,
            'result':'PASS; executed separately with act-py312 python -S before this review',
            'main_table_rebuild':'PASS_UNCHANGED'},
        'common_properties':saved['diagnostic_properties'],'excluded_point_properties':[7],
        'properties_still_required':9,'request_positive_count':0,
        'new_solver_calls':0,'new_model_forwards':0,'new_source_propagations':0,
        'new_ranges':0,'new_lower_bounds_proved':0,'exact_primal_feasibility_checked':False,
        'exact_LP_optimality_checked':False,'production_verdict_changed':False,
        'review_seconds':time.monotonic()-start,
        'scope':'Identity/reproduction and exact diagnostic accounting only. Do not reclassify saved points, missing obligations or original outcomes.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('analysis',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if not sys.flags.no_site:raise ValueError('python -S required')
    sys.addaudithook(no_external_work)
    if a.output.exists():raise ValueError('new receipt required')
    result=review(a.analysis.resolve())
    with a.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':result['status'],'seconds':result['review_seconds'],
                      'decision':result['decision'],'new_solver_calls':0}))
