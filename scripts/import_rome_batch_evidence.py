"""Import compact independent archive; separate clean errors from attack gains.

No new model evaluation, attack, data download or missing-result imputation.
Historical ATTACK_FOUND union labels remain unchanged in the parent artifact.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
from recent_moe_deployment import sha256
from robust_experts_workflow_control import write


def kind(row):
    if row['status'] != 'ATTACK_EVALUATION_COMPLETED':
        return 'INCOMPLETE_NOT_ROBUST'
    correct = row['adversarial_prediction'] == row['label']
    if correct != row['empirically_correct_after_attack']:
        raise ValueError('classification ledger inconsistent')
    if row['clean_prediction'] != row['label']:
        return 'PREEXISTING_CLEAN_ERROR'
    return 'NO_BREAK_FOUND_COMPLETED' if correct else 'PERTURBATION_BREAK_REPLAYED'


def run(config,source,audit_root):
    cfg=json.loads(config.read_text());archive=json.loads(source.read_text())
    if archive['audit']!='INDEPENDENT_EMPIRICAL_BATCH_REVIEW_PASS' or archive['config_sha256']!=sha256(config):
        raise ValueError('not the frozen batch audit')
    for file,digest in cfg['files'].items():
        if sha256(file)!=digest:raise ValueError('source changed')
    if sha256(Path(cfg['output_root'])/'summary.json')!=archive['raw_summary_sha256']:
        raise ValueError('raw terminal summary changed')
    if len(archive['rows'])!=12 or len(cfg['requests'])!=12:
        raise ValueError('incomplete registered roster')
    rows=[];clean={}
    for i,(row,request) in enumerate(zip(archive['rows'],cfg['requests'])):
        if sha256(audit_root/f'audit{i:02d}.json')!=row['audit_sha256']:
            raise ValueError('independent record changed')
        request_path=Path(request['config']);r=json.loads(request_path.read_text())
        if (r['index'],r['norm'])!=(row['index'],row['norm']):raise ValueError('request identity')
        prepared=Path(r['output_root'])/'prepared.json'
        p=json.loads(prepared.read_text())
        if p['config_sha256']!=sha256(request_path) or p['index']!=row['index']:
            raise ValueError('prepared telemetry binding')
        value={'label':p['label'],'clean_prediction':p['clean_prediction'],'input_sha256':p['input_sha256']}
        if r['index'] in clean and clean[r['index']]!=value:raise ValueError('cross-norm clean mismatch')
        clean[r['index']]=value
        if row['status']=='ATTACK_EVALUATION_COMPLETED' and (
                row['label']!=p['label'] or row['clean_prediction']!=p['clean_prediction']):
            raise ValueError('prepared/completed discrepancy')
        rows.append({'index':row['index'],'norm':row['norm'],'interpretation':kind(row),
            'parent_status':row['status'],'prepared_sha256':sha256(prepared)})
    counts=Counter(row['interpretation'] for row in rows)
    report={'scope':'derived interpretation only; unchanged denominator12norm requests/4inputs',
        'parent_archive_sha256':sha256(source),'counts':dict(counts),'rows':rows,
        'clean_telemetry':clean,'clean_telemetry_correct':sum(v['label']==v['clean_prediction'] for v in clean.values()),
        'completed_clean_errors_are_not_new_attacks':True,
        'timeout_is_not_robust':True,'configured_attack_list_is_not_executed_attack_count':True,
        'formal_SAFE':False,'ACT_comparison_score':None,
        'trust':'completed witnesses independently replayed; timeout clean outputs are identity-bound telemetry only'}
    write('docs/rome_multinorm_archive_20260922_r1.json',archive)
    write('docs/rome_multinorm_interpretation_20260922_r1.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--audit-root',type=Path,required=True)
    a=p.parse_args();run(a.config,a.source,a.audit_root)
