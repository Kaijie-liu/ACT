"""Saved-record coverage/identity audit, NOT independent reproof of HZ bounds."""
import argparse
import json
from pathlib import Path
from recent_moe_deployment import sha256


def check_record(cfg, config_hash, request, receipt, result, terminal):
    if not receipt['source_unchanged'] or receipt['source_before']['head'] != cfg['commit']:
        raise ValueError('source identity')
    if receipt['deadline_seconds'] != cfg['request_seconds'] or not receipt['cpu_only']:
        raise ValueError('budget/device')
    if receipt['total_with_postflight_seconds'] < receipt['execution_including_preflight_seconds']:
        raise ValueError('cost accounting')
    if receipt['status'] != 'COMPLETED':
        if terminal['status'] != receipt['status'] or terminal['evidence_grade'] != 'NONE':
            raise ValueError('late/partial result accepted')
        return
    if (result is None or result['config_sha256'] != config_hash or result['request'] != request or
            terminal['status'] != result['status'] or terminal['evidence_grade'] != result['evidence_grade']):
        raise ValueError('request/terminal binding')
    if result['status'] == 'POSITIVE':
        cand, excluded = result['candidates'], result['excluded']
        obligations = result['nonzero_obligations']
        if (not cand or result['unresolved'] or set(cand) & set(excluded) or
                set(cand) | set(excluded) != set(range(len(result['class_counts']))) or
                len(obligations) != len(cand) or {r['expert'] for r in obligations} != set(cand) or
                set(result['expert_statuses']) != {str(i) for i in cand} or
                any(s != 'certified' for s in result['expert_statuses'].values()) or
                any(not r['accepted'] or not (r['lower'] <= r['upper'] and
                      (r['lower'] > 0 or r['upper'] < 0)) for r in obligations) or
                result['evidence_grade'] != 'HZ_POLICY_ACCEPTED' or result['source_complete']):
            raise ValueError('incomplete positive obligation set')
    elif result['status'] == 'UNSAFE_REPLAYED':
        if not result.get('witness') or result['evidence_grade'] != 'FULL_MODEL_REPLAY':
            raise ValueError('missing complete-model witness')
    elif result['status'] not in {'UNKNOWN', 'TIMEOUT'} or result['evidence_grade'] != 'NONE':
        raise ValueError('unexpected status/grade')
    if result['property_rows'] != sum(result['class_counts']) - 1:
        raise ValueError('global classification properties missing')


def collect(config):
    cfg = json.loads(config.read_text())
    for p, h in cfg['files'].items():
        if sha256(p) != h:
            raise ValueError('frozen source/input changed')
    root = Path(cfg['output_root'])
    summary = json.loads((root / 'summary.json').read_text())
    if summary['config_sha256'] != sha256(config) or len(summary['terminals']) != len(cfg['requests']):
        raise ValueError('batch denominator/binding')
    records = []
    for request, terminal in zip(cfg['requests'], summary['terminals']):
        folder = root / request['id']
        receipt = json.loads((folder / 'receipt.json').read_text())
        result = json.loads((folder / 'result.json').read_text()) if (folder / 'result.json').exists() else None
        check_record(cfg, sha256(config), request, receipt, result, terminal)
        for stream in ['stdout', 'stderr']:
            if sha256(folder / f'{stream}.txt') != receipt[f'{stream}_sha256']:
                raise ValueError('log changed')
        records.append({'terminal': terminal, 'result': result,
            'cost_seconds': {k: receipt[k] for k in ['execution_including_preflight_seconds', 'total_with_postflight_seconds']},
            'record_hashes': {str(p): sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}})
    return {'audit': 'PASS', 'scope': 'record identities and coverage only; no independent HZ reproof',
            'config_sha256': sha256(config), 'records': records}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    args = p.parse_args()
    value = collect(args.config)
    if args.check:
        if value != json.loads(args.output.read_text()):
            raise ValueError('archive differs')
    else:
        with args.output.open('x') as f:
            json.dump(value, f, indent=2, allow_nan=False)
            f.write('\n')
    print('MetaMoE saved-record audit PASS')
