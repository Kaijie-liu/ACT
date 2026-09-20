"""Rebuild manuscript tables from committed reviews, without models or solvers.

This checks table accounting, not the truth of archived SAFE bounds. It never
follows private paths embedded in reviews and uses only the standard library.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
REVIEWS = 'act/pipeline/moe/results/'
SOURCES = {
    'confirmation': REVIEWS + 'schedule_confirmation_100_review_20260914_r1.json',
    'external': REVIEWS + 'external_pair_comparison_review_20260914_r1.json',
    'conv': REVIEWS + 'conv_full_v2_review_20260915.json',
    'evidence': 'docs/general_evidence_execution_v1_results.json',
}
HZ = 'HZ_POLICY_ACCEPTED'
CROWN = 'CROWN_NUMERICAL_FILTER'
RATIONAL = 'CHECKED_RATIONAL_CONDITIONAL'


def require(value, message):
    if not value:
        raise ValueError(message)


def outcome(cohort, model, arm, grade, states, n, seconds):
    allowed = {'SAFE', 'UNSAFE', 'UNKNOWN', 'TIMEOUT'} if grade == HZ else {
        'POSITIVE', 'UNSAFE', 'UNKNOWN', 'TIMEOUT'}
    require(set(states) <= allowed, 'unexpected state/evidence grade')
    require(all(type(v) is int and v >= 0 for v in states.values()), 'invalid count')
    require(sum(states.values()) == n, 'incomplete denominator')
    require(math.isfinite(seconds) and seconds >= 0, 'invalid cost')
    return dict(cohort=cohort, model=model, arm=arm, grade=grade, n=n,
                positive=states.get('SAFE' if grade == HZ else 'POSITIVE', 0),
                unsafe=states.get('UNSAFE', 0), unknown=states.get('UNKNOWN', 0),
                timeout=states.get('TIMEOUT', 0), mean_seconds=seconds)


def aggregate(records, expected, cohort, model, arm, grade):
    require(len(records) == expected, 'missing/extra requests')
    require(len({r['index'] for r in records}) == expected, 'duplicate input')
    require(all(math.isfinite(r['seconds']) and r['seconds'] >= 0 for r in records),
            'invalid individual cost')
    return outcome(cohort, model, arm, grade, Counter(r['status'] for r in records),
                   expected, statistics.mean(r['seconds'] for r in records))


def build(documents):
    rows = []
    d = documents['confirmation']
    require(d['independent_review']['status'] == 'PASS' and
            not d['independent_review']['issues'], 'confirmation review failed')
    a = d['full']['audit']
    require(a['status'] == 'PASS' and not a['issues'] and a['rows'] == 900,
            'confirmation audit failed')
    require(set(a['models']) == {'seed0', 'seed1', 'seed2'}, 'model inventory')
    for model, data in sorted(a['models'].items()):
        require(set(data['methods']) == {'adaptive', 'matched', 'legacy'}, 'arm inventory')
        for arm in ('adaptive', 'matched', 'legacy'):
            m = data['methods'][arm]
            rows.append(outcome('MLP confirmation', model, arm, HZ,
                                m['states'], 100, m['mean_observed_seconds']))
        for arm in ('matched', 'legacy'):
            for metric in ('SAFE', 'solved'):
                c = data['contrasts'][arm][metric]
                gain, loss = c['gained'], c['lost']
                require(c['denominator'] == 100 and len(set(gain)) == len(gain)
                        and len(set(loss)) == len(loss) and not set(gain) & set(loss)
                        and all(type(v) is int and 0 <= v < 100 for v in gain + loss),
                        'invalid paired ranks')
                def count(method):
                    s = data['methods'][method]['states']
                    return s.get('SAFE', 0) + (s.get('UNSAFE', 0) if metric == 'solved' else 0)
                require(count('adaptive') - count(arm) == c['net'] == len(gain)-len(loss),
                        'paired delta mismatch')
    for key in ('external', 'conv', 'evidence'):
        require(documents[key]['status'] == 'PASS' and not documents[key]['issues'],
                key + ' review failed')
    ext = documents['external']['full_requests']
    require(len(ext) == 30 and {r['model'] for r in ext} == {'seed0', 'seed1', 'seed2'},
            'external inventory')
    for model in ('seed0', 'seed1', 'seed2'):
        for arm, grade in (('adaptive', HZ), ('crown', CROWN)):
            records = [dict(index=r['dataset_index'], status=r[arm], seconds=r[arm+'_seconds'])
                       for r in ext if r['model'] == model]
            rows.append(aggregate(records, 10, 'MLP external', model, arm, grade))
    conv = documents['conv']['terminals']
    require(len(conv) == 90 and {r['method'] for r in conv} == {'adaptive', 'monolithic', 'crown'},
            'convolution inventory')
    conv_indices = {r['dataset_index'] for r in conv}
    require(len(conv_indices) == 30, 'different convolution cohorts across arms')
    for arm, grade in (('adaptive', HZ), ('monolithic', HZ), ('crown', CROWN)):
        subset = [r for r in conv if r['method'] == arm]
        require(all(r['evidence_level'] == grade for r in subset), 'mixed evidence grades')
        rows.append(aggregate([dict(index=r['dataset_index'], status=r['status'],
                                    seconds=r['wall_seconds']) for r in subset],
                              30, 'Conv V2', 'seed17', arm, grade))
    ev = documents['evidence']
    require(len(ev['terminal_bindings']) == 60, 'evidence terminal inventory')
    require({r['arm'] for r in ev['terminal_bindings']} == {'matched', 'evidence', 'crown'}
            and len({r['dataset_index'] for r in ev['terminal_bindings']}) == 20,
            'different evidence cohorts across arms')
    require(set(ev['summary']['methods']) == {'matched', 'evidence', 'crown'}, 'evidence arms')
    for arm, grade in (('matched', HZ), ('evidence', RATIONAL), ('crown', CROWN)):
        m = ev['summary']['methods'][arm]
        require(m['positive_evidence_level'] == grade and m['denominator'] == 20,
                'evidence contract mismatch')
        records = [r for r in ev['terminal_bindings'] if r['arm'] == arm]
        require(len(records) == 20 and len({r['dataset_index'] for r in records}) == 20,
                'evidence missing/duplicate terminal')
        require(Counter(r['status'] for r in records) == m['states'], 'terminal count mismatch')
        rows.append(outcome('Conv evidence', 'seed17', arm, grade, m['states'], 20, m['mean_seconds']))
    return rows


def render(rows, hashes):
    lines = ['# Rebuilt main outcome tables', '',
             'Generated from committed reviews; accounting reconstruction, **not independent reproof**.',
             'Positive columns retain their evidence grade. Costs include unsuccessful requests.',
             'Cohorts are separate; model–input pairs are not independent images. No pooled success rate.', '',
             '| Cohort | Model | Arm | Positive evidence grade | N | Positive | UNSAFE | UNKNOWN | TIMEOUT | Mean seconds |',
             '|---|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| {cohort} | {model} | {arm} | {grade} | {n} | {positive} | {unsafe} | {unknown} | {timeout} | {mean_seconds:.2f} |'.format(**r))
    lines += ['', '## Committed source identities', '',
              'No raw checkpoints, private paths or numerical libraries are read. Confirmation counts',
              'come from archived per-model aggregates with paired-delta checks; external/Conv V2',
              'counts are recomputed from committed rows. This is not a raw-run re-audit.', '']
    for name, digest in sorted(hashes.items()):
        lines.append(f'- `{SOURCES[name]}` — SHA-256 `{digest}`')
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='reject a stale committed table')
    args = parser.parse_args()
    data = {k: (ROOT / p).read_bytes() for k, p in SOURCES.items()}
    rows = build({k: json.loads(v) for k, v in data.items()})
    result = render(rows, {k: hashlib.sha256(v).hexdigest() for k, v in data.items()})
    if args.check:
        require((ROOT/'paper/results/main_tables.md').read_text() == result, 'stale main tables')
        print('PASS: main tables match committed evidence and accounting')
    else:
        print(result, end='')


if __name__ == '__main__':
    main()
