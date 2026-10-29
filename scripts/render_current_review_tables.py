"""Reconstruct current review tables from committed summaries, without solvers.

This checks accounting and produces LaTeX; it does not reprove numerical bounds.
Historical submission_tables.tex and its renderer remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'paper/results/current_review_tables.tex'


def read(name):
    return json.loads((ROOT/'docs'/name).read_text())


def render():
    primary = read('moe_aware_b4_records_20261027.json')['main_and_ablation_review']['main']
    multilayer = read('multilayer_stage1_records_20261022.json')['final_result_20261025']
    revised = read('guard_dual_refinement_result_20261029.json')
    radius = read('moe_aware_b4_records_20261027.json')['radius_final_review']
    c = primary['contrast']
    assert c['complete'] and c['denominator'] == 300 and not c['missing']
    assert all(sum(states.values()) == 300 for states in c['states'].values())
    assert len(c['gained'])-len(c['lost']) == c['net']
    assert multilayer['full_calls'] == 600 and multilayer['unexecuted'] == 0
    assert revised['completed_calls'] == 22 and len(revised['pair_details']) == 11
    rows = []
    for label, value in (('primary', primary), ('multilayer', multilayer), ('guard', revised), ('radius', radius)):
        digest = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        rows.append('% Relevant committed summary '+label+' sha256 '+digest)
    def table(caption, label, columns, header, body):
        return [r'\begin{table}[ht]', r'\centering\small', r'\caption{'+caption+'}',
            r'\label{'+label+'}', r'\begin{tabular}{'+columns+'}', r'\toprule',
            header+r' \\', r'\midrule', *[s+r' \\' for s in body],
            r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    rows += table('Complete-request cascade: 300 calls per arm. P is numerical POSITIVE, not a strict certificate. All terminals and costs are retained.',
        'tab:primary', 'lrrrrrr', r'Method & P & UNKNOWN & TIMEOUT & ERROR & Mean s & Median s',
        [' & '.join([name, *[str(c['states'][arm].get(s, 0)) for s in ('POSITIVE','UNKNOWN','TIMEOUT','ERROR')],
            f"{c['cost']['means'][arm]:.3f}", f"{c['cost']['medians'][arm]:.3f}"])
         for arm, name in (('off','Coupled'),('on','Cascade'))])
    rows += table('Family breakdown of the same primary cohort: 60 model--input objects per row. Gain/loss counts are paired, not differences between unmatched totals.',
        'tab:families', 'lrrrr', 'Family & Coupled P & Cascade P & Gained & Lost',
        [' & '.join([name, str(v['states']['off'].get('POSITIVE',0)), str(v['states']['on'].get('POSITIVE',0)),
            str(v['gained']), str(v['lost'])]) for name,v in primary['by_family'].items()])
    assert multilayer['multilayer_positives']['act'] == multilayer['multilayer_positives']['genbab_naive'] == 0
    rows += table('Separate multilayer bottleneck study: 60 objects per configuration/arm. The single-layer cost anchor is a separate pruning-only 60 calls.',
        'tab:multilayer','lrrr','Configuration & ACT v1 P & Naive P & Pruned P',
        [f"{name} & 0 & 0 & {multilayer['pruned_positives_by_configuration'][name]}" for name in ('L2E8','L3E8','L2E16')])
    sums = {}
    for arm in ('alpha','guard'):
        values = [d[arm] for d in revised['pair_details']]
        assert all(v is not None for v in values)
        sums[arm] = sum(v['closed'] for v in values)
    assert sums['guard']-sum(d['guard']['baseline_positive'] for d in revised['pair_details']) == len(revised['gained'])
    rows += table('Revised guard diagnostic on the same eleven histories. Each continuation retains its complete twenty-evaluation baseline. A local history is not a full MoE request.',
        'tab:guard','lrr','Continuation & Closed histories & UNKNOWN',
        [f"{name} & {sums[arm]} & {11-sums[arm]}" for arm,name in (('alpha','Extra alpha'),('guard','Joint alpha/guard'))])
    r = radius['contrast']
    assert r['complete'] and r['denominator'] == 120 and not r['missing']
    assert all(sum(states.values()) == 120 for states in r['states'].values())
    assert len(r['gained'])-len(r['lost']) == r['net']
    assert radius['reviewed_completed_calls'] == 240 and radius['primary_decision'] is False
    rows += table('Separate double-radius cohort at $4/255$: 120 calls per arm, the same six models and twenty input blocks. P is numerical POSITIVE; both arms have zero complete UNKNOWN, UNSAFE and ERROR terminals. Public counterexamples are not arm successes.',
        'tab:radius', 'lrrrr', 'Method & P & TIMEOUT & Mean s & Median s',
        [' & '.join([name, str(r['states'][arm].get('POSITIVE',0)), str(r['states'][arm].get('TIMEOUT',0)),
            f"{r['cost']['means'][arm]:.3f}", f"{r['cost']['medians'][arm]:.3f}"])
         for arm,name in (('off','Coupled'),('on','Cascade'))])
    return '\n'.join(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(); value = render()
    if args.check:
        assert OUTPUT.read_text() == value, 'current table drift'
        print('PASS: current tables reproduce committed accounting; no numerical reproof')
    else:
        print(value, end='')
