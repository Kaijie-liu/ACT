"""Render the completed S2a primary endpoint; no model or solver queries.

The compact export supports rendering without server-only evidence paths.
Source hashes identify the accepted primary records, not an independent proof.
Layer widths are descriptive, unprojected and on learned neuron scales.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ARMS = ('D-act', 'D-tot', 'M', 'M-reg')
LABELS = ('Dense active', 'Dense total', 'MoE + reg.', 'MoE no reg.')
COLORS = ('#276484', '#638858', '#b64945', '#c18b35')


def derive(path):
    raw = Path(path).read_bytes()
    source = json.loads(gzip.decompress(raw))
    if (source['decision'] != 'STOP_S2A_WRITE_RESULTS'
            or source['input_count'] != 10000 or source['paired_seeds'] != 5):
        raise ValueError('complete registered S2a primary endpoint required')
    if len(source['sources']) != 20:
        raise ValueError('all twenty accepted primary measurements required')
    details = []
    for record in source['sources']:
        name = record['id'].removeprefix('primary_')
        arm, seed = name.rsplit('_seed', 1)
        seed = int(seed)
        if arm not in ARMS or seed not in range(5):
            raise ValueError('unknown model')
        batches = record['batches']
        if ([(b['start'], b['stop']) for b in batches]
                != [(i, min(i+256, 10000)) for i in range(0, 10000, 256)]):
            raise ValueError('complete ordered batch coverage required')
        layers = {}
        for b in batches:
            if b['count'] != b['stop']-b['start']:
                raise ValueError('batch denominator')
            diagnostic = b['layer_diagnostics']
            if (diagnostic['schema'] != 'IBP_LAYER_WIDTHS_V1'
                    or diagnostic['count'] != b['count']):
                raise ValueError('frozen diagnostic contract')
            seen = set()
            for row in diagnostic['rows']:
                key = (row['network'], row['layer'], row['operator'], row['neurons'])
                if key in seen:
                    raise ValueError('duplicate layer row')
                seen.add(key)
                if not math.isfinite(row['mean_width']) or row['mean_width'] < 0:
                    raise ValueError('finite nonnegative width')
                unstable = row['relu_unstable_fraction']
                if unstable is not None and not 0 <= unstable <= 1:
                    raise ValueError('unstable fraction')
                target = layers.setdefault(key, dict(count=0, width=0., unstable=0.))
                target['count'] += b['count']
                target['width'] += b['count'] * row['mean_width']
                target['unstable'] += b['count'] * (unstable or 0.)
        if len(layers) != (5 if arm.startswith('D-') else 43):
            raise ValueError('all dense/router/expert layers required')
        normalized = []
        for (network, layer, operator, neurons), value in sorted(layers.items()):
            if value['count'] != 10000:
                raise ValueError('layer input coverage')
            normalized.append(dict(network=network, layer=layer, operator=operator,
                neurons=neurons, mean_width=value['width']/10000,
                relu_unstable_fraction=value['unstable']/10000 if operator == 'ReLU' else None))
        clean, certified = (sum(b[k] for b in batches) for k in ('clean', 'certified'))
        if (clean != source['arms'][arm]['clean_per_seed'][seed]
                or certified != source['arms'][arm]['certified_per_seed'][seed]):
            raise ValueError('seed/batch primary counts disagree')
        details.append(dict(arm=arm, seed=seed, count=10000, clean=clean, certified=certified,
            single_route=sum(b['single_route'] for b in batches) if arm in ('M', 'M-reg') else None,
            mean_margin_width=sum(b['count']*b['mean_margin_width'] for b in batches)/10000,
            request_seconds=record['request_seconds'],
            diagnostic_seconds=sum(b['layer_diagnostics']['seconds'] for b in batches),
            layers=normalized, ledger_sha256=record['ledger_sha256'],
            array_sha256=record['array_sha256']))
    if {(d['arm'], d['seed']) for d in details} != {(a, s) for a in ARMS for s in range(5)}:
        raise ValueError('complete paired seed roster')
    for arm in ARMS:
        values = source['arms'][arm]
        if (sum(values['clean_per_seed']) != values['clean_correct']
                or sum(values['certified_per_seed']) != values['certified']
                or values['denominator'] != 50000):
            raise ValueError('aggregate denominator/count mismatch')
        if arm in ('M', 'M-reg'):
            fraction = sum(d['single_route'] for d in details if d['arm'] == arm)/50000
            if fraction != values['single_route_fraction']:
                raise ValueError('single-route aggregate mismatch')
    return dict(source=str(path), source_sha256=hashlib.sha256(raw).hexdigest(),
        manifest_sha256=source['source_manifest_sha256'],
        decision=source['decision'], gates=source['gates'], arms=source['arms'],
        comparisons=source['comparisons'], bootstrap=source['bootstrap'],
        details=details, secondary_stage_complete=source['secondary_stage_complete'],
        model_or_solver_calls=0,
        scope='completed primary endpoint only; seed dots are observations, not confidence intervals; layer widths are descriptive, all experts on the full input box')


def render(data, output):
    plt.rcParams.update({'font.size': 10.5, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.6), gridspec_kw={'width_ratios': [1.2, 1.25, 1]})
    ax = axes[0]
    for arm, label, color in zip(ARMS, LABELS, COLORS):
        row = data['arms'][arm]
        ax.scatter(np.asarray(row['clean_per_seed'])/100,
                   np.asarray(row['certified_per_seed'])/100, color=color, s=16, alpha=.55)
        ax.scatter(100*row['clean_accuracy'], 100*row['certified_accuracy'],
                   color=color, marker='D', s=45, edgecolors='white', linewidth=.6, label=label)
    ax.set(xlabel='Clean accuracy (%)', ylabel='IBP certified accuracy (%)',
           title='(a) IBP primary endpoint')
    ax.grid(alpha=.15)

    ax = axes[1]
    for index, arm in enumerate(('D-act', 'D-tot', 'M-reg')):
        row = data['comparisons']['M_minus_'+arm]
        mean, (lo, hi) = 100*row['rate_difference'], [100*v for v in row['interval95']]
        ax.errorbar(mean, 2-index, xerr=[[mean-lo], [hi-mean]], fmt='o', color='#333333', capsize=4)
    ax.axvline(0, color='#999999', linewidth=1, linestyle='--')
    ax.set_yticks([2, 1, 0], ['M − D-act', 'M − D-tot', 'M − M-reg'])
    ax.set(xlabel='IBP accuracy difference (pp)', title='(b) Paired 95% intervals', ylim=(-.6, 2.6))
    ax.grid(axis='x', alpha=.15)

    ax = axes[2]
    for index, arm in enumerate(('M-reg', 'M')):
        color = COLORS[ARMS.index(arm)]
        value = 100*data['arms'][arm]['single_route_fraction']
        ax.bar(index, value, width=.55, color=color, alpha=.8)
        dots = [d['single_route']/100 for d in sorted(data['details'], key=lambda d:d['seed']) if d['arm'] == arm]
        ax.scatter(index+np.linspace(-.12, .12, 5), dots, color='black', s=12, zorder=3)
        ax.text(index, value/2, f'{value:.2f}%', ha='center', fontsize=10, fontweight='semibold')
    ax.axhline(90, color='#666666', linestyle='--', linewidth=1)
    ax.text(.5, 92, 'Gate for M: 90%', ha='center', fontsize=9)
    ax.set_xticks([0, 1], ['M−reg', 'M'])
    ax.set(ylabel='Single-route inputs (%)', title='(c) Route stability', ylim=(0, 105))
    ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .09, 1, 1), w_pad=2)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'), metadata={'CreationDate': None})
    fig.savefig(output.with_suffix('.png'), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.source:
        data = derive(args.source)
        args.data.parent.mkdir(parents=True, exist_ok=True)
        args.data.write_text(json.dumps(data, indent=2)+'\n')
    else:
        data = json.loads(args.data.read_text())
    render(data, args.output)
    print(json.dumps(dict(decision=data['decision'], models=len(data['details']),
                         model_or_solver_calls=0, source_sha256=data['source_sha256'])))
