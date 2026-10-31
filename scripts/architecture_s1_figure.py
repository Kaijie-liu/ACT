"""Render the archived S1 clean/positive decomposition without model queries.

Conditional fractions describe different clean-correct populations. They do
not replace the frozen all-input endpoint, identify an architectural cause,
or change STOP_S1. Seed dots are observations, not confidence intervals.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

GROUPS = ('L1E8', 'Dense_active', 'Dense_total')
LABELS = ('MoE L1E8', 'Dense active', 'Dense total')


def derive(source):
    if source['decision'] != 'STOP_S1_RETURN_TO_REDUCTION_PAPER':
        raise ValueError('expected frozen S1 stop record')
    panels = []
    for mode, count in (('complete', 200), ('incomplete', 1000)):
        groups = []
        for group in GROUPS:
            rows = [r for r in source['model_tables'] if r['mode'] == mode
                    and r['radius'] == 2 and r['model'].startswith(group + '_seed')]
            rows.sort(key=lambda r: int(r['model'].rsplit('seed', 1)[1]))
            if ([r['model'] for r in rows] != [f'{group}_seed{s}' for s in range(3)]
                    or any(r['denominator'] != count or r['clean_observed'] != count
                           or r['clean_unknown'] != 0 for r in rows)):
                raise ValueError('complete fixed-seed input/clean records required')
            positives = [r['states']['POSITIVE'] for r in rows]
            clean = [r['clean_correct'] for r in rows]
            if any(not 0 <= p <= c <= count for p, c in zip(positives, clean)):
                raise ValueError('positive must be a subset of clean correct')
            total, npos, nclean = 3 * count, sum(positives), sum(clean)
            aggregate = next(r for r in source['grouped_verification']
                             if (r['mode'], r['radius'], r['group']) == (mode, 2, group))
            if (aggregate['denominator'], aggregate['states']['POSITIVE'],
                    aggregate['clean_correct']) != (total, npos, nclean):
                raise ValueError('seed/aggregate counts disagree')
            groups.append(dict(group=group, positive=npos, clean=nclean, total=total,
                all_input_percent=100*npos/total,
                clean_conditional_percent=100*npos/nclean,
                all_input_seed_percent=[100*p/count for p in positives],
                conditional_seed_percent=[100*p/c for p, c in zip(positives, clean)]))
        panels.append(dict(mode=mode, groups=groups))
    return panels


def render(source_path, output):
    source_path, output = Path(source_path), Path(output)
    source = json.loads(source_path.read_text())
    panels = derive(source)
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    colors = ('#276484', '#c77725')
    for ax, panel in zip(axes, panels):
        x = np.arange(3)
        for j, (field, seedfield, label) in enumerate((
                ('all_input_percent', 'all_input_seed_percent', 'All inputs (registered)'),
                ('clean_conditional_percent', 'conditional_seed_percent', 'Clean correct (descriptive)'))):
            centers = x + (-.19 if j == 0 else .19)
            heights = [r[field] for r in panel['groups']]
            ax.bar(centers, heights, .34, color=colors[j], alpha=.85, label=label)
            for position, row in zip(centers, panel['groups']):
                ax.scatter(position + np.array([-.07, 0, .07]), row[seedfield],
                           s=13, color='black', linewidths=0, zorder=3)
                denom = row['total'] if j == 0 else row['clean']
                ax.annotate(f"{row['positive']}/{denom}", (position, row[field]),
                            xytext=(0, 5), textcoords='offset points',
                            ha='center', va='bottom', fontsize=8,
                            bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1))
        ax.set_xticks(x, LABELS)
        ax.set_ylabel('Numerical positive fraction (%)')
        ax.set_title('Complete verification' if panel['mode'] == 'complete' else 'Incomplete CROWN-IBP')
        highest = max(max(r['conditional_seed_percent']) for r in panel['groups'])
        ax.set_ylim(0, highest * 1.22)
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc='lower center', frameon=False)
    fig.tight_layout(rect=(0, .08, 1, 1))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.pdf'), metadata={'CreationDate': None})
    fig.savefig(output.with_suffix('.png'), dpi=180)
    plt.close(fig)
    derived = dict(source=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        panels=panels, scope='descriptive clean decomposition; three fixed seed dots; no CI or causal claim',
        original_decision=source['decision'], model_or_solver_calls=0)
    output.with_suffix('.json').write_text(json.dumps(derived, indent=2)+'\n')
    return derived


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.source, args.output), indent=2))
