"""Untrusted, outcome-blind row selection; same fixed quota in both arms."""
from fractions import Fraction as F
from source_enclosure.format import unpack, identity, rational
from full_source.graph import operator

ARMS = ('prefix', 'property')


def select(source, rows, bias, classifier, context, classes, label, arm):
    if arm not in ARMS or not rows or len(rows) != len(bias):
        raise ValueError('selection policy/rows')
    if type(label) is not int or type(classes) is not int or not 0 <= label < classes or classes < 2:
        raise ValueError('classification properties')
    shape, weights, _ = operator([1, len(rows)], classifier)
    if shape != [1, classes]: raise ValueError('classifier dimensions')
    h, _, _ = unpack(source)
    inventory = []
    for j, (row, b) in enumerate(zip(rows, bias, strict=True)):
        center = rational(b); continuous = {}; binary = {}
        for k, value in row.items():
            value = rational(value); center += value * h['c'][k]
            for result, name in ((continuous, 'Gc'), (binary, 'Gb')):
                for factor, coefficient in h[name][k].items():
                    result[factor] = result.get(factor, F(0)) + value * coefficient
        radius = sum(map(abs, continuous.values()), F(0)) + sum(map(abs, binary.values()), F(0))
        lo, hi = center-radius, center+radius
        gap = -lo*hi/(hi-lo) if lo < 0 < hi else F(0)
        negative = max([F(0)] + [weights[k].get(j, F(0))-weights[label].get(j, F(0))
                                for k in range(classes) if k != label])
        inventory.append({'row':j, 'range':[str(lo), str(hi)], 'triangle_gap':str(gap),
                          'negative_coefficient':str(negative), 'score':str(gap*negative)})
    ranked = sorted(range(len(rows)), key=lambda j: (-F(inventory[j]['score']), j))
    chosen = list(range(min(2,len(rows)))) if arm == 'prefix' else sorted(ranked[:min(2,len(rows))])
    return {'schema':'PROPERTY_RANGE_SELECTION_V1', 'arm':arm, 'source_sha256':identity(source),
            'context':context, 'classifier_sha256':identity(classifier), 'classes':classes, 'label':label,
            'quota':2, 'query_order':'ascending_row', 'rows':inventory, 'selected':chosen}
