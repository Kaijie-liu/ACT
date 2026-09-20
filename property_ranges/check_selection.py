"""Independent exact selection replay, no solver or producer imports."""
from fractions import Fraction as F
try:
    from proof_format import unpack, identity, rational
    from router_check import tensor
except ImportError:
    from source_enclosure.format import unpack, identity, rational
    from router_source.checker import tensor


def check(source, rows, bias, classifier, context, classes, label, arm, record):
    if arm not in ('prefix', 'property') or not rows or len(rows) != len(bias):
        raise ValueError('selection configuration')
    if type(classes) is not int or type(label) is not int or classes < 2 or not 0 <= label < classes:
        raise ValueError('property inventory')
    width = len(rows)
    info, weights = tensor(classifier['weight'])
    if classifier['kind'] != 'Linear' or classifier['training'] is not False or info['shape'] != [classes,width]:
        raise ValueError('classifier scope/dimensions')
    state, ci, bi = unpack(source); inventory = []
    for j in range(width):
        # Separate projection implementation: one combined shared factor map.
        offset = rational(bias[j]); projection = {}
        for coordinate, coefficient in rows[j].items():
            coefficient = rational(coefficient)
            offset += coefficient * state['c'][coordinate]
            for key, shift in (('Gc',0), ('Gb',len(ci))):
                for factor, value in state[key][coordinate].items():
                    index = factor+shift
                    projection[index] = projection.get(index,F(0)) + coefficient*value
        radius = sum((abs(v) for v in projection.values()), F(0))
        lower, upper = offset-radius, offset+radius
        excess = F(0)
        if lower < 0 and upper > 0: excess = (-lower)*upper/(upper-lower)
        harm = F(0)
        for competitor in range(classes):
            if competitor != label:
                harm = max(harm, weights[competitor*width+j]-weights[label*width+j])
        inventory.append({'row':j, 'range':[str(lower),str(upper)], 'triangle_gap':str(excess),
                          'negative_coefficient':str(harm), 'score':str(excess*harm)})
    chosen = list(range(min(2,width)))
    if arm == 'property':
        priority = sorted(inventory, key=lambda item: (-F(item['score']),item['row']))
        chosen = sorted(item['row'] for item in priority[:min(2,width)])
    expected = {'schema':'PROPERTY_RANGE_SELECTION_V1', 'arm':arm, 'source_sha256':identity(source),
                'context':context, 'classifier_sha256':identity(classifier), 'classes':classes, 'label':label,
                'quota':2, 'query_order':'ascending_row', 'rows':inventory, 'selected':chosen}
    if record != expected: raise ValueError('selection identity/score/roster mismatch')
    return chosen
