"""Untrusted sparse equality lift; does not optimize or certify a bound."""
import copy
from fractions import Fraction as F
from source_enclosure.format import unpack, pack, identity, clean


def affine(state, operator, bias, tag):
    s, ci, bi = unpack(state)
    h = copy.deepcopy(s); ids = list(ci)
    h['c'] = []; h['Gc'] = []; h['Gb'] = []
    for i, (row, b) in enumerate(zip(operator, bias)):
        center = F(b); gc = {}; gb = {}
        for j, w in row.items():
            center += w * s['c'][j]
            for result, key in ((gc, 'Gc'), (gb, 'Gb')):
                for k, v in s[key][j].items():
                    result[k] = result.get(k, F(0)) + w*v
        gc, gb = clean(gc), clean(gb)
        radius = sum(map(abs, gc.values()), F(0)) + sum(map(abs, gb.values()), F(0))
        h['c'].append(center); h['Gb'].append({})
        if radius:
            slot = len(ids); ids.append(f'{tag}/value/{i}')
            h['Gc'].append({slot: radius})
            gc[slot] = -radius
            h['Ac'].append(gc); h['Ab'].append(gb); h['b'].append(F(0))
        else:
            h['Gc'].append({})
    return pack(h, ids, bi), {'source': identity(state), 'tag': tag}
