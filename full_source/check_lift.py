"""Independent exact affine-image lifting check. Never imports the producer."""
from fractions import Fraction as F
try:
    from proof_format import unpack, identity, rational
except ImportError:
    from source_enclosure.format import unpack, identity, rational


def check(source, target, operator, bias, proof, tag):
    s, c, b = unpack(source); t, ct, bt = unpack(target)
    if proof != {'source': identity(source), 'tag': tag}:
        raise ValueError('lift source/tag binding')
    if t['frame_id'] != s['frame_id'] or ct[:len(c)] != c or bt != b:
        raise ValueError('lift frame/factor identity')
    if len(operator) != len(bias) or len(t['c']) != len(operator):
        raise ValueError('lift output dimensions')
    if any(s[k] != t[k] for k in ('Auc', 'Aub', 'ub')):
        raise ValueError('lift inequalities changed')
    n = len(s['b'])
    if any(t[k][:n] != s[k] for k in ('Ac', 'Ab', 'b')):
        raise ValueError('lift equality prefix changed')
    ids = list(c); added = 0
    for i, weights in enumerate(operator):
        center = rational(bias[i]); continuous = {}; binary = {}
        for j, value in weights.items():
            if type(j) is not int or not 0 <= j < len(s['c']):
                raise ValueError('operator index')
            w = rational(value); center += w*s['c'][j]
            for dst, key in ((continuous, 'Gc'), (binary, 'Gb')):
                for k, v in s[key][j].items(): dst[k] = dst.get(k, F(0)) + w*v
        continuous = {k:v for k,v in continuous.items() if v}
        binary = {k:v for k,v in binary.items() if v}
        radius = sum(map(abs, continuous.values()), F(0)) + sum(map(abs, binary.values()), F(0))
        if t['c'][i] != center or t['Gb'][i]: raise ValueError('lift exact center/output')
        if not radius:
            if t['Gc'][i]: raise ValueError('constant lift')
            continue
        column = len(ids); ids.append(f'{tag}/value/{i}')
        if t['Gc'][i] != {column:radius}: raise ValueError('lift range/slot')
        continuous[column] = -radius; at = n+added
        if at >= len(t['b']) or (t['Ac'][at], t['Ab'][at], t['b'][at]) != (continuous, binary, F(0)):
            raise ValueError('lift exact graph equality')
        added += 1
    if ct != ids or len(t['b']) != n+added: raise ValueError('lift missing/extra factors or rows')
    return {'status':'CHECKED_EXACT_AFFINE_LIFT','rows':len(operator), 'new_factors':added,
            'state_sha256':identity(target)}
