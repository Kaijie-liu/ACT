"""One-residual replacement composition; pure checking, no producer imports.

The original rational request must have exactly one nonpositive residual.
Recheck the ENTIRE original inventory before replacing that identified row.
This establishes only the original, explicitly conditional proof contract.
"""
from fractions import Fraction as F

from act.back_end.solver.lp_certificate import identity
from act.back_end.solver.check_hz_lp_export import check_export
from act.back_end.solver.check_rational_mccormick import check_construction
from act.pipeline.moe.check_request_lp import aggregate, RATIONAL_TRUSTED
from checked_gate.checker import check as check_gate


def check(manifest, read, replacement, gate_proof, record, certificate, *,
          expected_request_id, expected_manifest_identity):
    if (identity(manifest) != expected_manifest_identity or
            manifest['request_id'] != expected_request_id or
            manifest['schema'] != 'request_lp_rational_v3'):
        raise ValueError('original request identity/contract mismatch')
    if (replacement['schema'] != 'SINGLE_GATE_REPLACEMENT_V1' or
            replacement['request_id'] != expected_request_id or
            replacement['manifest_identity'] != expected_manifest_identity):
        raise ValueError('replacement identity mismatch')
    baseline = aggregate(manifest, read)
    if baseline.get('counts', {}).get('unknown') != 1:
        raise ValueError('replacement requires exactly one unresolved obligation')
    matches = [r for r in manifest['obligations'] if r['pair'] == replacement['pair']
               and r['property_index'] == replacement['property_index']]
    if len(matches) != 1 or matches[0]['kind'] != 'residual':
        raise ValueError('not a unique residual replacement')
    row = matches[0]
    item = manifest['proofs'][row['source']]
    old = read(item['export'])
    old_certificate = read(item['certificate'])
    old_result = check_construction(old, old_certificate, source_hash=item['hz_sha256'],
        q=old['q'], offset=old['offset'], gate=row['lambda_bounds'],
        difference=row['difference_bounds'])
    threshold = F.from_float(1e-7)
    old_bound = F(old_result['bound']['checked_lower_bound'])
    if old_bound > threshold:
        raise ValueError('replacement targets an already positive obligation')
    # aggregate has independently bound these supports to the same request,
    # pair, score orientation and property, including the shared HZ identity.
    endpoints = []
    for key in (row['gate_lower'], row['gate_upper']):
        proof = manifest['proofs'][key]
        checked = check_export(read(proof['export']), read(proof['certificate']),
                               expected_source_sha256=proof['hz_sha256'])
        endpoints.append(F(checked['bound']['checked_lower_bound']))
    context = {'request_id': expected_request_id, 'ordered_pair': row['pair'],
        'margin_lower_proof': identity(read(manifest['proofs'][row['gate_lower']]['certificate'])),
        'margin_negative_upper_proof': identity(read(manifest['proofs'][row['gate_upper']]['certificate']))}
    gate = check_gate(gate_proof, expected_context=context,
                     expected_margin=[str(endpoints[0]), str(-endpoints[1])])['gate']
    new_result = check_construction(record, certificate, source_hash=item['hz_sha256'],
        q=old['q'], offset=old['offset'], gate=gate, difference=row['difference_bounds'])
    bound = F(new_result['bound']['checked_lower_bound'])
    counts = dict(baseline['counts'])
    if bound > threshold:
        counts['unknown'] -= 1
        counts['residual'] += 1
    return {'status': ('CHECKED_REQUEST_CONDITIONAL_ON_TRUSTED_LOWERING'
                       if counts['unknown'] == 0 else 'UNKNOWN'),
        'request_id': expected_request_id, 'required': baseline['required'], 'counts': counts,
        'baseline_counts': baseline['counts'], 'replaced_pair': row['pair'],
        'replaced_property_index': row['property_index'],
        'old_replaced_bound': str(old_bound), 'new_replaced_bound': str(bound),
        'checked_gate': gate, 'trusted_base': RATIONAL_TRUSTED,
        'complete_strict_network_certificate': False, 'production_verdict_changed': False,
        'scope': 'All supplied output obligations checked; network-to-HZ, guard lowering and route exclusions trusted. Not deployed floating-point proof.'}
