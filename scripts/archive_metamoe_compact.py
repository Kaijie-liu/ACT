"""Same saved-record audit, with a hash reference instead of image coordinates.

Original frozen server results are never modified. The early full-coordinate
Git summary is superseded, not relabeled; its exact serialization hash remains.
"""
import argparse
import hashlib
import json
from pathlib import Path
from audit_metamoe_full_intake import collect


def compact(config):
    full = collect(config)
    prior = hashlib.sha256((json.dumps(full, indent=2, allow_nan=False)+'\n').encode()).hexdigest()
    for record in full['records']:
        result = record['result']
        if result is not None and 'witness' in result:
            witness = result.pop('witness')
            encoded = json.dumps(witness, separators=(',', ':'), allow_nan=False).encode()
            result['witness_reference'] = {'canonical_json_sha256': hashlib.sha256(encoded).hexdigest(),
                'location': 'original server result.json, bound by record_hashes',
                'inline_coordinates_omitted': True}
    return {'archive_schema': 'compact_witness_reference_v1',
            'superseded_full_coordinate_archive_sha256': prior, **full}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    value = compact(a.config)
    if a.check:
        if value != json.loads(a.output.read_text()):
            raise ValueError('compact archive differs')
    else:
        with a.output.open('x') as f:
            json.dump(value, f, indent=2)
            f.write('\n')
    print('Compact saved-record audit PASS')
