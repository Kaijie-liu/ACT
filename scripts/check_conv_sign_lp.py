"""Isolated rational check of a supplied F0-HZ property, not a network proof."""
import argparse
from fractions import Fraction
from pathlib import Path
import sys
import types

from scripts.conv_sign_lp_contract import ROOT, read, sha, save, validate_job


def isolate():
    for name in ('act', 'act.back_end', 'act.back_end.solver'):
        if name in sys.modules: raise RuntimeError('fresh -S checker process required')
        p = types.ModuleType(name); p.__path__ = [str(ROOT.joinpath(*name.split('.')))]; p.__package__ = name
        sys.modules[name] = p


def check_record(record, cert, capture, job):
    from act.back_end.solver.check_hz_lp_export import check_export
    if (capture['scope'] != job['expected_scope'] or capture['parent_request_sha256'] != job['parent_request_sha256']
            or record['q'] != [1] or record['offset'] != 0 or len(record['source']['c']) != 1):
        raise ValueError('wrong property/source binding')
    checked = check_export(record, cert, expected_source_sha256=capture['source_sha256'])
    value = Fraction(checked['bound']['checked_lower_bound'])
    # A nonpositive lower bound is not a violating point or proof of UNSAFE.
    return {'status': 'CHECKED_POSITIVE_SUPPLIED_F0_LP' if value > Fraction.from_float(1e-7)
            else 'CHECKED_NONPOSITIVE_LOWER_BOUND', 'lower_bound': str(value),
            'lower_bound_float_descriptive': float(value), 'lp_sha256': checked['lp_sha256'],
            'source_sha256': checked['source_sha256'], 'factors': checked['n_factors'],
            'relaxed_binaries': checked['n_relaxed_binaries'], 'full_request_SAFE': False,
            'trusted': ['model_input_to_HZ', 'guard_and_route_lowering', 'floating_F0_construction',
                        'capture_binding_to_model_and_property'],
            'checked': ['supplied_F0_HZ_to_continuous_LP', 'signed_duals_and_exact_box_residual'],
            'scope': 'One supplied-HZ property only; no solver call, MILP tree, full request or deployed-float proof.'}


def check_directory(root):
    job = validate_job(root); capture = read(root / 'capture.json')
    for name, key in [('export.json', 'export_sha256'), ('budget_journal.jsonl', 'journal_sha256')]:
        if sha(root / name) != capture[key]: raise ValueError('capture artifact drift')
    proposal = read(root / 'proposal.json')
    if proposal['export_sha256'] != capture['export_sha256']: raise ValueError('wrong proposal LP')
    if proposal['status'] == 'UNAVAILABLE':
        return {'status': 'PROPOSAL_UNAVAILABLE', 'full_request_SAFE': False,
                'reason': proposal['reason'], 'source_sha256': capture['source_sha256']}
    if proposal['status'] != 'PROPOSED' or sha(root / 'certificate.json') != proposal['certificate_sha256']:
        raise ValueError('certificate identity changed')
    return check_record(read(root / 'export.json'), read(root / 'certificate.json'), capture, job)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root', type=Path); p.add_argument('output', type=Path)
    a = p.parse_args(); isolate()
    if a.output.exists() or not a.output.resolve().is_relative_to(ROOT): raise ValueError('new project output required')
    save(a.output, check_directory(a.root))
