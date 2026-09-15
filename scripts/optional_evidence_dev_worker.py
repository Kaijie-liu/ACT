"""Opt-in driver adapters. Frozen ACT algorithms and accept gates unchanged."""
import argparse
from pathlib import Path
import sys
import time

from scripts.optional_evidence_dev_contract import ROOT, read, save, validate_job, verify_freeze
from scripts.optional_evidence_budget import EvidenceBudget, EvidenceBudgetExpired


def capture(directory, budget):
    # Only process-local entry/clock binding changes. No on-disk old code edit.
    import scripts.run_conv_pre_f0_r2 as old
    import scripts.budget_contract_v2 as v2
    original_validate, original_verify = old.validate_job, v2.verify_v2
    old.validate_job = validate_job
    def charged(*args, **kwargs):
        kwargs['started'] = budget.started
        return original_verify(*args, **kwargs)
    v2.verify_v2 = charged
    try:
        budget.remaining(2)
        old.capture_worker(directory)
    finally:
        old.validate_job, v2.verify_v2 = original_validate, original_verify


def propose(directory, budget):
    import scripts.run_conv_pre_f0_r2 as old
    import act.back_end.solver.lp_certificate as lp
    freeze = verify_freeze()
    original_validate, original_propose = old.validate_job, lp.propose
    old.validate_job = validate_job
    calls = []
    def limited(problem, *, time_limit=None):
        cap = min(float(time_limit), freeze['proposal_cap_seconds'])
        grant = budget.grant(cap, freeze['proposal_terminal_check_reserve_seconds'])
        calls.append({'entered_seconds': time.monotonic()-budget.started, 'grant_seconds': grant,
                      'deadline_seconds': budget.total, 'check_reserve_seconds': 80})
        save(directory / 'evidence_grants.json', calls)
        # Recheck after the durable grant write; writing is not free.
        grant = budget.grant(grant, freeze['proposal_terminal_check_reserve_seconds'])
        result = original_propose(problem, time_limit=grant)
        budget.remaining(2)
        return result
    lp.propose = limited
    try:
        old.proposal_worker(directory)
    finally:
        old.validate_job, lp.propose = original_validate, original_propose


def precheck(directory, budget):
    from scripts.check_conv_pre_f0_r2 import aggregate
    from portable_proof.runtime import digest
    from act.back_end.solver.lp_certificate import identity
    job = validate_job(directory)
    manifest = read(directory / 'manifest.json')
    def load(ref):
        budget.remaining(2)
        p = directory / ref['file']
        if p.parent != directory or digest(p.read_bytes()) != ref['sha256']:
            raise ValueError('proof reference mismatch')
        return read(p)
    snapshot = load(manifest['common_facts'])
    if identity(snapshot['payload']) != snapshot['payload_sha256']:
        raise ValueError('snapshot identity changed')
    result = aggregate(manifest, snapshot['payload'], job, load)
    budget.remaining(2)
    save(directory / 'independent.json', result)


def package(directory, budget):
    from scripts.build_portable_conv_proof import build
    budget.remaining(2)
    report = build(directory / 'portable', source=directory)
    budget.remaining(2)
    save(directory / 'packing.json', report)


def production(directory, budget):
    from scripts.budget_contract_v2 import verify_v2
    from act.pipeline.moe.external_pair_worker import load
    from act.pipeline.moe.staged_verifier import write_evidence_package
    from act.pipeline.moe.common_fact_snapshot import publish_snapshot
    from scripts.optional_evidence_dev_contract import git
    from portable_proof.runtime import digest
    req = validate_job(directory)['parent_request']
    model, tensors = load(req)
    report = verify_v2(model, tensors['center'], req['epsilon'], read(req['config']['path']),
        journal_path=directory/'budget_journal.jsonl', started=budget.started,
        identity={'request_sha256': digest((directory/'job.json').read_bytes()), 'mode': 'production_matched_v2'},
        expected_clean_prediction=req['sample']['label'],
        checkpoint_identity={'path': req['subject']['checkpoint'], 'sha256': req['subject']['checkpoint_sha256']},
        common_fact_callback=lambda v: publish_snapshot(directory/'common_facts.json', v))
    report.evidence['execution'] = {'git_head': git('rev-parse','HEAD'), 'config_path': req['config']['path'],
        'config_sha256': req['config']['sha256'], 'dataset_index': req['sample']['dataset_index']}
    report.evidence['execution_budget_contract']['journal_sha256'] = digest((directory/'budget_journal.jsonl').read_bytes())
    write_evidence_package(report, directory/'package')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['capture', 'propose', 'precheck', 'package', 'production'])
    p.add_argument('directory', type=Path)
    p.add_argument('--started', type=float, required=True)
    args = p.parse_args()
    budget = EvidenceBudget(args.started)
    validate_job(args.directory)
    budget.remaining(2)
    try:
        globals()[args.stage](args.directory, budget)
    except EvidenceBudgetExpired:
        sys.exit(3)
