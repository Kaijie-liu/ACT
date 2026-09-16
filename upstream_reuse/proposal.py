"""Opt-in proposal research adapter; not wired into frozen production runners."""
from pathlib import Path
from evidence_handoff.proposal import ProposalBudget, ReserveHandoff
from scripts.optional_evidence_dev_contract import save
from upstream_reuse.operations import Operations


def run(root, request, budget, *, source_enabled=False, matrix_enabled=False):
    root = Path(root)
    if any((root/name).exists() for name in ('query_log.json', 'handoff.json', 'upstream_reuse_started.json')):
        raise FileExistsError('proposal is one-shot; no mixing or resume')
    budget.remaining(2)
    with (root/'upstream_reuse_started.json').open('x') as stream:
        stream.write('{}\n')
    ops = None; reason = 'ERROR'; error = None
    try:
        ops = Operations(root, request, budget, source_enabled=source_enabled, matrix_enabled=matrix_enabled)
        try:
            ops.timers.call('proposal_loop', ops.loop, root, ProposalBudget(budget), cap=60, reserve=80)
            reason = 'PROPOSAL_LOOP_RETURNED'
        except ReserveHandoff:
            reason = 'RESERVE_EXHAUSTED_HANDOFF'
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        if ops is not None: ops.close()
        report = {**(ops.report() if ops else {}), 'schema': 'UPSTREAM_REUSE_CONTROLS_V1',
                  'reason': reason, 'error': error, 'total_seconds': 300, 'cap_seconds': 60,
                  'reserve_seconds': 80, 'elapsed_seconds': budget.clock()-budget.started,
                  'next_stage': 'UNCHANGED_COMPLETE_CHECK', 'not_a_proof_verdict': True}
        # Diagnostic write is charged. Never replace an original failure with a
        # diagnostic-write failure, nor convert either into an accepted proof.
        try:
            budget.remaining(2); save(root/'upstream_reuse_report.json', report); budget.remaining(2)
        except BaseException:
            if error is None: raise
    return report
