"""Opt-in reserve handoff around the immutable V1 proposal algorithm.

This changes control flow only. The same deterministic LP schedule, 60-second
cap, 80-second reserve, mathematical construction and independent checker are
retained. No exception from a solver/checker/serializer is converted to success.
"""
from collections import Counter
from pathlib import Path

from portable_proof.runtime import digest
from scripts.optional_evidence_budget import EvidenceBudgetExpired
from scripts.optional_evidence_dev_contract import read, save

SCHEMA = 'EVIDENCE_RESERVE_HANDOFF_V1'


class ReserveHandoff(RuntimeError):
    """Only a proposal grant exhausted its fixed reserve, not the request clock."""


class ProposalBudget:
    def __init__(self, budget):
        if budget.total != 300:
            raise ValueError('unchanged 300-second request budget required')
        self._budget = budget
        self.started, self.deadline, self.clock = budget.started, budget.deadline, budget.clock

    def remaining(self, reserve=0):
        return self._budget.remaining(reserve)

    def grant(self, cap, reserve=0):
        # Do not disguise actual total-budget exhaustion as a successful handoff.
        self._budget.remaining(2)
        try:
            return self._budget.grant(cap, reserve)
        except EvidenceBudgetExpired:
            self._budget.remaining(2)
            if reserve != 80:
                raise
            raise ReserveHandoff('proposal reserve exhausted; inspect committed partial manifest') from None


def propose_with_handoff(root, budget):
    from moe_evidence.generate import propose_all
    root = Path(root)
    if (root/'handoff.json').exists():
        raise FileExistsError('handoff stage is one-shot; no resume')
    before = read(root/'manifest.json')
    before_hash = digest((root/'manifest.json').read_bytes())
    reason = 'PROPOSAL_LOOP_RETURNED'
    try:
        propose_all(root, ProposalBudget(budget), cap=60, reserve=80)
    except ReserveHandoff:
        reason = 'RESERVE_EXHAUSTED_HANDOFF'
    # Anything incomplete remains incomplete in the original manifest. The
    # unchanged checker decides UNKNOWN/positive only from committed references.
    budget.remaining(2)
    after = read(root/'manifest.json')
    if after['request_id'] != before['request_id']:
        raise ValueError('proposal changed request identity')
    result = {'schema': SCHEMA, 'reason': reason, 'request_id': after['request_id'],
              'manifest_before_sha256': before_hash,
              'manifest_after_sha256': digest((root/'manifest.json').read_bytes()),
              'total_seconds': 300, 'proposal_cap_seconds': 60, 'check_reserve_seconds': 80,
              'elapsed_seconds': budget.clock()-budget.started,
              'remaining_request_seconds': budget.deadline-budget.clock(),
              'supports': dict(Counter(v['status'] for v in after['supports'].values())),
              'weighted': dict(Counter(v.get('weighted_status', 'REUSED_OR_PENDING') for v in after['obligations'])),
              'next_stage': 'UNCHANGED_PRECHECK', 'acceptance_override': False}
    save(root/'handoff.json', result)
    budget.remaining(2)
    return result
