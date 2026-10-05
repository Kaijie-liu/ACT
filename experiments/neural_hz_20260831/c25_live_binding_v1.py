"""Bind fresh generated fields to an independently archived complete proof.

No completed HZ is loaded or substituted. The expected hash is an enclosing
manifest trust input, never a representation trigger or an instance whitelist.
"""

from dataclasses import fields
import hashlib
import time

from experiments.neural_hz_20260831.c24_closed_state_v1 import Draft, Closed, restore


def bind(draft, proof_bytes, *, expected_proof_sha256, enabled=False):
    if not enabled:
        return None
    if type(draft) is not Draft:
        raise ValueError('fresh dense construction draft required')
    if type(proof_bytes) is not bytes or hashlib.sha256(proof_bytes).hexdigest() != expected_proof_sha256:
        raise ValueError('independently anchored complete proof is missing or changed')
    started = time.monotonic()
    draft.validate()
    values = {f.name: getattr(draft, f.name) for f in fields(Closed)
        if f.name not in {'origin_binding', 'receipt', 'seal'}}
    closed = restore(values, proof_bytes, expected_proof_sha256=expected_proof_sha256)
    draft.validate()
    if closed.hz is not draft.hz or closed.expression is not draft.expression or closed.owners is not draft.owners:
        raise ValueError('proof binding substituted a completed external HZ/source')
    return closed, {'schema': 'c25_fresh_closed_proof_binding_v1',
        'proof_sha256': expected_proof_sha256, 'closed_identity': closed.fingerprint(),
        'fresh_generated_numeric_owners_preserved_by_identity': True,
        'archived_HZ_loaded_or_substituted': False,
        'all_source_content_and_sharing_match_complete_proof': True,
        'authentication_elapsed_s': time.monotonic() - started,
        'authentication_is_not_free_generation_work': True, 'formal_gain': 0}
