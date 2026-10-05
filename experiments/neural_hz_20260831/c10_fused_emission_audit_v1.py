"""Independent original-DAG proof composed with exact Fraction quotient proof.

The original completed HZ is an EXTERNAL oracle artifact, never stored by the
candidate. Its proof is rechecked against the newly constructed original DAG.
"""

import numpy as np

from experiments.neural_hz_20260831.c9_integrated_suffix_v1 import Integrated
from experiments.neural_hz_20260831.c9_integrated_suffix_audit_v1 import audit as original_audit
from experiments.neural_hz_20260831.c10_alias_quotient_v1 import Quotient
from experiments.neural_hz_20260831.c10_alias_quotient_audit_v1 import audit as quotient_audit
from experiments.neural_hz_20260831.c10_fused_rows_v1 import aliases
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def audit(candidate, original_hz, maps):
    candidate.validate()
    keys = ('eq_roots', 'eq_scales', 'ineq_roots', 'ineq_scales', 'def_rows')
    if set(maps) != set(keys):
        raise ValueError('unregistered external original-row oracle maps')
    external = Integrated(candidate.expression, candidate.origin_binding, original_hz,
        candidate.nodes, candidate.root, candidate.old_n_cont, candidate.old_n_bin,
        candidate.old_n_eq, candidate.logical_n_cont, candidate.keep,
        {'external_oracle_only': True}, *(maps[key] for key in keys))
    external.seal = external.fingerprint()
    original_proof = original_audit(external)
    cols, parents, ratios, tagged = aliases(candidate)
    erased = maps['eq_roots'][tagged].copy()
    temporary = Quotient(candidate.hz, cols.copy(), parents.copy(), ratios.copy(), erased,
        candidate.old_n_cont, candidate.logical_n_cont, source_digest(original_hz), {})
    temporary.seal = temporary.fingerprint()
    quotient_proof = quotient_audit(original_hz, temporary)
    keep = np.ones(original_hz.n_eq, dtype=bool)
    keep[erased] = False
    rowmap = np.cumsum(keep, dtype=np.int64) - 1
    surviving = candidate.eq_roots >= 0
    if (not np.array_equal(candidate.eq_roots[surviving], rowmap[maps['eq_roots'][surviving]])
            or not np.array_equal(candidate.eq_scales[surviving], maps['eq_scales'][surviving])
            or not np.array_equal(candidate.def_rows, rowmap[maps['def_rows']])
            or not np.array_equal(candidate.ineq_roots, maps['ineq_roots'])
            or not np.array_equal(candidate.ineq_scales, maps['ineq_scales'])):
        raise ValueError('fused original/radix physical-row lineage mismatch')
    if not np.array_equal(np.sort(np.r_[candidate.eq_roots[surviving], candidate.def_rows]),
                          np.arange(candidate.hz.n_eq)):
        raise ValueError('missing/duplicate/hidden physical fused row')
    candidate.validate()
    return {'status': 'EXACT_ORIGINAL_DAG_AND_QUOTIENT', 'original_affine_proof': original_proof,
        'quotient_proof': quotient_proof, 'tagged_physical_lineage_checked': True,
        'persistent_extra_reconstruction_arrays': 0, 'external_original_hz_retained_by_candidate': False,
        'live_publication_proved': False, 'formal_gain': 0}
