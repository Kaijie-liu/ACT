"""Full stored-column union for unchanged ordinary prune-unused lowering."""
import numpy as np


def dimensions(hz, *, pool):
    """Count every stored column, including explicit zeros as native lowering does.

    This predicts dimensions only. The fresh worker must still independently
    ingest and compare every native coefficient, bound and integrality entry.
    """
    result = {}
    for label, width, names in (('cont', hz.n_cont, ('Gc', 'Ac', 'Auc')),
                               ('bin', hz.n_bin, ('Gb', 'Ab', 'Aub'))):
        pool.charge('c100_complete_native_stored_column_union',
                    128 + width + sum(getattr(hz, n).nnz for n in names))
        used = np.zeros(width, bool)
        for name in names:
            matrix = getattr(hz, name)
            if matrix.shape[1] != width or not matrix.has_canonical_format:
                raise ValueError('complete original canonical column domain required')
            used[matrix.indices] = True
        result['native_lowered_n_' + label] = int(np.count_nonzero(used))
    return dict(result, native_coefficients_not_yet_ingested=True,
                full_fresh_ingestion_still_required=True)
