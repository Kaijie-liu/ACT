"""Full owner bytes, interval-union CSR entries; no mutation or compaction."""

from experiments.neural_hz_20260831 import c5_explicit_schema_ledger_v2 as schema

core = schema.core


class PartialCSRVisitor(schema.KnownBufferVisitor):
    def __init__(self):
        super().__init__()
        self.csr_data_context = None
        self.csr_data_spans = {}

    def _visit_numpy(self, array, role, *, entry_semantics):
        if entry_semantics != "csr_data":
            return super()._visit_numpy(array, role, entry_semantics=entry_semantics)
        previous = self.csr_data_context
        lower, upper = core._byte_bounds(array)
        self.csr_data_context = (role, lower, upper, array.dtype.itemsize)
        try:
            # Reuse ALL owner/dtype/span/alias validation. The index visitor
            # charges full owner bytes without requiring data to fill it.
            return super()._visit_numpy(array, role, entry_semantics="csr_index")
        finally:
            self.csr_data_context = previous

    def _register_storage(self, **kwargs):
        context = self.csr_data_context
        if context is None or kwargs["role"] != context[0]:
            return super()._register_storage(**kwargs)
        if kwargs["entry_semantics"] != "csr_index_bytes_only":
            raise core.WholeStateReject("unexpected_csr_entry_adapter_semantics")
        role, lower, upper, itemsize = context
        if itemsize <= 0 or (upper - lower) % itemsize:
            raise core.WholeStateReject("unaligned_csr_data_span")
        spans = self.csr_data_spans.setdefault(kwargs["key"], set())
        for start, stop in spans:
            if max(start, lower) < min(stop, upper) and (start, stop) != (lower, upper):
                raise core.WholeStateReject("partial_csr_data_overlap")
        spans.add((lower, upper))
        entries = sum((stop - start) // itemsize for start, stop in spans)
        existing = self._storage_records.get(kwargs["key"])
        if existing is not None:
            if (existing.storage_kind != kwargs["storage_kind"] or existing.resident_bytes != kwargs["resident_bytes"]
                    or existing.entry_semantics != "stored_csr_data_elements"):
                raise core.WholeStateReject(f"incompatible_storage_alias:{role}")
            if entries < existing.resident_entries:
                raise core.WholeStateReject("decreasing_csr_data_union")
            existing.resident_entries = entries
            existing.roles.add(role)
            return
        kwargs.update(resident_entries=entries, entry_semantics="stored_csr_data_elements")
        return super()._register_storage(**kwargs)


def snapshot_partial_csr_owners(roots):
    snapshot = core._snapshot_roots(roots)
    visitor = PartialCSRVisitor()
    for namespace in ("sparse_hz", "affine_expr", "precomputed_relu", "phase_bounds",
                      "descriptor", "artifact", "active", "pending"):
        items = [(core._safe_key_label(key), value) for key, value in snapshot.maps[namespace].items()]
        for label, value in sorted(items, key=lambda item: item[0]):
            role = f"{namespace}[{label}]"
            expected = {"sparse_hz": core.SparseHZono, "affine_expr": core.SparseHZAffineExpr, "phase_bounds": core.Bounds}.get(namespace)
            if expected is not None and type(value) is not expected:
                raise core.WholeStateReject(f"malformed_registered_root:{role}")
            if namespace == "precomputed_relu":
                core._visit_precomputed(visitor, value, role)
            else:
                visitor.visit(value, role)
    return visitor.ledger()
