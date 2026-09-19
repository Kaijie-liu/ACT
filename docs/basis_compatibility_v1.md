# Real LP compatibility: freeze withheld

## Decision

**NOT_FROZEN_INCOMPATIBLE_CURRENT_LIMITS.** The read-only inventory covers all
30 previously analyzed nonpositive weighted obligations, not a newly selected
subset. All30 fail the current native/basis interface's necessary limits. The
four previously executed diagnostics are identified separately within those30.
No new diagnostic is frozen or launched; no solver, rational reconstruction,
rank computation, point repair or new bound check was performed.

Starting HEAD: `cfc7f22246fcdb98c1902fefd7284c68a445606c`.
See [machine-readable inventory](basis_compatibility_v1.json) and
[fresh-process audit](basis_compatibility_v1_audit.json). The audit reproduces
the report from the original hash-bound files: PASS,0 issues. Six analytic
dimension/CSR/lower-limit/eligibility controls pass. Old source freezes and
diagnostic outcomes remain unchanged. This is a compatibility result, not a
negative scientific result about the LPs or networks.

## Measured original sizes

The four previously queried LPs have the following original dimensions.
Rows refer to `E x=h` and `A x<=b`; the basis constructor uses all `E+A` rows
in `E x=h, A x+s=b`. CSR entries are the stored data count used by the actual
size gate, not a new sparsification of the matrix.

| Prior job | Variables | E rows | A rows | E+A rows | Stored CSR entries |
| --- | ---: | ---: | ---: | ---: | ---: |
| input220/p0 |7,397|1,441|2,890|4,331|246,558|
| input222/p1 |7,682|1,536|3,080|4,616|236,502|
| input230/p2 |9,482|2,136|4,280|6,416|348,915|
| input232/p0 |9,095|2,007|4,022|6,029|329,664|
| Current gate |**64**|—|—|**64**|**8,192**|

Across all30, variables range7,397–9,482, augmented rows4,331–6,416 and stored
entries236,502–348,915. The original denominator remains9 obligations from
input220,7 from222,5 from230 and9 from232. No search for a different, easier
real LP is performed to bypass these observations.

Two further necessary limits also fail for all30. They are deduced from the
frozen constructor loops, not measured runtime or elimination predictions:

- Before basis assembly, `validate` calls `Budget.value` on `c/lower/upper`,
  `b/h`, the offset and all stored coefficients. Its initial scalar-visit count
  alone is `3n + (A_rows+E_rows) + 1 + stored_nnz`:264,165–383,778, exceeding
  the200,000-operation cap. Later assembly and elimination would add work.
- A completed nonsingular m-row square basis has m normalized pivot rows,
  each with at least one stored pivot coefficient. Thus at least4,331–6,416
  live pivot entries would be needed, already beyond the4,096 live-nnz cap.
  A singular basis is unresolved instead; it is not a way around this limit.

This does **not** estimate actual fill-in, time, rank or bit growth. Rational
coefficient bit size was not scanned, and native basis statuses were not
generated. Even removing the first size check would not make the present
constructor capable of completing these obligations. Raising caps without a
separate sparse-construction design and controls is not an approved remedy.

## Native basis mapping cannot be inferred from old success

The four saved `proposal/native.json` files have a point, solver success/status,
residuals/marginals, objective and time metadata. None has the current adapter's
bound `basis_valid`, `column_status`, `row_status`, native version/options or
submitted/before/after model snapshots. Every file is matched to its archived
hash and original LP/statement identity; exact dimensions agree with the prior
execution archive.

Their mapping status is therefore
**UNDETERMINED_NO_BOUND_BASIS_CAPTURE**, not `MAPPED_HINT_ONLY` and not an
observed `UNSUPPORTED_MAPPING`. In particular, the fact that these LPs have
equalities does not prove their native bases contain basic equality-row
variables. Conversely, small residuals, marginals and optimal native status do
not identify a unique original-coordinate basis. No near-active guessing is
used to fabricate the missing basis.

The old code used SciPy `linprog(method='highs', options={'time_limit':...})`.
It did not bind the new adapter's highspy1.14.0/simplex/presolveOFF/scaling0
configuration. A future native capture would be a new execution identity,
not a relabeling of those four old solves. Their existing
`NOT_EXACTLY_FEASIBLE` and unresolved outcomes remain sealed.

## Reproducibility and evidence boundary

`basis_compatibility/review.py` traverses the existing ordered30 obligations,
resolves each weighted export by its unique archived SHA256, checks the parent
archive binding, reads its original CSR dimensions and binds its canonical LP
hash. For the four prior jobs it additionally checks their original dimensions,
native record hashes and obligation identities. The result contains compact
metadata only; no raw source matrices, checkpoints or native vectors are added
to Git.

```
/data1/Kane/miniconda3/envs/act-py312/bin/python -m unittest basis_compatibility.tests -v
```

The report/audit CLI uses no-overwrite outputs. `collect()` can be invoked in a
fresh process to recompute and compare to the archived JSON without creating a
new report or executing a solver. A shape-pass control explicitly remains
`NO_STATIC_BLOCKER_NOT_RUNTIME_APPROVAL`: dimensions alone never approve a
diagnostic freeze. All runtime compatibility decisions would also need their
own basis mapping, bounded arithmetic and supervision evidence.

No inference is made about network safety, LP infeasibility or whether the
current relaxation can prove the property. Network→HZ, guards, route exclusions
and F0 lowering remain upstream trusted components. No real bound or evidence
was newly constructed; the only counts are from existing serialized objects.

## Required next research decision

The requested compatibility prerequisite is complete, but freezing a meaningful
real diagnostic is blocked. Before such a freeze, separately scope an
original-coordinate **large sparse basis/anchor construction** design, including
how any redundant-equality/native row-variable mapping is handled without
silently deleting obligations, and explicit fill-in/bit/operation budgets.
Validate it on controlled structures before changing the supported-size
contract. There is no need to extend solver time or repeat old optimization
merely to demonstrate the already-determined size rejection.

This report is not permission to implement that extension, raise limits,
re-select a more favorable LP, alter the original relaxation, or reopen the
sealed diagnostic execution. Those would be separate research decisions.
