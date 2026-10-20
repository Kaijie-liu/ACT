# Gate margin relation implementation and control readiness

2026-10-20, implementation started from clean `e56515001` on
`feat/moe-route-verification`. The default-off relation is wired into native F0
and the existing N1 segmented builder. It has had separate AI source reviews and
an AST parse, but **no operator execution or frozen M3 control run yet**.
This is not evidence of performance, native correctness or G1/G3 completion.
The [static audit](gate_margin_relation_static_audit_20261020.md) supplies the
mathematical claim and its stronger-baseline limitation.

## Implemented path

`act/back_end/moe/gate_margin_rows.py` captures independently owned source
snapshots and binds entry, two expert blocks, joint HZ, padded input and router.
The caller declares the canonical expert-slot identities and supplies complete,
injective router-constraint maps into the entry. The checker reconstructs the
shared/private continuous and binary layout, every output and retained row.
It does not infer checkpoint or expert identity merely from a block position.
The caller must explicitly declare selected-softmax top-2 at unit temperature.
This semantic premise is bound and reported, not independently proved from
the network; arbitrary convex mixture gates do not justify these sigmoid rows.

The binding is immutable canonical bytes with a hash, limited by the existing
8 MiB header cap before parsing. Native CSR indices/pointers must be integral
one-dimensional buffers. Unique entries may be sorted without arithmetic;
duplicate coordinates are refused rather than merged in floating point.

`build_weighted_top2_f0` accepts optional `gate_margin_binding` and
`gate_margin_work`. Both must be provided together. The hook receives the
actual lambda slot, product slot, center and radius from that construction.
Absent both arguments, no new source capture, checker, Work ledger or support
query runs. The returned encoding has an additional optional evidence field.
The existing N1 builder forwards the same binding and Work to every retained
segment; it does not create a per-segment allowance or change query order.

The first version adds exactly two inequalities, on one justified sign side:

```
m >= 0:  lambda >= 1/2, lambda <= 1/2 + m/4
m <= 0:  lambda <= 1/2, lambda >= 1/2 + m/4
```

Margin coefficients are reconstructed exactly from the original stored router
rows. Only exact factor-box sign or one directly proportional, retained
inequality is used. The latter is checked on the active joint domain, not just
on the router. No solver, additional support query or new sigmoid envelope
computation is used by this mechanism.

Only unchanged domains or the same two appended inequalities in router, joint
and input are supported. Private expert rows may precede these added rows.
This validates retention, not that their union covers the parent request:
closed-slice coverage remains an explicit upstream premise.

## Independent acceptance and fallback

`scoped_source/check_gate_margin_rows.py` has standard-library dependencies
only. It captures a bounded private JSON copy, reconstructs sources and margin,
checks the sign premise, and independently assembles the two ideal rows.
It checks the binary64 row coefficients with exact residual compensation and
compares the complete old/new F0 snapshots. Output, equalities, old
inequalities and factor widths must stay unchanged.

The three successful inspection statuses have different meanings:

| Status | Native action | Meaning |
| --- | --- | --- |
| `APPLIED` | Append the two checked rows | Conditional row inclusion only |
| `NO_SIGN_PREMISE` | Return the original F0 | No supported exact sign proof |
| `GATE_EMBEDDING_REFUSED` | Return the original F0 | Stored center/radius do not outwardly embed the supplied gate interval |

Identity, schema, mapping, wrong native lambda recipe and other malformed-source
failures are errors, not normal fallback. Work/deadline exceptions propagate.
The receipt cannot establish SAFE and explicitly sets source completeness,
deployed floating-point SAFE and hard-budget supervision to false.

The native lambda recipe is checked against the supplied binary64 gate bounds,
but the checker does not rederive the original four McCormick rows or output
projection. Its guarantee remains
`CHECKED_ADDED_ROWS_GIVEN_SHARED_ROUTER_AND_F0`.
Network/input/guard lowering, canonical expert identity, original gate/F0
construction and full slice/route/property coverage are not newly proved.
Native solver acceptance remains the existing numerical policy.

## Accounting and finite capacity

All new capture, traversal, exact construction and reception use the same
caller Work object. JSON/CSR scalar visits and exact checked expressions are
charged; serialization and native sparse operations also consume the enclosing
wall-clock budget. No new Work instance or resource policy is created in these
modules. This is not a claim that unmodified upstream native instructions have
been converted to ACT Work units.

The given-source limit retains the old finite pair-map boundary: at most 128
total factors, 128 output coordinates and **256 total constraints**, not 256 of
each kind. Original F0 may add its two continuous factors and four product rows;
the enhanced F0 adds two further inequalities. Structural checks require exactly
those differences. This is not eight-expert full-shape admission.
The existing 8M Work, 128 MiB batch artifact, 8 GiB AS, 2 GiB sampled RSS and
300-second limits are not changed.

## Static review corrections

Separate read-only AI reviews found and the implementation corrected:

- Integral CSR buffer checks before conversion to Python integers.
- Binding length/hash checks before JSON parsing.
- An independently owned checker input instead of relying only on rehashing a
  caller-owned mutable record; external context/property checks run against
  that owned copy and separately captured expectations.
- A combined 256-constraint source limit, avoiding accidental admission of
  256 equalities plus 256 inequalities.
- A conservative JSON size estimate including non-BMP Unicode surrogate pairs.

The reviews checked the two sign directions, direct-row premise, lambda
substitution, binary-column shift, outward compensation and unchanged-F0
condition. They are not an independent human review or an executed test.
The four changed/new Python modules passed AST parsing using existing act-py312
with `-I -S -B`; the command imported no ACT modules and executed no operator.

## Next finite control

Freeze an exact case/subfault/call roster before one execution. Cover native
off/on construction, positive and negative sides, constant/equal experts,
private continuous/binary factors, retained-row order, unsupported-sign fallback,
source/property identity, duplicate or malformed CSR, inward rows, lambda/product
column mistakes, source pollution and deadline/Work/receipt handling.
Source and gate inputs must be literal synthetic controls, not checkpoints.

Keep support, optimizer and model-forward counts explicit; supplying ranges to
construct F0 must not be reported as a solved native request. Analytical
two-slice separation and the positive four-slice N1 reference remain separate
from measured complete-request cost. Missing full-request obligations are not
validated by this row-only receipt.

Reuse the public existing owned-session supervisor and hard limits. Its
live-RSS-unavailable refusal remains possible; do not suppress it, add an exit
delay to evade it, or repair/retry M1b. If it recurs in M3, archive the prefix and
report overall failure even if worker algebra succeeds. No M3 attempt has yet
been consumed; no real model, GPU, external deployment or M2 launch occurred.
