# S0-C1 Graph-Lineage Audit

Recorded on 2026-08-31 on branch `redu-hz` at commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. This is a read-only topology and
specification audit. It changes no production source or result, gives no
formal/E0 credit, and leaves the formal baseline at 1,870/2,413.

## Finding

The frozen S0-C1 rule says that composed Conv fusion never crosses ADD or
reshape. Production currently cannot prove that rule: `SparseHZAffineTerm`
stores only `source` and `operators`; graph ADD concatenates terms, and
reshape-like nodes return the expression without an operator marker. The
isolated `Barrier("ADD")` tests model an object that the production graph never
creates.

A type-only terminal matcher can therefore see identical tuples for a legal
straight-line tail and for an inner Conv before ADD followed by an outer Conv
after ADD. Absence of a marker is not evidence that no boundary occurred.

## Minimal exact metadata

Each production term needs immutable
`barrier_cuts: tuple[int, ...] | None`. `None` means lineage is unknown and
rejects the entire candidate request; production constructors explicitly use
`()`. A cut `c` represents:

```text
operators[:c] | graph boundary | operators[c:].
```

Cuts must be non-bool integers, strictly increasing, unique and within
`[0, len(operators)]`. A proposed core from `inner_index` through
`outer_index` crosses a hard boundary exactly if:

```text
exists c: inner_index < c <= outer_index.
```

This admits a core wholly before or wholly after a boundary and rejects only a
core spanning it. A single `composition_floor` is conservative but not exact:
it would also reject a complete tail that occurred before a later ADD.

State transitions are:

- a new materialized/nonlinear source starts at `()`;
- linear append preserves cuts;
- ADD returns new left/right terms with their current operator length appended
  and then concatenates them, never mutating shared inputs;
- reshape, flatten, squeeze and unsqueeze return a new expression with the
  current length appended to every term;
- repeated cuts at one position are deduplicated;
- a materialization/checkpoint resets lineage on the new source; and
- a successful core replacement preserves cuts at/before the inner and shifts
  cuts after the outer left by `outer_index - inner_index`.

Lineage belongs to a term, not an interned operator: two terms can share an
operator object while having different graph histories. Descriptor content
identity therefore excludes cuts, but every term is checked before interning.

Missing/malformed lineage rejects before reserve or compile. A normal
cross-boundary nonmatch preserves the original term, source, operator tuple and
bias by identity and proceeds only through the existing capped fallback.

## Tiny iid143 ReLU36 consequence

The strict v1 target is structurally zero-hit, independent of the unfinished
runtime census. Trial 8 and the graph trace establish:

```text
ReLU28
 -> Conv29 / Scale30 / Bias31
 -> ADD32                 (four lazy terms)
 -> Conv33 / Scale34 / Bias35
 -> ReLU36                (2,180 selected rows)
```

There is no lazy checkpoint. Conv33 is the sole Conv after ADD32 and before
ReLU36. The terminal matcher must choose Conv33 as the outer Conv, so every
possible inner Conv lies before ADD32. For all four candidate terms:

```text
inner_index < cut_ADD32 <= outer_index.
```

Hence the literal frozen S0-C1 rule yields zero legal two-Conv tails at this
target. Trial 9 can record the authoritative real tuple/counter evidence but
cannot alter the topology.

The preregistration also says each residual term is planned separately.
Mathematically, distributing a shared outer affine map is exact:

```text
B * (sum_i T_i s_i + d) = sum_i (B * T_i) s_i + B * d.
```

But under graph-lineage semantics that operation does cross ADD. It cannot be
silently reclassified after observing the target. If pursued, it must be a
new S0-C2 preregistration with its own theorem, bias/predicate/witness tests,
transaction gate and closure conditions. S0-C1 v1 remains a structural miss.

## Required lineage tests

At minimum, the real-term test matrix includes:

| Operators | Cuts | Result |
|---|---:|---|
| `C,D,C` | `()` or `(0,)` | accept |
| `C,D,C` | `(1,)` or `(2,)` | reject cross-boundary |
| `C,D,C` | `(3,)` | accept core before boundary |
| `C,D,C,Dout` | `(3,)` | accept core; preserve `Dout` |
| `prefix | boundary | C,D,C` | cut before inner | accept |
| `C,D | ADD/reshape | C` | cut inside core | reject |
| repeated/multiple boundaries | normalized cuts | reject iff any cut is inside |
| rewritten core plus later ops | remapped cuts | pass a second lineage audit |
| `None`, bool, duplicate or out-of-range cut | invalid | reject transaction before compile |

A real-shaped `Conv29 -> D -> ADD32 -> Conv33` fixture must report zero matched
and four cross-boundary terms without constructing full Conv CSR. Default-off
behavior remains byte-identical. No candidate is eligible for target, family
or full replay until these real-lineage tests pass.

## Decision

The S0-C1 production patch map is corrected to use exact cuts rather than a
single floor. The correction does not rescue the registered target. Formal
gain remains zero; the combined isolated adapter explicitly records that it
does not represent production ADD/reshape lineage.
