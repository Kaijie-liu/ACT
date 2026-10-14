# Consumer only changes cannot close full topology admission

**Do not implement another isolated traversal or zero-encoding optimization as
a full-topology capacity fix.** Even giving all map and buffer access away for
free leaves a 9,289,728 Work lower bound for checking 504 present endpoints with
the current three logical factor passes. The unchanged source path independently
refuses before the consumer. This narrows the next decision from a local buffer
API to a joint source, evidence and resource contract.

This static assessment binds `17c982185a4b24b7788036069ca340585adbaa9f` and the
[implemented consumer capacity](hz_pair_scoped_capacity_20261015.md).
The [analysis](../scripts/assess_consumer_only_options.py) and
[machine record](hz_consumer_only_decision_20261015.json) read source text and
fixed shape metadata only. The candidate-8/check-2 implementation, all quotas,
mathematical checks, numerical gates and 300-second budget are unchanged.

## Eliminating access overhead is not enough

Consider a deliberately optimistic counterfactual: all forward/reverse mapping,
presence and payload reads, buffer writes, proposal generation and pullbacks cost
zero. Keep the existing merge and two norm passes in `Session._consume_block`.
For each factor, each of these three passes still has one logical coefficient
visit and one exact sum check. The lower bound is therefore `6*n*E`, not merely
the `3*n*E` arithmetic-call count. Final scale-group sums and all auxiliary
factors are omitted.

| Fixed endpoint inventory | Input factors | Logical body Work lower |
| --- | ---: | ---: |
| Minimum 252 endpoints | 3,072 | 4,644,864 |
| Maximum 504 endpoints | 3,072 | **9,289,728** |

The maximum inventory cannot fit 8M even under this free-access assumption.
The minimum row **does not by itself prove refusal** for an improved consumer.
With the unchanged producer and independent source checker, however, the two
first-layer weight scans add `2*6684672`, giving a lower of **18,014,208** even
at 252 endpoints. Source checking also retains its independent
`126632 > 65536` root-membership refusal.

These are implementation-and-schedule lower bounds, not complexity lower bounds
for every equivalent MoE verifier. The full conservative roster does not assert
that all pairs are reachable; the actual gate degeneracies were not inspected.
Fusing passes, proving an algebraic shortcut, changing the source construction,
or changing the counting policy would require a separately specified contract.
Removing a charged logical visit while still doing the same work is not an
access optimization. No lower bound here supplies a sufficient new quota.

## A zero descriptor can fix one native byte envelope

A concrete draft can replace the stored zero-role vector with an explicit
full-roster zero descriptor, while preserving both candidate roles. Use a new
magic/schema; retain the 128-byte header, 72-byte endpoint binding and both
1040-byte advisory scalars. For `M` ordered dual rows and scalar width `kappa`,
the endpoint stride would be `2152 + kappa*M` instead of `2152 + 2*kappa*M`.
The role tag, request/source/contract identity, pair-derived ordered equality
and inequality dimensions, query and slot must still be checked. Missing both
roles remains distinct from this present zero role.

| Draft at 504 endpoints and 84 tiles, M at most 2700 | Maximum frame | Whole stream |
| --- | ---: | ---: |
| Native binary64 proposed values | 190,144 B | **11,981,760 B** |
| General rational proposed values | 22,481,344 B | **1,416,327,360 B** |

The native envelope fits the existing frame and cumulative byte caps; the
general-rational envelope does not. General values must not be rounded or
silently dropped to use the native profile. These are conditional format
uppers, not actual generated files or a new implementation.

The expansion argument is simple: for every bound ordered row, the descriptor
denotes exactly the rational zero that the old frame stored. Keep the proposed
role dense and unchanged. The consumer still checks both role dimensions,
visits every row, checks signs and zero semantics, and calculates both lower
bounds. A legacy adapter must canonical-decode and validate **each original
zero value before conversion**; replacing an unvalidated or polluted old vector
with the descriptor would silently turn rejected evidence into accepted evidence.

This distinction matters for costs. `Tile.weight` decoding itself does not call
`Work.visit`; the consuming row/role loop does. A descriptor can reduce bytes and
decode work without reducing that logical loop's Work. A new native descriptor
producer could avoid encoding/writing each zero value; an adapter from old
vectors still pays to inspect them. No such native or adapter path is implemented
here, and the old 22,868,160-byte stream envelope remains the current format's
bound. Shorter zero encoding never removes the second norm calculation.

## Checked traversal requires stronger storage ownership

The current `Scope._sealed` prevents replacing scope fields, but its forward and
reverse `Block` objects are not sealed. `RootTable` buffers are also writable.
Existing per-access membership/inverse checks remain necessary under this API.
A new cursor must not infer immutable bytes from an immutable object wrapper.

A reviewable future traversal contract would require:

- truly sealed mapping/membership bytes, or a private exclusive lease with an
  invalidation generation; no writable aliases reachable from proposal code;
- binding to the same checked forest, factor identity, pair, owner and cumulative
  Work object, with no reuse after transfer, release, abort or source replacement;
- independently checked forward/reverse correspondence, ordered full global
  factor coverage and outside-scope zero semantics, never merging equal scales;
- exact scalar validation and unchanged intermediate check points, with time
  checks throughout; preparation, sealing overlap and invalidation are charged;
- objective and dual pullbacks remain separate. No cancellation before a check
  and no unregistered change to factor-pass order or receipt acceptance.

This describes why blind caching of a local index is insufficient, not approval
to add a cursor. Even a correct cursor cannot remove the logical-body lower
above, and it does not affect source root or first-layer checking.

## Next decision and stopping rule

The [direct-source and unique-root draft](hz_unique_root_protocol_20261014.md)
is the relevant whole-path design; the new `RootTable` is **not** its source
implementation. The next contract must select the actual source identity and
unique-value/membership representation, preserve all ordered source uses, and
specify how generation and independent checking are charged. It must then bind
the same source and common resource policy to both complete internal methods.

The old [complete Work ledger](hz_full_resource_ledger_20261014.md) binds the
earlier `produce(width=1/8)` path, not the current owned Driver. It cannot silently
supply the new complete upper. Reconciliation still includes RootTable, per-tile
Scope, sparse descriptors, scoped access/scalar checks, the 252 dual check blocks
at the maximum inventory, wire decoding, typed-frame publication and release,
source/native coexistence and portable replay. Full runtime/RSS and capture
remain part of that contract, not components that can be omitted because a local
arena fits.

**Keep the current complete path refused. Do not freeze more isolated consumer
controls or a full-shape retry on this evidence.** The zero descriptor is a
useful format subdesign to retain in the joint contract, not a sufficient next
implementation. Complete Work/RSS and an absolute policy ceiling remain unset.
A resource-policy decision cannot be hidden in a representation change; real
or GPU execution remains separately authorized after complete admission.

Four new static arithmetic/binding tests pass alongside the existing static and
navigation tests. Two read-only AI reviews independently checked the logical
lower, format calculations and upstream separation; they are not human review
or reproduction. No source, model, native optimizer or GPU was run, and no
scientific result or G1–G6 completion claim changes.

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/assess_consumer_only_options.py --check-report docs/hz_consumer_only_decision_20261015.json
```
