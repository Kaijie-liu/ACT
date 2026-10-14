# Bounded exact workspace and candidate ownership controls

Frozen implementation scope, 2026-10-14, based on
`2d99db96068b559baabd23089d47ea236e64bd5b`. The user approved optional bounded
representation, buffer and ownership interfaces and differential/fault controls.
No cap increase, dependency change, real model or GPU execution is authorized.

## Implementation contract

1. Add request-owned fixed-size exact-value buffers. Persistent values are
   encoded integer magnitudes, not retained Fraction/PyLong objects. Support
   dyadic vectors and general rationals without rounding. Existing stored and
   working bit limits remain 4096 and 8192. Fraction arithmetic temporaries
   remain an explicit unclosed part of full runtime memory.
2. Add a unique-root lookup for an already independently checked V3 forest.
   Bind root ID to node ID and its owner; check every ordered view membership.
   Aliases do not merge different nodes, centers, owners or dual obligations.
   This is the consumer-side interface: existing upstream source construction
   and checking are still charged and are not claimed to have bounded storage.
3. Use a direct typed candidate tile with request/forest/contract/tile/slot/query
   bindings, complete dual dimensions and explicit proposed/zero roles. Native
   binary64 and generic rational encodings are distinct, exact profiles. Do not
   reconstruct old candidate JSON or charge its hypothetical logical bytes.
   Keep the old JSON path unchanged as a reference.
4. Both accessors must call one mathematical consumer: original dual sign,
   residual, scale grouping, zero candidate, strict threshold and complete
   aggregation checks. Opt-in dense exact maps replace persistent objective,
   alpha and residual dict values; preserve the original arithmetic check points.
5. One owner and one live tile per invocation. Publish immutable candidate bytes
   before consuming them; count the sealing copy, publication and check. A
   stale/cross-owner/partial/late tile or any exception poisons that invocation.
   Cleanup releases owned references even after the deadline, never resets Work
   or cumulative byte accounting, and never turns partial evidence into success.

The existing 4-MiB tile, 16-MiB candidate total, 128-MiB component/arena ceiling,
8M cumulative Work and at-most-300-second absolute deadline remain ceilings,
not promises of full-model admission. New memory counters describe owned bytes,
not RSS or external references. Existing source, native and Python temporaries
must still be included in a later complete cost contract.

## Fixed control matrix

Use only existing synthetic captured-source fixtures `mixed`, `deep`, `half`,
and `guarded`, widths 1 and 8. No trained checkpoint, dataset, native optimizer,
Torch or GPU is loaded. Compare every proposed/zero/selected/lower value and
complete result with the old Session and the independent expanded-matrix oracle.
Check a generic non-dyadic candidate separately; it must not be rounded into the
native profile. Different candidate encodings have different evidence hashes.

Fault controls cover bit/byte/allocation limits; root owner/constant/membership
identity; immutable versus mutable aliases; cross-request/owner and stale tiles;
truncation/trailing data, dimensions, nonfinite and noncanonical values; wrong
dual signs and zero descriptors; missing evidence; publication failure; cutoff
before/within/after checking; repeated terminal calls; cleanup and cumulative
accounting. The original source mutation and wire portability regressions remain
required. No timing competition or full-capacity workload is included.

Acceptance means optional interfaces work on these bounded controls and reject
the specified faults. It does not mean full RSS is closed, a captured source is
bound to a trained checkpoint, new mathematical certification power is shown,
or any G1–G6 objective is complete. Record exact code/protocol identities, tests,
remaining costs and failure boundaries before considering a later stage.
