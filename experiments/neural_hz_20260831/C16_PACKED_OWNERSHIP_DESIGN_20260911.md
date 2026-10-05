# Next generation hypothesis: exact packed consumption ownership

This is a derived design after completed C16, not a target preregistration,
implementation, work saving or promotion. Same S0 structure and original caps.
The required norm premise now has a complete authenticated compositional proof.
The remaining expensive discovery must identify each factor's predicate consumer
without another full CSR/CSC incidence pass. No target-specific row list.

## Why simpler counters do not suffice

C8.graph already computes exact support counts during reverse needed-mask
propagation, but discards the counts to a boolean. Counts can prove one use,
not identify its row. Keeping only a unique-row-or-conflict flag loses information
when a later alias removes edges. Rescanning coefficients to recover the row
would reinstate the cost we need to avoid. A weighted sum alone is also not
a sound uniqueness test. Do not propose these lossy metadata alternatives.

## Exact add/remove invariant to implement and prove

Assign every physical predicate row a stable row UID in [0,R); do not relabel
UIDs when compacting storage. With proved R<=2^20 and at most one canonical
nonzero coefficient for each (row,column), choose B=2^40. For each column keep

    packed = count*B + sum(consumer_row_UIDs).

The sum is <R^2<=B (strict because UIDs<R), count<=R, so no carry contaminates
count and packed<2^61, safely inside int64. Decode count=packed//B and sum=packed%B.
Exactly when count==1, sum is the unique consumer UID. Adding/removing a known
incidence changes packed by +(B+UID) or -(B+UID). Enforce nonnegative, count/sum
and complete-UID-domain bounds; no wraparound, hash collision or probabilistic
fingerprint is involved. This proof applies only to canonical incidence, not
uncollapsed duplicate summands or arbitrary multiplicity masks.

An integer addition can therefore carry both count and row sum during an
existing structural edge visit. A new ownership kernel may reuse the boolean
support traversal geometry, but must actually replace the discarded reverse
count operation, not run a second full weighted traversal and call it free.
The original SupportEngine.compute rejects masks>256M: do NOT weaken that API
or pass a packed word through it. A separately proved internal packed-ownership
operation has its own int64 range proof, the SAME work/entry limits, and distinct
cache keys including mode, B, UID layout and support binding. Its returned word
is metadata, never an HZ coefficient or a changed coefficient-window threshold.

## Integration obligations, still open

1. Establish stable physical row UIDs despite needed-mask-dependent MAIN slots,
   original predicate rows and radix insertion. A logical-graph label is not
   automatically the final physical consumer: a packed/radix row can move an
   edge. Either prove/charge the exact remapping and small emission-time deltas
   or reject before publication. Do not silently ignore those rows.
2. Account for actual alias rewrites: removing a defining row removes its known
   parent incidence; substituting in a surviving consumer adds that consumer's
   parent incidence; exact merging/cancellation removes duplicate or zeroed
   incidences. Update from the same already-owned hit/collision records with
   explicit precharged additions, not a second full scan. An erased selected
   factor's remaining count must become zero.
3. Root value liveness changes only when native ReLU consumes it. The69 existing
   affine-internal and199 ReLU-consumed C15 pairs require the SAME ownership
   invariant through this event sequence. A preactivation-only implementation
   does not qualify all268. Stable-positive output uses remain protected; all
   phase binaries and original global IDs must survive unchanged.
4. Use C16 only within an authenticated complete generation transaction. Bind
   scalar row-box queries to that transaction, not just exact=True. Complete
   proof authentication must be genuinely reused from the existing boundary
   or charged again. No optimistic capacity subtraction for unimplemented reuse.
5. Emit the surviving CSR payload once at the native ReLU assembly boundary;
   redirect an erased producer block to its consumer destination rather than
   build a complete old final HZ and copy it through a postpass. Charge actual
   row metadata/reconstruction certificate and source ownership. All temporary
   full ownership tables must be discharged before publication or fully counted;
   retaining~2MB to save6005bytes would fail the physical reduction condition.

Before a new target: prove packed arithmetic and signed add/remove behavior
against explicit incidence sets, then test alias collision/cancellation, DAG
sharing, radix insertion, UID compaction, value-to-predicate ReLU transition,
source/phase mutation and overflow/cap fail-closed guards. Derive a complete
coupled work bound BEFORE the target; C10 headroom remains1249923, not a promise
that these operations fit. Then a new preregistration must compare the COMPLETE
generated ownership census against actual sparse incidences and all268 C15 pairs.
If it fails, retain the negative result without narrowing to the easiest subset.

Even affordable fusion of268 pairs is not a large capability claim: C15 removed
only536 of~11M coefficients. The full objective remains stronger Neural-HZ and
new validated decisions with1870/13-family zero regression, never receipt/count
microbenchmarks in place of the required terminal/shadow/full-replay evidence.
