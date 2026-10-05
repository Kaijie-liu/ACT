# Next exact sparse append-ownership overlay hypothesis

Design only; no C23 implementation/target yet. C22 proved a11038-entry MAIN run
index on ALL reserved UIDs and physical rows, but did not retire any original
graph fields. C21's failed full storage gate remains CLOSED. Do not rerun it,
alter its comparator, release a required oracle or count this design as a pass.

## Source-backed opportunity

C21's completed phase audit copied243162 ownership words (1945296bytes) to add
only200 tracked MAIN incidences: its unchanged4-per-incidence ledger charged
800. This is a whole-vector copy for a genuinely sparse append event. The full
new EQ/INEQ row counts were200/400, but only references to existing MAIN factors
enter this ownership ledger; fresh phase/slack slots are outside its domain.

Represent the phase query state as immutable original ownership plus a sorted
packed list of actual appended (MAIN column index,row UID) events. Both fields
are below2^20, so one40-bit pair fits exactly in uint64. Canonical rows cannot
repeat a column, and fresh row-UID ranges cannot overlap old ones. For a queried
column with k events and UID sum s, the exact word is

    original_word + k*2^40 + s.

This is a sparse integer metadata overlay, NOT an approximate HZ relation or
probabilistic membership structure. Handle multiple appended rows consuming
the same MAIN column; do not assume one event per column from this target.
Binary coefficients, RHS, shared latent slots and native phase equations stay
unchanged. Query event ranges by packed column bounds; checked count/sum/domain
limits must guarantee exact nonoverflow. No base-owner mutation or rollback is
needed. This supersedes the more complicated reversible-copy-avoidance idea
unless evidence shows it cannot meet the same complete proof/resource gates.

## Required proof and measurement before integration

Freeze a NEW standalone overlay version and test zero events, repeated columns,
EQ/INEQ and binary consumers, negative/nonzero coefficients, invalid/duplicate
UIDs, domain/overflow and before-operation caps. Verify exact prefix/frame/RHS
against the full native post-HZ before extracting only its appended rows.
Independently compute every actual post-HZ ownership word and compare queries
for ALL243162 MAIN columns, including untouched and erased ones. Discover ALL
268 unit pairs using the same structural rule, checking the sealed table only
after discovery. A saved post-HZ remains a diagnostic oracle, not a generator
shortcut or permission to execute a new NN phase.

Every event scan/packing/sort, query binary search and scalar sum, numeric
buffer and diagnostic oracle must be charged; no hiding arrays in metadata or
returning a dense overlaid vector while counting only the sparse list. Keep
both complete external HZ oracles and the original native snapshot under the
eventual same boundary. The independent actual-vector oracle is an explicitly
temporary proof buffer as in C21, not a retained algorithm state.

At most200 retained event entries are suggested by C21's actual incidence work,
but the new standalone must verify its own complete event population. If this
count is proved, then node retirement+11038 run entries+200 sparse events
instead of the dense phase copy would give the OLD ledger arithmetic52240284,
188516 below the fixed52428800 bound. This is NOT a measured compact-state
result; receipts, owners, transient allocations and actual binding may consume
that margin. All graph consumers still need replacement as documented in
C22_UID_CONSUMERS_20260911.md.

The independent run builder itself costs1469998 and is not an affordable
postpass. Even a successful overlay therefore does not authorize a fresh whole
generator until run emission/final-row filtering and proof-state closure are
actually fused with a complete work budget. Source proof must happen before
retirement; a self-sealed digest/flag cannot replace the independent checker.
No existing production path, historical data, binary phase, or full-replay
retention requirement changes. Formal1870/2413 and separate E061/400 stay fixed.
