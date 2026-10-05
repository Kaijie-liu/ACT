# C10 defining-predicate census: material structure found, no rewrite yet

Recorded on branch redu-hz,2026-09-08. Exclusive completed transaction:
results/c10_predicate_census_20260908_v1/.405 tests passed in4.49s; test/worker
exits0; no source/provenance drift; supervisor8.791663751006126s. result.json SHA:
d3b470ce9eb923ce013f2a75236a5507c39a75d398c7db83010ef823ba824e9a.

Both input checkpoints were hash-checked before unpickling. Final and live
post-ReLU content hashes matched their sealed results. The full C9 predicate
prefix embedded in the final HZ was checked byte-for-byte, including absent
new-column coupling and unchanged right hand sides. All four source HZ content
hashes remained unchanged after the census. No solver or model-ingestion call,
no HZ transformation or publication, no source-checkpoint rewrite.

## Exact counts

- Raw final HZ255298 continuous/1350 binary factors,244540 equalities,
  2700 inequalities;11201930 total coefficient nonzeros, including40000 Gc.
- Original continuous prefix11708 protected; C9 MAIN factors243162;
  radix28; later native ReLU continuous400. Only200 continuous columns appear
  in the final value map. ALL243162 MAIN factors are final-value-dead.
-243158 MAIN variables have a directly represented defining pivot;4 logical
  definitions are radix-packed and are not misclassified as ordinary pivots.
-203173 dead MAIN variables have a single remaining predicate consumer.
  204116 have a strictly negative INDIVIDUAL structural nnz upper bound.
  These are NOT realized or jointly additive compression counts.
-106721 homogeneous two-continuous-column aliases have an exactly reversible
  ratio of magnitude<=1; their local eliminated-variable box is redundant.
  Only357 have a power-of-two ratio, so restricting to power-of-two aliases
  alone would miss most of the measured ordinary structure.
- All322936 remaining coefficient occurrences of these aliases were tested.
  101089 aliases have exact float64 products at EVERY occurrence;
  105925 have all products within unchanged[2^-20,2^40];
  100727 satisfy BOTH conditions.

The odd-significand exact-product test was checked against Fraction, including
2048 generated arithmetic pairs across normal/subnormal ranges. These are toy
arithmetic tests, not sampling or attack attempts on benchmark inputs.

Logical census work upper118064528 <256M. Measured analysis0.15608117077499628s,
traced peak93326908 plus48544 metadata; conservative resident-growth upper
311992320 bytes <1GiB. Worker2.4069372517988086s; diagnostic peak1835908KiB.
No full-pipeline speedup is inferred from these timings.

Per-MAIN-factor numeric table14350178 bytes, SHA:
47a88ec370e60fbee44cb5a54217355df2e33a17cb86cc5c28a07e888073ea8a.
It records columns, owning rows, support/degree, directness, liveness, alias
parents/ratios, individual nnz bounds and exactness/window flags; aggregate
degree/row-width/exponent histograms are in result.json.

## Boundaries and next action

100727 is NOT a proved removable-factor count. Different aliases may map to
the same parent column, requiring exact addition; aliases may depend on other
aliases, requiring a simultaneous substitution proof. Neither obligation was
established by this census, which explicitly reports both false. No HZ was
rebuilt and no global candidate-storage reduction is claimed.

The next substantive hypothesis is a frame-preserving, continuous-only exact
alias quotient. Its selection must depend only on defining equations, liveness,
box redundancy and exact arithmetic; protect all original-prefix/radix/ReLU
phase slots and every binary factor. A deterministic dependency-independent
alias frontier can avoid compound ratio products while an independent exact
sum check covers parent-column collisions. Preserve original global column
identities (potentially leave eliminated slots unused for ordinary lowering)
and retain a bounded reconstruction map rather than renumbering shared frames.

A live implementation must also discharge its temporary original matrix after
checking the transformation and retain the reconstruction certificate. Merely
keeping both full old/new matrices behind proof objects could INCREASE whole
live storage/entries despite a smaller local matrix; that cannot be counted as
the required simplification. Exactness, all boxes, coefficient window, strict
total nnz, complete reachable-state and unchanged construction gates still
precede any new ordinary-terminal attempt. This is a design constraint, not
permission to delete native prefix caches or change the frozen comparator.

Formal1870/2413 and independent E061/400 unchanged; formal/capability gain0.
C9's terminal UNKNOWN/base_unknown is not rerun or reclassified. The goal is
active, but no CIFAR/shadow/family/full-replay advancement is claimed here.
