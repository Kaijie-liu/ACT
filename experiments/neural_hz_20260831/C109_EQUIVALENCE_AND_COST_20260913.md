# C109 exactness, complete numeric costs and source relevance

## Exact shared partial identity

Let each original input-transform definition be p_v v=sum_j s_j P_j, with
P_j=sum_i T_ai 2^e_i xi_i and fixed source IDs. C109 canonicalizes the FULL
two-parent dyadic term tuple, removing only a common sign. If a tuple has at
least four occurrences, introduce one positive dyadic bounded factor u by
p_u u=P. Replace every selected occurrence of P by p_u u, with its original
sign. Original V pivots remain; all old auxiliary IDs shift bijectively above
the new prefix. No matrix equality follows from a hash alone.

For every old source assignment, each new p_u>0 gives one unique extension.
Exact absolute row norms bound u in[-1,1]. Conversely, its exact equation
forces P=p_u u, so substitution recovers the original V definition. Existing
channel sums, output equations/RHS and source EQ/INEQ predicates keep their
meaning under the coordinate bijection. Binary identity and all old scalar
inverse data are retained, not independently boxed or removed. Thus the two
extended HZs project to the same original coordinates wherever all native,
box and structural guards pass. This proof is independent of source phase.

Actual-literal Fraction projection checks EVERY output coefficient and
constant against a separate direct3x3 spatial sum in all15 complete fixtures.
Every introduced row's redundant box is proved from actual native literals.
Reconstruction uses actual equations, includes both binary phases and checks
unchanged source EQ/INEQ and output rows. The fixed assignments illustrate
reconstruction; they are not the universal-equivalence proof or neural attacks.
Nonzero centers are retained in source arrays and exact unchanged output RHS,
not lost in the homogeneous input-deviation transforms.

## Why ordinary spatial overlap loses the physical gate

For T from C85, the absolute column-use counts of the four horizontal input
transforms are(1,3,3,1). With tile origins spaced by2, an interior physical
column is used3+1 or1+3=4 times; outer columns are used at most3 times. For
injective channel/spatial IDs, a two-parent vertical tuple fixes its channel,
vertical origin and transform (the +/- middle transforms have distinct full
coefficient tuples). It therefore has at most4 horizontal occurrences.
Masking or dropping unused V rows only removes occurrences. Fixed nonuniform
coordinate scales cannot identify distinct source IDs.

Retaining the old V factors, one new partial with q uses adds3 defining nnz
and removes q input terms: delta nnz=3-q. Dense positive candidates have q=4,
so save only1 nnz but add one equation, factor and its records. For the FULL
declared native numeric payload, per added factor:

| Added numeric component | Bytes | Entries |
| --- | ---: | ---: |
| CSR row pointer |4|1|
| Row lower and upper |16|2|
| Variable lower and upper |16|2|
| Integrality |1|1|
| Owner occurrence count |8|1|
| Source ID |8|1|
| Inverse row/slot pair |16|2|
| Inverse pivot ID |4|1|
| Total excluding changed nnz |73|11|

Each native nnz has one float64 value and int32 column:12B/2entries. Therefore
delta bytes=12*delta_nnz+73*P and delta entries=2*delta_nnz+11*P, where P is
new partial count. In the dense injective case this is +61P bytes/+9P entries,
despite -P nnz. On gy by gx full tiles with C channels,
P=8*C*gy*(gx-1). All five preregistered dense grids exactly match this formula.
This is a proof for THIS representation retaining old V nodes and the declared
complete native payload, not a lower bound for every possible HZ encoding.
Even CSR values/indices/rowptr/RHS and variable bounds alone grow in this case;
discarding required witness/source records would not make it an admitted win.

The three-channel genuinely shared-ID fixture has32 new partials and336 uses:
delta nnz=-240, bytes=-544, entries=-128. This is a small complete NUMERIC
prototype saving. It is not full production Python/LIVE accounting, a real
source hit, runtime payment, an extra solved instance or a PLDI novelty result.

## Current source excludes that particular positive premise

Actual C107 birth emission allocates slots to all needed coordinates using
cursor+arange(rows.size), advances cursor across nodes and checks the complete
reserve. C107 tile_maps reads those slots and their powers directly; circuit
installation requires the parents to be retained original MAIN coordinates.
It does not identify separate channel coordinates merely because their values
or equations might happen to agree. C99's qualified row bound explicitly
requires these unique original slots and disjoint channel coordinates.

Hence the fixture's three channels with IDENTICAL physical latent IDs are not
a plug-in case in the current qualified source representation. This is a
source-level implication of its allocation/mapping contract, not a fresh scan
of all current arrays or proof that source functions can never be equal.
The per-operator new-partial proposal on these injective IDs is closed before
real-source/runtime integration: its ordinary overlap cannot pay its own rows.

Read sources:
- c107_birth_emission_v1.py allocation around lines94-103,
  SHAe6f79284a496b071b03eebdf9c42942c0a6faadd7901b2103ae790119ab18f01.
- c107_circuit_stream_v1.py tile_maps and complete MAIN postconditions,
  SHA18b1155750cfd64f745e3a7393f86a52e77e4ea4daef2fbdf8736a1c51d80f90.
- c99_unique_row_bound_v1.py actual uniqueness prerequisite,
  SHAcdb33e100bf65c931f1fd9f4157ef6bb741231070f91dcbbb01d40818703f102.

## Distinct follow-up: reuse existing generated factors

A read-only full scan of all15 SHA-authenticated C109 before-fixtures groups
actual input-only defining rows by their FULL normalized native coefficient
tuple (zero RHS, at least2 parents). It reads all auxiliary rows, not just the
positive case. Ordinary three-channel MASKED case08 has192 input-transform
rows,32 two-parent rows and4 proportional pairs:4 redundant-row candidates.
Shared-ID case09 has64 triple groups/128 candidates, but is source-inapplicable
as above. The other13 records have no full-recipe duplicates. Candidate counts
are not all-consumer/native-window/box/physical elimination certificates.

This observation reads only completed fixtures; no new actual network source
or MILP is loaded. SHA-bound complete NPZs are loaded with allow_pickle=False.
Posthoc measurement0.156930s, trace740442B+metadata286048B, RSSgrowth0; all
whole loaded file entries<=22040. Work11887072 includes the prior10023648 plus
4096metadata,1317632all-saved-entry work and541696all-auxiliary-row scans. It is
not a free append to a real source generator or an all-CPU/hash-work claim.
The raw stdout and normalized JSON are both saved; startup informational log
lines were separated from JSON, with no experimental value changed.

Existing-factor reuse is a DIFFERENT exact quotient from C109's added partials.
It might avoid their positive storage cost, but needs its own preregistered
proof, complete native/box/inverse/cost checks and real-source relevance test.
Do not treat it as permission to revive C84's zero ORIGINAL-recipe census,
invent source aliases, relax current caps or proceed to a network retry.
