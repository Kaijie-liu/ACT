# C14 deterministic complete-population proof with early rejection

Same S0, same sealed final HZ/maps, same complete structural population. C13's
negative result changed the next action: whole-row certificate-key variants
are closed. Previous turn is PROGRESS, not a wait/restatement. Revalidated its
manifest and current redu-hz production state before this new implementation.
No failed algorithm is rerun and no cap/window/default is changed.

## Mathematical rule and scope

Individual exact single-use affine elimination requires a CONJUNCTION of
necessary acceptance guards. A single rigorously false guard rejects that
definition; later guards need not be proved. If accepted, every guard must be
proved true. This is the SAME exact guard conjunction as C13, with its exact
coefficient-grid L1 box bound, not a weaker tolerance or approximate transform.
The change is a deterministic evaluation algorithm, not a new target subset.
All candidate columns are processed in ascending logical order. No coefficient
cache, instance identity, label, solver status, margin or historical verdict is
used to choose order or applicability. No benchmark sampling or solver call.

For each candidate, in this fixed order:

1. Charge scalar work and form the exactly reversible consumer multiplier.
2. For nonzero definition RHS, prove its product exact; prove the consumer RHS
   update exact with Fraction. Zero RHS is an exact identity, not sampling.
3. Check continuous coefficients in canonical column order, then binaries,
   in FIXED blocks of8 terms. Charge all8 (or final shorter block) before
   computing generic exact products and unchanged[2^-20,2^40] window checks.
   At the first failing block reject; exactness has diagnostic priority over
   window failure within the same block. A universal guard may be marked false
   from one observed counterexample, but true only after all its terms pass.
4. Only if products pass, construct exact coefficient L1 numerator on2^-72
   grid and prove N/2^72+abs(h)<=a with each definition's OWN h and pivot a.
5. Only if that box passes, join the already-proved product blocks and check
   EACH actual consumer-column collision with the existing exact-sum guard.

Store guard values1=proved true,0=proved false,-1=NOT EVALUATED. Do not turn -1
into true or pretend a rejected definition's untested guards were themselves
false. All-products exactness is derived with the same three-valued meaning.
Only accepted definitions get evaluated exact nnz deltas. No simultaneous
substitution, HZ construction, witness or capability result is implied.

## Same hard capacities, a new precharged execution ledger

Same256M whole work,64M input coefficient entries,1GiB measured construction,
tests60s/worker240s,address space16GiB,CPU1/GPU0. The fixed structural charge
8*input coefficient nnz+32*logical MAIN is reserved BEFORE payload validation,
hashing/liveness/degree classification. Reject before arithmetic if it fails.
Subsequently charge BEFORE each operation, with unchanged generic64/term cost:

-32 per definition for scalar/metadata/RHS bookkeeping;
-64 per nonzero RHS product;
-64 per ACTUALLY TESTED coefficient (the entire predeclared block is charged);
-16 per coefficient for exact L1 construction, only if reached;
-2 per coefficient to join accepted product workspaces, only if reached;
-32 per actual consumer coefficient for collision checks, only if reached.

No full-row product workspace is allocated before necessary early rejection.
The eager all-guard upper bound is reported, but is NOT substituted for actual
precharged dynamic work. Unlike C11-C13, this new algorithm does not execute
the unused tail of a failed conjunction. This is execution sharing within a
proof, not reduced tariffs or a larger ceiling. Actual wall/HWM/tracer gates
remain unchanged and can independently reject it.

Coverage is atomic: classify ALL structural candidates, or reject the WHOLE
census if the pool is exhausted. Never publish an admissible/processed prefix,
even if it contains true individual proofs. Intermediate events report only
coverage and charged work, explicitly acceptance_published=false. Exhaustion
throws; no partial factor table is returned. Only a complete successful census
can save a factor table. There is no second schedule or restarted candidate.

## Retention and promotion boundary

Exclusive results/c14_early_rejection_census_20260910_v1/. Source final file
192f3a5a95637933bbbed8b57b10fd20f7d237b4c6f2066573e5dfa14e6b1971 and live
map file1c242085191040cdea7db9975aba5a8c2134e4b2776213cfee2c99a7e3ec1c9d
are unchanged. Both complete checkpoints stay reachable during measurement.
Inherit577 frozen tests/sources, add eager-equivalence, tri-state, deterministic
block order, before-operation charging and whole-prefix-discard tests. Freeze
branch/commit/config/source hashes; automatically retain tests/events/result,
table if COMPLETE, failures/exits and hashes. Production/HyZor remain untouched.

No HZ rewrite, base-feasibility bypass, lowering, native ingestion, optimizer,
presolve or new family/shadow/full replay is authorized. A completed census
would still need a separately preregistered simultaneous/reconstruction and
whole-generation/live-state proof; it cannot be appended to254750077-work C10.
Formal1870/2413, separate E061/400, gain0 regardless of census completion.
