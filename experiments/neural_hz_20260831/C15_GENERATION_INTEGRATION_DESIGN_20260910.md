# After C15: generation integration design, not execution permission

Read-only source inspection plus the already sealed complete C14 table and
the completed C15 result. No new benchmark, optimizer or live target was run.
This note is not a new preregistration, source promotion or relaxed work gate.

## Measured ownership boundary

Read the C14 table (SHA0208ecdb11a35896c1ecb86a6fabb1c1e9f3c49bf7b47d6248a18d52c6e3328e)
with allow_pickle=False; filter its complete individually_admissible mask.
All268 defining rows lie in the preactivation prefix (row<143737). Of their
unique consumers,69 are also in that prefix;199 are in the new ReLU equality
suffix (rows143737..143936). Column range222968..254869. These are diagnostic
cohort counts, never a hardcoded row/column selection in a future algorithm.

Consequently adding splicing only inside C10.fold_rows before ReLU cannot
capture all268:199 consumers do not exist yet. Adding the completed C15 pass
after ReLU instead consumes253357392 work on top of C10's254750077, total
508107469. It is disallowed, regardless of the standalone16.9s timing. Even
copying only its19971152-work exact norm scan exceeds the current1249923-work
C10 headroom. No postpass, expanded budget, cheaper renamed tariff or skipped
verification is a justified integration.

## Concrete implementation direction to prove before any live run

1. Generate row ownership/liveness and a sound defining-box certificate while
   emitting the affine DAG. C10 already computes an upward box exponent for
   every fresh MAIN factor in c8_dyadic_balance_v1.box_exponent. Reuse that
   established construction fact only through a sealed source/row/scale binding;
   hz.exact=True alone is NOT a checked redundant-box certificate. Radix packed
   rows need separate reasoning; a direct MAIN case must be structurally proved.
2. Transfer the bound through exact row scaling and independent C10 aliases.
   For |r|<=1, substituting one alias cannot increase the coefficient L1 envelope;
   exact collision cancellation cannot increase it either. This is a candidate
   compositional theorem to test independently, not permission to silently omit
   the guard. Account for the certificate and metadata in the whole-state union.
3. Fuse ordinary native ReLU equality creation with the existing final CSR
   assembly. sparse_hz_apply_relu_exact constructs the equality from the old
   value map and fresh phase/slack slots, then stacks predicates once. A scoped
   replacement must prove the same binary phase relations and frame allocation,
   and splice eligible consumed MAIN definitions at that emission boundary.
   Original preactivation roots cannot be erased while still output-live.
4. Reuse graph incidence only if it proves sole predicate consumption, residual
   sharing and all prior aliases; graph fanout alone does not imply predicate
   degree after rewriting. No omitted degree scan may be assumed free. The69
   affine-internal and199 ReLU-consumed rows require one structural ownership
   rule over their lifetimes, not iid-selected branches or a two-path rescue.
5. Final reconstruction must compose new in-row extension BEFORE prior C10
   alias extension, with explicit global-frame padding. The old C10 helper
   expects its preactivation width254898 while the final width is255298; it
   cannot simply be called on the wrong-width vector. Protect the original
   input prefix and independently prove the entire combined relation.

Before freezing any new target: exact EQ/INEQ/binary/branch-sharing tests,
before-operation coupled whole/branch work accounting, closed numeric and
Python ownership, source+certificate identity, fail-closed cap tests, and
ordinary native all-coefficient fidelity. Then separately preregister fresh
original-network generation and its complete live comparator; no old completed
HZ may be the algorithm's shortcut. Inherited failed terminal runs stay closed.

## Why this is not yet the requested leap

C15 removes only536 coefficients from roughly11M and154362->154094 lowered
continuous variables. Reusing its proof can make an integration affordable;
it does not imply a new solved case or substantial compression. Any next
capability claim needs actual terminal evidence. Broadening non-unit products
is still blocked by102012 C14 exact-product failures and291 window failures;
rescaling alone cannot restore lost significand precision. Do not revive
whole-row cache-key-only variants disproved by C12/C13. A substantially better
Neural-HZ representation remains an open task under the original charter.
