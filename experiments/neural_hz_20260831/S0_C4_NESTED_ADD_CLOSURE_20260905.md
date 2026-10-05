# C4-v1: graph match, support-independent compilation budget failure

The separately preregistered complete additive-core rule was screened against
the same pinned, independently BN-corrected Tiny143 graph. All four operand
occurrences are present; no affine cut, nonlinear crossing, skipped SCALE or
ADD-as-identity rewrite was introduced. This is a graph-only necessary screen,
not a runtime or exact-set proof.

| Nearest source ReLU | Inner Conv | Outer Conv | ADDs on path | Descriptor products |
|---:|---:|---:|---|---:|
| 28 | 29 | 33 | 32 | 169,869,312 |
| 20 | 21 | 33 | 24, 32 | 169,869,312 |
| 9 | 10 | 33 | 16, 24, 32 | 169,869,312 |
| 5 | 13 | 33 | 16, 24, 32 | 9,437,184 |
| **Total** | | | | **519,045,120** |

Every descriptor separately fits the inherited coefficient-entry, coefficient-
byte lower-bound and contraction caps. The whole transaction does not:
**519,045,120 > 256,000,000 products**. This count precedes selected-row emission,
bias work, source-factor multiplication and whole-state accounting. Selecting
fewer spatial output rows cannot reduce the full-channel descriptor compilation
count used by V1. No descriptor cache discount was preregistered or applied.

The 32,440,320-byte sum counts coefficients only. It is not a complete resident
ledger, not measured memory savings and not a strict physical-reduction claim.
Passing individual caps does not compensate for failing the whole transaction.

Consequently **C4-v1 is closed with gain 0** before descriptor compilation,
abstract propagation, runtime-adapter development or any solver call. The
budget is not doubled after seeing this result; no favorable branch subset is
selected, and no jump to ReLU63 or another instance occurs. C1/C2/C3 closures
remain intact. The 8 new graph/cost tests pass, including exact channel-group
intersection counts, complete duplicate operand expansion and nonstationary/
cycle/extra-Conv/post-outer-ADD rejection. They are not HZ equivalence tests.

The next hypothesis must address the measured repeated cost itself: avoid
unneeded full-channel/full-tap descriptor products using proved support or
algebraic sharing, while preserving the complete affine DAG, source predicates
and shared latent frame. Merely accepting nested ADDs, retaining expanded
operators behind lazy references, or switching solver threads does not do so.
Any successor requires its own preregistration and must achieve actual work and
whole-state reduction before advancement. No successor is claimed implemented.

Evidence: `evidence/s0_c4_nested_add_preflight_20260905_v1.json`, SHA-256
`fd95f79e5712ad4544061129e72ffd13ff871da89eaf9c68fa4a10130fba3848`.
The record binds graph/model/spec, source hashes, branch/base and preregistration.
The corrected production BN option remains disabled. Formal 1870/2413 and
external E0 61/400 are unchanged; the full Neural-HZ goal remains active.
