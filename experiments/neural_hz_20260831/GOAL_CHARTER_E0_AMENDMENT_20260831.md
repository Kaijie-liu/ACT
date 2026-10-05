# Neural-HZ Goal Charter: E0 Ledger Amendment

Authorized and locked on 2026-08-31 for branch `redu-hz`, starting from commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. This versioned amendment supplements
`GOAL_CHARTER.md`; it does not overwrite that hashed charter, change the formal
score, or promote a candidate.

## Unchanged formal objective and constraints

The sole formal baseline remains **1,870/2,413** across 13 families: 1,063 CERT
plus 807 concretely validated ADV. Every old CERT and ADV, and every per-family
solved count, is a zero-regression constraint. The formal headline stays 1,870
until one candidate source manifest, one configuration and one execution path
complete all 2,413 cases, preserve the full baseline with zero invalid ADV, and
add at least one sound CERT or independently validated ADV.

The campaign remains structure-by-structure. The active S0 target is the
repeated residual Conv/BN/Add/ReLU structure, using exact phase slicing and
implicit/composed convolution operators. Selection may use only mathematical
state and frozen work/storage bounds; it may not use family, model, iid,
historical verdict, or a per-instance menu. The representation must retain
continuous factors, nonconvex binary phases, equality/inequality predicates,
shared latent identity, reversible witness reconstruction, and fail-closed
semantics. Zonotope/CZ/box degeneration and attack, split, BaB, backward, dual,
or LP-status rescue credited as Neural-HZ gain remain forbidden.

## Frozen external evidence baseline E0

The two source-universe manifests continue to freeze identity only. Their 400
`baseline_verdict` fields remain null and their
`baseline_vector_status=UNFROZEN`; they must not be rewritten with verdicts.

A separate content-key evidence vector is now frozen in the two v2 ledgers
under `evidence/`:

- CIFAR100: 0 CERT + 25 independently replayed historical-origin ADV + 175
  UNKNOWN;
- TinyImageNet: 0 CERT + 36 independently replayed historical-origin ADV +
  164 UNKNOWN;
- combined E0: **0 CERT + 61 validated ADV + 339 UNKNOWN = 61/400**.

Every E0 ADV was replayed against the content-matched ONNX model and original
VNNLIB property at literal zero tolerance. Every such row has
`neural_hz_gain_credit=false`. E0 is neither the historical 59/400 ledger nor a
Neural-HZ score, and it is never added to 1,870/2,413.

A future CIFAR100/TinyImageNet candidate must use one source/configuration/path
for all 400 rows. It must preserve all 61 E0 ADV; returning CERT on any of them
is a soundness conflict, not retention. Neural-HZ gain is possible only on the
339 E0 UNKNOWN rows and counts only as a candidate milestone until the complete
400-row replay has zero invalid ADV and both external families have no
regression.

## Data boundary

`/data1/Kane/HyZor` remains read-only. New work is written only beneath
`experiments/neural_hz_20260831/` with unique filenames and provenance. Existing
results, ledgers, manifests, and hashed charters are never overwritten to make
a later candidate look stronger. The authoritative E0 hashes and row semantics
remain those recorded in `BASELINE_LOCK.md` and `evidence/SHA256SUMS`.
