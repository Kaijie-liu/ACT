# Immutable-source endpoint supervision

The thirteen-call roster completed once, with expected outcomes and an
independent standard-library reread audit **PASS**. The
[finite protocol](hz_store_supervision_design_20261013.md) was committed and
pushed as `4997964b7` before implementation/process calls. Execution was frozen
and pushed as `b6526c9c4`; no numerical retry or budget change occurred.
Archive: `/data1/Kane/MOE/baseline_runs/hz_store_supervision_20261013_r1`.
The [compact receipt](hz_store_supervision_20261013_r1.json) binds saved outputs.

## Outcomes

| Invocation | Execution | Checked mathematical result | Observed whole seconds |
|---|---|---|---:|
| half | complete | 3/3 positive duties; 5/5 endpoints | 3.049770 |
| negative | complete | 3/6 positive duties; 6/6 endpoints; UNKNOWN | 3.015290 |
| partial | complete | 2/3 positive duties; 4/5 endpoints; UNKNOWN | 3.088423 |
| import / source delays | TIMEOUT | no acceptance | 15.295156 / 15.383112 |
| candidate / serialization delays | TIMEOUT | no acceptance | 15.391840 / 15.392005 |
| check / receive delays | TIMEOUT | no acceptance | 17.424982 / 17.769109 |
| producer / checker exceptions | ERROR | no acceptance | 2.457237 / 2.798621 |
| invalid candidate reference / missing checker receipt | ERROR | no acceptance | 2.574160 / 2.799074 |

Every registered fault was reached and every started owned group was cleaned.
The half source's (0,2) guard is empty. Its bound is valid on a vacuous domain,
not improved classifier margin. All experts are individually safe; this does
not measure relational benefit. Negative bounds do not establish UNSAFE.

Half/negative/partial cumulative row/coefficient/arithmetic counts are
487/637/1279, 627/774/1548 and 483/633/1270. They include producer and checker work
under one 96M ceiling, not a fresh allowance per process. Reception independently
checks query-derived slots and both half-gate orientations.

The serialization cutoff retains one 475-byte published tile with its hash and
counters (235/310/710), but no final manifest or accepted result. Incomplete
counters are null/censored, not charged as zero. Open operations retain
right-censored duration lower bounds.

## Cost and evidence scope

In half, numerical imports take 1.374315 seconds, source construction 0.007985,
preparation/source checking 0.007151, tiled candidate generation/exact selection/
publication 0.120048, and independent exact checking 0.007658. Nested diagnostic
times are **not added again** to process costs. Process clocks also include
worker binding, cleanup and unclassified overhead. Parent clocks include
launch/receipt/hash/inventory work. Final publication-to-observed-return tails
(about 1–2 ms) are retained separately; the table uses the whole observed clock.

The offline audit takes 0.180867 seconds **outside** request budgets. It rechecks
all normal source/proof bounds, identities, observer timing, fault reach, costs
and incomplete prefixes. It uses the same standard-library mathematical checker,
not a second independently implemented proof kernel or human review. Protocol/
code archiving and test startup are batch costs, not hidden request costs.

After audit, the archive contains 656 regular files totaling 1,180,800 bytes,
before later small bookkeeping: below the 64 MiB stage cap. No old evidence,
weights or unrelated caches were deleted. No GPU was invoked.

## Implementation boundaries

The new policy reuses `device_lifecycle`/`owned_bounded`; only the candidate
worker imports numerical packages. Source, complete property, algorithm and
tile width are externally bound. Receiver identities are recomputed from the
frozen half-gate/negative fixture, including the missing endpoint's duty.
Production is still limited to this synthetic supervision protocol, not a
general trained-model intake.

The process engine adds opt-in isolated-launch and nested-evidence inventory
hooks. Existing policies keep their old launch/inventory behavior. Observer
checking is factored without changing its rule: no observed on-time return,
no budget acceptance. Atomic exclusive tile publication retains completed
prefixes; the serialization fault is before the second tile so the first can be
checked after timeout. No final manifest is synthesized after interruption.

Exact-check work carries across producer/checker processes under one 96M
ceiling. Small manifests, not numerical proof arrays or source dictionaries,
cross the phase receipts. Interrupted counters are censored, never charged as
zero. All nested evidence files, including unpublished partials, are inventoried
under file/count/byte caps. An inventory exception fails closed at the API,
stops the batch and preserves pending slots; it may prevent a normal terminal
file. This is not a promise about arbitrary filesystem stalls or escaped jobs.

The observed pre-execution test command used `act-py312`, `-B -S -m unittest`,
the 12 new supervision tests plus the prior 83 endpoint/source/store/capacity/
layout/proof-closure tests: 95 tests, 1.731 seconds, OK. This paragraph records
tool-observed output, not a raw archived subprocess log. The execution runner
separately saved all 12 test outcomes (all PASS, none skipped) before launching
the process controls. Pre-freeze review caught and fixed foreign endpoint IDs,
inconsistent partial positivity, counter-reset and algorithm/tile metadata gaps.
No frozen result was overwritten to obtain this outcome.

All G1–G6 remain OPEN. This stage does not add a real-model certificate, GPU
execution, speedup, external win or wider source/operator support.

## Continue / stop

Stop this synthetic supervision study. The fixed control policy is not general
trained-model admission; declared affine/ReLU and tiny positive sources are not
trained-density capacity. Next implement the **same common/private block
candidate algorithm** on an optional GPU device, first CPU-equivalence and fault
controls under this schema. CPU exact checking remains on original coefficients.
A physical run needs its own freeze and fresh admission; do not retry old GPU
slots automatically. Captured checkpoint/source streaming and convolution remain
separate blockers. No sealed real input or backend search reopens.
