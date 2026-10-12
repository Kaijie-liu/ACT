# Request-owned HybridZ rows: given-HZ component completed

The first component of the [frozen design](hz_request_store_design_20261012.md)
now passes its controls and a separate-process saved-evidence audit. Shared
rows are stored once, inherited by immutable views, and consumed directly when
checking a pair's exact lower bound. This is an implementation result, not a
new real-model certificate, measured acceleration or network-source proof.

The coefficient recipe was committed at
`6fc6cddec0917250855af5c1e80201271c2ee8cf` before execution. R1 is preserved as a
pre-hardening run; R2 checks the same mathematical objects without regenerating
them. The [compact result and code bindings](hz_request_store_20261012_r2.json)
bind the 47 audit events, 38 refusals and all normal-object identities.

## What changed in the representation and consumer

`scoped_source/hz_request_store.py` reads an externally anchored request store
using only the standard library. Continuous/binary factor names, ownership,
common/private scopes, inherited constraints and pair guard suffixes are checked.
It retains immutable `Row` objects rather than expanding each pair into a joint
HZ or LP. Common constraints are visited once per pair calculation; private
suffixes start after the common prefix, not after the guard entry.

The residual checker directly computes, for equality multipliers z and
inequality multipliers y ≤ 0,

    r = c − Eᵀz − Aᵀy
    L = d + hᵀz + bᵀy − Σ |r_j|.

Shared-column terms are summed before the single box-support subtraction.
Binary factors are explicitly relaxed to [-1,1]. Every original coefficient
is consumed even when its multiplier is zero. The query and candidate are
privately captured, bound to the external query/request/store identities, and
the claimed bound is checked against the exact rational calculation.

This is a **given-HZ** contract. Hashes do not establish that stored coefficients
correctly lower a network, that a supplied guard is a router condition, or that
all actual network routes have been discovered. Those links remain the next
source-consumer work. The additive store does not change production defaults,
the legacy parsers, candidate optimization, mathematical relaxation or acceptance
gates.

## Frozen layout result

| Object | Factors | Stored rows | Stored nonzeros | Members | Bytes |
|---|---:|---:|---:|---:|---:|
| Small, 64 rows/member | 10 | 27 | 40 | 15 | 7,259 |
| Same small coefficients, 1 row/member | 10 | 27 | 40 | 27 | 10,828 |
| Complete large layout | 25,648 | 13,088 | 67,456 | 234 | 3,110,510 |

The large registry has 22,448 continuous and 3,200 binary factors. Every factor
occurs in an output coefficient. All 28 declared pair views use 8,504 continuous
and 896 binary factors, and 2,700 constraint rows. Across the request, constraints
are stored as 9,936 distinct rows, plus 3,072 common outputs and 80 expert outputs.
There are 37 views: one common, eight expert templates and 28 guard entries.

The independent auditor checks the complete registry and **every** frozen
output/constraint/guard coefficient and RHS, not only an aggregate objective.
All 56 large pair/dual checks match an independent index-formula oracle in every
residual coordinate and final bound. The small flattened differential and both
page partitions also agree. Actual object identity checks confirm inherited row
sharing and disjoint expert-private coordinates.

These dimensions follow the registered producer upper-bound layout, but its
sparse coefficients are deterministic controls, **not trained network density**.
The 3.11 MB store does not predict real evidence size. The fixed objectives are
not complete classification properties: 56 checks are not 504 endpoint queries
or 252 output obligations. No SAFE count or speedup is inferred.

## Refusals, repairs and portability

All 38 archived negative cases refuse with their registered reasons: identity,
scope, private-factor pollution, guard promotion, missing/duplicate obligations,
row/member mutation, malformed coefficients, finite capacities, changed query,
increased bound claim, missing dual and incorrect dual sign. R2 records complete
mutated member inventories and rechecks the exact R1 objects. It separately
exercises actual 8,000,001-visit exhaustion, rational-bit refusal, expired and
mid-calculation deadlines. Partial arithmetic never returns a completed bound.

R1's initial checks missed an ordinary deletion bypass of the Store seal and
did not capture caller query lists before arithmetic. R2 rejects deletion as
well as assignment, and demonstrates snapshot-consistent checking when an
instrumented caller mutates query/candidate lists during arithmetic. It also
adds full per-row recipe checks: aggregate cancellation alone could hide a
coefficient permutation. R1 code is retained and matches its recorded hashes.

Both small and large objects recheck after copying under `python -B -I -S`.
The bundle contains only the bound standard-library checker, control auditor and
store. No checkpoint, training data, repository import, optimizer or numerical
library is used. Returned exact bounds/residual identities and complete layout
records match. This is portable given-HZ evidence, not portable network proof.

## Cost and archive boundary

R1's per-control times were sampled before some publication/scan operations, so
they are not full generation-cost measurements and are not used for speed claims.
R2 rechecks saved inputs; its event costs include reading, checking, record
publication and stage-size scanning. The large original-location check took
about 6.55 s; its relocation including copy, isolated child startup and checking
took about 6.64 s. The complete R2 audit through final audit publication took
21.34 s. These are offline audit observations, **not end-to-end verification**.
The final tiny terminal-receipt write follows that last timestamp.

The large check accounts for 291,424 row visits, 1,228,448 coefficient visits
and 2,632,448 exact arithmetic operations. All per-control deadlines are 300 s;
the first-component deadlines are cooperative, not process-memory guarantees.
The observed 347,360 KiB process peak includes the deliberately oversized nnz
refusal, so it is not the large normal object's peak-memory measurement.

The retained stage occupies about 37.46 MB of regular-file payload, below its
64 MiB cap, including R1, its code snapshot, R2 and portable copies. R2 references
unchanged R1 inputs rather than duplicating the full refusal archive. New full
generation attempts have a conservative admission check; failure receipts no
longer depend on rerunning the admission condition that failed.

Raw evidence is retained outside Git in `/data1/Kane/MOE/baseline_runs/`:

- `hz_request_store_20261012_r1` — original controls and all negative inputs;
- `hz_request_store_20261012_r1_code_snapshot` — exact executed R1 sources;
- `hz_request_store_20261012_r2` — independent audit, hardening controls and
  relocated bundles.

The directories were moved from their temporary in-checkout location without
changing file contents. No scientific evidence or failed attempt was deleted.
The stage-end workspace allocated-size observation was 225,954,906,112 bytes;
this is not an attribution of all workspace growth to this stage. The filesystem
still has about 83 GiB free. Python was run with `-B`; no new cache cleanup,
environment installation or checkpoint copy was performed.

## Verification and next decision

57 regressions pass, including nine new store/identity/resource tests and the
48 existing capacity/navigation/proof-contract tests. Independent static review
found no remaining algebra or scope blocker for this component. Neither that
review nor the JSON audit is an independent human technical review.
Navigation additionally checks 340 local links, 619 unchanged frozen source
identities and 4,632 preserved historical handoff lines. The earlier 46-file
static capacity report still re-derives identically.

Recheck the preserved bundle without starting a new generation run:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S /data1/Kane/MOE/baseline_runs/hz_request_store_20261012_r2/portable_large_page64/check_hz_store_control.py --store /data1/Kane/MOE/baseline_runs/hz_request_store_20261012_r2/portable_large_page64/store --manifest 7d0fdeabc49366dfc06ddf77fa45b6dec2cb28e7f635addc3cb26391c3135523 --request 617c4e8ccf1d623210dcb64467056ac2aa2e7e131484e7c932442e4c12357583 --layout large
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_hz_request_store_units scripts.test_hz_combined_capacity scripts.test_hz_block_capacity scripts.test_hz_real_intake scripts.test_project_layout scripts.test_proof_closure scripts.test_proof_closure_costs
```

Stop this component's width/timing study. Next implement the exact source and
binary64 enclosure consumers against these views, with globally ordered factor
allocation and actual SparseHZono matching; then tile **all** endpoint queries
through the unchanged block proposal/exact acceptance algorithm. Do not turn
views back into repeated whole dictionaries. Complete-source supervision,
convolution support, trained density, full obligations and the same-algorithm
GPU block kernel remain open. No physical GPU admission/retry, real request,
sealed input, native solve or new training occurred. All G1–G6 remain OPEN.
