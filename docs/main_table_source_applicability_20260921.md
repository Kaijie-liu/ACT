# Main-table source applicability: saved-only audit

Status: **COMPLETED_WITH_SOURCE_CONTAINMENT_GAPS**. This implements priority 1
of `/data1/Kane/MOE/Advice/ee.md`. No model, data loader, ACT propagation,
optimization query, new output bound, or input98 follow-up was executed.
Historical outcomes are preserved, not re-certified or relabeled.

Machine-readable [ledger](main_table_source_applicability_20260921.json) and
[separate arithmetic review](main_table_source_applicability_20260921_review.json)
contain the 100 box records, 23 gain-to-evidence mappings, and file hashes.
The review uses a different arithmetic implementation but the same researcher;
it is **not third-party mathematical review**.

## What was read and what was not

The original execution is `bc0791976b00879c28c268692aecc5854b3bc091`.
All 376 Python blobs under `act/` at that commit reproduce runtime source digest
`63a810b9841c8785ad9530dda3f1d5d299443be69c161d0b9abfe50f29016600`.
The selected eight-file call path is recorded with frozen/current hashes;
current factory, experiment1 and staged-verifier files have changed since then,
so the analysis reads **Git's frozen blobs**, not an assumed current version.
The input constructor, seed helper and HybridZ transfer file remain identical.
Recorded runtime: Python 3.12.12, torch 2.9.1+cu128, NumPy 2.3.5, SciPy 1.16.3.

We bind the old selection and three method configurations, rehash the raw
runtime/final audit, and compare all 900 terminal files to the saved ledger.
All 739 `request.pt` files are hash-checked and decoded with CPU
`torch.load(weights_only=True)`; tensor identities agree with selection and
request identities. They contain center/lower/upper tensors, **not a saved
network-to-HZ derivation**. The saved copies cover all 100 selected inputs;
the 161 missing packages are retained outer TIMEOUTs, not missing images that
were silently reconstructed. The separate review rehashes 3,117 terminal,
manifest, evidence and request files.

No checkpoint or CIFAR raw data was loaded. Model/checkpoint identities are
bound as recorded, not rederived from model parameters. No UNSAFE witness was
replayed anew; previous replay evidence is preserved. We do not claim to have
checked raw-image preprocessing, the historical intermediate HZ trace, any new
network/guard containment, or the original solver's exact proof.

## Which sets are compared?

Let `x` denote the **exact binary value of the saved float64 center**, not a
claim about an ideal raw-pixel normalization. We separately consider

- `X_rat`: `[x - 2/255, x + 2/255]` clipped to `[0,1]`, in exact arithmetic;
- `X_bin`: the same real box using the exact binary64 value of runtime epsilon;
- `B`: exact real denotation of the saved, floating-materialized endpoints;
- `H_formula`: exact real denotation of the input HZ obtained by reconstructing
  the frozen binary64 midpoint/radius formula on `B`.

The runtime epsilon is `282578800148737/36028797018963968`, not literally
`2/255`. Both requested-box interpretations are examined; rounding the radius
is not conflated with rounding the subsequent endpoint arithmetic.

The frozen entry computes `(center - float(epsilon)).clamp(0,1)` and the
corresponding upper endpoint. Factory BOX specifications clone these endpoints;
`seed_from_input_specs` clones BOX bounds. The sparse input seed computes
`c = (lb + ub) * 0.5`, `r = (ub - lb) * 0.5`, retaining radii only when
`abs(r) > 1e-12`. Its `exact=True` flag does not prove containment under real
arithmetic. All these inputs have 3,072 positive radii, above the dense 1,024
generator limit; the sparse route is the relevant frozen path. No positive
radius was dropped by the cutoff on this cohort.

For the last comparison, endpoints are `Fraction(c) ± Fraction(r)`, **without
rounding that last addition back to float**. Merely computing rounded
`c-r` and `c+r` can conceal a gap. This is a formula reconstruction on saved
inputs, not recovery of historical source states. The independent implementation
decodes IEEE-754 bits and rounds each rational midpoint/radius operation
separately; it reproduces all coordinate counts and maximum gaps.

## Exact containment findings

An inward coordinate means at least one endpoint fails the tested inclusion.
Counts below have denominator 307,200 coordinates / 100 unique inputs, not
900 independent requests. Endpoint counts and first exact witnesses per box
are in the ledger.

| Required containment | Inputs failing | Inward coordinates | Largest exact gap | Gain requests failing |
|---|---:|---:|---:|---:|
| `X_rat ⊆ B` | 100/100 | 257,850 | `1/35888059530608640` (~2.79e-17) | 23/23 |
| `X_bin ⊆ B` | 100/100 | 197,704 | `1/36028797018963968` (~2.78e-17) | 23/23 |
| `B ⊆ H_formula` | 98/100 | 5,736 | `1/72057594037927936` (~1.39e-17) | 23/23 |

Ranks 59/index4279 and 91/index4373 have no inward coordinate in the last
comparison. This does not certify those requests: the requested-box comparison
still fails, and the downstream source chain was not checked.

The gaps are tiny but **are not a complete-output error bound**. Neither a
positive old margin nor the solver's `nextafter` policy supplies the missing
network-wide propagation argument. Conversely, an input containment gap is
not a network counterexample and does not establish that any old SAFE decision
would change under a new sound enclosure.

## All 23 primary gains: source and acceptance basis

Every gain retains its original request, checkpoint/model, center/box, config,
terminal, evidence, and per-property record hashes in the ledger. `F0` rows
are recorded obligations, not newly re-proved LP/MILP bounds. The last column
counts inward coordinates in `B ⊆ H_formula`; all rows also fail both
requested-box inclusions.

| Model | Rank / input | Recorded legal pairs | Decision | Reused / residual F0 rows | Inward HZ coordinates |
|---|---|---:|---|---:|---:|
| seed0 | 0 / 4088 | 2 | F0 | 7 / 11 | 86 |
| seed1 | 6 / 4104 | 3 | F0 | 12 / 15 | 10 |
| seed1 | 11 / 4128 | 2 | F0 | 6 / 12 | 23 |
| seed1 | 15 / 4141 | 2 | F0 | 4 / 14 | 26 |
| seed0 | 17 / 4145 | 2 | F0 | 5 / 13 | 8 |
| seed0 | 27 / 4172 | 2 | F0 | 7 / 11 | 60 |
| seed0 | 33 / 4188 | 2 | F0 | 9 / 9 | 89 |
| seed1 | 43 / 4223 | 3 | F0 | 7 / 20 | 39 |
| seed0 | 43 / 4223 | 2 | F0 | 7 / 11 | 39 |
| seed0 | 45 / 4232 | 3 | F0 | 12 / 15 | 46 |
| seed0 | 48 / 4236 | 2 | F0 | 3 / 15 | 28 |
| seed1 | 50 / 4248 | 2 | F0 | 4 / 14 | 50 |
| seed1 | 52 / 4256 | 2 | F0 | 12 / 6 | 51 |
| seed2 | 64 / 4297 | 3 | F0 | 19 / 8 | 76 |
| seed1 | 67 / 4311 | 3 | F0 | 7 / 20 | 7 |
| seed1 | 68 / 4314 | 2 | F0 | 1 / 17 | 207 |
| seed2 | 69 / 4316 | 3 | F0 | 11 / 16 | 7 |
| seed0 | 78 / 4340 | 2 | F0 | 6 / 12 | 21 |
| seed2 | 85 / 4360 | 3 | F0 | 13 / 14 | 89 |
| seed0 | 86 / 4361 | 2 | F0 | 9 / 9 | 69 |
| seed2 | 89 / 4371 | 3 | F0 | 20 / 7 | 50 |
| seed1 | 89 / 4371 | 4 | Tier 1 | — | 50 |
| seed1 | 99 / 4389 | 2 | Tier 1 | — | 32 |

The 21 F0 gains cover 450 recorded pair/property rows: 181 scoped interval
reuses and 269 residual `mip_dual_bound` acceptances. The audit checks pair and
property rosters, positive accepted minima and recorded status0 for residuals,
and reuse request/model/box/property/policy binding. This is **record inspection**,
not independent source validity or residual-dual checking. The two Tier1 gains
cover eight candidate branches: two `interval_fact_only` and six `certified`
expanded-violation infeasibility records. They are not eight scalar output
duals, and the scalar-minimum field remains null.

The frozen numerical policy requires status0, finite certifying bounds, zero
relative MIP gap, feasibility/integrality tolerances `1e-7`, absolute and
relative correction `1e-9`, and `nextafter`; accepted safety margin is `>1e-7`.
This policy does **not** check that floating input/affine/ReLU/guard/gate/F0
construction encloses its claimed source. Complete route coverage in the old
package is a recorded solver-policy result, not an independently verified
cover of the requested real network domain.

## Guarantee disposition (no retroactive repairs)

| Link / result | What this audit establishes | Still not established |
|---|---|---|
| Saved request identity | All 739 boxes match frozen selection and config; all 100 inputs covered | Raw-image/preprocessing equivalence |
| Requested real box to materialized box | Exact containment tests fail for both epsilon interpretations | A real-ball certificate from those endpoints |
| Materialized box to reconstructed seed | Exact containment fails on 98 inputs, including all gains | Historical full HZ trace; downstream enclosing semantics |
| Network/guard/F0 to accepted bounds | Hash-bound historical acceptance metadata and complete recorded gain obligations | Independent containment and lower-bound proof for these outputs |
| Main outcomes | 179/156/141 historical HZ-policy SAFE; primary 23 gains unchanged | 179 source-complete or native-float certificates |

Even restricting the interpretation to a floating-generated input HZ would not
by itself validate subsequent floating propagation, guard construction or
solver reasoning. Therefore the main table is an empirical **frozen-policy
acceptance comparison**, conditional on the unclosed source/numerical
obligations, not a proven coverage rate for any of the three proposed domains.
It remains evidence of the implementation's comparative behavior, with that
contract stated next to the table. We do not subtract gains, relabel SAFE as
UNSAFE, recompute confidence intervals, or transfer new source checks to old
positive matrices. Shared defects across comparison arms do not cancel a
soundness obligation.

## Reproduction and next action

With the original raw packages available, from this checkout:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/audit_moe_main_source.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/test_moe_main_source.py
```

The ledger generator and separate review use exclusive output creation; they
will not overwrite their first result. The separate review source is
`scripts/review_moe_main_source.py`. First generation took 4.193 seconds; the
separate arithmetic/hash review took 0.719 seconds **after importing torch**.
These are offline audit costs, not verifier speedups or full-request budgets.
No dependence on a solver, checkpoint or training/test dataset was introduced;
the saved `.pt` decoder still needs the existing torch environment.

An initial fresh-reread equality check rejected a tuple/list representation
difference in JSON witness endpoints. The output file was unchanged; the
generator now returns a list, with a roundtrip regression. This was an audit
serialization defect, not an altered bound, formula or acceptance threshold.

Priority 1 is closed. **No new input98 intervention or experiment is proposed.**
Next, a person uninvolved in implementation should review the theorem-to-result
contract, including these gaps, using the paper and artifact entry alone.
Reviewer assignment/contact and public/anonymous empirical artifact release
remain PI-managed. Neither actual third-party review nor a clean external
installation has been performed by this audit. Those coordination choices,
not more same-case solving, are now the next handoff.
