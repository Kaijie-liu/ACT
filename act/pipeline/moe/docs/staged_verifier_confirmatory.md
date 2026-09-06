# Fixed-radius staged-verifier confirmatory cohort

## Purpose

This experiment tests the production staged-verifier entry point on a new HZ
cohort after the outcome-selected development closure met its predeclared
signal. It does not revise the seed-2 R1 boundary-adaptive experiment. The
mathematical method and numerical policy remain those in
`staged_verifier_v1.json`.

## Frozen request population

The model is the hash-bound seed-2 balanced weighted top-2 checkpoint. The
selection is the first 100 ordered official CIFAR-10 test indices at or after
dataset index 2000 that are clean-correct for this checkpoint and absent from
the listed seed-0 and multi-seed HZ cohorts. The selection manifest was created
before any verification endpoint was queried. It uses no route-stability,
candidate, boundary-radius, guard, solver, or certificate predicate.

Every input receives exactly one direct top-1 robustness request at `2/255`.
There is no boundary search and no route-instability prefilter. Consequently,
all 100 selected inputs remain in the main denominator. Route-stable and
route-changing subsets are derived only after exact route-set coverage is
available.

The freezer failed twice before writing a manifest: first because one side of
the interpreter symlink comparison was not resolved, then because a local
logits variable shadowed the output path. Both failures occurred before
`_write_json`; the successful run created the only selection artifact.

## Production execution

For each request the runner invokes `staged_verifier.py` in `act-py312` with a
300-second outer deadline. The production path performs exact candidate and
unordered top-2 route-set analysis, guarded Tier 1 gate elimination, and F0
only when Tier 1 ends in registered semantic incompleteness. It does not run
the no-support comparison, unguarded accounting propagation, or any boundary
experiment. Results and a partial summary are flushed after every row.

Each completed request writes an immutable package with literal represented
box tensors, model/config/property identities, route coverage, property
obligations, accepted bounds, transitions, and any validated witness. A resume
may append only after an ordered result prefix whose runtime config hash is
unchanged; it cannot overwrite an existing package.
Each attempt has a distinct package/log/progress name, so an interrupted
partial directory is preserved rather than deleted before a resume.

## Endpoints and interpretation

The primary endpoint is the number of complete SAFE requests with more than
one exact feasible unordered top-2 route set, divided by all 100 selected
inputs. The predeclared existence-replication signal is at least one such
request, subject to zero independent-audit issues and full-model replay of
every UNSAFE verdict. The exact count and full denominator must always be
shown; the threshold must not replace them.

Secondary endpoints are the full status/reason table, Tier-1 and F0 SAFE
counts, F0 invocation and completion, exact route-stable/route-changing
counts, and stage time. Incomplete rows are right-censored for timing. This
clean-correct cohort is not certified accuracy. It is not a speed comparison
with historical experiment runners, whose work and scheduling differ.

## Independent audit

The external auditor reconstructs the ordered clean-correct selection from the
checkpoint and official test set, checks all bound hashes, requires exactly the
100 registered rows, audits every completed evidence package, and replays every
UNSAFE witness through the full selected-softmax model. A process error is an
audit failure; an explicit outer timeout remains a recorded endpoint.

## Commands

```text
ACT_TORCHVISION_DATA_ROOT=/data1/Kane/MOE/ACT/data/torchvision \
  /data1/Kane/miniconda3/envs/act-py312/bin/python \
  -m act.pipeline.moe.run_staged_verifier_confirmatory

ACT_TORCHVISION_DATA_ROOT=/data1/Kane/MOE/ACT/data/torchvision \
  /data1/Kane/miniconda3/envs/act-py312/bin/python \
  -m act.pipeline.moe.audit_staged_verifier_confirmatory
```

## Result

The run completed all 100 requests at implementation commit `3a3a334af`. The
independent audit reconstructed the selection, audited all 100 packages,
replayed all 42 UNSAFE witnesses, and reported zero issues.

The status table is 21 SAFE, 42 replay-validated UNSAFE, 27 UNKNOWN, and 10
solver TIMEOUT, for 63/100 complete semantic outcomes. Exact route coverage is
available on every row: 62 requests are route-stable and 38 are route-changing.
Six of the 38 route-changing requests are SAFE. The preregistered full-cohort
endpoint is therefore 6/100 (Wilson 95% interval 2.78%--12.48%); the conditional
6/38 = 15.79% is descriptive only. The six certificates comprise two Tier-1
gate-elimination results and four F0 weighted-range results.

Across all inputs, Tier 1 proves 14 SAFE requests. F0 is invoked on 56 requests
and reaches 32 complete outcomes: 7 SAFE and 25 replay-validated UNSAFE. No
outer hard deadline fires. Ten solver-returned timeouts remain endpoints rather
than process failures. Median verifier time is 43.24 seconds (IQR
16.90--79.49; p90 124.15), with no historical-speedup interpretation.

The compact audited result is
`results/staged_verifier_seed2_fixed2_confirmatory_20260906_r1.{json,csv}`.
This result passes the preregistered existence-replication signal. It does not
alter the seed-2 boundary-adaptive R1 endpoint, and it is not CIFAR-10 certified
accuracy because selection is conditioned on clean correctness.
