# Neural-HZ Goal Resume Amendment: S0-C3

Authorized on 2026-08-31 for branch `redu-hz`, starting repository commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.  This amendment records the
user-authorized continuation after the pause and narrows the current work to
one repeated structure.  It supplements but does not overwrite the hashed
`GOAL_CHARTER.md` or its E0 amendment.

## Goal

Develop a PLDI-strength, exact, genuinely nonconvex Neural-HZ representation
that raises the sole formal baseline beyond 1,870/2,413 and produces a large
capability gain on both CIFAR100 and TinyImageNet.  The long-range formal
target is 2,413/2,413 on one candidate source/configuration/path.  Work proceeds
one repeated structure at a time; no later structure starts until the current
version is either promoted through its frozen gates or honestly closed.

The sole current structure is the residual large-CNN chain:

```text
Conv -> exact diagonal* -> ADD
     -> exact shared diagonal* -> shared Conv -> ReLU frontier.
```

The current sequence is fixed:

1. prove/repair BatchNorm SCALE/BIAS graph-event, variable-producer and HZ
   operator-lineage bijection without changing production defaults;
2. implement the independent S0-C3 same-frame residual-distributive `D*` rule;
3. prove exact set, factor, predicate, bias and witness preservation;
4. run Tiny iid143 ReLU36, then ReLU63 only after ReLU36 passes;
5. run preregistered same-structure Tiny/CIFAR shadows and zero-hit guards;
6. replay the affected formal family/families and four-concurrent gate; and
7. run all 2,413 formal rows before any score or default promotion.

S0-C1 and S0-C2 remain closed real-graph zero-hits with gain zero.  S0-C3 may
reuse their proven infrastructure but may not rewrite their grammar, evidence,
hashes or closure.  The corrected graph is a prerequisite, not HZ gain.

## Non-negotiable limits

- Formal baseline: exactly 1,870/2,413 = 1,063 CERT + 807 concretely validated
  ADV.  Every old solved row and every one of the 13 family solved counts must
  remain solved; invalid ADV/SAT is zero.
- Score: UNKNOWN, TIMEOUT, ERROR, intentional stop, synthetic test, component
  speed and disconnected run have gain zero.  Only a complete 2,413 replay
  retaining all 1,870 and adding a sound CERT or validated ADV changes the
  headline.
- Representation: preserve continuous factors, binary nonconvex phases,
  equality/inequality predicates, shared latent/frame identity, exact source
  ancestry and reversible concrete witnesses.  No Zonotope, constrained
  Zonotope, box or other convex-domain degeneration is allowed.
- Uniformity: one structural/state rule only.  iid, family, model, layer,
  margin, historical verdict and elapsed behavior cannot select the rule.
  Attack/PGD, BaB, splitting, backward/dual rescue and LP status cannot be
  credited as HZ representation gain.
- Fail closed: any missing graph event, unproved identity, nonfinite payload,
  alias ambiguity, budget overflow, frame/predicate/bias mismatch, resource
  failure or witness mismatch returns the identical baseline path/UNKNOWN.
- Physical gate: candidate bytes and entries must each be strictly smaller at
  the same complete strong-root and consumer-GC boundary; remaining residual
  successors, old aliases, descriptors, artifacts, bounds and pending plans
  all count.  Four-concurrent performance cannot regress.
- Data: `/data1/Kane/HyZor` and every historical model/log/table/result are
  read-only.  New outputs use exclusive/no-overwrite names only beneath
  `experiments/neural_hz_20260831/` with branch, commit, config, dependency and
  baseline provenance.
- Trial 9: PID/start-ticks-bound source freeze remains untouched; the worker is
  neither killed nor altered, and its sealer retains an exit record
  automatically.  Its intermediate stop has formal gain zero.

## Promotion meaning

The external E0 ledger stays separate at 61/400 (25 CIFAR + 36 Tiny validated
historical-origin ADV and 339 UNKNOWN).  A future one-path 400-row replay must
retain all 61; only newly solved rows among the 339 UNKNOWN are external
Neural-HZ gain.  External gains demonstrate generality but are never added to
the 13-family 1,870 score.

The current S0-C3 rule, tests, target order and closure conditions are frozen
in `S0_C3_IDENTITY_MIDDLE_PREREG_V1.md`.  Until its graph-faithfulness
certificate passes, no Tiny/CIFAR target execution or production integration
is authorized and its gain remains zero.
