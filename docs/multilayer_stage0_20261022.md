# Multilayer stage 0: frozen semantic comparison

This is the first stage of the user's fourth research revision, not evidence of
algorithmic novelty, real-model performance or source-complete certification.
The adjacent JSON fixes nine calls before any numerical outcomes: three cases
in SAFE, UNSAFE, tie order, each with ACT v1, naive GenBaB and generically pruned
GenBaB. One cumulative 300-second CPU batch includes interpreter startup,
conversion, point reference, solving, audit, reception and cleanup. Existing
DEV_MARGIN_20261021_V1 and owned-session supervision apply: 8 GiB per-process
virtual address limit, RSS telemetry, 128 MiB total artifacts, unchanged
numerical gates. No training data/checkpoint/GPU or dependency installation.

## Fixed objects and analytic answers

Input x is in [-1,1]. Each of two layers selects any tie-legal unordered top-2
of three experts and uses selected softmax. First-layer scores are (x,-x,-1/2),
and experts are (x+1,x-1,x). Every resulting hidden z is in [-2,2]. Second-layer
scores are (z,-z,-4). Its third expert is never legal: both z and -z exceed -4.
The output has two logits; its property is first minus second >0.

- SAFE: second experts are (z+3,0),(-z+3,0),(-100,0).
  The only legal second pair is {0,1}; its first output is
  3+z*tanh(z), at least 3. Even independent convex mixing gives the weaker
  lower bound1 since |z|<=2. All three first pairs remain coverage obligations.
- UNSAFE: replace both second-layer offsets3 by -1/4. At x=0 the first selected
  experts cancel, z=0 and the original dynamic model margin is -1/4. This
  concrete witness must replay; a static relaxation violation is insufficient.
- Tie: both routers are identically zero, so all nine histories are legal.
  First-layer z is in [-3/2,3/2]. Second experts are (z+3,0),(-z+3,0),(3,0).
  Each expert's first output is at least3/2, so every legal mixture is positive.
  No tie history may be pruned.

The reference partitions [-1,1] at the first-router boundaries -1/2 and1/2
and includes257 dyadic grid points. For each of nine histories it compares
ACT's native static expression with the independent external compiler and
checks the scalar bad-set direction. Sampling is differential evidence, not
the analytic proof above. No margin, case or domain is selected by outcomes.

## Three paths and compatibility boundary

ACT v1 enumerates all nine full histories, jointly propagates final outputs and
all routers, then imposes every selected>=outside guard. The default-off
weighted_matmul option expresses the same selected-softmax sum through the
existing shared-factor softmax/value MATMUL transfer. Local conversion handles
MATMUL explicitly instead of the generic dispatcher's `mul` substring. This is
necessary compatibility, not a new domain, cross-layer union or partition rule.

External static histories independently compose actual earlier expert mixtures.
Each bad-set output is max(property margin, all outside-minus-selected guards)
and must be strictly positive over the entire input box. This includes ties.
The pruned frontend independently computes all ordered next-router margin upper
bounds with plain CROWN over the static prefix's original input box. It ignores
previous guards conservatively, excludes a pair only with upper<-1e-9, and
retains zero, missing and nonfinite bounds. It never reads ACT coverage.
Naive and pruned use the same frozen B2 alpha-CROWN/nonlinear GenBaB engine,
CPU/double configuration and four disclosed compatibility patches.

Both frontend coverage catalogs retain attempted, verified, pruned and missing
histories. Enumeration coverage is not output-proof closure. Structural audits
do not reprove CROWN/HZ bounds. Numerical positives retain separate evidence
grades; only original-model replay can accept UNSAFE.

## Readout and stopping

All nine expected statuses must agree with analytic answers, with complete
positive-history coverage and valid UNSAFE replay, before stage1. An incorrect
conclusion or coverage/identity error stops for scientific review. An execution
or conversion tooling error preserves its prefix, is repaired and retested
under the user's standing authorization; it cannot be relabeled success or
used to adjust the cases. UNKNOWN/TIMEOUT is not an incorrect safety claim but
does not pass this stage. Runtime outcomes will be appended here and bound in
the existing one-stage record, not expanded into per-interface protocols.

## Later-phase clarifications registered before results

The current goal incorporates the user's explicit three-arm stage2 budget
540calls/45h, separate on/off and on/GenBaB gain gates, gray uncertainty case,
and identical budgets for each stage3 comparison. Stage1 common fixed PGD and
full-model replay are independent of arms and separately charged; replay-valid
UNSAFE from any source is excluded from primary n', with raw60 denominator,
source overlap and both cost versions also reported. Those later decisions do
not change this synthetic batch.
