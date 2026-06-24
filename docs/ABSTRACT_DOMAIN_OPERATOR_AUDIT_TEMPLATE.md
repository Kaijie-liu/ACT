# Abstract Domain Operator Audit Template

This template is the gate for adding or changing an ACT abstract-domain
operator.  A benchmark improvement is not enough; each operator must first be
checked on controlled toy networks where the first divergence can be localized.

## Rules

- Use degenerate boxes (`lb == ub`) for point consistency before any interval
  or HZ-width experiment.
- Compare every nontrivial operator with an independent exact oracle.  Gurobi
  is allowed here only as a diagnostic oracle, never as a proof path for final
  benchmark results.
- Separate soundness, tightness, and solver runtime.  Do not call a solver
  timeout a soundness bug.
- ORT may audit an engine-produced witness, but it must not upgrade an engine
  UNKNOWN into a pure HybridZ verdict.
- Do not tune a single iid into the final result.  Any benchmark number must
  come from a frozen dataset-level config and artifact.

## Eight-Step Ladder

1. Point consistency: `lb == ub` must match the real operator to about `1e-6`.
2. Per-layer width: compare domain bounds with an exact oracle and find the
   first layer where the ratio jumps.
3. Affine Jacobian check: when all nonlinear phases are stable, the domain
   should be exact for affine structure.
4. MILP-tightness: compare the domain's exact binary encoding with the
   independent exact oracle.
5. Binary-count audit: unstable nonlinearities should create exactly the
   expected binary variables.
6. Raw VNNLIB assert check: validate canonicalized `C, t` against the raw
   property to catch sign or disjunction mistakes.
7. Blast-radius test: run old vs fixed code across representative benchmarks
   and record both gains and regressions.
8. Toy first, production second: do not run full benchmark experiments until
   the toy audit is clean or the remaining gap is explicitly classified.

## Minimum Report

For each operator or structural pattern, record:

- point-consistency max error;
- exact-oracle width and domain width;
- ratio and first layer where the ratio changes;
- binary/continuous/equality counts;
- effect of exact redundancy removal;
- effect of any lossy reduction, if enabled only as an ablation;
- runtime and memory notes;
- final classification: `sound+tight`, `sound+loose`, `unsupported`, or `bug`.
