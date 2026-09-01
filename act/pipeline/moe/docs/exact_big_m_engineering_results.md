# Exact-support big-M engineering result

The repaired r2 execution completed all 20 paired sample/radius identities and
passed independent audit with zero issues. Candidate semantics matched on all
20 pairs.

Constraint-aware support reduced aggregate membership selector width from 657
to 631 binaries. Twelve samples improved, with 26 eliminated selectors in
total (3.96%). It did not reduce branch-and-bound nodes in this feasibility
workload. The median exact/fast feasibility-time ratio was 1.009; once support
cost was included, the median total-time ratio was 7.142.

Numerical capping was material to the audit trail: 124 support sides across 52
expert conditions used the minimum of the solver-derived and independent fast
sound bounds. This is a sound minimum of two upper bounds and prevents a
numerically valid but looser support result from increasing M.

The engineering decision is therefore negative for default scheduling. Fast
generator bounds remain the default membership encoding. Exact support remains
available as a correctness/tightening ablation and for cases with an explicit
selector-width bottleneck. No certificate or Experiment 1D solved-rate change
is attributed to this experiment.

The failed r1 execution is retained separately and excluded. Frozen evidence
for r2 is in
`act/pipeline/moe/results/exact_big_m_engineering_20260901_r2_audit.json`.
