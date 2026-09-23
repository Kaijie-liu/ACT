# BN dataflow repair and same-object conformance control

Scope authorized 2026-09-23: repair BN expansion edges, then check the same
physical object before integrating fast feasibility. No historical relabel.

Repair only converter BN dependency edges: SCALE reads the producer of BN
input variables (or a placeholder); BIAS reads that specific SCALE. Existing
FX predecessor overrides preserve branch identities, including a later branch
from the original input. Coefficients, intervals and solver policy unchanged.
Fourteen initial regressions cover historical failure, first-layer/multiple/
shared-module/branched BN, negative/unit scale, BN1d/2d/3d and nonaffine BN.
Old defective control JSON remains, tests now require repaired conformance.

New control: exactly old MNIST0 and expert1, same materialized normalized
float64 box/2/255, checkpoint and source adaptation. 30 s outer, 8 GiB sampled
RSS, existing 2 GiB CSR representation budget. No native query; one new
zero-free-factor proposal with 3 s construction/check cap inside the outer
budget. No imported old assignment or certificate. Save fresh matrix,
assignment, input map, all layer point arrays and source forward arrays.

Both source -> concrete IR and concrete IR -> HZ at this fixed point must
agree within diagnostic 1e-9; this is NOT an altered proof acceptance margin
or an all-domain floating execution equivalence theorem. Final model identity
is NEW. Source hashes are explicitly rebound ONLY for torch2act.py; every
other old executable/input/environment binding is checked unchanged. Old
manifests continue to fail drift checks on the repaired checkout; use their
historical commits to reproduce them, never rewrite their hash lists.

Commit/push implementation, freeze config, commit/push, inspect resources,
run once, saved-only independent audit/archive. Failure preserves partial
evidence and does not open certification. Success permits development of an
opt-in checked-assignment base-query path; it does not itself produce SAFE.
Native property tests and any representation study require their own freeze.
