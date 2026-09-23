# BN expansion edge defect found during feasibility research

## What changed in the diagnosis

The current-request feasible-assignment work did identify reusable existence
evidence cheaply, but also exposed a source-correspondence problem. This is
**not** a conclusion that generic HZ relaxation is too weak, nor a reason to
spend more native time on the old output matrix. The original MNIST0 outcome
remains UNKNOWN. No source SAFE or source UNSAFE was added.

See `metamoe_current_assignment_result_20260923_r1.md` for the 0.01859 s
construction/check, 16 abstract violation regions, original point replay, and
12.839 s of observed duplicate-query work. The following additional bounded
diagnostic separates the conversion and HZ stages.

## Same-point real-model localization, not new solving

Implementation `c0e951d86`, R1 freeze `c30de592f`: the first diagnostic failed
because exclusive-create progress writing reused one filename. R1 ERROR and
partial INPUT observation remain in their own directory. They did not pass
the final matrix identity gate and are not a conformance result.

Recorder-only R2 implementation `f8ac95713`, freeze `32f1cfc5d`, config SHA256
`e2ebb434d9cbc62ad4d81226290c0105937f25878ed21d5968e378afc02f7b0d`.
It uses a distinct progress file per layer, same original replay worker,
same old input/assignment, 30 s / 8 GiB, zero new native queries/proposals.
No production transfer/conversion code changed. It completed in **7.026 s**
outer charged time, peak sampled RSS **1,806,925,824 bytes**.

All 26 non-ASSERT expert layers were observed. For every layer, the stored HZ
value of the fixed factor assignment was compared with independent concrete
torch execution of the **converted IR** and its actual predecessor graph:

- Maximum across all layers: **9.663381206337363e-13**.
- Final HZ versus concrete IR: **2.3092638912203256e-14**.
- Final concrete IR versus original padded expert: **3.203507818200451**.
- Final HZ matrix exactly matches the frozen base model hash; saved point
  passes the unchanged full constraint check.
- Fresh original point, outputs, input box and properties exactly match the
  earlier point replay arrays. No new source violation.

Thus, at this point, HZ follows the converted graph; the converted graph does
not follow the source output. This localizes the next investigation to
conversion rather than blaming the factor proposal. It is not an all-domain
equivalence proof, an exclusion of every numerical error, or a proof that
there is only one conversion defect.

Saved-only audit: `scripts/audit_metamoe_assignment_layers.py`; compact result
`metamoe_assignment_layers_archive_20260923_r1.json`. It independently recomputes
differences, checks all 26 append-only progress records, final matrix/input/
source bindings, streams, deadline and failed R1. It performs no propagation,
original forward or solver call. Raw records remain under
`/data1/Kane/MOE/baseline_runs/metamoe_assignment_layers_20260923_r{1,2}`.

## Concrete source-level defect, independently reproducible on a tiny model

`act/pipeline/verification/torch2act.py::_convert_batchnorm()` correctly creates
SCALE then BIAS, with the BIAS `in_vars` naming the SCALE outputs. But
`_register_node()` maps the FX BatchNorm node to its last layer, BIAS.
`_build_preds_succs()` gives that mapped layer the original FX predecessor
(e.g. CONV2D), and only gives **unmapped** SCALE its sequential edge. It keeps
the already-present BIAS edge, bypassing SCALE. Consequently `preds` and
declared `in_vars` disagree. HZ propagation follows `preds`.

The toy control uses one identity convolution, eval BatchNorm with exact
scale 2 and bias 1 (eps 0, variance 1, mean 0), then flatten/sum. The four
inputs equal 2. No model/data/optimizer is involved:

| Point computation | Output |
| --- | ---: |
| Source `sum(2*x+1)` | 20, 20 |
| Current converted graph, BIAS predecessor CONV2D | 12, 12 |
| Local toy interpreter with BIAS predecessor SCALE | 20, 20 |

Other controls cover negative scale, unit scale, BN as the first operator,
and one spatial element. All show the structural bypass; unit scale masks
the numerical error, explaining why identity-like controls alone are
insufficient. Rewiring only the toy BIAS edge restores its source output.

Code: `scripts/diagnose_bn_expansion_edges.py`; recorded outputs:
`metamoe_bn_edge_controls_20260923_r1.json`; five diagnostic regressions:
`tests/test_bn_expansion_diagnosis.py`. These tests deliberately assert that
the **sealed converter defect is detected**, not that it has been fixed.
The production converter is unchanged. This proves the existence and toy
mechanism of the edge bug; a corrected real-object replay is still needed
to determine how much of the real mismatch it explains.

## Impact and required next step

1. Do not treat this MetaMoE expert IR as source-equivalent. A numerical
   feasibility point from it cannot bypass original full-model replay.
   Source-level positive conclusions using an affected BN expansion need
   separate impact review; do not erase or relabel frozen records.
2. No universal retraction of unrelated models follows from this finding.
   BN-free MLPs, other converters and other graph paths must be checked by
   actual source/graph identity rather than assumed affected or unaffected.
3. Next repair is a separately versioned **conversion/dataflow** change, not
   a relaxation tweak. Preserve internal expansion edges and external FX
   dependencies together. Add checks against input-variable producers and
   controls for first-layer BN, negative/non-unit affine coefficients,
   branching/shared predecessors and multiple BN expansions. Avoid a generic
   "every BIAS connects to previous layer" rule, which would corrupt branches.
4. Before any new native property solving, replay the same physical request
   through source, corrected concrete IR and corrected HZ under a fresh
   identity. A corrected matrix must receive its own current-request proposal;
   the old point/proof is hash-bound and cannot be imported as valid evidence.
5. Only after conformance controls should the fast checked-base assignment
   enter a separately frozen scheduling control. Byte-identical property
   deduplication remains another isolated optimization. Keep all old 19
   obligations, numerical gates and full-model UNSAFE replay requirements.

Combined focused controls: **73 tests in ACT**, including the five
known-defect diagnosis tests; corresponding new suites also pass in the
pinned intake environment. Zero additional real native queries, no enlarged
budget/cohort, no production default change, no extra output certificate.
