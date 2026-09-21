# Multilayer whole-box history verifier v1 — 2026-09-21

## Delivered contract

`act.back_end.moe.multilayer.verify_multilayer_box` accepts a CPU/float64 eval
model, explicit finite `center/lower/upper`, and all requested linear rows
`rows @ output >= thresholds`. “Whole domain” means every input in this box
and every tie-legal complete history, NOT only a clean forward trace, and NOT
all of R^d. This is a new API, not a change to frozen output-level top-2 F0.

The model is a pure, statically traceable acyclic tensor program. Each routing
call is per-input, has homogeneous expert output shapes and a finite discrete
top-k choice. Residuals and repeated invocation of a shared routing module are
supported; repeated calls have separate call-site identities. Routing inside
experts/routers, token/patch routing, batch-dependent normalization, stochastic
training execution, active hooks and dynamic Python control are not v1.
FX/network/input/guard lowering remains trusted, including functional tensor
semantics. This is not an independent source-equivalence proof.

Generic `RoutedLayer` supports hard top-1 and selected-softmax graphs that the
existing HZ backend can retain. Compilation also preserves raw scores and the
author's raw-score/(selected-sum+1e-5) expression, but unsupported nonlinear
graphs return `UNSUPPORTED`, not a substituted independent-box certificate.
The current raw-epsilon control compiles concretely but loses the backend
joint HZ and is explicitly unsupported. Do not describe all weighted graphs
as verified just because they can be compiled.

The pinned Robust Experts `MOELayer`/`SkipMOELayer` and `TopKGate` have an
explicit source adapter. It recognizes their ORIGINAL gate semantics: k=1
STE is nonzero-score masking, not softmax; k>1 normalization is raw-score
division, not selected softmax. Zero-score STE cannot silently become weight
one. Source hash mismatch/fixed-expert/variance modes reject. RoME's dense
continuous low-rank mixture is deliberately NOT reinterpreted as top-k.

## Coverage argument (conditional on lowering and numerical policy)

For sites t=1,...,T let H be the Cartesian product of all unordered k_t-subsets
of E_t experts. A history h defines a complete static model F_h: at each site,
selected experts receive the intermediate value produced by its predecessors
in THIS history. Its score vector r_t^h is evaluated on that same intermediate
value. Compile `(F_h, r_1^h, ..., r_T^h)` together, never propagate a later router
on a clean-route tensor. The joint HZ thus retains a single shared input frame,
branch-dependent hidden activations, output and every route guard.

Define X_h by all `r_t^h[i] >= r_t^h[j]`, selected i and unselected j. Ties are
included. Induction in call-graph order shows that any legal concrete execution
has a history h in H, its input belongs to X_h, and its output is F_h. Conversely,
a defined static history satisfying its guards is a legal real-arithmetic
execution under ANY_LEGAL_TOPK. Consequently, proving every requested property
on every nonexcluded X_h proves the full routed output property on X.

For STE simplification, each selected score must be proved nonzero on the
corresponding PREFIX domain, before applying downstream guards. Otherwise a
later guard could hide an earlier zero-score input on which the simplification
changed the function. An undecided prefix condition prevents exclusion or
acceptance of that history. Raw denominator obligations are similarly explicit;
the current backend may reject the nonlinear graph before reaching them.

Only solver-policy infeasible histories are excluded. Unknown feasibility is
retained and may still be discharged by positive output bounds. A cap or hard
deadline cannot turn a partial history list into complete coverage. If any
needed obligation remains, the result is UNKNOWN/TIMEOUT/UNSUPPORTED.

## Evidence levels and limitations

- `POSITIVE / HZ_POLICY_ACCEPTED`: all complete histories excluded or all rows
  accepted under the existing optimal-status, finite-bound, corrected-positive
  margin policy (1e-7). No acceptance threshold or backend search changed.
- `UNSAFE_REPLAYED / FULL_MODEL_REPLAY`: concrete in-box violation replayed on
  the ORIGINAL full model, not on an expert or relaxation alone. This uses the
  recorded floating execution, not an independent exact-arithmetic witness.
- `audit_multilayer_result`: stdlib-only structural audit reconstructs the
  history/product cardinality, property and prefix-definedness coverage,
  request/policy binding, and acceptance conditions. It does NOT reprove solver
  bounds, infeasibility, network lowering, or deployed floating-point semantics.

The package says `source_complete=false` and lists trusted components. Do not
upgrade the paper to a source-complete/strict floating multi-layer certificate.
The full-history strategy is exponential; v1 defaults to at most 4,096 complete
histories. It does not claim scalable certification of the full author ResNet.

## Hard budget / evidence interface

Use `verify_multilayer_box_supervised(..., output_dir=<fresh>, total_seconds=300)`
behind `if __name__ == '__main__'`. The API starts with an already constructed
CPU model and request tensors; upstream checkpoint loading by the caller is
outside this API boundary and must be charged by an experiment runner.

Child startup/transfer, static compilation, propagation, guards, feasibility,
definedness, all property solves, progress serialization, original replay and
the child structural audit are inside ONE deadline. Parent identity/structural
acceptance is also charged and late acceptance rejected. Native child calls
are terminated at the outer deadline; bounded termination/kill cleanup and
terminal receipt publication are separately disclosed, not extra solving time.

Fresh directory contains request, atomic progress, candidate or error, and a
terminal receipt with hashes and cost fields. Partials are not deleted. A
candidate published late cannot replace a TIMEOUT. Progress includes active
history/phase/property. Per-history compile/propagation/guard/feasibility,
definedness and solver timings are diagnostic components; unclassified
framework/serialization overhead stays in total wall cost, never subtracted.
Public acceptance does not run a solver in the parent. Independently coded
structural auditing and full-model witness replay execute inside the child;
the parent rechecks identities/structure and the audit receipt.

## Controls and usage

Run in the unchanged ACT environment:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 /data1/Kane/miniconda3/envs/act-py312/bin/python \
 -m unittest discover -s tests -p 'test_multilayer*.py' -v
```

Controls include four complete two-layer histories with changing routes;
non-clean-history full-model UNSAFE; earlier outputs constraining later routes;
ties (including unsafe unchosen ties); repeated module calls; convolution,
ReLU and residual paths; selected-softmax concrete differentials; raw division
fail-closed; prefix nonzero obligations; budget/cap/solver-limit rejection;
missing history/property/wrong identity mutations; partial, exception, startup
and late terminal controls; real supervised positive and unsafe requests.

The full-size ORIGINAL Robust Experts architecture compatibility check is
separately frozen in `configs/recent_moe/multilayer_author_history_control_r1.json`.
It uses the already installed author environment only for author forward and
FX/static compiler intake (ACT environment lacks einops; no dependencies are
installed or injected). ACT solving/tests remain in act-py312. Its two fixed
probes are NOT a certification experiment and do not exercise all 1,024
histories. It is initialized, not a trained checkpoint. Results will be archived
separately without overwriting any existing author workflow or holdout.

## Next boundary

1. Preserve this implementation/control result independently of frozen top-2
   experiments. No previous UNKNOWN becomes SAFE.
2. Before a real multilayer proof experiment, bind a trained source model,
   supported operator graph, input selection, complete property rows, history
   cap and total budget. Full-size compatibility alone is not sufficient.
3. Missing joint-HZ nonlinear support and exponential history cost are explicit
   engineering limits; future relaxations/prefix reuse require separate proofs,
   controls and experiments, not silent changes to this v1.
