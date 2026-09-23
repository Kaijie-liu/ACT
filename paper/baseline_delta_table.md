# Baseline delta table

Current scope (2026-09-21 review response): historical HZ SAFE counts below
are frozen-policy acceptances, not source-complete real-box certificates.
Shared input construction warrants caution across cohorts; exact containment
failures have been established only for the specifically audited inputs.
See the [proof boundary](README.md#current-proof-boundary) and
[review response](../docs/external_ai_review_response_20260921.md).

This table fixes the comparison target for each neighboring MoE or verifier
family. “Delta” means a capability or audited evidence difference. It does not
mean formal common-task numerical superiority. B3 now provides an audited
official-scale numerical-conformance comparison; formal superiority remains
prohibited because the installed CROWN bounds are not outward rounded.

| Comparator | What it provides | Audited delta in this project | Execution status |
|---|---|---|---|
| Zhang et al., ICML 2025 (RT-ER) | Analytic Lipschitz-gate certification theorems and official training/model code | Theorems 5.4/5.5 have no released numerical instantiation or constants protocol; the released CIFAR-10 pipeline uses hard argmax and does not update the router. Our exact K=20 census quantifies the resulting radius-dependent applicability and initialization lottery. | Two official-code compatibility reproductions land at 34.22%/32.70% and 32.01%/30.51% SA/PGD-50 RA, both outside frozen paper-reference intervals with 0 audit issues. B3 r5 completes all 318 expert branches: fixed-radius Route A versus route-invariance numerical filters are 17/12, 14/8, 7/3, 2/0, and 0/0. Formal SAFE remains zero pending validated numerical bounds. |
| MetaMoE-style route invariance | Composition after proving a fixed route | With the same downstream verifier and budget on the frozen 100-sample cohort, staged Route A resolves 56 additional samples and yields 36 route-changing HZ-policy SAFE results unavailable to the route-invariance precondition. | Historical executed reimplementation, not author-tool reproduction. A separate author-checkpoint/backend sufficient-adapter comparison initially stopped on ERROR; its disclosed-repair, same-cohort full20 followup is now archived: ACT4 policy positives versus9 author numerical filters, no ACT-only positive. Do not conflate these experiments or numerical grades. |
| SpecSphere and unavailable certification artifacts | Published certification claims | The audited case series records artifact/retrieval availability and machine-checkable assumptions without inventing an executable comparison. | Survey-only; not presented as a runnable baseline. |
| alpha,beta-CROWN | State-of-the-art static-network verification | Its frontend rejects the full dynamic-dispatch ONNX graph at `GatherElements`; Route A specialization converts the same model into four accepted static expert graphs. This project extends the verifier's input domain rather than competing with its expert bounds. | Parser rejection and 4/4 specialization acceptance executed; official-scale numerical conformance completed on 318 branches, while formal expert certificates remain gated on outward rounding. |
| Monolithic MILP/HZ | Standard single-formulation exact or mixed-integer reference | Route-conditioned branches have substantially smaller structural binary width when multiple experts are feasible. On the frozen 20-row common cohort, Route A solves 12 rows and the true single-formulation F0 baseline solves 8; discordance is 5 versus 1 (exact paired p=0.21875), so the result is descriptive and not a dominance claim. | Executed with shared F0 semantics and 900-second row deadline; all 2 UNSAFE rows replay in both systems. Runtime is not treated as paired because executions were not interleaved. |
| robust-moe-cnn / V-MoE | Empirical robustness or sparse-MoE scale | They establish architecture relevance but do not provide the same route-conditioned formal-verification task. | Provenance/background only; no borrowed checkpoint or superiority claim. |
| Hash Layers / THOR | Static or randomized routing as a legitimate design choice | They calibrate interpretation: the audit concerns the released artifact, theorem assumptions, and training narrative, not a claim that static routing is intrinsically invalid. | Related-work control, not a verification baseline. |

The five evidence-backed contribution axes are: exact affine route-boundary and
applicability measurement; retained-path-condition verification with positive
and negative representation controls; staged normalized top-k semantics across
two gate families; artifact-level certification-gap auditing; and a replayable,
hash-anchored MoE verification artifact. Numerical “outperformance” is reserved
for the B3 common-task table.

## MetaMoE intake repair control (2026-09-23; separate from the table)

The author checkpoint intake exposed an ACT BN expansion edge defect. A new
converter version restores SCALE -> BIAS and passes same-object source/IR/HZ
point conformance; this is not an all-domain source equivalence proof. On
one old MNIST0 request, both corrected-source arms exclude all 19 output
violations, but native base feasibility times out. A new opt-in checked
current assignment closes that nonvacuity gate, yielding
`POSITIVE / HZ_POLICY_ACCEPTED` under unchanged budgets and numerical gates.
See the [audited control](../docs/metamoe_checked_base_result_20260923_r1.md).
This is one route-stable request, NOT route-changing or source-complete
certification and NOT a new author-tool competition table. Historical defective
runs are preserved, not retrospectively relabelled; any affected source claims
must be reviewed using their actual converter/model identities.

A separate two-call routing-feasibility control kept expert checked-base on in
both arms and retained all 19 output obligations. Both accepted the same old
route-stable request; full charged costs were 132.264 and 39.453 seconds.
The checked arm supplied a fresh, fully checked point for the current guarded
routing matrix instead of the 92.759-second native feasible-point search;
score-support optimization stayed unchanged at about 30 seconds. Identical
routing/expert matrices and complete obligations pass the saved-only audit.
This is a single-input engineering cost control, not a new certificate gain,
population speedup, source-complete guarantee or author-tool comparison.
See the [result and limitations](../docs/metamoe_checked_routing_result_20260923_r1.md).

The subsequent, separately frozen nonzero-precheck control kept both checked
feasibility paths on. The already available generator enclosure excluded zero;
an independently recomputed rational sign of the stored row confirmed it.
Both arms retained 19/19 output queries and identical routed/expert/source
matrices; complete policy-positive endpoints were unchanged. Request costs
were 39.407 versus 8.764 seconds, with one versus zero support optimization
calls. This demonstrates avoidance of unnecessary optimization on one old
request, not tighter relaxation, new coverage or author-tool superiority.
The upstream source/guard assumptions and native infeasibility trust remain.
See the [separate result](../docs/metamoe_nonzero_precheck_result_20260923_r1.md).

Fresh combined-configuration author smoke retains two old inputs and complete
end-to-end charging. Both arms replay the CIFAR10/0 center violation and both
accept MNIST/0 under their distinct numerical contracts. ACT takes 9.446 s
versus 7.826 s for the unchanged author-backend route-invariance sufficient
adapter. This validates the comparison path but supplies no ACT speed or
coverage advantage. The three shortcuts remain default-off, all 19 output
properties retained, and no source-complete claim is made. See the
[four-request archive](../docs/metamoe_checked_paired_smoke_result_20260923_r1.md).

## New-input author sufficient-adapter comparison (2026-09-23)

The separately frozen5CIFAR10+5MNIST cohort has20 terminal rows, but only
15 executions:14 normal returns,1 authorBaB ERROR and5 registered fail-stop
omissions. On the7 normally returned input pairs, ACT accepts MNIST1,3 under
HZ_POLICY_ACCEPTED and returns UNKNOWN on all5CIFAR10 inputs; the author
strict-route-invariance sufficient adapter returns numerical positive filters
on all7. There is no ACT-only positive, new route-changing certificate or
performance victory. Author MNIST7 fails a BaB lAs consistency assertion;
this cost and all omitted rows remain visible, not recoded as model failures.
The missing3input pairs prevent a full-cohort comparison. These are different
numerical contracts, not an unconditional formal-SAFE competition.

ACT's133 recorded output rows comprise56 native infeasibility acceptances and
77 local query TIMEOUTs; routes/base assignments/nonzero checks completed.
This identifies an execution limit, not proven relaxation insufficiency.
All attempted costs include the author error:ACT190.163s/7calls,
author309.331s/8calls; these unequal, differently terminated denominators are
not a speedup ratio. On seven complete pairs ACT is slower by19.108s mean;
the two common positive MNIST pairs differ by2.606s mean. Source/guard lowering
and native numeric bounds remain trusted. See the
[stopped-run result, audit and limits](../docs/metamoe_checked_small_result_20260923_r1.md).

The subsequent separately frozen compatibility control reproduces the author
BaB failure as9initial lA keys versus4updated keys:5disconnected expert entries
were zero initially and omitted asNone after pruning to a router obligation.
An explicitly labelled runtime wrapper restores only structurally justified
zero branching metadata, retaining original bounds/assertions. Old MNIST1
keeps its numerical filter; MNIST7 completes7original domain insertions, then
hits300s outerTIMEOUT. AuditPASS0 opens a compatibility gate, not a coverage
or performance claim. The failed original20batch is not relabelled or completed
by these controls. See the [repair limits](../docs/metamoe_las_repair_result_20260923_r1.md).

## Repaired same-cohort full comparison (2026-09-23)

A newly frozen full20 execution on the same observed10inputs completes without
ERROR or omissions. ACT has4 HZ-policy positives/6 completedUNKNOWN; the author
backend plus disclosed lAs wrapper and strict route-invariance sufficient
adapter has9 numerical positive filters/1 outerTIMEOUT. Common positives are
MNIST1,3,9,10; author-only all5CIFAR10; ACT-only0. There is no new route-changing
or source-complete certificate. This is not a fresh holdout or a literal
unchanged author-tool execution. The original failed batch remains separate.

Full charged costs are285.577s ACT and365.374s author. The latter includes the
unresolved300.033s MNIST7 run; early ACT UNKNOWN cannot be sold as equal-success
speedup. On nine normally returned pairs ACT costs16.026s more on average;
on four common positives mean costs are10.552s versus7.171s. Thus this fixed
comparison does not establish ACT coverage or successful-solving speed benefit.

ACT records209 output rows:116 excluded under its native numerical policy,
93 localTIMEOUT. MNIST7 additionally lacks one route's reachability/base proof.
Saved logs identify local scheduling/receipt limits, not a demonstrated
representation defect. Structural auditPASS0 and empty original UNSAFE replay
do not close source conversion or numerical proof gaps. See the
[full result, costs and bounded diagnosis](../docs/metamoe_las_followup_result_20260923_r1.md).
