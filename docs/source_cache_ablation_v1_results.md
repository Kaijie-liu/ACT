# Source-cache attribution — controls and freeze

**FROZEN, NOT EXECUTED.** No new real-model verification queries were run.
The previous eight-request experiment remains sealed. This follow-up uses the
same four observed inputs; it is not new-sample confirmation or cohort expansion.

- [Protocol](source_cache_ablation_v1.md), [freeze](source_cache_ablation_v1_freeze.json).
- [Controls attempt 001](source_cache_ablation_controls_attempt001.json):
  **110/110 PASS**, including inherited regressions and four offline-analysis
  controls. No skipped tests. Analytic tests are not real-cohort measurements.
- [Separate-process identity review](source_cache_ablation_v1_selection_review.json):
  PASS, zero issues; exact parent input/configuration/tensor hashes reconstructed.
- Inputs **220,222,230,232**, same convolutional epoch-89 checkpoint, epsilon
  **2/255**. Eight interleaved requests: matrix-only versus both caches.

Both arms retain matrix parsing reuse and the identical single portable tail
check. Only the upstream source-cache flag differs. The analytic full chains
produce exactly the same checked result; each retains all checks and the original
start time. Each package relocates and verifies with `python -I -S`, without
loading checkpoint/data/solver. Re-signed source/matrix flag mutations are
rejected. Real watchdog, expired/late-publication, invalid identity, missing-cost,
partial-JSON, fail-stop and complete-denominator controls pass.

No performance conclusion or default change is made from these controls.
Current 23.21% combined-cache reduction remains the old small-cohort observation;
it is not evidence that source caching individually helps or hurts. Source
decode/freeze/copy costs and changes in generated obligation counts require this
separate comparison. Report full cost and evidence coverage jointly.

After clean commit/push and a decision to execute, launch once with the command
in the protocol. New output directory is reserved but **not created by this
stage**. Do not resume/retry the old study, adjust checks/order/precision, or
combine this attribution with representation changes. Follow-up summary records
source equality and common checked-row equality where observable; missing data
are not agreement. No source-cache default is switched automatically.
