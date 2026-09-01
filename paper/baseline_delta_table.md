# Baseline delta table

This table fixes the comparison target for each neighboring MoE or verifier
family. “Delta” means a capability or audited evidence difference. It does not
mean common-task numerical superiority. The latter remains pending the
official-scale B3 table and must not be claimed before B3 completes.

| Comparator | What it provides | Audited delta in this project | Execution status |
|---|---|---|---|
| Zhang et al., ICML 2025 (RT-ER) | Analytic Lipschitz-gate certification theorems and official training/model code | Theorems 5.4/5.5 have no released numerical instantiation or constants protocol; the released CIFAR-10 pipeline uses hard argmax and does not update the router. Our exact K=20 census quantifies the resulting radius-dependent applicability and initialization lottery. | Seed-0 official-code compatibility reproduction landed at 34.22% SA / 32.70% PGD-50 RA with 0 audit issues, outside the frozen paper-reference intervals; the preregistered seed-1 follow-up is queued. Theorem-provider and official-scale B3 comparisons remain pending. |
| MetaMoE-style route invariance | Composition after proving a fixed route | With the same downstream verifier and budget on the frozen 100-sample cohort, staged Route A resolves 56 additional samples and yields 36 route-changing SAFE certificates unavailable to the route-invariance precondition. | Executed follow-up comparison; labeled reimplementation because an executable author artifact is unavailable. |
| SpecSphere and unavailable certification artifacts | Published certification claims | The audited case series records artifact/retrieval availability and machine-checkable assumptions without inventing an executable comparison. | Survey-only; not presented as a runnable baseline. |
| alpha,beta-CROWN | State-of-the-art static-network verification | Its frontend rejects the full dynamic-dispatch ONNX graph at `GatherElements`; Route A specialization converts the same model into four accepted static expert graphs. This project extends the verifier's input domain rather than competing with its expert bounds. | Parser rejection and 4/4 specialization acceptance executed; official-scale expert certificates pending B3. |
| Monolithic MILP/HZ | Standard single-formulation exact or mixed-integer reference | Route-conditioned branches have substantially smaller structural binary width when multiple experts are feasible. A common-task runtime/coverage comparison is still needed. | Verification-scale width census complete; monolithic solver experiment pending after B1. |
| robust-moe-cnn / V-MoE | Empirical robustness or sparse-MoE scale | They establish architecture relevance but do not provide the same route-conditioned formal-verification task. | Provenance/background only; no borrowed checkpoint or superiority claim. |
| Hash Layers / THOR | Static or randomized routing as a legitimate design choice | They calibrate interpretation: the audit concerns the released artifact, theorem assumptions, and training narrative, not a claim that static routing is intrinsically invalid. | Related-work control, not a verification baseline. |

The five evidence-backed contribution axes are: exact affine route-boundary and
applicability measurement; retained-path-condition verification with positive
and negative representation controls; staged normalized top-k semantics across
two gate families; artifact-level certification-gap auditing; and a replayable,
hash-anchored MoE verification artifact. Numerical “outperformance” is reserved
for the B3 common-task table.
