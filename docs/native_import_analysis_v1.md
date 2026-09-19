# Native import warning: saved matrix inventory and analytic reproduction

User authorized only the first two follow-ups: read-only saved-matrix analysis
and minimal analytic import controls. Starting HEAD was
`2164708986b4a5b74495243bff4e28131a48fafb`, clean/pushed feature branch.
No production/frozen module, native numeric option, time budget or acceptance
rule was changed. No real LP was submitted again. **Zero optimization calls**.

## Evidence

[Attempt001](native_import_analysis_attempt001.json) preserves seven analytic
imports and five passing unit controls. [Attempt002](native_import_analysis_attempt002.json)
adds the exact observed negative coefficient and a second semantic witness:
eight analytic imports and five unit controls pass. Across both attempts there
were15 small analytic imports, zero real-model imports and zero optimizations.
`Highs.run` is forbidden in the probe; dimensions are capped at4 variables,
4 rows and16 entries, so it cannot submit the actual7,397-variable LP.

[Fresh review](native_import_analysis_v1_review.json) reconstructs the real
inventory,16 saved analytic files,8 cases and2 exact-arithmetic semantic witnesses.
PASS,0 issues. It performs no native import or optimization. Old23-artifact
execution archive and all frozen interface sources remain hash-identical.

## 1. Read-only inventory of the actual saved submission

Only `input220_p0/native/input.json` is analyzed as a real submitted model.
The other three jobs were never started and are not silently imported here.
The saved intended binary64 model is reconstructed from its original rational
LP and matches exactly under the frozen conversion rule.

| Item | Observed value |
| --- | ---: |
|Variables|7,397|
|Equalities / inequalities|1,441 /2,890|
|Stored submitted matrix entries|246,558|
|Minimum nonzero absolute matrix coefficient|3.6294188569593725e−10|
|Maximum absolute matrix coefficient|15.774675658747613|
|Nonzero entries with magnitude≤1e−9|**16**|
|Invalid CSR / duplicate indices / inverted boxes|0 /0 /0|
|Matrix entries reaching default large-matrix threshold|0|
|Costs or finite bounds reaching their default infinity thresholds|0|

All16 small entries are in **A row2 (zero-based)**, at columns
`0–3,32–35,64–67,96–99`. Their exact original coefficient is
`−107121537265/295147905179352825856`; the submitted value is
`−3.6294188569593725e−10`.

Runtime option readback, without changing them, gives highspy1.14.0:
`small_matrix_value=1e−9`, `large_matrix_value=1e15`,
`infinite_cost=infinite_bound=1e20`. Runtime binary hash is recorded.

Separately,12,084 matrix entries undergo rational→binary64 rounding in the
already registered conversion. This is **not** the16-entry deletion count or a
newly introduced conversion bug: the intended submitted snapshot reproduces.
The exact original LP remains authoritative for feasibility checking.

## 2. Minimal analytic import/readback, no solve

Use two variables, one row and the frozen native options (presolveOFF,
simplex scaling0, threads1, etc.). Only diagnostic logging is enabled, to new
artifact files; no coefficient threshold is changed. Import via `passModel`,
then inspect `getLp` without optimizing.

| Tiny second coefficient | Status | Matrix readback |
| --- | --- | --- |
|1e−8|kOk|unchanged|
|nextafter(1e−9,0)|kWarning|second entry removed|
|1e−9|kWarning|second entry removed|
|nextafter(1e−9,+∞)|kOk|unchanged|
|−5e−10|kWarning|second entry removed|
|+3.6294188569593725e−10, inequality|kWarning|second entry removed|
|+3.6294188569593725e−10, equality|kWarning|second entry removed|
|−3.6294188569593725e−10, inequality|kWarning|second entry removed|

The logs explicitly say small absolute matrix values at or below1e−9 are
**ignored**. In these controls, row counts, row bounds, variable bounds, objective
coefficients, offset and sense remain unchanged. Presolve being disabled does
not prevent this import-time filtering. The strict frozen adapter therefore
correctly refuses a non-identical imported matrix; accepting the warning alone
would still violate its subsequent full-readback equality contract.

Let `a=107121537265/295147905179352825856>0`, variables in[0,1]. Exact arithmetic
checks two examples:

- `x+a y≤0`: point(0,1) is infeasible originally but feasible after deleting `a`.
- `x−a y≤0`: point(a/2,1) is feasible originally but infeasible after deleting `−a`.

Thus filtering is neither generally equivalent nor guaranteed to be a safe outer
relaxation. These are LP semantic counterexamples, **not** network witnesses or
claims about the feasibility of the saved real LP.

## What is established, and what is not

There is a concrete, reproducible **import-contract mismatch**: the saved real
submission includes16 entries in a range that this runtime demonstrably filters
in analytic models with the same numerical options. This explains a specific
warning mechanism compatible with the frozen failure; it is not a time-budget
or exact-elimination result.

The original run stopped before real-model readback and suppressed detailed
native messages. Since this stage deliberately does not resubmit that full
model, it does **not** establish an exhaustive list of all warnings or all changes
that would occur during its import. It also does not measure real basis quality,
rank, fill-in, reconstruction or LP feasibility. Those remain unresolved.

## Does this need new guidance?

No new scientific direction or larger experiment is needed. The next technical
question is now well specified: **can a separately versioned native interface
preserve the intended submitted matrix while still proposing a useful basis?**

Recommended next decision (not implemented here): first develop a fidelity-
preserving import V2, with an explicit, supported small-entry policy and complete
coefficient/constraint/bound readback checks. Validate its controls and resource
accounting before any new real diagnostic freeze. If a threshold choice cannot
preserve all intended values, study a documented equivalent scaling separately.
Do not silently set a new threshold in V1, accept `kWarning`, or resume its run.

An alternative—allowing a filtered native model purely as an untrusted basis
hint while checking the unchanged rational LP—requires a **different identity
and mapping contract**. Native bounds/feasibility would not transfer. This is
not the current frozen interface and should not be adopted implicitly.

Accordingly, the next stage needs authorization of the interface/configuration
decision and a new protocol, not new model training, more samples, more time or
a relaxed exact-feasibility threshold. This turn completes only the two requested
diagnostic steps and preserves the sealed1 ERROR+3 NOT_RUN_AFTER_ERROR outcome.
