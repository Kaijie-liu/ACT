# Selected-score nonzero precheck avoids unnecessary support optimization

Implementation `2a12d5156`; freeze `0f6f4beee`; config SHA256
`7acce02cd428a5842e38668c1853b96dd4ed6e30d2e0970dce999f5e63351455`.
[Frozen protocol](metamoe_nonzero_precheck_protocol_20260923_r1.md),
[execution config](../configs/recent_moe/metamoe_nonzero_precheck_control_r1.json),
[independent audit and inventory](metamoe_nonzero_precheck_archive_20260923_r1.json).
Both requests completed, audit PASS with zero issues. No new cohort opened.
Raw records remain in
`/data1/Kane/MOE/baseline_runs/metamoe_nonzero_precheck_control_20260923_r1`.

## Result: same obligation, same bounds, zero support optimizer calls

Same old MNIST0 checkpoint/input/`2/255`, same repaired BN, same 300-second
request cap, 30-second support allocation, 19 global output properties and
numerical policy. Checked routing and expert base are ON in both arms. Only
the definedness-specific support precheck changes. No relaxation, order,
property coverage, tolerance or total budget was adjusted.

| Measurement | Original support | Nonzero precheck |
|---|---:|---:|
| Full terminal | POSITIVE / HZ_POLICY_ACCEPTED | POSITIVE / HZ_POLICY_ACCEPTED |
| Charged request, seconds | 39.407129 | 8.763545 |
| Definedness query including snapshot/check, before terminal publication | 30.053890 s | 0.012939 s |
| Actual native support optimization calls | 1 | 0 |
| Returned score range | [3.345338179462453, 4.30685851511294] | same |
| Range status | fast_fallback | fast_nonzero_checked |
| Candidate analysis, seconds | 0.108097 | 0.109150 |
| Expert verification, seconds | 1.698051 | 1.709832 |
| Output violation queries excluded | 19/19 | 19/19 |
| Peak sampled process-group RSS, GiB | 2.262508 | 1.823761 |

Charged cost decreased by **30.643584 seconds (77.76%)** on this old request.
The original optimizer used its allocation and returned the existing fallback
range. The enabled arm obtained exactly that same range before optimization,
checked its sign, and discharged this definedness obligation without a native
support solve. This is not an improved margin or stronger relaxation.

This control is native-first, one old input, one execution per arm. It does
not estimate population speedup, add a certificate, or establish an advantage
over the author verifier. Do not combine earlier 132-second timings into this
fresh pair, or compare its timing to an independently run historical author arm.

## Why the shortcut is admissible

For the CURRENT stored guarded router HZ,

`score_i = c_i + Gc_i xi_c + Gb_i xi_b`,

each continuous factor lies in `[-1,1]`, and each original binary factor in
`{-1,1}`. Therefore `c_i +/- (sum |Gc_i| + sum |Gb_i|)` encloses the coordinate
on the unconstrained factor box, hence also on its guard-constrained subset.
Only exclusion of zero is needed for selected raw top-1's `score/score`
definedness. Optimizing that already-sign-definite range is unnecessary for
this obligation. A feasible point would not justify this argument and is not
used by the precheck.

The float enclosure is the unchanged support routine's zero-budget fallback.
In addition, a binary-rational check over the stored generator coefficients
can only veto the shortcut. The independently recomputed sign interval is:

- lower: `505533212561844664846095 / 151115727451828646838272`;
- upper: `650834057543394445441265 / 151115727451828646838272`.

Both are positive; all **4,332 stored generator entries** in the selected row
participate. This exact sign is about the stored affine-generator object, not
exact optimal support or an independently proved enclosure of the original
floating-point network. The reported numerical range remains the old range.

## Identity, coverage and rejection behavior

Both arms have identical guarded routing matrices, expert matrices and full
nonzero-source HZ snapshots. The latter hash is
`4787c1f18dffd7919c8c6c91e92367e7e08a96e44944bac138b253db5eb69ac9`.
The audit independently maps that snapshot's original +/-1 factors, equality
rows and guard inequalities to the separately saved routing MILP object; it
does not call the production HZ lowering or sign-check function to do so.
Request/input/row/nonces, remaining budget and evidence inventory are checked.

Candidate `[1]`, excluded `[0]`, unresolved `[]` are identical. All 19 expert
properties remain explicit native queries, not reused as unproved facts.
This is still **route-stable** and `source_complete=false`. Network-to-HZ,
guard lowering and native infeasibility remain in the trusted basis. The
earlier BN repair establishes finite point conformance only. No source-complete
MoE certificate or new route-changing result is claimed.

For crossing/touching/nonfinite ranges, unsupported representations, failed
sign checks or exhausted precheck time, the enabled path does not assert
nonzero. Native support receives remaining original allocation when available;
otherwise the result is unresolved. A late published shortcut is rejected.
Nonfinite unresolved observations are serialized as null and never accepted.
Default support behavior and all other support callsites are unchanged.

## Costs, tests and reproducibility boundary

Charged requests total **48.170674 s**. Batch wall through final summary,
including postflight inventory, is **48.765437 s**; final cost-file publication
and independent audit are explicitly separate. The saved audit took
**0.632169 s**. Inventory takes 0.016843/0.016769 s. All 301 raw files are kept,
total 43,485,871 bytes; no missing request or open trace span occurs. Raw arrays,
checkpoints and external source trees are not committed.

121 focused ACT controls and 86 overlapping pinned-environment controls pass.
They include both signs, zero/touching/crossing, guard-only positivity needing
optimization, binary factors, duplicate coefficients, exact-sign disagreement,
wrong identities, dimension changes, every tie-legal route, untouched default
support, partial records, deadlines, remaining-budget fallback and full outer
cost/stop behavior. An initial invocation without explicit CPU initialization
hit seven existing CPU-entry guards; the corrected test harness initializes
CPU/float64 as production does, with no gate or threshold change.

The new protocol explicitly rebinds only the small class-separated entry
change; historical manifests remain sealed. They must be replayed at their
historical source versions, not silently rebound to this one.

## Decision

The precheck is effective on the authorized fixed case. Keep it **opt-in**;
there is no reason to seek tighter score bounds or adjust this already-closed
expert's relaxation. Do not extend this case with more time or sample tuning.
For wider use, separately freeze a same-object comparison (including the
current author path) and preserve UNKNOWN/timeout cases. Source-enclosure
closure remains an independent research task. No follow-up is automatically
queued by this control.
