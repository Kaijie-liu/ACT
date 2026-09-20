# All fresh output bounds attempted: no complete positive proof

Frozen execution `fe8ec8bf1df3d1742ba0f2b24b168ef394a6a0b6` ran once, under
the [registered protocol](full_bounds_v1.md). All nine NEW LPs were attempted.
Independent relocated review **PASS, 0 issues**; mathematical endpoint
**UNKNOWN_MISSING_BOUND_EVIDENCE**, not SAFE. Seven available lower bounds
are independently checked and nonpositive; two calls provide no candidate.

Read the [full review](full_bounds_v1_review.json) and the separately produced,
[saved-record diagnostic](full_bounds_v1_analysis.json). Raw artifacts remain
at `data/moe/results/full_bounds_conv98_20260920_v1`. This follow-up does not
replace the old conditional positive proof, whose matrices differ, or any
formal experiment. There is no change to production acceptance or main counts.

## Complete denominator and fresh evidence

Same input98, label0, sole legal pair[1,2], represented2/255 box and complete
source chain as `full_source_v1_results.md`. All competitor properties1..9
remain necessary. Each new LP has26,085 variables; no old dual, basis or
certificate is imported. SciPy's bundled HiGHS1.8.0 proposes multipliers;
the independent rational checker establishes their bounds without trusting
the native objective or optimality status.

| Competitor | Native seconds | Native outcome | Independently checked lower bound |
|---|---:|---|---:|
| 1 | 16.0546 | time limit | missing |
| 2 | 1.8344 | reported optimal | -66.222524 |
| 3 | 1.5521 | reported optimal | -59.349889 |
| 4 | 4.7814 | reported optimal | -65.159780 |
| 5 | 8.0730 | reported optimal | -64.665522 |
| 6 | 3.3341 | reported optimal | -76.337621 |
| 7 | 1.8224 | reported optimal | -74.752783 |
| 8 | 16.0453 | time limit | missing |
| 9 | 4.3650 | reported optimal | -70.789609 |

The native16s limit is a solver option, not an exact wall-time bound: small
call-return overheads are charged to the common request budget. All nine call
records are durably published; the two time-limit records explicitly contain
no candidate. Thus **nine published records is not nine checked bounds**.
No call was retried, extended, replaced or excluded from the denominator.

The checker first revalidates all7 source-prefix steps,16 remaining expert
steps, final factor map,5 exact route exclusions and9 LP constructions. It
then checks all7 available multiplier vectors against their exact NEW LP
identities. The missing two obligations are not waived. None of the seven
nonpositive bounds is a counterexample or a proof of an intrinsic LP gap.

## What the saved records diagnose, and what they do not

For all seven checked candidates, the floating reported objective minus the
exact corrected bound lies between approximately9.91e-15 and4.69e-14. Their
negative magnitudes are59.35–76.34. Therefore the observed negative outcomes
are **not positive native objectives destroyed by the independent residual
correction**. This comparison does not prove exact primal feasibility or
native optimality, and does not rule out every better dual.

The exact residual-box contributions are approximately -16.22 to -19.39.
These include legitimate finite-variable-bound terms. They are not all
floating roundoff; subtracting that term alone would be an invalid diagnosis
or an invalid change to the lower-bound acceptance formula.

A separate read-only analysis evaluates the registered expressions at the
seven stored, **unverified** approximate LP vectors. Arithmetic is exact on
the stored binary64 numbers; feasibility and network correspondence are not
checked by this analysis. Gate ranges stay[0,1]; difference interval widths
are197.45–277.64 for these properties.

| Competitor | Difference interval | `lambda*d - w` at saved vector | `u + lambda*d` at saved vector |
|---|---|---:|---:|
| 2 | [-100.094, 130.240] | 53.998 | -12.225 |
| 3 | [-82.933, 114.519] | 45.403 | -13.947 |
| 4 | [-100.884, 127.455] | 53.859 | -11.301 |
| 5 | [-94.954, 125.755] | 51.374 | -13.291 |
| 6 | [-113.994, 140.597] | 58.397 | -17.940 |
| 7 | [-109.325, 168.314] | 63.311 | -11.442 |
| 9 | [-118.323, 126.829] | 58.170 | -12.620 |

Here `u+w` is the LP objective, with `w` relaxing `lambda*d`. The recorded
product discrepancies make the weighted outer envelope a concrete inspection
target. However, merely substituting the product at these vectors leaves
negative values: **a gate/McCormick-only root-cause claim is not established**.
Such substitution also does not enforce actual softmax dependence, binary
ReLU choices, or exact LP feasibility. These are not real-input violations,
checked feasible upper witnesses, or a controlled alternative verification
experiment. No new solve, forward pass, model or sample is used in this analysis.

Reproduce only the saved-record arithmetic:

```sh
python -m full_bounds.analysis data/moe/results/full_bounds_conv98_20260920_v1
```

## Full costs, portability and controls

| Measurement | Seconds |
|---|---:|
| Prepare child, including copy/hash validation | 0.2011 |
| Proposal child, including startup, conversion and all calls | 60.1124 |
| Seal child, including publication/hashing | 0.1609 |
| Full source + bound checker child | 52.2809 |
| Total through terminal publication | **112.8448** |
| Source checks inside first checker | 40.0865 |
| Exact bound checks inside first checker | 11.9887 |
| Separate relocated checker | 52.0157 |
| Separate full review | 52.3217 |
| Post-result saved-record analysis | 0.3770 |

Nested times are not added twice. The original source generation is excluded;
this is a stored-source proof attempt, **not production end-to-end timing or
a speedup**. The fresh package is153,312,482 bytes. Proposal peak RSS is
418,984KiB; fresh relocated check peak RSS is530,620KiB. The two missing
candidates arise from native sublimits, not an outer300s cutoff. The remaining
outer time is not retrospectively donated to them.

The complete package moves independently of checkpoint, training data, old
run directories or numerical solvers. Run on a copied `relocated/` directory:

```sh
python -I -S /new/location/verify_bounds.py --manifest-hash 3db4683baf764a3b813bd01897ea8a6896074e3bdedc4ba67ceaf7e86ae0e4a0
```

This reproduces the UNKNOWN and all exact bounds; portability does not imply
positivity. Manifest and checker identities must be pinned externally. Terminal
identity is `0dbd52d8d063342793712187f524de1893c09df10f92688e224ca97586a27f26`.
All44 pre-execution controls/regressions pass; two additional post-result
arithmetic controls pass. The moved full-source toy does produce a complete
positive bound, and missing/duplicate/wrong-source/property/LP/sign mutations
reject or retain explicit UNKNOWN. The toy is not empirical coverage evidence.

## Updated research boundary and next decision

The complete **declared source → exact enclosure → all output LPs → checked
candidate bounds** path now executes. What it does not deliver here is a
positive complete-request proof. It is no longer accurate to say the only
remaining work is generating any duals, or that all unknowns are checking or
serialization delays. Two candidates are missing and seven independently
checked bounds are negative; those are distinct failure modes.

No claimed high-accuracy, cross-family route-changing strict certificate is
added. This is still a67.06% model and a single-route request. Native floating
execution and captured-graph/program correspondence remain distinct boundaries.
Historical certificates cannot be transplanted onto these new matrices.

Do not restart modular elimination, expand this cohort, repeat these calls or
change the native cap to obtain a prettier outcome. A subsequent improvement
should first use finite controls to study **checkable ranges on the NEW source
and the weighted/activation outer envelopes**, preserving shared factors and
all obligations. This is a candidate research direction, not a uniquely
established cause; any new range or representation needs its own checked
evidence and separately frozen comparison. Keep the present negative result
visible beside the established main-method evidence.
