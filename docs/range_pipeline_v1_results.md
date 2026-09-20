# Complete scoped-range pipeline: actual two-arm result

Executed the [frozen protocol](range_pipeline_v1.md) once at
`8bec4ce7d0232a6697d9ac91503a72faf617aafd`, in
`data/moe/results/range_pipeline_conv98_20260920_v1`.
Both300s arms completed, then passed separate relocated mathematical rechecks
and terminal/cost/identity audit with zero issues. See the
[review](range_pipeline_v1_review.json) and
[saved-only analysis](range_pipeline_v1_analysis.json).

**The complete proof remains unclosed in both arms.** Four checked hidden
ranges and one fewer ReLU binary are real progress in source integration, not
a new complete safety certificate. No sample/row/limit retry followed these
outcomes. The historical full_bounds run and all main tables remain unchanged.

## Complete request outcomes

| Same input98 | Range off | Range on |
|---|---:|---:|
| Required classification properties | 9 | 9 |
| Checked positive | 0 | 0 |
| Checked nonpositive | 7 | 9 |
| Missing candidates | 2 | 0 |
| Complete positive requests | 0 | 0 |
| Range native calls / admitted two-sided facts | 0 / 0 | 8 / 4 |
| Output native calls | 9 | 9 |
| Complete stored-source execution | 141.5569s | 153.7974s |
| Portable proof bytes | 153,350,854 | 155,451,140 |
| Separate moved-check and audit | 54.4758s | 68.3801s |

Off returns `UNKNOWN_MISSING_BOUND_EVIDENCE`; on returns
`UNKNOWN_NONPOSITIVE_BOUNDS`. All necessary source and output-construction
checks run: seven prefix transitions,16remaining expert transitions, the
shared/private join, five exact route exclusions and all nine output LPs.
No checkpoint load, forward or old LP-certificate reuse occurs in either arm.
The range-on branch is thus a real **complete evidence** result, not a
complete positive result. Nonpositive lower bounds still do not prove an
intrinsic LP gap or model unsafety; there is no checked primal upper witness.

## What the four ranges changed

All numbers below are rounded displays of exact rational bounds in the
archive, on the newly built source before hidden Linear layer6.

| Expert / row | Generator box | Checked range | Activation consequence |
|---|---|---|---|
| 1 / 0 | [-10.774208, -4.593483] | [-8.039313, -6.983890] | Already inactive |
| 1 / 1 | [-4.563975, 0.001447] | [-2.578263, -1.834587] | Unstable becomes inactive |
| 2 / 0 | [-8.101112, -2.871629] | [-6.094308, -4.902504] | Already inactive |
| 2 / 1 | [-3.924248, -2.149274] | [-3.159534, -2.845605] | Already inactive |

Each hidden layer contains64rows; the other62per expert keep explicit
generator-box fallbacks. Selection remains the predeclared prefix, not a
post-result selection of promising neurons. Source dimensions change from
24,464 continuous /1,619 binary /11,581 equalities /3,242 inequalities to
24,462 /1,618 /11,580 /3,240. Shared input factors remain3,072 and both
experts still reach all ten outputs.

The range-off joint exactly reproduces the earlier full-source identity:
`93b15af1a0ff299aa60a6be0525513947ad0640f51749471e596a8709db0e3f3`.
The independently checked range-on joint is:
`cdc671b5ba8bf1bb8b10b288169247bb86ed0c9cc99b9221a66577682c16496a`.
These are not interchangeable sources; all output evidence is newly generated.

## Every output lower bound

| Competitor | Off checked lower bound | On checked lower bound |
|---|---:|---:|
| 1 | Missing | -73.10270276 |
| 2 | -66.22252373 | -66.22245016 |
| 3 | -59.34988859 | -59.34977209 |
| 4 | -65.15978019 | -65.15967917 |
| 5 | -64.66552172 | -64.66546093 |
| 6 | -76.33762139 | -76.33747910 |
| 7 | -74.75278324 | -74.75273589 |
| 8 | Missing | -67.34927758 |
| 9 | -70.78960851 | -70.78956668 |

Seven common recorded bounds improve by4.18e-5 to1.42e-4; none approaches
zero. Competitors1and8 now have independently checked candidates. This is
observed finite-budget candidate availability, not proof that every repeated
run would have the same availability or that the exact LP optimum improved by
the same amount. It also does not establish that other hidden ranges would
be ineffective. The current fixed four-row intervention has failed to produce
the desired complete positive endpoint.

## Complete and nested cost accounting

| Phase, includes subprocess cost | Off | On |
|---|---:|---:|
| Rebuild source, optional ranges, all new LPs | 25.8876s | 59.2812s |
| Output proposals | 60.4488s | 26.8321s |
| Seal | 0.1809s | 0.1810s |
| Complete independent check | 54.9368s | 67.4024s |
| Total through terminal publication | 141.5569s | 153.7974s |

Source-check portions are42.252s /52.086s; output dual-check portions are
12.469s /15.108s. These are nested within the check phase, not extra charges.
The four on-arm range procedures total32.9783s before their final receipt
publication: assembly4.0505s, conversion2.7365s, native calls1.3695s,
exact prechecks23.7719s, with the remainder in identity/serialization/import
and other procedure overhead. Layer receipts separately retain propagation
and delta-serialization times. The final checker then independently repeats
necessary range checks; the precheck is not used as a trusted shortcut.

Output proposal time decreases in this run, but the new range-generation and
checking costs outweigh that decrease: total rises12.2405s (about8.65%).
There is one fixed-order trial per arm, so this is **not** an unbiased speedup
or slowdown estimate. Frozen capture/model loading and the separate
post-terminal review remain excluded; all subsequent source propagation is
charged. These costs cannot be compared directly with old stored-final-HZ
runs that excluded prefix/remaining propagation.

## Evidence identities and reproduction

Off package manifest:
`1599a15e0d0059e4cd680ed23e0b6548ca79f028d7b399061cff806ead3a8689`.
On package manifest:
`794fa32d7c4d19ef20b56e2149b662810fea706b773da827ab73af4338cda54d`.
Terminal identities, exact rational rows, source identities, per-call limits,
environments, row rosters, costs and archive sizes are in the review. Raw
sources/checkpoints/matrices stay local and are not committed.

To rebuild the compact analysis using only committed review data:

```sh
python scripts/analyze_range_pipeline.py docs/range_pipeline_v1_review.json
python scripts/test_analyze_range_pipeline.py
```

For a moved raw proof, run its `verify_bounds.py` with `python -I -S` and the
corresponding externally pinned manifest hash. The checker uses neither the
checkpoint nor historical data/solver directories. The preparation suite
passed56 tests plus3 main-table tests; three saved-analysis tests additionally
check exact reproduction, missing/duplicate/count errors and widened ranges.
The early fixture/import test mistakes are preserved in the control receipt.

## Research disposition

This closes the **complete integration and finite effect check** requested
after the scoped-range interface: fresh source propagation, supplied-range
proofs, all new weighted output obligations and exact checking now run under
one budget. It does not close full positivity. Three selected rows were
already inactive; the only new inactive row yields small output-bound changes.
This gives a concrete local explanation of this intervention's limited effect,
not a unique root cause for the whole convolutional failure.

Do not automatically extend the row prefix, add time, repeat native calls or
turn checking overhead into another arithmetic/cache project. Any subsequent
range/envelope intervention should first identify, from saved source and
obligation records, which still-unstable relation could materially affect a
blocking complete property, then specify a separate finite control. That
selection would be development, not untouched evidence. Retain unresolved
causes where exact optimality/feasibility has not been established.

Input98 remains single-route at67.06% clean accuracy. High-accuracy,
cross-family route-changing strict certificates and native floating-program
equivalence remain open. No production result or paper main table is upgraded.
