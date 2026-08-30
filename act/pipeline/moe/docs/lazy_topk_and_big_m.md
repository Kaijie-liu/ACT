# Lazy tie-inclusive top-k enumeration and support-derived big-M

Status: implemented correctness stage; E-scaling and timed solver rerun pending.

## Semantic contract

For router scores `r` and selector variables `z_i in {0,1}`, the encoding uses
`sum_i z_i = k`. For every ordered pair `(i,j)` it adds

```
r_j - r_i <= M_ji (1 - z_i + z_j).
```

Thus `z_i=1,z_j=0` enforces the weak inequality `r_i >= r_j`. Weakness is
intentional: at a tie, every legal unordered top-k set must remain enumerable.
Every `M_ji` is at least a sound upper bound on `r_j-r_i`. The implementation
supports three explicit sources:

- `fast`: the unconditioned generator bound retained for compatibility;
- `lp`: constraint-aware HZ support with binary factors relaxed;
- `exact`: constraint-aware HZ support with binary factors integral.

An incomplete support query falls back per side to the sound generator bound.
The metadata records the mode, every upper-side status, and whether the whole
support call was exact. No timeout or mixed fallback is labelled exact.

## Incremental enumeration

The sparse HZ plus `E` selectors is lowered once into one HiGHS model. After a
validated solution selects set `S`, the next query adds

```
sum_{i in S} z_i <= k - 1
```

to a cumulative scratch-row pool. Enumeration is complete only when the
cut-augmented model is proved infeasible. A timeout, numerical status, replay
failure, duplicate selector, or `max_sets` budget returns `complete=false`.
Solver points are independently checked against variable bounds, integrality,
the full base row system, and every no-good row before a route set is exposed.

The same model is reused for every solve. A partial MIP start containing the
previous non-selector HZ variables is submitted after each new cut. HiGHS
accepted all submissions in the E=8 correctness run. The artifact does not
claim that HiGHS used the submitted start internally; that behavior is not
observable through the public telemetry. Likewise, MIP basis reuse is not
claimed when no valid basis is returned.

## Frozen E=8 correctness result

The all-tied E=8, top-2 instance has a closed-form legal set family containing
all `C(8,2)=28` pairs. Exhaustive branch checking and lazy enumeration both
return exactly those 28 sets. The lazy path builds one model, accepts 28
partial MIP starts, adds 28 no-good cuts, and performs 29 solves; the final
solve proves infeasibility. An independent combinatorial audit reports zero
issues:

`act/pipeline/moe/results/lazy_topk_e8_correctness_20260830_r2.json`.

The same frozen run contains a guarded three-score control. Fast bounds ignore
`x>=0.5` and allocate two selector binaries for expert 0. Integral
constraint-aware support proves both competitors globally dominated and
allocates zero binaries (`2 -> 0`). This is a correctness/tightening witness,
not a runtime or prevalence result.

The first local `_r1` output predates explicit partial MIP-start submission and
is superseded. It is excluded from reported results and is not committed.

## Remaining registered work

The correctness stage does not establish scalability. The registered scaling
study remains `E in {4,8,16,32,64}` on frozen model families, with exhaustive
comparison only where tractable. It must report set count, completeness,
model builds, cuts, solves, support cost, solve cost, and total wall time.
The timed exact-support big-M engineering rerun remains deferred until B1 ends
so it does not contaminate training or paired runtime measurements.
