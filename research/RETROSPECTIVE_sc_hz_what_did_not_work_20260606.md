# SC-HZ Post-Experiment Retrospective: What Worked, What Did Not, and Why 2000+ Was Not Reached

**Date**: 2026-06-06 morning  
**Companion to**: `research/INNOVATION_BRIEF_sc_hz_20260604.md`  
**Audience**: advisor / collaborator handoff  
**Status**: revised after Phase F1/F2b/F3/G audits. Use this file as the
honest post-practice account of the SC-HZ research arc.

This retrospective explains what changed between the original SC-HZ
innovation brief and the experiments that followed. It is intentionally
conservative: it separates real verified gains from failed hypotheses,
withdrawn claims, parser work, and future research ideas.

---

## 0. Executive Summary

The original goal was to push ACT/HyZor from the canonical **924 V/A**
toward **1500-1900**, and eventually toward **2000+**, while preserving the
project's binding principles:

1. No CROWN-style backward bound propagation.
2. No gradients or autograd-derived information.
3. Continuous LP only; no Gurobi/MILP/integer solving.
4. No fallback verifier.
5. No branch-and-bound or input-box splitting.
6. No PGD/random/corner-sample-then-check falsification. FAL must come from
   structured forward-HZ/LP evidence and strict ORT replay.

The experiments produced a real improvement:

```text
Canonical ACT/HZ GPU STRICT baseline:        924 V/A
SC-HZ forward-coeff sidecar on safenlp:     +536 NEW A
S1 structured PHANTOM repair on safenlp:     +12 NEW A
------------------------------------------------------
Frozen current headline:                    1472 V/A
```

This is a substantial result, but it is not the broad breakthrough the
original brief forecast. The **entire +548 NEW V/A comes from
`safenlp_2024`**. Dense-conv benchmarks such as `cifar100_2024`,
`tinyimagenet_2024`, `yolo_2023`, `traffic_signs`, and `vggnet16_2022`
still receive **0 NEW V/A** from SC-HZ. Small dense / case-heavy benchmarks
such as `acasxu_2023`, `linearizenn_2024`, and `relusplitter` also do not
move under the tested SC-HZ sidecar mechanisms.

The corrected conclusion is:

> **1472 V/A is the audited ceiling of the current pipeline**
> `(SC-HZ directional witness sidecar + DeepZ triangle + continuous-LP
> sidecar cuts)`, not the theoretical ceiling of all forward-only HZ-like
> verifiers.

To go materially beyond 1472, especially toward 2000+, we need a genuinely
new abstraction or a large parser/model-coverage effort. More local
triangle cuts on the current pipeline are not enough.

---

## 1. What the Original Brief Expected

The original file `research/INNOVATION_BRIEF_sc_hz_20260604.md` proposed a
mechanism called **Spec-Conditioned HZ (SC-HZ)**, centered on PRUNE.

The proposed pain points were correct:

| Pain point | Meaning |
|---|---|
| PP1 | HZ degenerates toward Star-like forward propagation in conv bodies; binary structure often does not help during affine/conv layers. |
| PP2 | Deep ReLU triangle slack accumulates across many layers. |
| PP3 | Generator reduction is query-blind and can discard generators relevant to the target rival margin. |
| PP4 | Verification only needs a 1-D rival projection, but the full output HZ is propagated. |

The original PRUNE idea was:

1. Compute a per-rival, per-layer linear direction `d_L^r`.
2. Keep generators whose `|d_L^r * Gc_L[:, j]|` contribution is largest.
3. Absorb all other generators into a sound interval tail.
4. Verify the lower-dimensional projected HZ by LP.

The staged expectation was:

| Stage | Original expectation |
|---|---|
| Phase A | Sentinel signal. |
| Phase B | 924 -> 1100/1300. |
| Phase C | 1500/1700 after binary/GPU integration. |
| Phase D | 1700/1900 after engineering. |

This expectation was only partially realized. The final lift to 1472 is
real, but it came from a different mechanism discovered while testing PRUNE,
not from PRUNE itself.

---

## 2. What Actually Worked

### 2.1 SC-HZ Forward-Coefficient Sidecar on `safenlp_2024`

The successful mechanism was:

1. Run forward HZ to obtain a rival-margin affine form.
2. Compute the closed-form LP maximizer over the input box for that
   one rival direction.
3. Decode the resulting root-box corner as a concrete input.
4. Strictly replay the candidate in ONNX Runtime at zero tolerance.
5. Count only witnesses that violate the original spec under strict replay.

This produced:

| Component | NEW A |
|---|---:|
| Forward-coefficient closed-form sidecar | 536 |
| S1 structured PHANTOM repair | 12 |
| **Total SC-HZ lift** | **548** |

All counted witnesses are audited in:

- `audit_results/sc_hz_final_1472_aggregate.json`
- `research/sc_hz_FREEZE_1472_20260605.md`
- `research/sc_hz_hard_gates_for_v_a_results.md`

Why this worked on `safenlp_2024`:

- Input dimension is moderate.
- Specs are wide enough that many true adversarial witnesses exist.
- The forward rival projection is aligned with actual violating directions.
- FAL only needs one valid witness; it does not need to prove all inputs safe.
- The strict ORT replay gate prevents phantom LP witnesses from being counted.

### 2.2 S1 Structured PHANTOM Repair

Some `safenlp` iids had LP-SAT directions whose first decoded candidate did
not replay. The S1 repair tried deterministic, structured candidates:

- base sign direction
- reverse sign direction
- top-K single flips
- pair flips
- zero-top-3 plus flip-next-3

This added **+12 NEW A**. It was useful, but it saturated quickly. Triple and
quad variants did not create further meaningful gains, so S1 is closed.

### 2.3 Engineering and Audit Infrastructure

Even where there was no new V/A, several engineering pieces are valuable:

- strict zero-tol ORT replay discipline
- provenance bundles for every counted witness
- withdrawal discipline for the bad relusplitter claim
- dense-conv memory pilots
- constrained-LP prototype and tests
- pairwise hull soundness tests
- FC-HZ toy diagnostics

These are important because they prevent the project from confusing a faster
or tighter relaxation with an actual verified result.

---

## 3. What Did Not Work

### 3.1 PRUNE Was Empirically Falsified

**Hypothesis**: per-rival generator selection by `|d_L^r * Gc[:, j]|` would
retain the proof-relevant directions and safely absorb the rest into an
interval tail.

**Result**: On 40/40 dense sentinels, LP upper bounds got worse as K shrank.
PRUNE was sound, but precision dropped.

**Reason**: after several affine/conv/ReLU layers, generator directions are
homogenized. The ranking signal is weak; discarded generator mass is not
recoverable by the interval tail. In practice, K-pruning behaves like lossy
dimension reduction, not proof-relevant compression.

**Status**: closed negative. Do not revive PRUNE as a scoring mechanism
without a new ranking theorem and an independent monotonicity gate.

### 3.2 Dense-Conv Memory Was Improved, But Verification Did Not Improve

Phase E made dense-conv runs feasible:

- streaming/limited generator budgets
- liveness-style memory control
- peak memory reduced from roughly 60-80 GB to roughly 26-30 GB on sentinels
- OOMs reduced in the pilot

But this produced **0 NEW V/A** on dense-conv sentinels.

This distinction matters:

> The dense-conv problem is no longer only "we cannot run it".
> It is now "we can run it, but the forward relaxation is still too loose".

### 3.3 F1: Per-Neuron Triangle-Constrained LP

**Hypothesis**: explicitly adding last-ReLU triangle constraints as
continuous LP rows would tighten output bounds enough to flip dense-conv
PHANTOM cases.

**Evidence**:

| Metric | Result |
|---|---:|
| Synthetic median tightening | 15.7% |
| Real CIFAR median tightening | about 17% |
| Best observed real drop | about 44% |
| NEW V/A | 0 |

F1 is sound and useful diagnostically, but the remaining gap is too large.
For example, one CIFAR case dropped from `+0.261` excess to `+0.146`, still
above the strict CERT threshold.

**Status**: keep as diagnostic infrastructure; do not expect it to generate
large headline lift on dense-conv.

### 3.4 F2b: Pairwise Multi-Neuron Joint Hull

**Hypothesis**: the residual looseness after F1 comes from pairwise
correlation between unstable ReLUs. Adding pairwise joint-hull cuts should
bind.

**Evidence**:

| Setting | Result |
|---|---|
| 2-neuron toy | exact; F2b beats F1 |
| CIFAR iid 113, top_k=4/10/20/25 | 0% additional gain over F1 |
| All 300 pair cuts for 25 unstable neurons | still 0% gain |

**Reason**: real dense-conv LP optima spread the objective across many
unstable neurons. No single pair is tight enough to bind. Pairwise cuts work
when one pair controls the objective; they do not work when slack is diffuse
across dozens of units.

**Status**: closed negative for dense-conv current pipeline.

### 3.5 F3: Small-Dense / Control Scout

**Hypothesis**: if F1 is not enough for dense-conv, it might still help small
dense/control networks where the total unstable count is smaller.

**Evidence**:

| Benchmark | Result |
|---|---|
| `acasxu_2023` | 186 walker OK, 65 near-CERT checked, 0 NEW V |
| `tllverifybench_2023` | 32 walker OK, 0 NEW V |
| `linearizenn_2024` | blocked by Slice parser |

Some ACASXU cases tightened by 80-90%, but still ended at the strict boundary
or slightly above it. Under the strict G4 replay/proof gate, boundary equality
does not count.

**Status**: F3 did not justify a 5-day parser sprint by itself. Parser work
can still be done as engineering coverage, but it should not be sold as a
likely 2000+ path.

### 3.6 Phase G: FC-HZ Multi-Layer Triangle Constraints

**Hypothesis**: carrying all per-layer ReLU triangle constraints into the
output LP would recover the slack lost by applying only the final ReLU
constraints.

**Evidence**:

| Test | Result |
|---|---:|
| 20 random two-layer instances | FC-HZ tighter than F1 on 19/20 |
| Median additional drop over F1 | 8.1% |
| Mean additional drop over F1 | 9.2% |
| Advisor gate | required at least 40%; failed |

**Reason**: later-layer triangle relaxations smooth earlier-layer slack.
Adding earlier-layer constraints restricts which hidden configurations can
reach a later endpoint, but often does not move the endpoint enough.

**Status**: FC-HZ is a real, sound tightening idea, but too weak in the tested
form to justify implementation as the main path to 2000+.

---

## 4. Withdrawn or Non-Scoring Claims

These must not be cited as improvements:

| Claim | Status | Reason |
|---|---|---|
| 1346 V/A | withdrawn | included bad relusplitter result |
| +64 NEW V on relusplitter | withdrawn | prune incoming-tail bug produced false CERTs |
| sat_relu 41 SC-HZ A | non-scoring | all already production FAL |
| malbeware 49 SC-HZ CERT | non-scoring | all matched production CERT |
| parser ERROR -> UNKNOWN | engineering only | improves audit cleanliness, not V/A |
| dense-conv memory pass | engineering only | enables running, but 0 NEW V/A |

The project should continue using the **1472 V/A** freeze unless a future
experiment passes the same hard gates and updates the aggregate bundle.

---

## 5. Why 1472 Is Still Far From 2000+

The current gap is not evenly distributed. It is concentrated in categories
where the leading tools use mechanisms that this project intentionally does
not use.

### 5.1 Dense-Conv Robust CERT

Examples:

- `cifar100_2024`
- `tinyimagenet_2024`
- `vggnet16_2022`
- `yolo_2023`
- `traffic_signs`

Other tools close many of these with backward bound optimization, BaB, or
MILP-like exact reasoning. Our forward-HZ pipeline sees many PHANTOM margins:
the LP relaxation says a rival may win, but decoded candidates either do not
replay or the proof margin remains positive.

The attempted local fixes were not enough:

| Mechanism | Real effect |
|---|---|
| F1 per-neuron triangle LP | about 17% median drop |
| F2b pairwise joint hull | 0% additional on real CIFAR all-pair test |
| FC-HZ multi-layer triangle | about 8% additional over F1 on random two-layer toys |

The dense-conv gap often needs something closer to a 100% removal of the
remaining relaxation slack. Local triangle-family cuts do not reach that.

### 5.2 Small-Dense Case Reasoning

Examples:

- `acasxu_2023`
- `linearizenn_2024`
- `relusplitter`

These benchmarks often need reasoning about activation cases. Exact-star
splitting, BaB, or MILP can prove them because they refine the activation
state space. Under no-splitting/no-MILP constraints, the forward domain keeps
too many activation cases merged.

### 5.3 Parser and Architecture Coverage

Examples:

- `cctsdb_yolo_2023`
- `linearizenn_2024`
- parts of `nn4sys`
- parts of `metaroom` / `ml4acopf`

Parser work can reduce ERROR counts and may unlock some V/A, but the scout
data suggests it is unlikely to contribute hundreds of new decisions by
itself. It is still valuable for engineering maturity and benchmark coverage.

### 5.4 FAL vs CERT Asymmetry

SC-HZ was very effective at finding FAL witnesses on `safenlp_2024` because
FAL needs one concrete counterexample. CERT requires proving no input in the
box violates the spec. The latter demands global tightness across all rivals
and all feasible activation configurations.

This explains why the lift is mostly A, not V:

```text
SC-HZ NEW V: 0
SC-HZ NEW A: 548
```

That is a strength for falsification-heavy benchmarks and a limitation for
robust-CERT-heavy benchmarks.

---

## 6. Correct Comparison With Other Tools

The comparison table should not be framed as "we beat abcrown overall". We do
not. Based on the current public comparison:

| Tool | V/A | High-level engine |
|---|---:|---|
| abcrown | 2460 | backward bound propagation + BaB-style complete reasoning |
| NeuralSAT | 2065 | SAT/BaB-style complete reasoning + bound propagation |
| ACT/HyZor + SC-HZ | 1472 | forward HZ + continuous-LP sidecars + strict replay |
| nnenum | 1445 | exact-star splitting |
| PyRAT con_z | 1393 | forward constrained zonotope |

The honest claim is:

- ACT/HyZor is not first overall.
- ACT/HyZor is competitive and now third in the listed total.
- ACT/HyZor is unusually strong among forward-only/no-splitting style
  methods.
- ACT/HyZor has a much stricter witness replay and provenance discipline than
  many comparison runs.
- The remaining gap to 2000+ is concentrated exactly where non-forward or
  splitting-based methods dominate.

This is still a publishable position if framed correctly. It is not enough if
the only target is "beat abcrown in raw V/A without using its mechanisms".

---

## 7. What We Should Not Claim

Do not claim:

1. "SC-HZ generally improves all benchmarks." It does not.
2. "1472 is the theoretical ceiling of forward-only verification." It is not.
3. "Dense-conv is solved." It is not.
4. "More pair cuts or more triangle layers will likely reach 2000+." Current
   evidence says no.
5. "Parser cleanup is V/A improvement." It is only V/A improvement if the
   newly parsed instances pass the same strict proof or replay gates.
6. "The withdrawn relusplitter result was validated." It was not.
7. "Boundary equality FAL is acceptable." It is not under the strict gate.

---

## 8. What We Can Claim

The paper/report can safely claim:

1. ACT/HyZor + SC-HZ reaches **1472 V/A** under a strict principle set.
2. The new lift over 924 is **+548 NEW A**, all audited and strict.
3. The lift is concentrated in `safenlp_2024`.
4. ACT/HyZor is strong on forward-friendly benchmarks such as:
   - `safenlp_2024`
   - `dist_shift_2023`
   - `nn4sys`
   - `collins_rul_cnn_2022`
   - `cgan_2023`
   - `malbeware`
   - `cora_2024`
5. Dense-conv and activation-case-heavy benchmarks remain weak.
6. F1/F2b/FC-HZ provide an empirical negative characterization of the
   current pipeline: local triangle-family tightening is insufficient.
7. Future 2000+ progress requires either:
   - a new forward abstraction, or
   - relaxing a binding principle, or
   - a large parser/coverage effort plus an unexpectedly high yield.

---

## 9. What Remains Worth Doing

### 9.1 Short-Term: Documentation and Audit Cleanup

This should be done before any new scoring campaign:

1. Keep `research/sc_hz_FREEZE_1472_20260605.md` as the numeric anchor.
2. Keep this retrospective as the failure/lesson document.
3. Ensure `PAPER_1472_CHARACTERIZATION_20260606.md` and the final paper do
   not overstate generality.
4. Keep the FC-HZ failed gate as an expected failure or separate gate test,
   not as a hidden failing unit test.
5. Maintain the withdrawn-claim list in every handoff document.

### 9.2 Medium-Term: Parser/Architecture Sprint With a Hard Stop

Scope:

- `cctsdb_yolo_2023`: dynamic Slice / shape handling.
- `linearizenn_2024`: Slice parser.
- selected `nn4sys`, `ml4acopf`, `metaroom` residual parser issues.

Gate:

```text
Timebox: 3-5 days
Success threshold: >= 30 NEW V/A
If below threshold: stop and return to paper
```

This work is worthwhile for engineering maturity. It should not be presented
as the main route to 2000+ unless it unexpectedly clears the gate.

### 9.3 Long-Term: Phase H, a New Forward Abstraction

If the real goal is 2000+, the next serious research direction cannot be
"one more sidecar cut". It needs a new domain design.

Promising directions:

1. **Output-projected constrained forward domain**  
   Carry only constraints that are relevant to final rival projections, but
   carry them as real constraints rather than discarded triangle slack.

2. **Block-level template polyhedra for residual/conv blocks**  
   Use templates over block outputs rather than neuron-local triangles.
   The target is aggregate slack, because pairwise/local cuts failed.

3. **Forward constrained zonotope with retained side constraints**  
   Track selected `A_c x <= b_c` constraints forward, not just interval tails
   and DeepZ triangle generators.

4. **Structured exactness for low-dimensional tails**  
   Where the final classifier tail has small dimension, compute a more exact
   forward projected hull without splitting the input region.

Phase H acceptance gate should be high:

```text
Minimum scout gate:
  - >= 100 NEW V/A across at least 3 benchmark families, OR
  - >= 50 NEW V/A on dense-conv with no regression elsewhere

Stop condition:
  - if first 2 mechanism prototypes give < 30 NEW V/A or only one-benchmark
    concentration, write closure and return to paper.
```

---

## 10. Practical Answer to "Why Did We Not Reach 2000+?"

Because the successful SC-HZ mechanism is a strong **structured falsifier**
for `safenlp_2024`, not a universal verifier. It finds many true violating
inputs under strict replay, but it does not remove the relaxation slack needed
to prove dense-conv safety or activation-case-heavy small dense networks.

The tools above 2000 use at least one of the following:

- backward output-aware bound optimization,
- branching/splitting over ReLU states,
- SAT/MILP/integer reasoning,
- exact-star splitting.

Those are precisely the mechanisms we excluded to preserve the project
thesis. Within the current pipeline, we tested the obvious continuous-LP
tightening options and they all fell short.

This is not "no improvement"; 924 -> 1472 is real. But it is also not enough
for the raw-score target. The next step is a design decision:

- write the paper at 1472 with an honest scope, or
- commit to a multi-month Phase H abstraction redesign, or
- explicitly relax one principle and compete closer to abcrown's territory.

---

## 11. Evidence Index

| Evidence | Path |
|---|---|
| Original SC-HZ brief | `research/INNOVATION_BRIEF_sc_hz_20260604.md` |
| Frozen 1472 memo | `research/sc_hz_FREEZE_1472_20260605.md` |
| 1472 aggregate bundle | `audit_results/sc_hz_final_1472_aggregate.json` |
| Hard gates policy | `research/sc_hz_hard_gates_for_v_a_results.md` |
| Honest paper characterization | `research/PAPER_1472_CHARACTERIZATION_20260606.md` |
| PRUNE bug disclosure | `research/sc_hz_prune_bug_disclosure_20260604.md` |
| S1 PHANTOM repair result | `research/s1_phantom_repair_result_20260605.md` |
| Dense-conv memory roadmap | `research/phase_e_roadmap_20260605.md` |
| Dense-conv memory gate pass | `research/phase_e_gate2_v2_PASSED_20260605.md` |
| F1 constrained LP prototype | `research/phase_F1_constrained_lp_prototype_20260605.md` |
| F1 closure and F2 plan | `research/phase_F1_closed_F2_plan_20260605.md` |
| F2b final closure | `research/phase_F2b_FINAL_CLOSED_20260605.md` |
| F3 day-one result | `research/phase_F3_day1_result_20260605.md` |
| FC-HZ design | `research/phase_G_forward_constrained_hz_design.md` |
| FC-HZ failed gate | `research/phase_G_FAILED_paper_1472_20260605.md` |

---

## 12. Recommended Advisor-Facing Summary

Use this wording:

> We improved ACT/HyZor from 924 to 1472 V/A under the strict forward-only
> principle set. The improvement is real and fully audited, but it is
> concentrated: all +548 new decisions come from safenlp falsification.
> The original PRUNE hypothesis was falsified. Dense-conv and small
> case-heavy benchmarks did not move under F1, F2b, F3, or FC-HZ. The data
> show that the current DeepZ-triangle plus continuous-LP sidecar pipeline
> has reached its practical ceiling. To reach 2000+, we need a new forward
> abstraction or a principled decision to relax one of the binding rules.

This statement is accurate, defensible, and does not hide the weakness.

---

## End of Retrospective
