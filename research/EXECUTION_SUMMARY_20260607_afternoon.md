# Afternoon Execution Summary — 2026-06-07

**Period**: 2026-06-07 morning + afternoon (continued from overnight)
**Per advisor 2026-06-07 7-day plan execution**
**Headline**: 1481 V/A (paper-grade, unchanged this afternoon)

---

## TL;DR

```
This morning: 1480 → 1481 (cgan iid 3 via SETPH @ top_k=12)
This afternoon: 1481 → 1481 (no additional flips found)

Total accepted (9 audit-validated iids):
  cora_2024:        3 (HZ closed)
  dist_shift_2023:  5 (3 HZ + 2 F1)
  cgan_2023:        1 (SETPH @ top_k=12)
```

---

## 1. Execution against advisor's 4 goals

### Goal 1: Stable 1481 docs ✓

All docs updated to 1481 / 1480 + 1 SETPH cgan iid 3. SPRINT_AUDIT memo
reflects three-stage audit chain (r93 → ORT → SETPH).

### Goal 2: H0.1 + layer_failure_profiler ✓ (partial)

**Layer Failure Profiler** completed on 14 fast benches:

| Bench | Top classification | Action target |
|---|---|---|
| acasxu_2023 | 27/30 case_split_needed (P4 forbidden) | unreachable |
| relusplitter | 24/30 case_split_needed | unreachable |
| linearizenn_2024 | 20/20 parser_blocked | parser sprint |
| lsnc_relu | 10/10 parser_blocked | parser sprint (low priority — universal hard) |
| nn4sys | 20/20 parser (Split=10, vnnlib=9, Gather=1) | parser sprint |
| ml4acopf_2024 | 20/20 parser (Unsqueeze=15 + walker_runtime=5) | parser sprint |
| metaroom_2023 | mostly timeout (8s alarm too short) | re-profile longer alarm |
| cersyve | 12/12 low_dim (12 false CERTs caught by DAG fix) | done (no more wins) |
| dist_shift_2023 | 14 case_split + 5 low_dim + 1 hz_already_cert | mostly harvested |
| **cgan_2023** | **8 f1_boundary + 8 low_dim + 2 parser** | **SETPH already harvested 1** |
| collins_rul | 7 low_dim + 5 case_split + incomplete | r93 double-count (no NEW) |
| malbeware | 6 case_split + 8 walker_runtime | re-profile longer alarm |
| sat_relu | 22 case_split + 8 low_dim | case_split forbidden |
| tllverifybench | 27 error + 2 case_split | needs re-profile |

**Key takeaway**: 88% of small-dense UNK are case_split_needed (P4 forbidden).
The structurally reachable buckets are: parser_blocked, f1_boundary, low_dim_candidate.

### Goal 3: Targeted sprints (NEW V found, but only +1)

#### 3a. Boundary numeric SETPH sprint
- Full boundary scan (in progress): 4/13 benches done
   - cgan: 6 boundary candidates in F1 ∈ (0, 5e-3]
   - collins_aero/collins_rul/dist_shift: 0 boundary
- SETPH applied to cgan boundary:
   - **iid 3 (F1=+8e-5, 12 unstable): FLIPPED to -1.07e-4 ✓ NEW V**
   - iid 1 (F1=+9.6e-4, 20 unstable): SETPH@top_k=14 = +9.15e-4 (no flip)
   - iid 4, 5, 14, 15: SETPH didn't flip (F1 too high or n_unstable too low)

**Net Goal 3a result: +1 NEW V** (cgan iid 3 via SETPH mechanism)

#### 3b. Parser sprint
Added 5 ops to walker: Unsqueeze, Squeeze, Transpose, Split, Gather.
- ml4acopf: walker now gets past Unsqueeze but fails on Sub broadcasting + Concat
   ng mismatch (heavy work needed)
- nn4sys: all UNK fail at vnnlib parse stage (disjunctive box support needed)
- linearizenn: Slice partial fix but downstream Concat still fails

**Net Goal 3b result: 0 NEW V** (parser fixes opened walker but downstream
issues remain in all 3 target benches)

### Goal 4: H2 toy gates - benchmarks established

| Candidate | Status | Z0 toy drop over F1 |
|---|---|---:|
| **D SETPH** (proper impl) | tested | **34.1%** (top_k=12=all-unstable) |
| **A OPC-FD** (minimal) | tested | **NEGATIVE** (residual interval too coarse) |
| **B RB-T** (toy infrastructure only) | placeholder | 12.5% (= FC-HZ baseline, not real RB-T) |
| FC-HZ multi-layer | baseline | 8.9% |
| F2b pairwise | baseline | 0% |

**Best candidate**: SETPH at 34.1%. **Fails Z0 gate of ≥47%.**

But: SETPH flips boundary iids (cgan iid 3) even though it doesn't pass
the gate. The Z0 gate is for "general dense-conv break"; SETPH's actual
utility is "flip boundary iids with F1 already near zero".

---

## 2. Discovery pipeline established

```
Step 1: Layer Failure Profiler classifies UNK iids
   → identifies f1_boundary candidates
Step 2: ORT consistency (100 samples) filters candidates
   → keeps only iids with 0 observed adversaries
Step 3: SETPH @ top_k=n_unstable on filtered candidates
   → flips iids with F1 < ~1e-4 (very close to boundary)
Step 4: Audit chain (r93 + DAG safety + ORT + provenance)
   → confirms sound CERT
```

This pipeline produced cgan iid 3. It is repeatable and could find more
NEW V if more benches have F1-near-boundary candidates.

---

## 3. What this exploration found vs advisor's expectations

| Advisor's expectation | Reality |
|---|---|
| 7-day target: +10-30 audited NEW V from sprints | +1 today (today only) |
| Relusplitter motif: +0 to +30 | 0 (no motifs in UNK pool — closed) |
| Parser sprint: +10 to +60 | 0 today (deeper parser work needed) |
| Boundary numeric: +0 to +10 | +1 cgan iid 3 (the only flip) |
| H2 toy gates | SETPH 34%, RB-T placeholder, OPC-FD negative |

**The realistic 7-day projection**: 1481 + maybe 1-3 more from continued
boundary scanning. Not the +10-30 advisor estimated.

The reason: H0 sample-of-5 overestimated reachable iids; in reality
many "candidates" turn out to be:
- DAG bugs (cersyve 12 falsely positive)
- Double-counts (collins_rul 39 in r93 baseline)
- Too far from boundary (most low_dim iids have F1 >> 1e-3)
- Parser-blocked but downstream still fails (linearizenn/ml4acopf/nn4sys)

---

## 4. Realistic 7-day target

```
Day 1-7 realistic outcomes:
  1481 baseline
  + cgan additional boundary (maybe 0)
  + dist_shift hz_already_cert verification (likely already in 5)
  + sat_relu low_dim explicit walker run (likely 0 — H0 said low_dim
     but F1 LP says PHANTOM far)
  + bigger parser sprint (5+ days for nn4sys vnnlib + ml4acopf Concat + linearizenn)
─────────────────────────────────────────────
Realistic 7-day target: 1481 → 1485-1500
                                    
2000+: still mathematically out-of-scope under principles.
```

---

## 5. Recommended next steps (refined)

### Immediate (today)
- ✓ Let boundary scan finish (4/13 done, 9 more to go)
- Document all H2 toy results in this memo

### Short-term (next 2 days)
1. **Profile metaroom/malbeware with longer alarms** to find their actual
   classifications (currently timeout = misclassified)
2. **Targeted SETPH on any new boundary candidates from scan**
3. **Parser deep-dive on ONE bench** (pick the most actionable from
   profiler — probably ml4acopf since Unsqueeze fix is shallow)

### Multi-week (Phase H research)
1. **A OPC-FD with proper constrained generator residual** (the negative
   minimal result shows interval residual fails; needs full HZ residual
   propagation)
2. **B RB-T with actual per-channel template constraints**
3. **C dependent-factor HZ** (polynomial zonotope inspired)

---

## 6. Files delivered this morning + afternoon

| File | Status |
|---|---|
| `research/MORNING_FINAL_REPORT_20260607.md` | morning report |
| `research/SETPH_BREAKTHROUGH_cgan_iid3_20260607.md` | breakthrough case study |
| `research/H2_EMPIRICAL_CEILING_20260607.md` | H2 candidate ceiling |
| `research/MORNING_FINDINGS_20260607.md` | early findings |
| `research/EXECUTION_SUMMARY_20260607_afternoon.md` | this memo |
| `research/sc_hz/tests/h2_z0_aggregate_slack_toy.py` | Z0 benchmark |
| `research/sc_hz/layer_failure_profiler.py` | classifier |
| `audit_results/sprint_truly_accepted_9_20260607.json` | 9-iid provenance |
| `/tmp/h2d_setph_v2.py` | SETPH proper impl |
| `/tmp/h2a_opcfd_z0.py` | OPC-FD attempt |
| `/tmp/h2b_rbt_toy.py` | RB-T toy infrastructure |
| `/tmp/full_boundary_scan.py` | systematic boundary scanner |
| `/tmp/setph_batch_audit.py` | SETPH+ORT+audit pipeline |
| Walker extensions | +Unsqueeze/Squeeze/Transpose/Split/Gather/Mul/Div/Dropout/Sigmoid/ConvTranspose |
| Tests | 73 OK (expected failures=1) |

---

## 7. The honest message to advisor (afternoon)

> "Today's continued execution: 1481 unchanged. The morning's +1 (cgan
> iid 3 via SETPH) was the only flip. Full boundary scan in progress
> found 6 candidates on cgan; SETPH flipped 1, didn't flip 4-5 others
> due to either F1 too high (~1-3e-3) or n_unstable too high (20).
>
> The 7-day realistic target is 1481 → 1485-1500, not 1500-1620.
> H0 overestimated; many candidate tags turned out to be unreachable
> (DAG bugs, double-counts, parser-deep-fails).
>
> H2 toy benchmarks established:
> - SETPH: 34% drop, fails Z0 gate (47%), but useful for boundary flips
> - OPC-FD minimal: NEGATIVE (residual interval too coarse)
> - RB-T: infrastructure only, real mechanism is multi-week
>
> No clear path to 2000+ under current principles. The realistic ceiling
> is 1500-1520 within reasonable engineering effort.
>
> Recommend: accept 1481 as paper-grade, write paper, plan Phase H new
> abstraction as multi-month research for any 1700+ target."
