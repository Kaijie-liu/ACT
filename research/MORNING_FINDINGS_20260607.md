# Morning Findings — 2026-06-07 (Step B+C+D Partial Execution)

**Period**: 2026-06-07 morning (continued from overnight 8hr exploration)
**Per advisor 2026-06-07 directive**: layer_failure_profiler + parser/motif/boundary
sprint + H2 toy math gates before any big engineering.
**Headline**: 1480 V/A unchanged. No new V/A. But systematic empirical data
established for advisor's 5-step plan.

---

## TL;DR

```
1480 V/A unchanged (no new tonight from this morning's work)

Z0 toy benchmark BUILT and reproduces dense-conv pattern:
   HZ closed-form: 575% loose
   F1:             23.5% drop over HZ (matches cifar 17%)
   FC-HZ:          +8.3% additional over F1 (matches cifar 8%)
   F2b:            +0% additional over F1 (matches cifar 0%)
→ Any H2 candidate must achieve ≥47% drop over F1 to pass Z0 gate.

L3 boundary numeric: NOT applicable
   acasxu 98 UNK: 0 iids in F1 ∈ (0, 1e-3]. All 85/98 have F1 > 100.

L5 motif simplification: NOT applicable
   relusplitter 213 UNK: 0 duplicated ReLU, 0 opposite ReLU pair.
   UNK pool is just simple sequential MLPs.

Profiler partial (fast benches, sample-of-30):
   acasxu_2023:     27/30 case_split_needed (P4 forbidden)
   relusplitter:    24/30 case_split_needed (P4 forbidden)
   → ~85% of small-dense UNK NEEDS case split, structurally impossible
     under our principles.
```

---

## 1. Z0 toy benchmark established

Built `research/sc_hz/tests/h2_z0_aggregate_slack_toy.py`. 2-block dense
network with aggregate ReLU slack diffusion. Tests HZ/F1/FC-HZ/F2b.

```
20 random trials, all matching cifar dense-conv pattern:
  HZ closed-form looseness vs brute (median): +575%
  F1 drop over HZ (median):                   23.5%
  FC-HZ drop over F1 (median):                8.3%
  F2b drop over F1 (median):                  0.0%
```

This is the **math gate** advisor mandated. Any Phase H2 candidate
implementation must pass:
- Toy Z0 PASS gate: drop ≥47% over F1
- cifar 113 Z1 PASS gate: F1 excess +0.146 → ≤+0.05 OR CERT
- 8-sentinel Z2 PASS gate: ≥1 NEW CERT OR ≥60% median drop

If a candidate fails Z0, do NOT proceed to implementation. Math says no.

## 2. L3 boundary numeric closed (acasxu)

Per advisor's plan, L3 (boundary exact-arithmetic LP) should target iids
with F1 excess in (0, 1e-3] (potential float-precision PHANTOM that exact
LP could flip).

**Empirical scan of acasxu 98 true UNK**:
```
F1 excess distribution:
  ≤ 1e-5:    0
  1e-5_1e-3: 0
  1e-3_1e-2: 0
  1e-2_1e-1: 0
  1e-1_1:    0
  1_10:      9
  10_100:    3
  100+:      85
```

**0 boundary candidates**. L3 mechanism doesn't apply to acasxu. The
earlier F3 day-1 claim of "iid 107 HZ=+0.001 → F1=+0.000" was for HZ
excess, not F1 excess. F1 LP on acasxu drives ALL iids to F1 > 1.

**L3 CLOSED on acasxu**. Not pursuing rational LP for headline V/A.

## 3. L5 motif simplification closed (relusplitter)

`/tmp/relusplitter_motif_scout.py`: detect duplicated/opposite ReLU patterns
across all 213 relusplitter UNK iids.

**Findings**:
- 0/213 iids with duplicated ReLU
- 0/213 iids with opposite ReLU pair

UNK pool is just simple sequential MLPs (4-7 Gemm/Conv + ReLU layers).
The benchmark name "relusplitter" appears to test verifiers that DO
split ReLUs into segments; we don't split, so these are structurally
hard.

**L5 motif simplification mechanism is empty for relusplitter UNK pool.**

## 4. Profiler classification (partial, fast benches in progress)

`research/sc_hz/layer_failure_profiler.py` running on 14 fast benches with
sample-of-30 each. Partial results:

| Bench | Sampled | case_split | dense_aggregate | low_dim | parser | other |
|---|---:|---:|---:|---:|---:|---:|
| acasxu_2023 | 30 | **27** | 0 | 2 | 1 (vnnlib) | 0 |
| relusplitter | 28/30 | **24** | 3 | 1 | 0 | 0 |

(profiler running on remaining 12 benches)

### Insight: case_split_needed dominates small dense

51/58 (88%) of small-dense UNK iids are classified as `case_split_needed`:
- shallow net (n_relus ≤ 6)
- HZ closed > 0
- not in boundary / low_dim / parser_blocked / dense_aggregate buckets

This is the EMPIRICAL confirmation that small-dense benches like acasxu/
relusplitter NEED the activation case-split mechanism we forbid by P4.
Within our principles, these iids are out-of-scope by definition. Honest
acceptance: these contribute to the 735 robust_blocked count from H0.

## 5. What this means for advisor's 5-step plan

### Step A ✓ Done
- All docs 1480 self-consistent (SPRINT_AUDIT memo three-stage chain)

### Step B 🏃 Running
- Profiler producing data; will complete in ~30 min
- Already classifies ~60 small-dense UNK as `case_split_needed`
- Final classification distribution will inform Step C parser targets

### Step C ⏳ Pending profiler
- Once profiler labels each bench, run parser sprint on benches dominated
   by `parser_blocked` tag
- Initial bench data suggests:
   - acasxu/relusplitter: case_split (no parser fix can help)
   - linearizenn/nn4sys/ml4acopf/cctsdb_yolo: parser-blocked (parser fix
      could unlock 0-30 NEW V/A across these — uncertain)

### Step D 🎯 Partially executed
- **Z0 toy benchmark built and validated** (reproduces cifar pattern)
- Gate set: ≥47% drop over F1
- Next: implement minimal A OPC-FD / B RB-T / D SETPH on Z0 toy ONLY
- DO NOT implement in production until Z0 passes

### Step E ⏳ Pending Z0 candidate pass
- RB-T / OPC-FD / dependent-factor HZ → research direction
- Multi-week effort, only worth it if Z0 toy shows ≥1 candidate passes

---

## 6. Honest distance to 2000+

Updated picture after morning's work:

```
1480 V/A (current paper-grade)
+ Step C parser sprint (uncertain): 0 to 30 NEW V/A
+ Step D H2 candidate (if Z0 passes): 0 to 100 NEW V/A (research-grade)
+ Step E long-term H2 (if D toys pass): 0 to 250 NEW V/A
─────────────────────────────────────────────────────────
2000+ target:                +520 needed
Realistic upper bound:       1480 + 380 = 1860 (most optimistic)
Realistic median:            1530-1650
```

**2000+ remains structurally unreachable within current principles**.
The H2-Z0 finding REINFORCES this — any candidate has to beat F1 by
~47% which the F2b/FC-HZ closure already showed is hard in continuous LP.

## 7. Hard gates per advisor (updated)

```
7-day:
  ✓ 1480 docs stable (DONE)
  ✓ Z0 toy benchmark built (DONE)
  🏃 profiler completes systematic classification (in progress)
  ⏳ ≥20 audited NEW V from parser+low_dim sprints

30-day:
  1480 → 1550/1650 realistic
  H2-Z0 candidate pass OR honest closure

60-90-day:
  H2 candidate ≥ Z2 passes → 1700-1850
  OR all candidates fail → paper at 1480-1650
```

## 8. Files

| File | Status |
|---|---|
| `research/sc_hz/tests/h2_z0_aggregate_slack_toy.py` | Z0 benchmark, 47% gate established |
| `research/sc_hz/layer_failure_profiler.py` | systematic per-iid classifier |
| `/tmp/relusplitter_motif_scout.py` | L5 motif scout (0/213) |
| `/tmp/layer_failure_profiler_fast/` | profiler receipts (running) |
| `research/MORNING_FINDINGS_20260607.md` | this memo |
| Tests: 73 OK (expected failures=1) | clean |

## 9. The honest message to advisor

> "Morning execution: 1480 stable. Z0 toy benchmark built and validates
> dense-conv pattern (HZ 575% loose, F1 23.5%, FC-HZ +8.3%, F2b +0%).
> Gate: ≥47% drop over F1 for any H2 candidate.
>
> L3 boundary numeric on acasxu: 0/98 candidates. L5 motif on relusplitter:
> 0/213. Both directions empirically closed for those benches.
>
> Profiler (running): 51/58 small-dense UNK classified case_split_needed
> (P4 forbidden). Confirms these benches are structurally out-of-scope
> under our principles.
>
> Step C parser sprint: pending profiler completion to identify which
> benches' UNK pool is parser-dominated.
>
> Step D: Z0 toy benchmark is the MATH GATE before any production
> implementation. No candidate built yet (need toy implementations of
> A OPC-FD / B RB-T / D SETPH minimal first).
>
> L4 walker remains 0 lines code.
>
> Distance to 2000+: structurally 520 V/A. Realistic ceiling within
> principles: 1530-1650 (median estimate). 2000+ requires either
> H2 candidate passing Z0+Z1+Z2 (uncertain research) OR principle
> relaxation (against thesis)."
