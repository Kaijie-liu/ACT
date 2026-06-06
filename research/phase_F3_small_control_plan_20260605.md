# Phase F3 — Small/Control + Parser + Constrained LP

**Date**: 2026-06-05 night
**Status**: PLAN. Execution starts immediately following F2b closure.
**Goal**: +30 to +100 V/A in 1-2 weeks (NOT 2000+).

---

## 1. Why F3 now

F2b decisively failed on dense-conv (0% gain even with all 300 pairwise
cuts on cifar 113). Dense-conv short line CLOSED. But:
- **F1 constrained LP DOES work on dense networks** — 17% real
   tightening, parity exact, sound.
- Small/control benches have small networks where F1 LP can directly
   apply without the multi-neuron washout problem.
- Several benches currently sit at low V because of PARSER fail-closed,
   not because the math is hard.

So F3 = engineering payoff: get F1 LP onto benches where its 17% drop
might actually flip verdicts (smaller networks have fewer unstable
neurons per layer, less washout).

## 2. Target benches (6 in priority order)

| # | Bench | Current V (canonical) | Reason F3 may help |
|---|---|---:|---|
| 1 | tllverifybench_2023 | partial | small dense, F1 LP applies directly |
| 2 | linearizenn_2024 | partial | small dense, F1 already proven on it |
| 3 | acasxu_2023 | varies | small dense, F1 + spec-aware refinement |
| 4 | ml4acopf_2024 | parser fail-closed | parser fix → ≥0 baseline → F1 LP |
| 5 | metaroom (cifar100) | partial | parser fix for correlation operators |
| 6 | nn4sys | partial | parser fix for Reshape/Slice/Gather |

DEPRIORITIZED:
- cctsdb_yolo_2023 — variable-shape Slice requires symbolic abstraction,
   high engineering cost, PyRAT also 0 conv, no leverage.

## 3. Day-by-day execution

### Day 1 (today/tomorrow)
- Parser audit across 6 benches: identify fail-closed reasons
- Run F1 constrained LP on tllverifybench + linearizenn + acasxu 20
   sentinels each (3 × 20 = 60 instances)

### Day 2
- ml4acopf parser fix + smoke test
- metaroom parser fix
- nn4sys Reshape/Slice/Gather parser

### Day 3
- 20-sentinel runs on parser-fixed benches
- Aggregate F3 day-3 result

### Day 4-5
- If ≥30 NEW V/A: expand each passing bench to full
- If <30 NEW V/A: switch to Phase G FC-HZ

## 4. Hard gates (per advisor)

For each bench:
```
20-sentinel run must produce ≥3 NEW V/A in that bench
  → bench passes, scale to full
≥3 NEW × 6 benches = ≥18 NEW theoretical floor
  → realistic target +30 to +100
```

Aggregate F3 floor:
```
day 5 total ≥30 NEW V/A → extend to 1-2 weeks
day 5 total <30 NEW V/A → switch immediately to Phase G
```

NO bench parser fix counts as "improvement" if its V/A doesn't lift.
Parser is plumbing, not headline.

## 5. Estimated yield per bench

| Bench | Current V | Realistic F3 yield |
|---|---:|---:|
| tllverifybench | partial | +5 to +15 |
| linearizenn | partial | +5 to +15 |
| acasxu | partial | +10 to +25 (spec-aware refinement) |
| ml4acopf | 0 (parser-blocked) | +5 to +20 |
| metaroom | partial | +0 to +10 |
| nn4sys | partial | +5 to +15 |
| **Total floor** | | **+30 to +100** |

## 6. Stop criteria

```
F3 day 5 yield < 30 NEW V/A
OR any bench parser fix doesn't lift V/A despite parse PASS
→ F3 closes for that bench; aggregate decision at day 5
```

If F3 fails entirely:
- Immediately commit to Phase G (FC-HZ new abstraction) OR paper at 1472

## 7. Files to produce

| File | Purpose |
|---|---|
| `research/sc_hz/small_control_f1_runner.py` | 20-sentinel batch runner for F1 LP |
| `research/sc_hz/parser_fixes/` | bench-specific parser patches |
| `audit_results/phase_F3_day1_*` | day-1 baseline + F1 results |
| `audit_results/phase_F3_day5_summary.json` | aggregate decision data |

## 8. Critical reframing (per advisor)

The 1472 number is the **SC-HZ + DeepZ triangle + F1/F2b sidecar LP**
ceiling. It is NOT the "forward-only HZ-like verifier theoretical ceiling".

F3 doesn't change the math; it gets existing math to existing places.
Phase G is where the real ceiling lift would come.
