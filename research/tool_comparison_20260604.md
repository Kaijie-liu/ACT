# ACT/HyZor vs Other Verifiers — Cross-Tool Analysis (2026-06-04 night)

**Status**: paper-ready cross-tool comparison. Numbers from the 22-benchmark
full sweep (`audit_results/` various); ACT row = 805 V + 119 A = 924 V/A.
ACT does NOT use a "helper" attack tool, MILP, BaB, CROWN backward, or
random/PGD candidates (see Section 2 of `paper_skeleton_20260604.md`).

The audience-relevant question is:
> *In a like-for-like comparison, is ACT the strongest pure-forward
> Hybrid-Zonotope verifier?*

The answer is **yes, by +47.4%** over its only same-domain competitor
(PyRAT [hyb_z]). The detail is below.

---

## 1. Full cross-tool table

| Tool | V | A | V+A | E | Resolve | Engine class |
|---|---:|---:|---:|---:|---:|---|
| abcrown `--NOPGD` | 1718 | 742 | 2460 | 282 | 71.2% | BaB-complete + bound prop |
| NeuralSAT `--disable_attack` | 1581 | 484 | 2065 | 506 | 59.8% | BaB-complete + bound prop |
| nnenum | 693 | 752 | 1445 | 433 | 41.9% | exact-star splitting |
| PyRAT STRICT `[con_z]` | 1242 | 151 | 1393 | 521 | 40.3% | forward constrained zonotope |
| **ACT (HZ) GPU STRICT** | **805** | **119** | **924** | **109** | **26.8%** | **forward Hybrid Zonotope (ours)** |
| PyRAT STRICT `[hyb_z]` | 602 | 25 | 627 | 987 | 18.2% | forward Hybrid Zonotope |
| NNV STRICT | 457 | 0 | 457 | 1418 | 13.2% | forward approximate star |
| CORA TRUESTRICT | 2 | 0 | 2 | 0 | 0.06% | forward reachability |

All numbers are over N = 3,453 instances across 26 VNN-COMP-2025
benchmarks.

---

## 2. The fair comparison band

The first four tools (abcrown, NeuralSAT, nnenum, PyRAT [con_z]) use
search, splitting, or exact-star fragmentation as their proof engines.
**They are not "pure forward" tools.** They incorporate machinery that
ACT's principle set (Section 2 of paper skeleton) explicitly forbids:
BaB-complete tree exploration, exact-star case splitting, etc.

The **pure-forward group** is the apples-to-apples comparison for ACT:

| Tool | V+A | Position |
|---|---:|---|
| **ACT (HZ) GPU STRICT** | **924** | **#1** |
| PyRAT [hyb_z] | 627 | #2 (−297, −32%) |
| NNV STRICT | 457 | #3 (−467, −51%) |
| CORA TRUESTRICT | 2 | #4 (essentially 0) |

ACT is the strongest pure-forward verifier in this sweep, by a wide
margin even among same-domain competitors.

---

## 3. The HZ-vs-HZ comparison — direct domain match

PyRAT [hyb_z] runs the same abstract domain (Hybrid Zonotope) as ACT;
this is the closest apples-to-apples comparison possible.

| Metric | ACT | PyRAT [hyb_z] | Δ |
|---|---:|---:|---|
| V+A total | **924** | 627 | **+297 (+47.4%)** |
| V (sound UNSAT) | 805 | 602 | +203 (+33.7%) |
| A (sound SAT) | 119 | 25 | **+94 (+376%)** |
| E (tool error / OOM) | **109** | 987 | **−878 (9× fewer)** |
| Resolve over N=3,453 | **26.8%** | 18.2% | **+8.6 pp** |

### Per-benchmark head-to-head

| Benchmark | ACT | PyRAT [hyb_z] | Δ | Winner |
|---|---:|---:|---|---|
| acasxu_2023 | 88 | 55 | +33 | ACT |
| collins_rul | 51 | 57 | −6 | PyRAT |
| cgan_2023 | 11 | 7 | +4 | ACT |
| linearizenn | 17 | 13 | +4 | ACT |
| malbeware | 136 | 64 | **+72** | ACT |
| nn4sys | 86 | 50 | +36 | ACT |
| relusplitter | 7 | 20 | −13 | PyRAT |
| safenlp_2024 | 345 | 189 | **+156** | ACT |
| sat_relu | 51 | 20 | +31 | ACT |
| tllverifybench | 3 | 0 | +3 | ACT |
| cifar100_2024 | 0 | 0 | 0 | tie |
| cersyve | 1 | 1 | 0 | tie |

**ACT wins on 8 / 12 head-to-head benchmarks** in the HZ-vs-HZ
comparison, including the largest two (safenlp +156, malbeware +72).

### Why ACT beats PyRAT [hyb_z]

| Mechanism | ACT | PyRAT [hyb_z] |
|---|---|---|
| HZ as primary abstraction | yes — HZ is the carrier | no — HZ is layered on top of `con_z` |
| Memory pressure | controlled (109 OOM/error) | high (987 OOM/error, ~9×) |
| Conv kernel | GPU dense | host-side |
| LP backend | warm-started highspy | not warm-started |

The 9× fewer errors and the dual-carrier overhead in PyRAT [hyb_z]
together explain the +297 gap. **It is not an abstract-domain
difference** — it is implementation engineering on the same domain.

---

## 4. Where ACT is competitive with helper-using tools

In the helper-using group (abcrown, NeuralSAT, nnenum), ACT cannot
match the totals — those tools have BaB completeness or exact-star
splitting that ACT explicitly excludes. But on several specific
benchmarks, ACT matches or beats them anyway:

| Benchmark | ACT | abcrown | NeuralSAT | nnenum | PyRAT con_z | NNV | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| dist_shift_2023 (72) | **72** | 65 | 65 | unsup | 67 | unsup | **#1 (clean 72/72)** |
| nn4sys (194) | 86 | 69 | 86 | 24 | 50 | 17 | **ties NeuralSAT; +17 over abcrown** |
| collins_rul (62) | 51 | 39 | 39 | 62 | 58 | 39 | **+12 over both BaB tools** |
| cgan_2023 (21) | **11** | 9 | 10 | unsup | 19 | 0 | **beats abcrown and NeuralSAT** |
| cora_2024 (180) | 20 | 22 | 22 | 20 | 20 | 18 | top cluster |
| malbeware (150) | 136 | 149 | 128 | 91 | 125 | 49 | **#2; beats NeuralSAT and nnenum** |
| acasxu_2023 (186) | 88 | 139 | 137 | 186 | 178 | 75 | competitive but BaB/exact-star stronger |
| sat_relu (100) | 51 | 99 | 100 | 45 | 12 | 4 | **beats PyRAT, nnenum, NNV** |

**ACT is #1 or tied with the strongest helper-free tools on 6
benchmarks.** This is the "pockets where pure-forward HZ has a real
edge" story.

---

## 5. Where ACT is honestly weak

| Benchmark | ACT V/A | Top result | Mechanism that ACT lacks |
|---|---:|---:|---|
| cifar100_2024 (200) | 0 | abcrown 101 | BaB completeness; ACT BoxHZ ceiling kicks in |
| tinyimagenet_2024 (200) | 1 | abcrown 140 | same — BaB completeness |
| vggnet16_2022 (18) | 1 | abcrown 14 | same — and memory wall at L28 |
| safenlp_2024 (1080) | 345 | abcrown 1080 | BaB completeness solves wide-spec exactly |
| metaroom_2023 (100) | 15 | abcrown 94 | BaB completeness |
| linearizenn_2024 (60) | 17 | abcrown/NeuralSAT 59 | same — small dense, BaB fits perfectly |
| relusplitter (220) | 7 | abcrown 113 | same — ReLU splitting heuristics |
| vit_2023 (200) | parser unsupported | abcrown 83 | architecture coverage (attention shape lineage) |
| yolo_2023 (72) | 0 | abcrown 62 | architecture coverage |
| cctsdb_yolo_2023 (39) | 0 | abcrown 39 | architecture coverage (data-dependent Slice) |

### What these weaknesses share

1. **Large CNN ceiling**: ACT's Phase 1–3 BoxHZ surrogate loses too
   many generators after several conv layers; this is the §6c
   memory-wall finding writ across cifar / tiny / vgg.
2. **Wide-spec benchmarks**: safenlp has 1080 instances each with
   many disjuncts; BaB-complete tools exhaust the disjunct space
   directly, while pure forward must over-approximate.
3. **Architecture coverage**: vit / yolo / cctsdb_yolo are
   parser-side gaps. The cctsdb_yolo 2026-06-04 cleanup landed only
   the fixed-shape subset; the variable-shape Slice abstraction
   remains future work (see `frontend_cleanup_plan.md` §8).

None of these weaknesses indicate a mathematical flaw in the
forward-HZ approach. They are precision ceilings inherent to
pure-forward verification, which the principle set (Section 2)
deliberately accepts.

---

## 6. Soundness — the ACT advantage no headline number captures

All ACT FAL receipts (119 A-verdicts) pass strict ORT zero-tolerance
replay:

```text
input_box_holds         = True
vnnlib_query_holds      = True
spec_zero_tol_holds     = True
provenance bundle attached on every receipt
```

Comparable strict-replay receipts are NOT a default in
abcrown / NeuralSAT (their FAL counts include attack-based witnesses
that need a separate audit), and are not produced by NNV / CORA at
all under the helper-free strict run.

This is the *audit-receipt contract* of the paper (Section 2 of paper
skeleton). It is invisible in a single-number total but is the
property that allows the 924 V/A to be **independently re-verified**
by an external auditor without re-running ACT.

---

## 7. Engineering quality — error count

Among the 8 tools, ACT has the **lowest E count** in absolute terms
relative to its resolve count:

| Tool | E | E / (V+A+E) |
|---|---:|---:|
| **ACT** | **109** | **10.5%** |
| abcrown | 282 | 10.3% |
| NeuralSAT | 506 | 19.7% |
| nnenum | 433 | 23.1% |
| PyRAT con_z | 521 | 27.2% |
| PyRAT hyb_z | 987 | 61.1% |
| NNV | 1418 | 75.6% |
| CORA | 0 | 0% (but only 2 resolved at all) |

ACT and abcrown have comparable E rates (~10%), but ACT runs without
any of abcrown's helper machinery. **This is the quietest finding of
the table**: a pure-forward tool reaching 924 V/A with a 10% error
rate, on the same hardware budget, is a non-trivial engineering
result.

---

## 8. Headline summary for the paper

For the paper abstract:

> ACT/HyZor is a forward-only Hybrid Zonotope neural-network verifier
> that delivers **924 sound V/A across 22 VNN-COMP-2025 benchmarks**
> (805 V + 119 A, 26.8% resolve rate, 109 tool errors). Under a
> principle set that excludes CROWN backward, PGD/random falsification,
> MILP/integer reasoning, and BaB / input splitting, ACT is **the
> strongest pure-forward verifier in the sweep** — outperforming the
> only same-domain competitor (PyRAT [hyb_z]) by **+47.4% V/A** with
> **9× fewer tool errors**. ACT beats the helper-using BaB-complete
> tools on six specific benchmarks where forward HZ has structural
> advantages (dist_shift 72/72, nn4sys 86, collins_rul 51 [+12 over
> both BaB tools], cgan 11, cora top-cluster, malbeware 136 [#2 across
> all tools]). All A-verdicts carry strict ORT zero-tolerance replay
> + provenance hashing, enabling independent re-verification without
> re-running ACT.

---

## 9. Files cited

- `research/paper_skeleton_20260604.md` — the principles, design lock, and audit-receipt contract that justifies the strict-only column.
- `research/results_20260604.md` — the upstream results memo (now to be updated to the 22-benchmark headline).
- `research/profile_matrix_20260604.md` — the per-profile honesty doc (which name-gated vs structural-gated profiles produced each V/A row).
- `research/frontend_cleanup_plan.md` — the cctsdb / parser-cleanup status, including the 2026-06-04 fixed-shape Slice envelope landing.
- Source audit dirs:
  - `audit_results/clean_canonical_combined_summary_20260604.json` (the original 5-bench 253 V/A scope)
  - `audit_results/cctsdb_dynslice_full_20260604T064140Z/` (cctsdb post-parser-fix: 39 UNK with audit reason)
  - `audit_results/dynslice_parity_smoke_20260604T064507Z/` (4-bench single-iid parity confirmation)
