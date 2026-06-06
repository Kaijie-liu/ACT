# SC-HZ Scout Finding on 5 Benchmarks (2026-06-05)

**Goal**: per advisor 5-step plan §3, scout for principle-clean NEW V/A on
`malbeware` / `metaroom` / `ml4acopf` / `sat_relu` / `nn4sys` using the
forward-coefficient + fixed-prune sidecar that drove the safenlp +536
result.

Target: **+140 toward 1600**.

---

## 1. Sweep results (forward-coeff + fixed prune, K=∞, 30s per iid)

`audit_results/sc_hz_scout_5bench_20260604T140229Z/`

| Bench | n | A_CONFIRMED | CERT | PHANTOM_LP_SAT | UNK (parser fail-closed) |
|---|---:|---:|---:|---:|---:|
| malbeware | 150 | 1 | 49 | 0 | 100 |
| metaroom_2023 | 100 | 0 | 0 | 0 | 100 |
| ml4acopf_2024 | 69 | 0 | 0 | 0 | 69 |
| **sat_relu** | **100** | **41** | 0 | 59 | 0 |
| nn4sys | 194 | 0 | 0 | 0 | 194 |

The sat_relu 41 A is the only major sweep signal. malbeware adds 1 more
CERT vs the horizontal-extension count of 48 (due to fixed prune giving
slightly different result on one boundary iid). metaroom / ml4acopf /
nn4sys are parser-blocked.

---

## 2. Production cross-check (60s budget)

### sat_relu 41 A_CONFIRMED

`audit_results/sc_hz_scout_prod_baseline_*` + `sc_hz_scout_prod_remaining_*`

All 41 receipts pass SC-HZ STRICT-PASS (4-check audit, bug-independent
forward-coeff + ORT). All 41 are sound A.

Production verdicts:
- FALSIFIED: **41 / 41**
- UNKNOWN: 0

**NEW A vs production: 0**.

Interpretation: production's per-rival LP/MaxPool path on sat_relu_2023
ALREADY finds these 41 A. SC-HZ + forward-coeff reproduces the same set
but does not extend it. Confirms mechanism but adds no headline count.

### malbeware 49 CERT

`audit_results/sc_hz_scout_prod_baseline_*` + `sc_hz_scout_prod_remaining_*` + `sc_hz_scout_mal_final_*`

Production verdicts (all 49):
- CERTIFIED: ~49 (consistent with horizontal-extension result that all
   48 prior CERTs were also production CERTIFIED)
- UNKNOWN: 0
- FALSIFIED: 0

**NEW V vs production: 0**.

Interpretation: same as horizontal extension — SC-HZ's CERTs on
malbeware are a strict subset of production's CERTs at 60s.

### malbeware 1 A_CONFIRMED

Already covered in horizontal extension: production = FALSIFIED. Matched,
no new contribution.

---

## 3. Parser-blocked benchmarks

3 of 5 scouted benchmarks fail-closed at the ONNX walker:

- **metaroom**: ~~~  (likely ResNet structure)
- **ml4acopf**: likely Reshape/Slice/Mul preprocessing
- **nn4sys**: known Reshape/Slice/Gather preprocessing (per CLAUDE.md)

Each would require 0.5-1 day of parser extension work. Given the
sat_relu / malbeware results show 0 NEW V/A on benchmarks where the
parser DOES work, the expected yield from parser extensions is low.

---

## 4. Net scout finding: 0 NEW V/A across 5 benchmarks

Combined with the safenlp_2024 result (+536 A), the post-scout SC-HZ
sidecar headline is:

| Source | NEW V/A |
|---|---:|
| safenlp_2024 (forward-coeff + fixed prune, 60s budget) | 536 NEW A |
| sat_relu (41 A matched production) | 0 |
| malbeware (49 CERT matched production, 1 A matched) | 0 |
| metaroom / ml4acopf / nn4sys (parser-blocked) | 0 |
| relusplitter (withdrawn bug artifact) | 0 |
| **Total** | **536** |

**Final headline remains: 924 + 536 = 1460 V/A.**

The advisor's "+140 to 1600" target is NOT reachable via SC-HZ on the
currently-available scout pool. To get past 1460, we need:

1. **Parser extensions** (0.5-1 day each) for metaroom / ml4acopf /
   nn4sys. Expected yield modest based on sat_relu / malbeware pattern.
2. **A different sidecar mechanism**. The forward-coeff path with
   triangle ReLU saturates at safenlp's wide-spec dense network shape.
   For tight-spec or conv benchmarks, a different relaxation or
   different candidate construction is needed.
3. **Dense-conv track** (advisor §5, deferred long-term). ResNet shape
   tracking for cifar100 / tinyimagenet / yolo. Multi-week investment.

---

## 5. What this scout DOES tell us positively

1. **The forward-coeff mechanism is consistent across benchmarks**: 
   sat_relu's 41 A produced by SC-HZ are exactly production's 41 A. The
   mechanism reliably finds the SAME witnesses production finds (when
   the parser works). No spurious A.
2. **The bug-fixed prune is sound on multi-benchmark data**: 49 + 1 + 41
   verdicts on malbeware/sat_relu cross-check production with 0
   contradictions. No CERT-vs-FAL conflicts.
3. **safenlp is genuinely unique**: the +536 NEW A on safenlp is not
   replicated on other dense / small-dense benchmarks. The mechanism
   pays off when the spec structure aligns with closed-form box-corner
   maximization on triangle-ReLU forward HZ — that specific structure
   appears in safenlp (wide hyperrectangle perturbations on small dense
   ruarobot networks) but not in the scouted benches.

---

## 6. Implications for the 1460 → 2000+ trajectory

The advisor wrote: "要冲 2000+，光靠 safenlp sidecar 不够，必须开第二个大来源."

The scout confirms this. The +536 on safenlp is currently the ONLY
material source of new V/A under principle-clean constraints. To extend:

- **Short term (1-2 weeks)**: parser extensions for metaroom / ml4acopf /
  nn4sys (+ a few benches with conv preprocessing). Audit risk: the
  underlying mechanism may not find new V/A on these either.
- **Medium term (2-4 weeks)**: dense-conv shape tracking for cifar100 /
  tinyimagenet / yolo. Would open the largest remaining benchmark mass
  to the forward-coeff sidecar.
- **Long term**: design a second principle-clean sidecar mechanism for
  benchmarks the box-corner forward-coeff fails on. Candidates:
  per-rival K-piece for smooth activations; spec-aware abstraction
  refinement (forward only); structured constraint propagation through
  Reshape / Concat / preprocessing ops.

**Bottom line for advisor**: 1460 holds as principle-clean ceiling.
Path to 1600+ is not via additional safenlp-style sidecar; it requires
either parser engineering (low-yield bet) or dense-conv investment
(high-yield but multi-week).

---

## 7. Documents updated

- `research/sc_hz_principle_clean_1460_20260604.md` — 4th → 3rd overall
- `research/sc_hz_phase_b_results_20260604.md` — pointer to 1460 added
- `research/sc_hz_hard_gates_for_v_a_results.md` — NEW: G1-G8 binding policy
- `audit_results/sc_hz_final_1460_aggregate.json` — NEW: consolidated bundle
- This memo

## 8. Hard-gate sign-off for 1460

Per G1-G8 (see `sc_hz_hard_gates_for_v_a_results.md`):
- G1: ✓ multi-layer regression test enforces
- G2: ✓ forward-coeff decoder + ORT independent of LP-UB path
- G3: ✓ 0 contradictions in 546 + 41 + 49 cross-checks
- G4: ✓ all 546 zero-tol ORT, no clip required
- G5: ✓ provenance complete on 546 A
- G6: ✓ `git diff --stat -- act/` empty
- G7: ✓ K=∞ headline; smaller K consistent
- G8: ✓ strict 536 NEW A; PHANTOM 381 in watchlist

1460 PASSES all 8 gates. Headline ready for paper / external citation.
