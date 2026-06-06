# CANONICAL STRICT AUDIT MEMO — 2026-06-07

**Status**: 3 tiers, in increasing confidence-relaxation.

---

## Tier 1 — STRICT FLOOR (conservative, defensible)

```
2013 V/A
```

**Source**: `strict_555_FINAL_20260607.json` (555 records, 0 dup, 0 MILP, 0 ORT viol)

**Accounting** (per advisor 2026-06-07):
- 1472 prior session strict baseline
- + 555 this session records
- − 13 (overlap with prior 1472 safenlp baseline, advisor caught)
- − 1 (`cora_2024/iid 5` r93 already CERTIFIED, advisor caught)
- = 1472 + 541 = **2013 V/A**

**Audit checklist**: r93 cross-check ✓, FIXED walker recompute ✓, ORT 500-sample 0 viol ✓, sha256/vnnlib_sha256 ✓, tail_radius unit tests 86/86 ✓.

**This is what we can publish externally TODAY.**

---

## Tier 2 — STRICT CANDIDATE (needs cifar+nn4sys overlap audit)

```
2107 V/A
```

**Source**: `strict_649_FINAL_20260607.json` (= 555 + 94: 84 cifar sparse + 10 nn4sys)

**Accounting**:
- 1472 baseline
- + 555 records (per above: 541 net new)
- + 84 cifar sparse-slack remaining (`push_cifar_sparse_20260607.jsonl`)
- + 10 nn4sys with new MatMul + Sub broadcast (`push_nn4sys_20260607.json`)
- − overlaps in the 94 added (to verify)
- = 1472 + 635 = **2107 V/A**

**Needs before publication**:
1. cifar sparse 84: verify NONE were r93 CERT (my analysis: 0 overlap, looks clean)
2. nn4sys 10: verify NONE were r93 CERT (my analysis: needs check)
3. Fresh recompute audit on 84 cifar sparse records (sparse-slack soundness)

**My finding**: cifar sparse 84 has 0 r93 CERT overlap (clean). nn4sys 10 needs careful overlap check.

---

## Tier 3 — STRONG CANDIDATE (needs relusplitter audit)

```
~2290-2295 V/A (estimate)
```

**Source**: `strict_925_FINAL_20260607.json` (= 649 + 276 added in last sweep)

**Per-record baseline label audit** (using strict_517 as proxy for "prior session strict accepted"):
```
LABEL                    Count
OLD_517_BASELINE         517
NEW (this session)       315
OLD_R93_CERTIFIED         92  ← double-counted with r93 baseline
OLD_R93_FALSIFIED          1
```

**NEW per bench**:
```
relusplitter:  183  ← NEEDS independent audit
cifar100_2024: 117  (84 sparse + 33 clean dedup, mostly verified clean)
nn4sys:         10  
cora_2024:       4
malbeware:       1
```

**Accounting** (best case if all NEW records are truly new vs 1472):
- Tier 2 (2107) + 183 relusplitter
- = ~**2290 V/A** (if relusplitter 183 all pass overlap audit)

**Per advisor**: "relusplitter +183 needs baseline overlap audit and fresh recompute audit before claim".

**This tier CANNOT be published until relusplitter audit complete.**

---

## REJECTED claim — 2384

The previously circulated `strict_925 → 2384` headline is **REJECTED**.

Root causes:
1. **per-row filter bug**: `per_instance.csv` has multiple source rows per iid (cpu/gpu/cpu_smoke/gpu_full). My filter `v != CERTIFIED` checked EACH ROW; an iid with UNK row + CERT row was wrongly added to UNK list. 92 r93-CERT iids slipped through.
2. **collins_rul 39**: ALL my "new" collins were r93 FALSIFIED (cpu + gpu sources). Filter bug above.
3. **metaroom 52**: 52 of 57 were r93 CERTIFIED (gpu_full source). Filter bug above.
4. **cora iid 5**: r93 CERTIFIED. Advisor caught.

**File renamed**: `SESSION_FINAL_2384_VA_STRICT_20260607.md` → `_CANDIDATE_NOT_FINAL_2384_20260607.md` with warning header.

---

## Action items

| Item | Status |
|---|---|
| Withdraw 2384 from external | ✓ done (file renamed) |
| Publish 2013 floor + 2107 candidate | this memo |
| Audit relusplitter 183 vs 1472 baseline + fresh recompute | TODO |
| GPU ORT re-validate all 832 records (2000-sample) | in progress (PID 2486187) |
| Single canonical bundle filename | this memo proposes `STRICT_FLOOR_2013` and `STRICT_CANDIDATE_2107` |
| Archive obsolete strict_555 / 565 / 649 / 702 / 924 / 925 | TODO |

---

## Comparison vs other tools (using 2107 candidate)

| Rank | Tool | V+A | Compliance |
|---:|---|---:|---|
| #1 | αβ-CROWN --NOPGD | 2460 | uses Gurobi + backward + BaB |
| #2 | NeuralSAT --disable_attack | 2065 | uses BaB + LiRPA |
| **HyZor 2013 floor** | strict P1-P5 | 2013 | between #2 and #3 |
| **HyZor 2107 candidate** | strict P1-P5 | 2107 | passes #2 NeuralSAT |
| #4 | nnenum | 1445 | input enum |
| #5 | PyRAT [con_z] | 1393 | |
| #6 | PyRAT [hyb_z] | 627 | same HZ family |
| #7 | NNV STRICT | 457 | |
| #8 | CORA TRUESTRICT | 2 | |

Even at conservative **2013 floor**, we are between NeuralSAT (2065) and nnenum (1445) — solidly in #3 position, very close to #2.

At **2107 candidate** (after audit), we PASS NeuralSAT and are clear #2 under strict P1-P5.

---

## GPU usage note

Per user's question 2026-06-07: GPU is now being used for ORT re-validation (2000-sample, CUDAExecutionProvider). Walker itself is CPU/numpy-bound. Going forward:
- ORT validation: GPU (faster, more samples)
- Walker propagation: CPU (numpy / scipy.linalg)
- Multi-bench parallelism: CPU cores (we have 24+)

Future optimization potential: walker GPU port via torch would give 5-10× on cifar/tiny.
