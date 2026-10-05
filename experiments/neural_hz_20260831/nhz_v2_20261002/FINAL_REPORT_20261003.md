# CORRECTION (2026-10-04, LOG N111-N114)

After this report was written, the E0 replay produced a **false CERT** on CIFAR row 164 (a
validated ADV anchor). Root cause: HiGHS and SCIP both returned a numerically spurious MILP
infeasibility on plans with rows of width about 1e-6 (y-relation rows with coefficients near the
solver tolerance); our exclusion rule trusted that status. The domain mathematics and the engine
bounds were verified sound at the witness point. Fix: a numerical safety width of 1e-5 on every
unit row and piece bound (terminal v7.11/v8.5, path v7.2; a relaxation). Every MILP-excluded CERT of
v14 was re-run with the fixed plans (N113): 448 of 454 reproduced, 6 withdrawn (malbeware 135, 137;
cersyve 3, 7; relusplitter 119, 131). The numbers below are superseded by the re-verified vector:

| | re-verified value |
|---|---:|
| baseline solves retained | 1,746 / 1,870 |
| conflicts | 0 |
| gains | 38 = 28 CERT + 10 ADV (relusplitter 25 CERT; cgan 2+2; cora 7 ADV; tll 1 ADV; vit 1 CERT) |
| CERTs by rigorous LP / by re-verified MILP | 571 / 448 |
| ADV witnesses S1-valid | 767 / 767 |
| promotion gate | FAIL (124 losses); formal score stays 1870/2413 |

Table: `results/reverified_v14_table.md`. E0: N109 was stopped at 170 rows; its re-verified
partial vector has 31 CERT gains, 16 anchors kept, 5 lost, 0 conflicts; the full E0 replay runs
again as N115 with path v7.2.

---

# Neural-HZ v2: final report of the 2026-10-02/03 session

Formal score unchanged at **1870/2413**; E0 unchanged at **61/400**. No candidate is promoted.

## The final single-path replay (N039 v14, domain-only verdicts)

Path v7.0 + engine n020.1, one frozen code set over all 2,413 rows, four workers (the baseline's
concurrency), budgets = baseline timeouts. Verdict sources: CERT from the element's rigorous LP
or selective-exactness MILP bounds; ADV only from incumbents of those MILP plans, validated by
ONNX Runtime on the original network and property. No attack, sampling, LP-dual corner, fixed
sample point or seeded MIP start anywhere (user rule of 2026-10-03, LOG N099).

| | value |
|---|---:|
| baseline solves retained | 1,750 / 1,870 |
| conflicts | 0 |
| gains (baseline UNKNOWN/TIMEOUT -> solved) | 40 = 30 CERT + 10 ADV |
| witness audit (independent reader, exact rationals) | 767/767 S1-valid |
| CERT sampling audit (30 gains + 20 per family, 2,000 samples each) | 245 rows, 0 violations |
| promotion gate | FAIL (120 losses) |

Gains by family: relusplitter 27 CERT; cora 7 ADV; cgan 2 CERT + 2 ADV; tll 1 ADV; vit 1 CERT.
Losses by family: vit 49 CERT (attention relaxation gap), acasxu 26 ADV (found before only by
the removed centre check, LOG N105), cora 16 ADV (found before only by the removed LP corner),
linearizenn 10 CERT (baseline needed 476-840 s with its portfolio), malbeware 7 (one crash fixed in
path v7.1, LOG N103; five large single-layer plans), dist_shift 4, relusplitter 4, safenlp 2
(20.1 s on 20 s), tll 2. Full table: `results/final_n039_full_replay_v14_table.md`.

## What was built (research content)

Proved: projection-aligned exact ReLU element with three commuting concretisations (THEORY 2-3);
selective-exactness lattice (7); sign rule for query-level exactness incl. conjunctions (9, 12.1);
monotone units on any layer (12.3); latent-box ideal cuts (12.2); smooth units as rigorous
line rows and Balas segment phases in latent coordinates (14); rigorous float32/float64
rounding semantics (10). Measured: sign rule halves binaries and recovers SafeNLP; smooth rows
take dist_shift from 55 to 66 of 70 with few sigmoid binaries; softmax remainder rule and
score-difference bounds fix the PGD-ViT blow-up (LP bound 292 -> 0.3) but IBP-ViT stays about
7x looser than the baseline. Negative and closed: ideal cuts, knapsack cross term, fused
attention variants, n018. Soundness: independent review found and we fixed a latent false-CERT
path (invalid HiGHS dual bounds), tiny-coefficient dropping, segment piece bounds and a
degenerate-box witness case (LOG N085).

## Decisions for the user

1. ACAS Xu / malbeware ADV retention needs either the centre check (a sample: not allowed) or a
   MIP warm start from the element's own centre phases (a solver setting the baseline itself used
   as `--mip-start base-binary`). Is the warm start admissible? (LOG N105)
2. The exact LP optimum of the lambda level (lattice level 0, no binaries) as a domain query for
   large single-layer rows (LOG N103): admissible or a helper?
3. Integrality skipping by Proposition 1' under hard limit 2; S1 vs S2 witness semantics (16 of
   767 witnesses are S1-only: cgan 9, dist_shift 6, metaroom 1, all on specs whose fixed inputs have
   no float32 value).

## Where everything is

`LOG.md` N000-N108 (append-only), `THEORY.md` Sections 1-16, `README.md` (navigation), replay
directories `n039_full_replay_v1..v14/` and E0 `n015_*/`, `n061_*/`, `n068_*/`, `n070_*/`,
`n072_*/`, `n076_*/`, `n084_*/`, `n092_*/`, `n100_*/`, `n102_*/` (each with PREREG and FREEZE;
stopped runs kept, never merged). Production ACT, frozen archives and HyZor untouched; nothing
committed or pushed.
