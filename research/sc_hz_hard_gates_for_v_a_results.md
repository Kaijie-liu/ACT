# Hard Gates for Any SC-HZ V/A Result — Binding Policy

**Date**: 2026-06-05
**Trigger**: 2026-06-04 prune.py soundness bug taught us self-audits can be circular
**Status**: BINDING for all future V/A claim publication

Any V (CERT) or A (FALSIFIED) verdict from SC-HZ — or any forward-only sidecar —
must pass ALL of the following before it can be counted toward a headline,
written into a memo for advisor review, or published.

---

## Gate G1: LP UB monotonicity

For any pruned forward propagation that produces a closed-form LP UB:

```
UB(K=K_small) >= UB(K=infty)   for all K_small <= ng
```

This must hold by construction. Empirical check: pin via unit test on a
synthetic multi-layer Dense+ReLU network with random weights. Already
implemented at `research/sc_hz/tests/test_prune_multilayer_soundness.py`.

If this gate ever fails (UB(K=small) < UB(K=infty)), STOP and investigate.
A sound over-approximation cannot tighten when columns are dropped into a
tail. Violation indicates under-approximation, which falsely creates CERTs.

This was the gate that caught the 2026-06-04 incoming-tail bug.

---

## Gate G2: Independent audit path

A V/A claim's audit MUST use a verification path independent of the
propagation that produced the claim. Concretely:

| Claim type | Audit path |
|---|---|
| A (FALSIFIED) | Strict ORT replay on the decoded x_star. ORT is the ground truth; bug-independent. |
| V (CERT) | (a) Re-derive LP UB with at least one alternative K (e.g. K=infty if claim was K=K_small). Both must give UB < threshold. (b) Adversarial corner: decode x_star with at least one alternative decoder; ORT must NOT confirm any unsafe condition. |

NEVER re-run the same buggy forward propagation as the "audit" of itself.
The 2026-06-04 V-side audit on 71 relusplitter CERTs failed this — it ran
the same buggy path and falsely STRICT-PASSed all 71.

If only the bug-prone path is available, the audit is INVALID and the
verdict cannot be published until an independent path is built.

---

## Gate G3: Production cross-check

For benchmarks where the canonical production verifier (act/) has a verdict
at the SAME budget, the SC-HZ claim must be consistent:

| SC-HZ verdict | Production verdict | Outcome |
|---|---|---|
| CERT | CERTIFIED | MATCHED — count as confirming, do not double-count |
| CERT | UNKNOWN | NEW V — count, but require G1 + G2 to be airtight |
| CERT | FALSIFIED | **CONTRADICTION — SOUNDNESS FAILURE; STOP** |
| A | FALSIFIED | MATCHED — count as confirming, do not double-count |
| A | UNKNOWN | NEW A — count if ORT replay confirms |
| A | CERTIFIED | **CONTRADICTION — SOUNDNESS FAILURE; STOP** |

Production CONTRADICTION (CERT vs FAL or A vs CERT) must trigger immediate
stop and diagnosis. Either SC-HZ has a soundness bug, production has one,
or there is a benchmark-spec interpretation gap.

---

## Gate G4: Strict zero-tolerance ORT replay (STRICT INEQUALITY)

For every A_CONFIRMED witness `x*`, the strict ORT replay must pass:

```
d_out · M(x*) >  threshold     STRICT inequality at zero tolerance
x* in [lb, ub]                 with NO clip required
```

**STRICT `>` is binding (not `>=`)**. Witnesses with `d_out · M(x*) == threshold`
EXACTLY are rejected as "near-miss boundary" cases — they do not strictly
violate the safety assertion `d_out · M(x*) <= threshold` (which permits
equality). G4 was tightened from `>=` to `>` on 2026-06-05 after the S3
scout caught 20 sat_relu candidates with margin exactly 0 (network output
hitting threshold on the boundary; cannot be counted as falsification).

If clip is required, the decoded maximizer was outside the box, indicating
a structural problem in the LP/decoder; flag as clip-caveat A and do not
count toward the strict headline.

### When is margin = 0 acceptable?

Never for A claims. The safety assertion in VNN-COMP vnnlib is usually
expressed as `(>= a b)` or `(>= a const)`. A violation requires the
opposite (`< b` strictly), so a witness must produce `d_out · M(x*) > threshold`
strictly to be sound.

Implementation in `s1_phantom_repair.py`, `s3_smallcontrol_scout.py`,
`audit_*.py`, `forward_witness.py` — all comparison sites in the FAL
verdict pipeline use `>` not `>=`.

---

## Gate G5: Provenance bundle

Every receipt MUST contain:
- `canonical_root` (absolute path to the VNN-COMP benchmarks tree)
- `instances_csv_sha256` (SHA-256 of the benchmark's instances.csv)
- `onnx_sha256` (SHA-256 of the ONNX model)
- `vnnlib_sha256` (SHA-256 of the VNNLIB spec)

Without all four, the claim is not independently re-derivable and cannot
be cited. This protects against benchmark drift and onnx/vnnlib hot edits.

---

## Gate G6: Production code modification audit

Before publishing any headline number:

```
$ git diff --stat -- act/
(must be empty)
```

If `act/` is dirty, the SC-HZ sidecar is no longer a pure overlay — it
modifies the production baseline. The canonical 924 V/A is then no longer
the baseline, and "924 + N" arithmetic is invalid.

---

## Gate G7: Multi-K sanity (headline is K=∞; smaller-K is sanity-only)

**Headline rule**: any V/A verdict counted toward the published headline
MUST be derived at K = ∞ (no prune). Smaller K values are NOT counted
toward the headline.

| K | Use |
|---|---|
| K = ∞ | The HEADLINE NUMBER comes from here. Tightest LP UB; ground truth for over-approximation. |
| K = 256 / 512 / 1024 / etc. | Sanity-only: each must satisfy `UB(K) >= UB(K=∞)` (this is G1). Used to spot-check the monotonicity invariant. NOT used to claim "full identical verdict" or as an alternative headline. |

Do NOT claim "K=256 sweep gave the same N V/A as K=∞" unless a complete
parity sweep has been run on the same input set and the verdicts are
identically labeled at zero numerical tolerance. Even then, the headline
stays anchored on K=∞.

If verdict changes between K and K=∞ (e.g. CERT at K=∞ becomes
FAL_CANDIDATE at K=256), the smaller K is more conservative — but the
SC-HZ headline always reports the K=∞ result. Smaller-K result, if
mentioned, must be clearly labeled "smaller-K sanity check, not headline".

If a non-monotone K behavior is ever observed (G1 violation), the
underlying prune / propagation is unsound; STOP and investigate.

---

## Gate G8: Two-tier reporting

Always report:
- **Strict**: only verdicts that pass G1-G7 at zero tolerance, with `atol = 0`
- **Watchlist**: verdicts that pass G1-G3 + G5-G6 but fail strict ORT
  zero-tol replay due to small numerical issues. These are PHANTOM-LP-SAT
  for A claims, and not countable toward the strict headline.

Strict headline goes into the paper / advisor report. Watchlist goes into
the appendix and Phase X+1 investigation queue.

---

## Gate G10: Resource budget (good-citizen on shared machine)

**Hard cap on SC-HZ research processes**:
- Per-iid worker peak RSS: **≤ 80 GB**
- Concurrent SC-HZ processes (parent + workers): **total RAM ≤ 100 GB**
- GPU concurrent allocation: **≤ 32 GB per worker**, no more than **2 workers** when other users are active
- CPU: **n_workers=1** for any worker with peak RSS > 40 GB (this single-thread serialization is the default)

**Why**:
- Shared 125 GB machine; other users run persistent VMs (libvirt qemu) that hold ~24 GB at idle. Leaving ≥ 25 GB free at all times keeps the machine responsive for them.
- An SC-HZ worker that grows to 80 GB + parent 5 GB + buffer ≈ 100 GB — within budget.
- Going past 100 GB triggers OOM-killer, which may evict OTHER users' workloads. This is unacceptable.

**Enforcement**:
- Before launching any sweep / pilot, run `free -g` and refuse to start if `available < 90 GB`.
- All worker scripts must set `resource.setrlimit(resource.RLIMIT_AS, (100 * 1024**3, ...))` so individual processes self-cap and die cleanly if they exceed.
- Long-running pilots must check `free -g` every 5 minutes; pause if `available < 25 GB` and wait for it to recover.
- ANY shared-machine OOM-kill event involving an SC-HZ process must be logged and the per-iid worker that caused it goes on a do-not-rerun list.

**Why we are adding G10 today**:
- The 2026-06-05 dense-conv pilot OOM-killed itself on tiny iids at K=∞ and K=60k. Local kill, no impact on others.
- But the 70 GB per-iid RSS was right at the threshold; one accidental n_workers=2 would have triggered system-wide OOM-killer and affected OTHER users' VMs.
- G10 makes the implicit "play nice" rule explicit and binding.

---

## Application to 2026-06-04 1460 headline

| Gate | Status |
|---|---|
| G1 | PASS — `test_prune_multilayer_soundness` enforces |
| G2 | PASS — forward-coeff decoder is independent of backward-chain; ORT replay is bug-independent |
| G3 | PASS — 546 SC-HZ A vs production: 536 UNK (NEW A), 10 FAL (MATCHED). No CERT vs FAL contradictions. |
| G4 | PASS — 546/546 strict zero-tol ORT replay, no clip required |
| G5 | PASS — 546/546 provenance bundle complete |
| G6 | PASS — `git diff --stat -- act/` empty |
| G7 | PASS — headline is at K=∞ (no prune); smaller-K only used for monotonicity sanity, NOT as alternative headline |
| G8 | Strict: 536 NEW A on safenlp; Watchlist: 381 PHANTOM_LP_SAT (not counted) |

**1460 V/A passes all 8 gates.**

---

## Application to withdrawn claims

| Claim | Gate failed |
|---|---|
| 1346 (relusplitter +64 V) | **G1 FAIL** — LP UB UNDER-approximated due to incoming_tail_radius bug |
| 1346 71/71 STRICT-PASS audit | **G2 FAIL** — audit re-ran the same buggy forward propagation; not independent |
| 1346 PyRAT[con_z] 5th-place position | derived from withdrawn 1346, also withdrawn |

---

## Future gate additions

If new failure modes are discovered, add a Gx clause here. Every gate must
be enforced by either a unit test, an automated check in the audit
pipeline, or an explicit step in the verification checklist.

This file is binding for all future SC-HZ work and advisor reporting.
