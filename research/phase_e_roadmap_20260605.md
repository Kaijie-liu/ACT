# Phase E Roadmap — Dense-Conv Memory + Forward-Only ReLU Tightening

**Date**: 2026-06-05
**Status**: ROADMAP (not implementation). Headline `1460 V/A` frozen and
unaffected by Phase E outcomes.
**Source**: advisor 2026-06-05 review after the day-of dense-conv pilot
returned YELLOW (0 NEW V/A).

This roadmap supersedes the earlier "park as Phase E candidate" line in
`research/day_denseconv_20260605.md` with a concrete 10-day plan, hard
gates, and a kill switch.

---

## 1. Strategic position going into Phase E

| | Value |
|---|---|
| Frozen headline | **1460 V/A** (audit-validated, principle-clean) |
| Source of the +536 lift | safenlp_2024 only (forward-coeff + fixed prune + strict ORT replay) |
| What today's pilot proved | ResNet walker works; LP UB median +0.5 above threshold on cifar; box-corner doesn't realize witness; tinyimagenet OOM-blocked |
| What today's pilot didn't prove | Whether tighter relaxation + memory headroom would unlock NEW V/A |
| Gap to next public tool (nnenum) | +15 (already achieved at 1460) |
| Gap to NeuralSAT | -605 |
| Gap to abcrown | -1000 |

**Honest read**: 1460 is real and defensible. Reaching 2000+ requires a
SECOND source of NEW V/A. Day-of pilot showed that the SAME mechanism
(forward-coeff box-corner) does not extend cleanly to dense-conv.

---

## 2. What Phase E will and will not do

### Phase E IS:
- Memory infrastructure for forward HZ propagation on dense-conv ResNets
- A forward-only ReLU relaxation tighter than DeepZ triangle, applied
   selectively (final 1-2 ReLU layers, or per-rival group)
- A controlled experiment on the existing 40-sentinel pilot set
- A 10-day investment with a hard kill switch at day 10

### Phase E IS NOT:
- A blind sweep of cifar100 / tinyimagenet under any K
- A parser engineering pilot on nn4sys / ml4acopf / metaroom / cersyve / cgan
- A backward-bound refinement (CROWN, α-CROWN, β-CROWN) — still forbidden
- A switch to MILP or binary encoding — still forbidden
- A modification to `act/` — production code remains untouched

### What Phase E will not touch:
- `act/` (production)
- `research/sc_hz/forward_witness.py`, `prune.py`, `onnx_walker.py`,
   `onnx_walker_resnet.py` — sound; only the new memory/relaxation
   modules are added
- The 1460 headline (frozen regardless of Phase E outcome)

---

## 3. Day-by-day plan

### Day 1-3: Dense-conv memory infrastructure (chunked Conv propagation)

**Goal**: lift the ~70 GB single-iid RAM ceiling so tiny/cifar-deep can
actually be measured.

Concrete deliverables:

- `research/sc_hz/conv_chunked.py` (NEW): chunked Conv propagation that
   does NOT materialize the full `(C_out × H × W × ng)` intermediate at
   once. Implementation pattern:
   - Iterate over groups of generator columns of size `chunk_size` (e.g. 256)
   - Allocate `(C_out × H × W × chunk_size)` per chunk
   - Accumulate per-row L1 contribution into the tail when over budget
   - Keep numeric equivalence to `forward_resnet`'s K=∞ path up to
       ordering of float-sum

- Two storage tracks, separated explicitly:
   - `G_root`: generators rooted at input coordinates (lineage carried,
       used by `decode_xi_star_forward`)
   - `G_relu_slack`: generators added by ReLU triangle (anonymous;
       collapsed earlier into the tail when ng grows)

- Tail-radius propagated through exact linear ops via `|W| @ tail`,
   same formula as `prune.py` (sound).

- Per-layer trace required: `{ng, nc, tail_norm, peak_mem, wall_s, op}`
   for memory-debugging.

**Hard gate at end of Day 3**: re-run the SAME 20 cifar + 20 tiny sentinels
with chunked Conv. Acceptance:
- 0 OOM kills
- peak_mem < 80 GB per iid
- verdict count NOT REPORTED yet (memory is the deliverable here)
- Walker center parity for all 40 iids at 1e-5 cifar / 2e-5 tiny (the
   tiny f32 floor diagnosed today)

If hard gate fails: stop Phase E, dense-conv mechanism declared
memory-infeasible at this hardware (125 GB RAM).

### Day 4-5: Pilot the chunked walker (40 sentinels) — no headline claim

**Goal**: verify the chunked walker produces the SAME LP UBs as
forward_resnet on the 9-iid cifar overlap.

Concrete deliverables:

- `research/sc_hz/goal2_chunked_pilot.py` (NEW)
- Run on the user-specified 40-iid sentinel set, K=∞ (no prune)
- Per-iid verdict + max_excess MUST match the K=∞ run on the cifar 9 iids
   we already have (sanity preservation)
- Memory chart: peak_mem distribution across 40 iids
- Re-run the cifar iid 2 STRICT-PASS audit; must remain STRICT-PASS

**Hard gate at end of Day 5**: 9/9 cifar verdicts identical to today's
pilot. If not, the chunking implementation has a soundness bug; STOP
and diagnose before any relaxation work.

### Day 6-10: Final-tail tighter ReLU relaxation (selective)

**Goal**: on the 8 cifar PHANTOMs (max_excess +0.315 to +1.95), apply a
tighter forward-only convex relaxation in the LAST 1-2 ReLU layers and
measure LP UB reduction.

Concrete deliverables:

- `research/sc_hz/relu_final_tail_kpiece.py` (NEW): k-piece linear
   relaxation of ReLU on the LAST 1-2 ReLU layers only. Method:
   - Per-coordinate, partition `[l, u]` into k=2 or k=3 pieces
   - Each piece carries an exact linear segment + a per-piece slack
   - Combine into the existing PrunedState via additional generator
       columns + interval tail
   - For k=2: 1 extra column per coordinate (sound, never widens UB)
   - For k=3: 2 extra columns per coordinate

   Soundness invariants enforced by test:
   - LP UB(k=2) <= LP UB(triangle) for every iid, every rival
   - LP UB(k=3) <= LP UB(k=2)
   - Per-coordinate range of the relaxed state contains the triangle range

- An alternative parallel track: `research/sc_hz/anderson_forward_facets.py`
   - Forward Anderson 2020 multi-neuron cuts using pre-activation bounds
       from the forward HZ (no backward, no MILP, no branch)
   - Group of 2-4 neurons per cut; choose groups by largest individual
       max_excess contribution
   - Continuous LP only

Both tracks run on the 8 cifar PHANTOM sentinels. We pick whichever
shows larger LP UB reduction at day 7 to invest the remaining 3 days in.

**Per-cut monotonicity gate (binding)**:
For every cut added: LP UB MUST not increase; per-coordinate box range
MUST not widen; nb (binary count) MUST not increase. Cut accepted only
if it strictly tightens or matches the prior state.

**Day-7 milestone**: on the 8 cifar PHANTOMs, the chosen relaxation
must reduce median max_excess by **≥ 30%** (from +0.519 to ≤ +0.36).

If milestone not reached by day 7: STOP and reassess. Either the
relaxation track is wrong, or the PHANTOMs come from a structural
geometry that tighter relaxations cannot reach.

**Day-10 final gate (Phase E kill switch)**:
Phase E ends with one of three states:

1. **GREEN**: median max_excess reduced ≥ 30% AND ≥ 1 NEW CERT or NEW A
    on cifar/tiny that production didn't have at 60s.
    → Promote relaxation to a sidecar candidate for the 1460 → 1600+
    push; keep iterating.

2. **YELLOW**: median max_excess reduced ≥ 30% but 0 NEW V/A vs
    production (e.g., the closed-PHANTOMs are all production CERTs).
    → Document as "principle-clean tighter relaxation works but
    production overlap blocks NEW V/A on cifar/tiny"; close Phase E;
    redirect to a second principle-clean mechanism (Phase F).

3. **RED**: median max_excess unchanged or grows, or new soundness
    invariants fail.
    → Close dense-conv forward-coeff entirely; the box-corner
    candidate is fundamentally not the right mechanism for
    cifar/tiny under any reachable relaxation; redirect to Phase F.

In all three cases, the 1460 headline is preserved.

---

## 4. What we will NOT do

### Not parser pilot
- 5-bench scout already established 0 NEW V/A on
   malbeware / metaroom / ml4acopf / sat_relu / nn4sys at the current
   mechanism level
- Parser extensions would convert `ERROR → UNKNOWN`, not
   `UNKNOWN → V/A`
- 1-day timebox on `nn4sys` parser is permissible as a side-project
   ONLY after Phase E day-10 gate, NOT during the 10 days

### Not blind cifar/tiny sweep
- Today's pilot already showed median max_excess +0.5 on cifar
- Re-running the same sentinels gives the same data; no information gain
- Larger sweeps only burn GPU/RAM without changing the headline

### Not third-bench scout
- The 5 scouted benchmarks each had specific reasons for 0 NEW V/A;
   adding more (cgan / linearizenn / cersyve / dist_shift / cora /
   tllverifybench / soundnessbench / yolo / traffic) would replicate
   the same pattern unless the mechanism changes
- Yolo/traffic/cifar_biasfield require conv-body parser AND memory
   work AND tighter relaxation simultaneously — out of scope

### Not backward/CROWN/PGD/MILP
- Hard ban remains (P1-P5). Phase E mechanism MUST stay forward-only,
   no gradients, no binary integers, no branch.

---

## 5. The strategic bet behind Phase E

CIFAR PHANTOMs have median LP UB of +0.5 above threshold on a classifier
with O(10-100) class scores. That is ~0.5% of the natural scale. The
relaxation is NOT off by an order of magnitude — it's off by a fraction.

The bet:
- Memory rewrite (3 days) unlocks tinyimagenet measurement.
- A modest tightening (k=2 ReLU or 2-neuron Anderson cuts) on the last
   1-2 layers could reduce that 0.5% margin to negative on a subset
   of the PHANTOMs.
- Even if only 1-2 NEW CERT or NEW A emerges, it validates the path.

If the bet fails (Day-10 RED or YELLOW), the conclusion is:
- Forward-only + box-corner + DeepZ-triangle saturates at 1460 for the
   benchmark set covered. To go past requires a structurally different
   mechanism (Phase F).

This is the honest "fail soft" plan. It does not over-invest before
the relaxation work has produced any data, and it has a kill switch.

---

## 5.5 Resource budget — G10 enforced

Per 2026-06-05 advisor reminder ("我们烧 CPU 太多"), Phase E adheres to G10:

| Resource | Cap | Action if exceeded |
|---|---|---|
| Per-iid worker peak RSS | ≤ 80 GB | self-RLIMIT_AS → clean kill of just that worker |
| Concurrent SC-HZ RAM total | ≤ 100 GB | only n_workers=1 by default; n_workers=2 ONLY when peak < 40 GB and other users quiet |
| System available RAM check pre-launch | ≥ 90 GB | refuse to launch otherwise |
| Mid-run check | available < 25 GB | pause sweep, wait for recovery |
| GPU | ≤ 32 GB per worker; ≤ 2 workers if others active | refuse to add a 3rd worker |

Practical implication for Phase E:
- All `chunked Conv propagation` work uses `n_workers=1`.
- All pilots in Day 4-5 are sequential single-iid subprocess (already the pattern we used 2026-06-05 night).
- No multiprocessing.Pool for SC-HZ ever exceeds 2 workers without explicit advisor approval.

Before any Phase E pilot, the launching script must call:

```python
import resource
resource.setrlimit(resource.RLIMIT_AS, (100 * 1024**3, resource.RLIM_INFINITY))
```

and the parent pilot driver must:

```python
import subprocess
free_gb = int(subprocess.check_output(["free","-g"]).decode().split()[12])
if free_gb < 90:
    print(f"REFUSE: only {free_gb} GB available, need ≥90 GB")
    sys.exit(1)
```

## 6. Process discipline (carryforward from G1-G8 hard gates)

Every Phase E receipt MUST pass:

- G1: LP UB monotonicity (K=small UB ≥ K=∞ UB; tests already pin this
   for `prune.py`; chunked Conv must also pass)
- G2: independent audit path (re-derive x_star independently and check
   via strict ORT replay)
- G3: production cross-check (60s budget) for any claimed NEW
- G4: strict zero-tol ORT replay for FAL claims
- G5: provenance bundle complete
- G6: `git diff --stat -- act/` empty
- G7: headline number from K=∞ only; smaller-K is sanity-only
- G8: PHANTOM not counted as V/A; watchlist only

Phase E adds:
- G9 (per-cut monotonicity): tighter-relaxation receipts must NOT widen
   LP UB / box range / increase nb relative to triangle baseline. Each
   accepted cut documented with before/after LP UB and box range.
- G10 (shared-resource budget): per-worker RSS ≤ 80 GB, total ≤ 100 GB,
   refuse-to-launch if available RAM < 90 GB. See §5.5.

---

## 7. Concrete next files (for tomorrow)

| File | Day | Purpose |
|---|---|---|
| `research/sc_hz/conv_chunked.py` | 1 | chunked Conv propagation |
| `research/sc_hz/tests/test_conv_chunked_parity.py` | 2 | tests vs forward_resnet on small input |
| `research/sc_hz/goal2_chunked_pilot.py` | 4 | re-run sentinels under chunked path |
| `research/sc_hz/relu_final_tail_kpiece.py` | 6 | k=2 piecewise ReLU on final layers |
| `research/sc_hz/anderson_forward_facets.py` | 6 | Anderson 2020 multi-neuron cuts (forward) |
| `research/sc_hz/tests/test_relu_relax_monotonicity.py` | 7 | G9 monotonicity invariant |
| `research/phase_e_day_<N>_<topic>.md` | each day | day-end mini-report |
| `research/phase_e_final_<DATE>.md` | 10 | final gate outcome |

---

## 8. Honest framing for the paper / external

If Phase E is GREEN: principle-clean +N on top of 1460; clearly attribute
to "chunked-memory + final-tail k-ReLU"; preserve forward-only stance.

If Phase E is YELLOW: write up "forward-only LP UB on dense-conv reaches
tight margins (median +0.5%); production overlap caps NEW V/A; mechanism
is sound and tight but tight-without-new-V/A is also a publishable
result for the abstract interpretation community."

If Phase E is RED: write up "forward HZ + DeepZ triangle + box-corner
saturates at safenlp-class wide-spec dense networks; dense-conv ResNet
PHANTOM margins are 0.5% but neither closes nor witnesses; mechanism
limits documented." Then propose Phase F directions.

In all three cases, the 1460 headline is the stable anchor and the
public claim does not depend on the Phase E outcome.

---

## 9. End-of-day status

```
Headline: 1460 V/A (frozen)
Phase D: closed, parked as Phase E candidate
Phase E: roadmap written, NOT started
Day-0 GPU usage tonight: returned to baseline (no active SC-HZ workers)
act/: clean
unit tests: 28/28 pass
```

Tomorrow's first action: implement `research/sc_hz/conv_chunked.py`
per Day 1 spec.
