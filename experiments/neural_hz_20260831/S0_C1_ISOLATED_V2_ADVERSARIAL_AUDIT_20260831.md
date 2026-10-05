# S0-C1 Isolated V2 Adversarial Audit

Recorded on 2026-08-31 on branch `redu-hz` at repository commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. This record concerns only the
default-off experiment implementation under `experiments/neural_hz_20260831`.
It is not imported by production and creates no formal or E0 score credit. The
formal baseline remains exactly 1,870/2,413.

## Version boundary

The first production-shaped V2 submitted for audit had source SHA-256
`c303f15ccd778379daa66af0462d603d2b4d05c132512267c661b4af6fd883a8`.
Its original focused test SHA-256 remains
`3665a3afcb5405f426dd6aae97bc24538f122568d7a7c0279f212262a5f2f2d2`.
The audit did not accept that source version for production integration.

The hardened isolated source has SHA-256
`5f7963fdec0924148ee683bf87289f4cd2bb8b5de5c46120f44de5ab3823c646`.
The separate adversarial test source has SHA-256
`6f6f5c0e40d2b61fcd5f8a2a79eaa13042b2a9c757f2e1e513f621a338d30c0b`.
No production source, Trial 9 source, historical archive, result ledger or
formal verdict was edited to obtain this version.

## Exact operator result

The group-intersection construction remains the exact real-algebra map

```text
H[t,q,O,I] = sum over ascending global m in J
              B[O,m,t] * sigma[m] * A[m,I,q].
```

Actual rank-one scalar products, the sum over group intersections and the
registered gate formula are checked for equality during compilation. The
original 23 V2 tests cover ordinary, aligned and misaligned groups, both
depthwise multiplier directions, asymmetric geometry, batch two, channel and
outer masks, signed/zero scales, bias propagation and both left-compose paths.
The new seeded audit adds a deterministic sweep over the first 64 shuffled
valid-group configurations; 58 have nonempty two-layer geometries and all 58
match explicit CSR composition. An independent read-only audit also exercised
80 seeded mixed group/depthwise/geometric cases with no mapping mismatch.

## Defects found and closed in the isolated source

1. The general sparse `left_compose` path previously formed the SciPy product
   before checking `max_nnz`. A reproduced case gathered 42 support nnz, then
   formed 4,800 result nnz before rejection; the output-row count could make
   this arbitrarily larger. V2 now sums a conservative contribution upper
   bound from gathered row widths and rejects with
   `left_product_contribution_limit` before `@` is invoked. A monkeypatch makes
   any premature sparse multiplication fail the test.
2. Two distinct open reservations could interleave. Rolling back the first
   restored its old peak even while the second remained live, so peak could
   fall below live transient bytes. A transaction now permits exactly one open
   reservation, binds it to a private owner token and validates all commit
   invariants before publishing state. Independent requests use independent
   transactions.
3. A `KeyboardInterrupt` or `SystemExit` after reservation bypassed the old
   `except Exception` cleanup and leaked work, reserved keys and live transient
   bytes. An unconditional `finally` now rolls back every unclosed reservation,
   including `BaseException` exits. The interrupt is still re-raised.
4. `ImplicitConv2DOp.content_key` is fixed at construction while its private
   kernel and row-mask arrays are currently writeable. Mutating a kernel after
   a first build therefore reused a stale cached descriptor. V2 now copies the
   current finite kernel/mask semantic payload, derives typed binary snapshot
   keys from those copies and compiles the same snapshots. The snapshot buffers
   are explicitly charged to the controlled-transient estimate. A regression
   mutates a weight while the source key stays stale, then proves a new V2 key,
   no descriptor reuse, preservation of the old descriptor and equality of the
   new descriptor with the changed explicit CSR reference.
5. Commit formerly published dictionary/set entries before all invariant
   checks, while rollback did not remove a possible partial publication. V2
   now prevalidates ownership, active identity, descriptor content and reserved
   keys; rollback also removes any partial cache publication before restoring
   the exact counter snapshot.
6. Reservation formerly mutated counters before allocations into its private
   set/dictionary had all succeeded. An injected allocation failure could
   therefore escape before the caller obtained a rollback token. The reserve
   mutation is now locally guarded and restores the exact prior snapshot on
   every `BaseException` before re-raising.
7. Earlier transactions recorded emission keys and work without constructing
   or retaining the corresponding CSR artifact. This was false accounting.
   V2 is now explicitly descriptor-only: emission is prospective and deferred,
   and no support key is committed. A real materializer must reserve around
   actual construction and either retain the identical artifact or charge
   every execution.
8. A preview followed by reserve could use stale cumulative resident/physical
   state. Reserve now repeats the authoritative quote while holding the
   transaction lock, includes every already committed descriptor in the
   physical-after operand and rejects if an intervening descriptor removes the
   strict reduction.
9. Reservation fields formerly admitted negative/bool-like values and a
   zero-work request could be mistaken for a cache hit. All integer budgets
   now use strict nonnegative validation, `needs_compile` is explicit, and the
   reservation is an opaque frozen same-owner token.
10. A successful `BuildDecisionV2` was once allocated after descriptor commit.
    Injected `MemoryError` could therefore report failure after publishing
    state. Both fresh and cached success decisions are now constructed before
    commit; failure returns a controlled rejection with the prior transaction
    snapshot intact.
11. Cleanup briefly tracked closed tokens in a growing set, so cleanup itself
    could allocate and mask the original exception. Rollback is now
    idempotent without a closed-token allocation. Commit also checks resident
    bytes and requires
    `gate_formula_products == actual_contraction_products == charged_delta`;
    undercharge, overcharge and stale cache quotes all reject.
12. A final asynchronous handoff gap remained between a successful
    `reserve_descriptor` return and assignment of its returned token in
    `try_build`. The caller now creates and retains a request token before any
    ledger mutation, passes it into reserve, and runs
    `rollback_if_pending(request)` in `finally`. A reproduced
    `KeyboardInterrupt` at exactly that handoff restores the prior snapshot.
    Cleanup tests also prove that a nonpending request never rolls back a
    different active reservation.

## Ledger boundaries that remain explicit

`exact_selected_logical_nnz` and `exact_full_logical_nnz` are exact structural
counts after colliding identical spatial offsets. Consistent with the existing
operator protocol, they include structural positions whose kernel, scale or
accumulated numerical value is zero. They are safe materialized-nnz upper
bounds, not actual canonical CSR nnz. Actual CSR nnz is recorded separately.

The frozen 64 MiB resident limit is per unique descriptor. The transaction
tracks cumulative descriptor-resident bytes and includes them in its locked
physical-after quote, but it does not prove the complete reachable program
state or enforce the C2 planner's cumulative whole-request resident limit.
Production integration must include every already cached unique descriptor in
the caller-supplied whole reachable state and independently require strict
reductions in both resident bytes and entries; otherwise the physical
comparator is unproven and the candidate must reject.

The transaction's cumulative work is descriptor contraction plus the current
request's prospective emission. It is not a ledger for multiple real
emissions. The future materializer must charge cumulative actual emission and
enforce the 256M complete-request gate around the real artifact. Reverse-prefix
identity, actual canonical CSR nnz and support-specific cache ownership do not
exist in V2.

The controlled-transient ledger covers owned NumPy/CSR buffers and the new
input snapshots. Python dictionary/list/object allocator overhead in row
materialization is not a strict RSS bound. The registered real-network worker
must therefore report the numeric ledger and measured peak RSS separately, as
required by the comparator amendment. Neither number may substitute for the
whole-state strict-reduction gate.

## Reproduction

Using the repository interpreter:

```text
python -m pytest -q \
  experiments/neural_hz_20260831/test_composed_conv2d_stencil_candidate_v2.py \
  experiments/neural_hz_20260831/test_composed_conv2d_stencil_candidate_v2_adversarial.py
39 passed

python -m pytest -q \
  experiments/neural_hz_20260831/test_composed_conv2d_stencil_candidate_v2.py \
  experiments/neural_hz_20260831/test_composed_conv2d_stencil_candidate_v2_adversarial.py \
  experiments/neural_hz_20260831/test_s0_c1_combined_v2_integration_prototype.py
60 passed

python -m pytest -q \
  experiments/neural_hz_20260831/test_s0_c1_pure_tail_planner_prototype.py \
  experiments/neural_hz_20260831/test_composed_conv2d_stencil_candidate_v2.py \
  experiments/neural_hz_20260831/test_composed_conv2d_stencil_candidate_v2_adversarial.py \
  experiments/neural_hz_20260831/test_s0_c1_combined_v2_integration_prototype.py
74 passed
```

One preliminary invocation through the standalone `pytest` executable used a
different interpreter and failed during collection with `ModuleNotFoundError:
act`. It executed zero tests and changed no source or result. Re-running through
`python -m pytest` produced the results above; this environment failure is not
counted as a candidate failure or a passing test.

## Advancement status

The hardened descriptor V2 has passed the current isolated operator,
transaction and adversarial gate. The combined adapter remains synthetic and
retains explicit no-claims for real emission, reverse-prefix artifacts,
whole-state roots, old-root release, strict RSS and mutable-source publication.

Independent graph lineage proves that frozen S0-C1 is a structural zero-hit at
Tiny iid143 ReLU36, so S0-C1 is closed without a production attempt. Only the
separately preregistered S0-C2 residual-distributive rule may reuse this core.
It has not passed the real target, same-structure shadows, any affected-family
replay or the full 2,413-case replay. Trial 9 still owns the nine-file source
freeze. Formal and E0 gain remain zero.
