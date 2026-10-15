# Next runtime decision: paged results, not another ownership-only patch

**Static decision, not an execution freeze.** Base
`447b301c0814b16186531717c4ef92c78ab59c04`, branch
`feat/moe-route-verification`; starting worktree clean. The approved shared-source
implementation and controls are [complete](hz_shared_source_controls_20261015.md).
This follow-up closes the next decision using the
[archive/lifetime findings](hz_shared_archive_contract_20261015.md), source review
and read-only environment inspection. No runtime code, limit or acceptance gate
changes; no source/candidate/model/solver/GPU execution.

## Decisions

| Proposed change | Decision and reason |
| --- | --- |
| Move dense buffers instead of copying them | **NO_GO_FOR_OWNERSHIP_ONLY**: the surviving vectors alone exceed the unchanged arena. Do not implement a patch that cannot close this refusal. |
| One paged logical result, referenced by production/replay/receipt | A bounded next implementation candidate; complete independent recomputation remains mandatory. Requires a separate, narrowly scoped freeze. |
| Treat sampled RSS as hard process memory enforcement | Rejected. Parent reception is outside the current sampling loop; current account has no identified writable dedicated cgroup. |
| Retry the registered full shape after report deduplication | Rejected. Work, membership, dense arena and full-source adapter restrictions remain independent. |

## Why ownership transfer alone is insufficient

The [existing arena calculation](../scripts/assess_shared_execution_envelope.py)
uses 2,065 bytes per rational slot (2,064 payload plus one presence byte)
and dense check width eight. Complete, all-present consumption
consumption retains an original objective for the zero-candidate norm, a
dual/residual factor vector and node accumulators. Even if the node vector is
moved into pullback with no duplicate, the current representation needs at least:

```text
Nmin = inputs + affine outputs + guards = 3072 + 3288 + 336 = 6696
Fmin = inputs = 3072
8 * 2065 * (Nmin + 2 * Fmin) = 212,116,800 bytes
unchanged owned-payload arena = 134,217,728 bytes
```

This excludes roots, descriptors, frames, Python objects and arithmetic
temporaries. It is a lower bound for complete, all-present eight-column consumption
**under this representation and simultaneous-map schedule**, not a universal
algorithmic memory lower bound or a bound for missing-candidate UNKNOWN paths. The prior
copying-path bound remains 322,734,720 bytes. Read-only views cannot accept
pullback updates; copy-on-write cannot remove the worst-case allocation.

Keep dense check width eight and scoped check width two distinct. A new scalar
representation or a revised resource contract would be a separate decision, not
an implicit change hidden inside ownership handling.

## Minimal proposed result interface

`SHARED_RESULT_STREAM_V1` is a proposed name, not an implemented capability.
The aim is to replace four serialized full-result copies with one logical result
and references. It does not remove either complete mathematical checking pass.

1. **Begin:** derive the ordered pair/property roster from the checked plan,
   never from the result's self-declared list. Bind source, forest, contract,
   method, invocation and ordered candidate identities. Preflight row, page,
   index, file-count and aggregate archive limits before allocating or encoding.
2. **Append:** publish checker-produced rows in exact roster order, retaining
   nonpositive and incomplete obligations as well as positive ones. Proposed
   layout: at most eight rows per existing 64 KiB page, hence 32 pages for 252
   duties. Under the previous row schema, the rows alone have a conservative
   bound of `1 + 8*(52 + 4936 + 1) = 39,913` bytes. This is **not** a complete
   page-size proof: headers, bindings, index and failure records must be bounded
   before implementation. Reject oversized admitted values; never raise caps.
3. **Seal:** only after all endpoint receipts, exact aggregation, source seal,
   row count and EOF checks succeed. Preserve the existing explicit `None`
   missing-candidate markers for complete UNKNOWN aggregation; distinguish them
   from interruption before the roster is fully processed. Partial pages are
   retained and charged but cannot establish a complete positive result. The index binds evidence
   identity, not its containing bundle hash; manifest and bundle reference the
   index without a circular hash dependency.
4. **Replay:** independently rebuild and check the source and **all endpoints**,
   then compare each newly computed exact row with the saved stream. Production
   rows are not proof inputs. Hash equality identifies bytes; it cannot replace
   recomputation. Relocated `python -I -S` checking must need no checkpoint,
   historical directory or solver.
5. **Receive:** reports and the parent receipt carry the same result reference
   and bounded summaries. Parent reception still validates the entire expected
   roster, exact signs, statuses, totals and execution identity. It is not a third
   full source/LP proof: fresh replay supplies the independent mathematical pass.

Both arms must bind the same output protocol without changing their candidate
or check widths. Preserve all current guarantee flags, UNKNOWN distinctions,
late-result rejection and online/offline separation. Every write/read, page
comparison, replay, reception, failed prefix and cleanup consumes the same
request's existing budgets. Offline checking cannot repair online timeout.

Before any performance claim, finite standard-library controls must cover:
same-candidate old/new exact results; negative/zero/partial rows; missing,
duplicate, reordered and extra obligations; wrong pair/property/source/method;
changed lower bounds **with all hashes recomputed**; alias/ownership pollution;
oversized declarations rejected before allocation; mid-page exceptions; failed
source seal; deadline at publication/reception; relocation and full cost totals.
No new scientific result follows from those controls alone.

## Memory enforcement: actual local capability, not a renamed metric

Read-only checks on 2026-10-15 (root-agent snapshot at 09:10:42 UTC) found UID
1002, cgroup v2 and membership in `/system.slice/ssh.service`. Its directory,
`cgroup.procs` and memory controls are root-owned; the read-only review found no
ACL write grant. `memory.max` and `memory.swap.max` are `max`.
`user@1002.service` is failed with an empty `ControlGroup`; a `Delegate=yes`
configuration alone is not an available delegated subtree. AS and RSS limits
are unlimited. No limit, service or permission was changed.

`RLIMIT_AS` constrains per-process virtual address space, not aggregate request
RSS; `RLIMIT_RSS` is not a modern Linux hard-RSS substitute. Do not relabel the
existing 2 GiB observation threshold as an AS quota. A future bounded receiver
helper and any AS value require their own frozen scope; the outer supervisor
must retain only bounded status/cost metadata. See the official
[Linux limit semantics](https://man7.org/linux/man-pages/man2/getrlimit.2.html).
Dedicated cgroup enforcement would require an administrator-provided subtree
and an explicit process/memory/swap contract, not modification of shared SSH
services; see [cgroup v2](https://www.kernel.org/doc/html/latest/admin-guide/cgroup-v2.html).
No new enforcement interface or administrator action is authorized here.

## Stop boundary and requested next scope

Two independent complete source checks still require at least 13,916,160 Work
against 8M; mandatory memberships remain 126,632 against 65,536. Paged result
storage does not alter either count. Complete RSS, general source/capture and
native proposal contracts remain open. The scientific G1–G6 status is unchanged.

The smallest useful next approval is **only default-off paged result publication,
reference-based reports, complete replay comparison and reception binding, with
fixed-candidate/standard-library controls**. First freeze its full metadata
bounds and finite controls; retain all checks, quotas and 300 seconds. Exclude
dense storage redesign, memory-policy changes, native sources, real models and
GPU. Do not implement the rejected ownership-only patch or rerun full shape.

Validation for this decision: independent read-only reviews of lifecycle,
ownership and supervision; exact integer recomputation of the new lower bound;
24 existing static ledger and 13 navigation regressions pass, as do exact saved-
report equality and current-document links. The repository layout check preserves
619 frozen source bindings and all 4,632 historical handoff lines.
These validate the decision and historical bindings, not a new runtime or proof.
