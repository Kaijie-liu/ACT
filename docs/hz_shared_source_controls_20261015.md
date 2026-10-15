# Shared source, complete checking and single-owner archive

The default-off implementation passes **156 controls/regressions**, plus **four
static cost-ledger tests**. The [protocol](hz_shared_source_protocol_20261015.md)
was pushed at `93a230630` before implementation. The
[result](hz_shared_source_controls_20261015.json) binds tested files; the
[first failed suite](hz_shared_source_controls_attempt001_20261015.json) is retained.
No native optimizer, numerical library, checkpoint, dataset, real model or GPU
ran. No dependency, acceptance threshold, resource ceiling, 300-second maximum
request budget or production default changed.

**This completes the approved interface/control stage, not full-size admission.**
The first complete source bank alone requires at least 13,926,024 Work on the
registered eight-expert shape, above 8M. Expanded memberships remain 126,632
above 65,536. Full successful Work/RSS uppers remain open.

## Implementation and guarantee

- [Shared source](../scoped_source/shared_source.py) stores binary64 rows in an
  immutable request-local bank. First-affine nodes reference exact graph/layer/
  row identities and the checked input basis; only the at-most-two compensation
  coefficients are explicit. This is not a cross-request cache.
- [Independent checking](../scoped_source/check_shared_source.py) reads the source
  anew and reconstructs every equation, zero position, factor, center,
  compensation, ReLU enclosure and guard. It never calls the builder. Even a
  numerically zero source alias outside the first-affine role is rejected.
  Complete inventories and final source reseals remain mandatory.
- [Paged format](../scoped_source/shared_format.py) shares complete ordered
  root-ID sequences, not duties. Every expanded use is still checked and charged
  against the old ceiling. All page lengths, hashes and section extents must
  match; missing, extra and corrupt pages are refused. No whole-forest join is
  performed in production or portable intake.
- Both new consumers use the same bounded objective-root descriptor. Their
  residual/pullback algorithms remain different. Candidate width eight,
  pair-scoped check width two, both proposed/zero roles, both norms, and complete
  pair/property aggregation remain. Shared/legacy method and forest identities
  cannot be interchanged; production and portable replay use the same method.
- [Single ownership](../scoped_source/shared_archive.py) transfers only this
  request's newly generated fixed source, on the same filesystem, without copy
  fallback. PID, Work, parent, inode, inventory, links, bytes and remaining member
  allowance are checked. Failed prefixes remain counted; no historical source
  is deleted. Production objects release before fresh online replay.

These checks concern a captured real-arithmetic declaration. Hash and terminal
audits do not prove deployment floating-point execution. No historical source
gap, external comparison or real-model certificate claim is upgraded.

## Results and failures

Four frozen sources (`mixed`, `deep`, `half`, `guarded`), two candidate profiles,
two consumers and old/new representations yield **32 configurations and 328
endpoint records**. Proposed/zero bounds and complete aggregation agree exactly.
Mixed/half stay positive; deep/guarded remain nonpositive UNKNOWN.

| Source | Old/new forest payload bytes | Logical/stored new memberships | Packed source bytes |
| --- | ---: | ---: | ---: |
| mixed | 9,610 / 9,656 | 81 / 65 | 720 |
| deep | 14,675 / 14,438 | 93 / 77 | 840 |
| half | 6,734 / 6,699 | 39 / 31 | 192 |
| guarded | 4,592 / 4,690 | 33 / 25 | 288 |

Alias metadata makes two tiny payloads larger. These sizes exclude the index,
source, candidates and code. Source packing adds work while objective-root
access removes other work. This joint implementation is not a single-factor
timing study; owned arena observations are not RSS or speedup measurements.

**29 new supervised invocations** comprise eight normal configurations, one
relocation preparation, one completed `UNKNOWN_MISSING_EVIDENCE`, eleven
expected errors (including Work exhaustion), and eight expected timeouts. All
terminal audits pass. Forced deadline controls use shorter timeouts, never a
larger budget. Partial, late or wrongly bound evidence is never accepted.
Relocated `python -I -S` checking succeeds with the old source path unavailable,
without checkpoint/data/solver access. That separately charged offline check
cannot rescue online failure.

The first 154-test suite failed in two test-harness locations: `Receipt.data`
should be `Receipt.record`; a Flatten mutation was actually unchanged. The
mutation helper now requires a real change. No mathematical acceptance gate was
weakened. Review also prompted sealing source-bank bindings, rejecting non-affine
aliases, source-directory symlink replacement and unindexed pages, and checking
remaining member allowance before ownership transfer.

An explicitly separate **equation-only** control supplements, not replaces or
extends, the endpoint roster: `deep(center=(1, 2^-1000), radius=1/4)`. Input
scales are `1/8, 1/4`; first-affine compensation has error `2^-55` and lift
`2^-1000`. Old/new equations agree; deleting or negating either tail is rejected
after rehashing. Three read-only AI reviews examined equations, lifecycle and
supervision. They do not replace independent human review.

## Actual resource and memory contract

The [static ledger](hz_shared_source_cost_20261015.json) and
[calculator](../scripts/assess_shared_source_cost.py) bind current code and two
frozen dimension inventories. They construct no source, candidate or worker.
For `P=6,961,368` affine scalars and `R=3,288` affine rows:

`Snapshot Work >= R + 2P = 13,926,024 > 8,000,000`.

Packing and finite-value scanning each visit every scalar. Replay, equations
and consumers are additional. Three complete banks alone would cost 41,778,072,
but earlier gates prevent that execution. This is an implementation-specific
conditional lower bound, not a measured run, universal bound or proposed quota.

For tensor bytes `T` including center `X`, forest pages `S`, index `G` and tiles
`C`, successful online logical reads include `17T+2X+9(S+G)+9C`; parent reception
contributes `4T+4(S+G)+4C`. Metadata, code, imports, in-memory copies/hashes and
offline audits are additional. Retained storage has **one** source `T+H`, plus
forest/index, candidates, code, manifests, results and supervision records.
Deduplication does not halve all reads.

| Lifetime | Bounded payload and explicit exclusions |
| --- | --- |
| Source bank | `8P` packed bytes; row inventory capped; Python objects additional |
| Per-row packing | Up to 4,096 weights plus bias; fill and immutable copy coexist |
| Construction/check | Bank, DAG, pages, metadata, norms and row/Fraction temporaries coexist |
| Page encoding | Immutable pages plus one fill of at most 65,536 bytes; not constant-memory streaming |
| Metadata intake | Raw header at most 8 MiB; parsed objects additional |
| Consumer | Exact arena plus graph, scales, roots and current candidate values |
| Online replay | Production references released first; independent bank/check rebuilt |

Reference release does not prove allocator/RSS release. The 128 MiB aggregate,
4,096 members, 1 MiB worker file limit and sampled 2 GiB RSS refusal are distinct.
No full process-RSS or successful request-Work upper is closed. Work counters
exclude much hashing/I/O/GC, which remains charged in wall time. OS-delayed
cleanup may overrun and must remain failed, not a late accepted result.

**Stop at this completed fixed-control stage.** A next design needs a complete
checked-traversal/membership contract preserving every check without current
repeated materialization. Do not rerun the full shape, raise quotas, remove
checks or infer real/native/GPU admission. G1–G6 and scientific claims stay open.

## Reproduce

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S -c 'import sys;sys.path.insert(0,"/data1/Kane/MOE/ACT");from scripts.run_shared_source_controls import main;raise SystemExit(main())'
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/assess_shared_source_cost.py --check-report docs/hz_shared_source_cost_20261015.json
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_shared_source_cost
```
