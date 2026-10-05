# C5 live transaction and first actual ReLU36 prefix

This is an additive audit, not a score or default change. Branch `redu-hz`,
base commit `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`; the modified candidate
is identified by each run's complete inherited `source_sha256` map, not by
that Git commit alone. Historical HyZor data and the five earlier seals remain
untouched. The prior reporting turn was a status report, not a new experiment.

## Closed versions and admitted evidence

- Live transaction V1 rejected the real model's OrderedDict state before
  candidate construction. Its failure and 41 passing tests remain in
  `results/c5_live_transaction_20260905_v1/` and its exclusive evidence file.
- Live V2 explicitly registers OrderedDict entries and version metadata;
  unknown attributes still reject. Its 44 tests pass. This is an accounting
  correction, not a numerical retry or a changed physical comparator.
- Integrated prefix V1 genuinely produced HZ at ReLU20, but selected C5 for
  the empty ReLU28 frontier and rejected at Conv25. Later loop/snapshot labels
  28/36 did NOT contain corresponding HZ. This version is closed, not a hit.
- Integrated V2 makes the native empty-row transfer a preselection condition.
  It requests no implicit operator rows, retains all predicates and factor
  identities, and leaves the compiler's zero-denominator quarter-work gate
  unchanged. No selected/poisoned C5 island can fall back to the old path.
  Its 49 tests pass. The native ReLU and cache-publication functions themselves
  are unchanged and execute in the fresh prefix.

## Actual live qualification at Conv17 / ReLU20

Evidence `evidence/c5_live_transaction_20260905_v2.json`, SHA-256
`5b2af877318515dd7daa06eb0c79181d13c3dd71a351c6cb4b2cbeddc91fde94`.
All 363 registered numeric roots include transfer-function state, facts,
constraints, graph caches, frame/slot metadata, concrete model registered
parameters/buffers, and input/output specifications. All incoming root
fingerprints remain unchanged. Arbitrary Python/library heap ownership is
not claimed; Python shallow metadata is reported separately and non-additively.

The native probe requests 318 rows and rejects phase-selective admission
(7,711 positive generator nnz versus threshold 25,875), so the unchanged native
follow-up requests all 1,097 non-negative rows. Both stages and both source
branches have byte-identical retained operator coefficients and all nine HZ
numeric fields versus the original ordered scalar oracle. Continuous/binary
widths, predicates, frame and exactness flags agree. The cumulative work is
80,639,104 channel products, within the frozen per-branch and whole-island
caps, and every stage/branch passes the quarter-work test.

| Live boundary | Candidate numeric bytes | Fixed expanded comparator bytes |
| --- | ---: | ---: |
| Entry | 73,461,928 | 418,946,740 |
| Probe result retained | 80,059,832 | 425,544,644 |
| Follow-up result retained | 94,876,964 | 440,361,776 |
| Both results retained | 101,474,868 | 446,959,680 |

Every boundary strictly reduces both bytes and resident entries with identical
consumer/cache policy. The comparator is `phase_selective_expanded_v1` for
the combined representation, NOT the already implicit Trial 9 implementation.
This is not a standalone C5 persistent-storage reduction against Trial 9.

Both candidate stages were measured before scalar/expanded oracle allocation.
The maximum traced allocation peak is 85,189,294 bytes plus 240,896 tracer
metadata bytes; conservative resident-growth upper bounds are below 31 MB.
Both pass the 1 GiB construction gate. The complete diagnostic, including test
oracles, peaked at 1,667,956 KiB and stayed inside the 16 GiB process cap.
Tracing timings (23.79/35.00 seconds) are not runtime speed-gate results.
Functional return/no-publication guards and injected faults are tested; this
live qualification deliberately stops before ReLU/cache publication.

## Fresh integrated V2 prefix

`results/c5_integrated_prefix_20260905_v2/result.json` SHA-256
`a42456761ac6dd925bf89b1fc2e4537b6f010294ac8ea28fe233ab3e20597081`.
Source/provenance drift is false, test and worker exits are zero. Native
propagation to the intentional ReLU36 stop took 6.039902 seconds; supervisor
wall time was 11.096038 seconds. This includes observation and is neither an
end-to-end solve nor a controlled speed comparison.

The actual pickles were independently reloaded, their seals verified, and all
nine numeric fields checked finite. Each following HZ is exact with frame 1:

| Actual cached layer | Continuous | Binary | Equalities | Inequalities | Gc nnz |
| --- | ---: | ---: | ---: | ---: | ---: |
| ReLU20 | 11,516 | 1,054 | 1,054 | 2,108 | 1,139,730 |
| ReLU28 | 11,516 | 1,054 | 1,054 | 2,108 | 0 |
| ReLU36 | 11,548 | 1,070 | 1,070 | 2,140 | 35,924 |

ReLU28's zero output generators do not remove its binary factors or predicates.
The Conv33 island executes two native 38-row stages across four source terms,
including this zero source. Its cumulative work is 9,939,968 products; all eight
stage/branch quarter-work gates pass. ReLU36's sealed pickle is
`f9332062fb6f85217258fdc00db47efc206647064917f16192b847136adfcd5b`.

The result remains UNKNOWN because terminal layer 79 is intentionally not
executed. No terminal witness, new CERT, or ADV is claimed. The ReLU20 live
owner/oracle qualification cannot substitute for the corresponding ReLU36
boundary proof. Those checks precede advancement to ReLU63/71/terminal.

## Remaining gates and archive status

Formal score stays 1,870/2,413; separate E0 stays 61/400. Neither full retention
replay nor four-concurrent no-regression is complete. Candidate remains opt-in.
All four runs have terminated; raw tests, logs, exits, source freezes, evidence,
snapshots and intentional hard-linked staging names are retained. No existing
production source or previous archive is changed by this audit. The additive
checkpoint binds these records without rewriting the earlier five manifests.
