# C23 complete sparse append-ownership diagnostic — PASS

2026-09-11, branch redu-hz, commit
f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. Production candidate hash remains
15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75.
Goal ACTIVE. This turn is PROGRESS in a reusable exact representation metadata
primitive, not a solved-instance gain. Formal1870/2413 and separate E061/400
remain unchanged. C21's failed complete-entry gate remains CLOSED.

## Measured result

Official tests978 passed7.43s (50 new +928 inherited); supervisor31.179037949s,
worker exit0, worker21.628675390s. No timeout, source drift or provenance drift.
All output artifacts are exclusively created in
results/c23_sparse_phase_overlay_20260911_v1/. No job remains active.

- All243162 old MAIN words were derived independently from the complete
  successful old HZ. The base is shared by identity and never modified/copied
  to make a dense phase-update vector.
- All200 new EQ plus400 new INEQ rows were inspected:1000 continuous nonzeros
  in total, but only200 reference the tracked old MAIN prefix. They affect200
  distinct MAIN columns on this target. Multiple new rows per MAIN remain
  supported and tested; the implementation does not assume the target's
  singleton-per-column property.
- The exact additional persistent update metadata is200 uint64 entries,
  **1600bytes**, instead of243162 int64 entries/**1945296bytes**. Saving is
  **242962 entries /1943696bytes** for THIS metadata component only.
- ALL243162 random-access queries and ALL243162 streamed words matched an
  independent complete post-HZ incidence oracle, including untouched/erased
  factors. No positive-only or selected subset was used.
- The complete stream found102571 structural single-consumer definitions and
  ALL268 C15 unit-pair columns. The sealed C14 table was read only after
  discovery and matched exactly. No splice was executed.
- Both complete successful C10 dictionaries, original graph arrays, source
  operators/HZs, lineage maps, binary phases, predicates and shared frame
  remained. Portable source identity and complete post-HZ digest were unchanged.

The formula is exact integer addition: base_word+k*2^40+sum(new_UIDs), with
fresh disjoint row-UID ranges, canonical incidence and checked sum/count
domains. One40-bit event stores(MAIN-index,new-UID). It changes NO HZ equation
or reachable set and does not relax to Zonotope/CZ/box. Count/sum or a
self-sealed digest is not a source-membership proof; tests explicitly reject
that interpretation. All queries were enclosed in a full source-bound proof
transaction using the independently archived C10 DAG/quotient/box proof.

## Work and memory, without a free-integration claim

Standalone construction1972696 =1945296 full base validation +27400 event
build. Event build comprises16800 appended-row scan,2400 packing,200 array,
6400 stable sort and1600 structural validation. The source C21 ledger had
only140566 whole-work headroom: this standalone build is NOT an affordable
append to it. The27400 event-only number cannot be used alone until a new
implementation genuinely supplies a checked immutable base without repeating
the full validation. No such integration is implemented here.

Whole diagnostic work242702342 <=256000000. Overlay branch40883416
<=200000000, charged additionally to the same whole pool (not omitted).
The rest is independent full proof/oracle work, explicitly itemized in
overlay_audit.json. Two full incidence scans cost158939308; UID metadata
29809376; all random searches35015328 plus2400 event sums; complete streamed
queries3890592 plus2400 event sums. All discovery/lookup/slice/array work is
included. These are offline diagnostic costs, not a new live-generation cost.

Measured complete construction/proof16.306005868s; entryRSS1675657216bytes,
lifetimeHWM before/after1981730816bytes. Conservative HWM-minus-entry growth
306073600bytes (NOT zero merely because the HWM did not increase). Traced
peak29583214bytes plus231616bytes tracer metadata, current2095918bytes.
Both unchanged1GiB transient gates PASS. Absolute maxRSS1935284KiB includes
source loading/authentication, separately timed4.642436015s. CPU1/GPU0,
16GiB address-space and240s wall limits stayed unchanged.

The independent full post-word oracle and full UID lookup are temporary proof
buffers and die only after the complete proof function returns, as in C21.
Both full source dictionaries remain reachable. No claimed whole-state saving
comes from releasing a required oracle, native cache or original source.
sparse_overlay.npz stores the shared old base plus200 events; it does not
store a newly copied dense post-word vector. unit_columns.npz is separately
labelled diagnostic output.

## Scope and next gate

No original-expression generator, new native ReLU, unit splice, proof-state
retirement, native ingestion, terminal solver, witness, full live-path pass,
CIFAR expansion, shadow or family replay occurred. Tests are NOT evidence of
2413-case/per-family retention for a promoted candidate. No formal score gain.

Replacing ONLY C21's dense phase copy in its OLD full-entry ledger gives
54398622, still1969822 above52428800. If one additionally retires all four
new node fields and adds the measured11038-entry C22 run index, the old-ledger
arithmetic is52240284,188516 below the unchanged bound. This remains a DESIGN
calculation, not measured compact complete state: checker receipts, proof
binding, retained metadata and actual ownership must all be counted afresh.

Next design constraints are in C23_INTEGRATION_HANDOFF_20260911.md. Neither
this component success nor the arithmetic authorizes rerunning C21 or the
closed C9/C10 terminal versions. All historical/production files are untouched.

## Artifact identities

Input portable identity:
4ca034b0ce88eae8bdf6b252e60124bd6b20dccc4dafe53d9f3940f8543c7423.
Post-HZ content:
82df62f1233ca8f34b5163ee3fafa88ae0afc65a9f36fe5b1b7f3b022cca2367.
Base raw words:
a000173017dc1a4df790982cd0abe9d0fa29126a1a6c8d7480b2d8f87e1a7794.
Event raw words:
ac42aef7f4d2acd7c645eb07bcb6203ad304622b932ec7243d9febcef4373a41.
Independent post words:
73b6e36934faf5b6e35090624f1985f8081fb3c1801cf4e219679e741355f568.

Artifact SHA256:

- preregistered.json d194bf80abb50cb9aa0574b87ef7dfa35ac5d357e9def4c774fb9a82c88f50f1
- sparse_overlay.npz d3a4baff55e71199762f05b08571a83cec040b1b0c4ce0222b2c602b6a80119f
- unit_columns.npz 417a7a85deb9d93d300791ec0135c721f4a585acd8d7c6879c14c940b034fba2
- overlay_audit.json 6a2fe75b47018d17dcf463e903812b8c224c0ccb13d32406d3f812349824c76b
- result.json 68b3edf0cfa0aab38017f1fa8ade21dde8c3b5546149f47176a38467ebf42aef
- exit.json 3db058e79bffa869dbaa03c7b5fb78d2810e4bde904e23808e78b85cd1b812dd
- events.jsonl 53e2ed9b8877b9352c531d22aed4df2b5fcaaa934fcfb0f4ff7bd11305c44cdc
- tests.log 677b182d387cbefecd24e0160d071324ff7aaf161cebfefbd2cbef2b3722c03e
- worker.log 4f11fa4cd907d1b201a136655deb1a886bef32127319bd1311f16f409c6f2901
