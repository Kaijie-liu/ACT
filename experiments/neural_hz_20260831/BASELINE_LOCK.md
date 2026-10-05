# Neural-HZ Formal Baseline Lock

Locked on 2026-08-31 for the `redu-hz` structure-by-structure research branch.
All referenced `/data1/Kane/HyZor` artifacts are read-only.

## Formal headline and promotion invariant

The only active 13-family baseline is **1,870/2,413**:

- 1,063 CERT;
- 807 concretely validated ADV;
- 269 UNKNOWN;
- 274 TIMEOUT; and
- 543 unsolved in total.

Every promotion must preserve the solved count of each row below, every old
CERT and validated ADV, and zero invalid ADV. UNKNOWN/ERROR/TIMEOUT and
disconnected capability results count as zero gain. The headline changes only
after a complete 2,413-case replay has no per-family regression and at least
one new CERT or concretely validated ADV.

| Family | Total | CERT | Validated ADV | Solved | UNKNOWN | TIMEOUT | Remaining |
|---|---:|---:|---:|---:|---:|---:|---:|
| safenlp | 1,080 | 432 | 647 | 1,079 | 1 | 0 | 1 |
| sat_relu | 100 | 50 | 50 | 100 | 0 | 0 | 0 |
| malbeware | 150 | 131 | 19 | 150 | 0 | 0 | 0 |
| metaroom | 100 | 94 | 1 | 95 | 0 | 5 | 5 |
| acasxu | 186 | 86 | 34 | 120 | 62 | 4 | 66 |
| linearizenn | 60 | 39 | 1 | 40 | 18 | 2 | 20 |
| relusplitter | 220 | 43 | 2 | 45 | 98 | 77 | 175 |
| dist_shift | 72 | 63 | 7 | 70 | 0 | 2 | 2 |
| tllverify | 32 | 5 | 12 | 17 | 15 | 0 | 15 |
| cgan | 21 | 5 | 8 | 13 | 8 | 0 | 8 |
| cersyve | 12 | 5 | 6 | 11 | 1 | 0 | 1 |
| cora | 180 | 20 | 20 | 40 | 9 | 131 | 140 |
| vit | 200 | 90 | 0 | 90 | 57 | 53 | 110 |
| **Total** | **2,413** | **1,063** | **807** | **1,870** | **269** | **274** | **543** |

## Authority and machine provenance

The unique active per-family table that exactly matches the locked vector is:

- `/data1/Kane/HyZor/VMCAI_2027___Kaijie_Guanqin/tables/verdict.tex`
- SHA-256:
  `de13942115a2a1ba79eb3af82a4c435a4919492c044d174cfdef6be1c41429d0`

There is no single archived 2,413-row CSV with a frozen manifest hash. The
machine-reproducible authority is therefore the following composite tuple:

1. generator
   `/data1/Kane/HyZor/VIT13_PAPER_UPDATE_20260828/generate_figures.py`,
   SHA-256
   `971f8c8c31a40a5c2b732338d9219572c9350e05b61ff4a496a3d4efe15222b0`,
   which asserts 1,870;
2. 12-family, 2,213-row overlay
   `/data1/Kane/HyZor/DIST_SHIFT_K3_HEADLINE_UPDATE_20260822/_DETAIL_K3_OVERLAY.csv`,
   SHA-256
   `05ab7e4b09c3285bb8d5f1d09ae2876fee7a5933b68c0a6af54b8e6bf951544a`,
   containing 973 CERT, 807 ADV, 212 UNKNOWN,
   and 221 TIMEOUT;
3. strict-wall 200-row ViT source
   `/data1/Kane/HyZor/vit_hz_legacy1_100s_20260826/consolidated_strict_100s.csv`,
   source commit `14d558953`, SHA-256
   `8d52519c65345c04639a2abc345be462c3e62f8edc810215151013321431c8c9`,
   containing 90 CERT,
   57 UNKNOWN, and 53 TIMEOUT; and
4. aggregate
   `/data1/Kane/HyZor/VIT13_PAPER_UPDATE_20260828/summary.json`,
   SHA-256
   `90b7f405b140add3c9461cb22cd5e483decc989261a36df45026e968c0921934`.

The dist-shift source set additionally has a checksum manifest at
`/data1/Kane/HyZor/CONFIGURABLE_FUSED_TIGHTENING_20260822/analysis/MANIFEST.json`
with SHA-256
`c3884d513798e04a9b21b3d87f455642c0f02f1f82a23aca32d0a272a407bbb3`,
covering 922 source, summary, log, and corrected-source files. The ViT strict
90 directory has no equivalent complete checksum manifest; this is why the
authority must remain the composite tuple rather than an invented single-file
2,413-row authority.

Concrete ADV provenance is also composite. The 2,213-row
`/data1/Kane/HyZor/SUBMISSION_R2/paper/artifact/witness_validation.csv`
has SHA-256
`9ea1caf1931d49baaa09dadca8a23918cdc203378b09f13238a92720a5a2ed5b`
and contains 805 historical ADV with `replay_checked=yes` and
`replay_passed=yes`; 800 of those are outside dist-shift. The active K3 overlay
replaces the old five dist-shift ADV with seven witness-gated ADV, as documented
by
`/data1/Kane/HyZor/CONFIGURABLE_FUSED_TIGHTENING_20260822/VALIDATION_REPORT.md`
(SHA-256
`6e7d46d7cc03183b5817f6cc30cb5f6beaf732734d8ace1b8769c2e3daceb7d5`).
Thus the active 807 consists of 800 replay-passed non-dist-shift witnesses plus
seven K3 witness-gated dist-shift examples. The older witness CSV under
`VMCAI2027/artifact` comes from the unsound historical dist-shift source and is
not admissible.

Before a full promotion replay, these hashes must be freshly checked and copied
into the replay manifest. They identify the frozen sources without copying or
changing the archive.

## Explicit exclusions

- The 1,871 claim in
  `/data1/Kane/HyZor/ACT_WIKI_LOCAL_20260827/HyZor.md` uses a later ViT speed
  run with 91 rather than the active strict-wall 90 and is not this baseline.
- The non-strict `summary.json` in the legacy ViT directory also reports 91;
  only `summary_strict_100s.json`/the strict consolidated source is admissible.
- Historical Stage-II `dist_shift` CERT=70 contains six known conflicts. The
  active row is the K3 result, CERT=63 and validated ADV=7.
- The current dist-shift decoder was validated for its continuous active input
  columns. Any future use on boxes with interleaved zero-radius coordinates
  must pass explicit active-column indices or fail closed; this does not alter
  the frozen snapshot but is a mandatory new-candidate guard.
- The old 12-family 1,776 table under `VMCAI2027_overleaf` is superseded.
- CIFAR100 and TinyImageNet are not members of these 13 rows. They are large
  CNN structure scouts/guards and independent target families for Neural-HZ
  innovation. Their outcomes cannot alter 1,870, and neither their 59/400
  ledger nor any disconnected result may be merged into this table.

## Independent non-13-family ledgers

The research goal also covers benchmark/network families not represented by
the active 13-family VNNCOMP table. Each such family receives its own frozen
instance universe, baseline verdict vector, concrete-witness protocol,
per-family zero-regression gate, configuration/commit hashes, and complete
replay before an external score is promoted. External gains are reportable
research results and may justify migrating a structural rule into the formal
13-family cohort, but they are never arithmetically added to 1,870/2,413.

TinyImageNet and CIFAR100 are the first external large-CNN targets. Their
current isolated trials remain capability records until the corresponding
full independent baseline and promotion manifests are frozen. The historical
59/400 large-classification ledger is kept separate as well; disconnected
results remain zero formal gain in every ledger.

There are two incompatible local 200+200 CIFAR100/TinyImageNet universes. They
must never be joined by iid:

- the universe actually used by the current Neural-HZ worker is rooted at
  `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks`, with CIFAR100
  `instances.csv` SHA-256
  `aa656d7a73529ba7c41b5618440f543ba4677418bb44115d384b644cc034f9ee`
  and TinyImageNet SHA-256
  `188058624df1122f32295f99d83380485a7d736212555a5e8214204459c22b7e`;
  its specifications are the isolated `vnnlib_v2` copies below this experiment
  directory; and
- the older `/data1/Kane/ACT/data/vnnlib` copies have different manifests:
  CIFAR100
  `ee3df1ea1f1beee15842177eb5ac7bcdf15cc49610c91b37704f2794ebd4ddae`
  and TinyImageNet
  `cdbb932ad476c95249afc8cca0af594f753ec47f3e54ee6ea78d8f1efe718ed1`.

Only 14/200 normalized CIFAR rows and 42/200 normalized Tiny rows overlap
between these universes. Current iid143 and every Trial 6--9 result refer to
the first, vnncomp2025-root universe; old per-iid claims from the ACT-copy
universe are not transferable. A promoted external ledger must additionally
freeze a deterministic full-file manifest for the isolated `vnnlib_v2`
specifications and the referenced ONNX models.

The original vnncomp2025-root source universes now have deterministic,
content-addressed manifests under `manifests/`:

- `cifar100_2024_universe_v1.json`: 200 rows, 202 referenced assets, no
  duplicate instance key, file SHA-256
  `fa30dafe17cdcafeb08b56da66189795e1623b5556ced7a8247903fef507d948`,
  ordered-universe SHA-256
  `17dba106e5fc160822d9e92de3c09fdcd895b1ff9b88867d1e6b1489c1d47069`;
- `tinyimagenet_2024_universe_v1.json`: 200 rows, 201 referenced assets, no
  duplicate instance key, file SHA-256
  `a8a0dc7504af2c6b89d099fd5c74f27aa5ae98458c5c151cbb6d0da2ef5c1f59`,
  ordered-universe SHA-256
  `d09befc11b3afd7f6db4c5e5359c67a96ed35a971c8c7614b4d887249fc2bdca`.

Each row is keyed by family plus model-content hash, specification-content
hash, and canonical timeout rather than by iid. All 400 `baseline_verdict`
fields are deliberately null and both manifests say `UNFROZEN`: these files
freeze source-instance identity, not a fabricated verdict vector.

The complete tensor-indexed execution specifications are now isolated under
`vnnlib_v2_full_v1/`. The conversion is a declaration removal plus flat-token
to tensor-index rewrite with an exact token-level round trip for every source:

- CIFAR conversion manifest file SHA-256
  `7e0567d32799dddde3236b00eb862d1817e496ed80ec4a93dbaf99651cac365b`,
  payload SHA-256
  `05a9c36a5902ddfef2b4dc1184e1072cc2959d69cfc262a353afce4798896e01`;
- Tiny conversion manifest file SHA-256
  `eb012c46b3a4437847b24125c41907255c9bf452d4d3cddabe2305a823ee7539`,
  payload SHA-256
  `f0556fbdfdd1d4f54b02f8bcb1568b9be0a5d88f5a016cdc9f8e325d46605275`.

A read-only closure verification rehashed 202 CIFAR and 201 Tiny original
assets, rehashed all 400 converted files, checked exact file-set equality, and
parsed 200 queries per family through ACT's VNNLIB 2.0 parser. The earlier
eight targeted `vnnlib_v2` files are byte-identical to their full-set copies.
Thus instance and executable-spec identity are frozen; the per-instance
baseline verdict vector remains the explicit missing promotion prerequisite.

### Historical large-classification verdict boundary

No archived 400-row verdict vector that sums to the cited 59 has been found.
The closest content-matched historical source is instead 61/375:

- `/data1/Kane/HyZor/audit_results/round4_worker1_complete.csv`, SHA-256
  `91ee850599fe3fbe550044346c29e00e513cdc8feb0edb10ce8dcdb5eb2cbbc1`,
  containing 25 CIFAR ADV over 200 rows and 36 Tiny ADV over 175 rows;
- the 29-row CIFAR SAT-slice file, SHA-256
  `434a059777950c52cde108549bc20ddf37f4377468ef7cf7e616be2dd9021b22`;
- the 175-row Tiny file, SHA-256
  `548bb35e6931f3326848b761aee1f58498b1488c4e9ca0f5ad569d4b5a268f65`;
- its recovery notes, SHA-256
  `358e2c86f267d565b7897532b73eedbee8a14256c8276853861f49eeec00e868`,
  reporting 25/29 and 36/38 SAT-slice recovery.

Those records include CIFAR iid166 and Tiny iid153 as `sat_zero_tol`, while the
current research queue treats them as UNKNOWN targets. There is no archived
decision record that explains a reduction from 61 to 59. It is therefore
forbidden to manufacture a 59-vector by subtracting those two rows or to import
the incomplete 61/375 vector into the current 400-row namespace. Historical
sidecars may be independently revalidated as evidence, but the ledger remains
`UNFROZEN` until one consistent protocol covers all 400 rows.

### Current content-key evidence baseline E0

The historical winning witnesses have now been independently mapped and
replayed under one current protocol. This creates a new external evidence
namespace; it does not reconstruct or replace historical 59/400 and it is not
a Neural-HZ gain:

| Family | CERT | Independently validated historical-origin ADV | UNKNOWN | Total |
|---|---:|---:|---:|---:|
| CIFAR100 | 0 | 25 | 175 | 200 |
| TinyImageNet | 0 | 36 | 164 | 200 |
| **E0 total** | **0** | **61** | **339** | **400** |

The authoritative v2 ledgers are:

- `evidence/cifar100_2024_evidence_baseline_v2.json`, file SHA-256
  `d5ff05325ed3c8182406a38faa364e98759acf96cb74f0aea5c867e3910baf15`,
  payload SHA-256
  `60a744b579edad986b5ac3e416e8d532b85a800eb10c34b334dd0ba9537c5420`;
- `evidence/tinyimagenet_2024_evidence_baseline_v2.json`, file SHA-256
  `76a9359e4760d1327fb9da4febfd0ac21306b7fb3c05571bab4110f40d80adc2`,
  payload SHA-256
  `ba0a10cdb5fb6e4555a0c4aa38337e05a7e2a4358e4363642c115a07815b6ce3`.

Mapping uses `(family, model content hash, original-spec content hash)` and
must resolve to exactly one current `instance_key`; iid and paths are not
identity. The replay does not trust the historical verdict or stored logits:
it feeds the witness to the current content-matched ONNX through single-thread
CPU ONNX Runtime and evaluates the original VNNLIB UNSAFE S-expression with a
new ACT-independent evaluator at literal zero tolerance. All 61 stored logits
also equal the current ORT logits element by element. The minimum UNSAFE slack
is about `9.352564811706543e-4` for CIFAR and
`1.4238357543945312e-3` for Tiny; invalid ADV is zero.

Every E0 ADV has `neural_hz_gain_credit=false` because its origin is a legacy
attack/sidecar witness. It is only a zero-regression retention anchor. A future
single-path Neural-HZ candidate must preserve all 61 as ADV (reporting CERT on
one is a soundness conflict) and can gain only by adding CERT or independently
validated ADV on the 339 E0 UNKNOWN rows. A full 400-row replay is required,
and its score remains arithmetically separate from 1,870/2,413.

## Structure-targeted research cohorts

Rules are selected by network/HZ structure, never by instance identifier.
Mixed families must first be partitioned by model structure for shadowing.

1. **Plain FC--ReLU:** safenlp, sat_relu, acasxu, and cora, with the FC
   subsets of malbeware and relusplitter as guards. Cora (140 remaining) and
   ACAS Xu (66) are the main targets; fully solved sat_relu/malbeware are
   strong zero-regression anchors.
2. **Conv--ReLU / sparse affine:** metaroom and the CNN subsets of malbeware,
   relusplitter, and cgan. CIFAR100/TinyImageNet scout the implicit Conv plus
   sparse-nonlinear-frontier rule before this formal cohort. A 2026-08-31
   per-model ONNX audit corrected the earlier coarse placement of LinearizeNN:
   its `AllInOne` models contain no Conv. The current ordinary-Conv rule has a
   direct formal-unsolved reach of 124 (MetaRoom 5 plus ReluSplitter-CNN 119);
   the four cGAN large-image cases require a separately proved ConvTranspose
   operator extension.
3. **Shared-ancestor residual/Add/Concat:** linearizenn and cersyve, followed
   by ViT residual blocks. LinearizeNN's repeated structure is
   `Gemm/ReLU + MatMul + dynamic Concat`, not Conv. This cohort targets
   persistent shared latent IDs and phase-separated affine paths.
4. **TLL/symmetric ReLU:** tllverify remains its own structural cohort for
   signed duplicates and dead exact subgraphs; its positive probes still need
   the formal replay gate.
5. **Smooth nonlinearities:** dist_shift Sigmoid and Tanh-bearing cgan
   substructures. Their formulas stay separate although structural selection
   may share a state-based framework.
6. **Attention:** ViT Softmax/MatMul/attention relational HZ is a separate
   class and must not be treated as an ordinary residual numeric variant.

The current execution order is: finish the CNN sparse-frontier/implicit-Conv
scout; migrate the resulting uniform rule into the formal Conv cohort; close
the existing TLL structural positive through its promotion gate; then attack
plain FC--ReLU (Cora/ACAS Xu), shared residuals, attention, and smooth tails.
Each accepted structure repeats the synthetic -> target -> same-structure
shadow -> per-family -> full-2,413 transaction.
