# C10 quotient checkpoint — 2026-09-08

Goal ACTIVE; formal1870/2413, independent E061/400, new formal/capability gain0.
Previous status-only turn was no algorithm progress. This turn implemented
and completed the preregistered exact alias quotient on the actual saved HZ.

- 442 tests pass, including 37 new proof tests; worker/supervisor exits0,
  provenance/source drift false. Default off; production/history unchanged.
- 100603 MAIN aliases exactly eliminated; original global latent indices and
  all1350 binary factors preserved. Independent Fraction audit of every erased
  definition/changed row and all unchanged predicates passes.
- Ordinary native lowering now154362 continuous variables vs254965. Native
 10960724/10960724 matrix coefficients retained without solve/presolve.
- Total coefficient nnz11201930 ->11000724. Including3219296-byte reconstruction
  certificate, component bytes138382224 ->137577400 (only804824 bytes net saving).
  Numeric entries11449370 ->11549973: INCREASE100603, not a whole-state win.
- Construction6.1242s; component logical work236441920 <256M, measured construction
  memory gates pass. This does NOT establish fused/live work or complete-state
  reduction. No terminal solve or full/family replay was run.

See C10_ALIAS_QUOTIENT_AUDIT_20260908.md and
results/c10_alias_quotient_20260908_v1/. All processes terminal; all artifacts
retained automatically. Next is fused emission/alias lineage with full work,
reconstruction and live-state accounting, NOT bolting this postpass onto C9
(registered bounds would sum to470885700 >256M), not deleting native caches,
not broadening the solver budget, and not reclassifying UNKNOWN as gain.
