# OBSOLETE — DO NOT USE FOR PAPER

This directory contains an EARLIER CORA "STRICT" sweep (2026-05-26 23:45 →
2026-05-27 03:53) that ran with `falsification_method='center'`. Despite
the "strict" label, **this run was NOT helper-free**: CORA's `center` mode
still performs a 1-point deterministic falsification (evaluates the input
box's geometric centre and checks if it violates the unsafe spec). All 26
A (sat) verdicts in this archive came from that 1-point eval, NOT from
pure reachability.

For arXiv submission we re-ran CORA with a TRUESTRICT patch that adds a
'none' option to `falsification_method` and skips the entire falsification
block in verify.m. That clean result lives in:
   /data1/Kane/ACT/audit_results/cora_truestrict_20260527/

This `_OBSOLETE_*` directory is kept for forensic/reproducibility evidence
only — it documents the inadequacy of `center` as a "no-helper" config.
