# C11 single-use census and lowering diagnostic

Formal1870/2413 and separate E061/400 unchanged; gain0. All518 tests pass
in5.39s. Worker exits1 because the complete arithmetic census is REJECTED,
not because a solver returned UNKNOWN. Supervisor24.918209025636315s,
source/provenance driftfalse. No optimizer, presolve, native passModel or HZ
rewrite was called. Both source checkpoints and the production path remain
unchanged. All19 preceding checkpoint manifests verify with sha256sum.

## Complete structural result, before any candidate arithmetic

The saved C10 final HZ has142559 remaining MAIN definitions, after100603
already-proved alias removals. Exactly102571 remaining direct positive-dyadic
definitions are output-dead and have ONE other predicate occurrence. Their
general affine substitutions require8200917 replacement/RHS terms and inspect
1921766 consumer coefficients. Under the preregistered complete generic cost,
work682142176 exceeds256000000. Therefore the census stopped BEFORE all
individual exact-product/box/RHS/collision checks. There is no per-factor
arithmetic table, no individually-admissible count and no simultaneous proof.
The structural102571 count is NOT an achieved reduction, solvability evidence
or an additive nnz saving. No subset was selected and no ceiling was changed.

## Independent unchanged-input lowering

As explicitly preregistered before launch, the rejected census was saved first,
then ONE ordinary _lower_hz_milp call ran on the original sealed final HZ with
prune/coalesce true and projection/phase fixing false. It completed in
4.609759916551411s; measured growth644956160bytes, traced peak508551820bytes
plus206976bytes tracer metadata, both below unchanged1GiB. It produced
154362 continuous,1350 binary,155712 total variables,146637 rows and
10960724 matrix coefficients. Exact source maps and array hashes were saved.
Full source checkpoints remained reachable throughout measurement. These are
construction measurements, NOT response-gate results or an optimizer result.

This does NOT retrospectively locate C10's outer240s timeout. The earlier
ordinary API call had no return or native-entry observation, so native setup,
optimization and other internal time remain unpartitioned. The new result only
shows that standalone ordinary lowering completed in this fresh diagnostic.
The rejected census stays rejected regardless of that diagnostic success.

## Provenance and retained results

- branch redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac
- production candidate15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75
- result ef5283cd2417ff918657414619a357bf850d89d8304b7dbbfecde96e35e097c7
- census2d05e52d0a6fbd6f7b213852079e180c1c65044391dbdec985cc88d905c9cb3e
- lowering c7430c9771ad22efe95a69d33457dbe10535744a3f075290d20f86cdc68c117b
- prereg6b64d8b13871a54c16b251e36863a0c52529e671f2d1901c2d2c1df9cc657bfd

All files reside in exclusive results/c11_single_use_census_20260910_v1/.
No process remains running, no failed version will be restarted, and no family,
cohort or full2413 replay is authorized by this result. The same S0 remains
open. A useful next hypothesis is sharing exact coefficient certificates for
bitwise-repeated affine definitions (as neural operators reuse weights), with
complete byte-identity checks and a new whole-cost proof under the SAME cap.
This is only a hypothesis, not evidence of reuse or authorization to append a
postpass to the already254750077-work C10 construction.
