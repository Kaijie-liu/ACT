# C7 V1 numerical carrier audit: model changes during unchanged ingestion

Result SHA-256:
`281e31ec287d55b4b14355043b2310fb184eabc1557910ded15c938de6de5467`
in `results/c7_checkpoint_ingestion_20260905_v1/result.json`.
231 tests pass; fresh worker/tests exit zero, source/library/provenance drift
false; supervisor wall 26.872087577357888 seconds. Checkpoint hash was verified
before loading. All 243162 definitions and original predicates were independently
re-audited after reloading, with process-local identities explicitly rebound.

The exact affine-factor representation passes its set identity but C7 V1's
coarse recursive box normalization produces an unsuitable numerical carrier:

- required auxiliary exponents range from -42 through 54;
- constraint Ac coefficients range in nonzero magnitude from
  4.170317434531852e-23 to 1;
- 1,764,561 Ac coefficients are <= the installed native small-matrix threshold;
- all 200 output Gc coefficients range from 2^51 to 2^54 and exceed the native
  large-matrix threshold. These output values were inspected, NOT passed as
  property constraints, so no property-model load failure is claimed.

The actual backend inspected is the SAME SciPy-bundled HiGHS binary used by
the local ACT solver: SciPy 1.17.1, HiGHS 1.12.0. Local wrapper/core hashes are
bound in the preregistration. Its unchanged defaults are small_matrix_value=1e-9
and large_matrix_value=1e15. ACT's ordinary lowering used existing unused-factor
pruning and exact row coalescing, without projection or binary phase fixing.
It retained 254537 continuous and all 1150 binary solver columns.

Only passModel/getLp were called: no run, presolve, optimize or solve. The
native warning explicitly reports ignored small entries. The full matrix
comparison confirms **11,160,274 submitted -> 9,395,713 retained entries**,
exactly **1,764,561 changed/removed nonzero coefficients**. Column/row bounds
and integrality remain identical. Thus successful loading is not equivalent
problem retention, and no downstream verifier verdict may be trusted from
this unqualified carrier. No solve or verdict was attempted. Full diagnostic
peak was 1,982,976 KiB; all logs and exit records are retained.

## Next same-structure hypothesis, not yet implemented

Do not change solver tolerances. Investigate exact dyadic balancing inside the
HZ representation: a tighter power-of-two bound on the SUM of term magnitudes
instead of number-times-maximum; a common minimum coordinate scale to avoid
manufactured tiny intermediate units; and positive power-of-two equality-row
scaling where a proved coefficient-range window exists. Every transformation
must remain exactly reversible, retain all binary/predicate semantics and
prefix reconstruction, and pass complete work/storage plus unchanged-backend
ingestion again. If no such window exists, reject rather than discard terms.
These are hypotheses, not a successful remedy or a new target run.

This is a routine representation-induced scale problem in the registered
network, not an excursion into unrelated extreme-case optimization. C7 V1 is
closed for live advancement; its exact-set/physical evidence remains valid
and immutable. Formal 1870/2413, E0 61/400 and defaults are unchanged. There
are no live jobs remaining, and the overall research goal is not blocked.
