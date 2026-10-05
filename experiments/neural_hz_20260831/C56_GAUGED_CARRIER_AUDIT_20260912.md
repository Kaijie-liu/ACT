# C56: exact common-carrier binary64 realization passes the fixed prototype gates

Previous goal turn: PROGRESS (C55 exactness plus physical rejection).
This turn: PROGRESS. A uniform dyadic row-gauged common-affine-carrier lift
and inverse-live exact scalar pool now pass all three unchanged mathematical/
physical prototype cohorts. This is NOT an actual network or solver result.

Branch redu-hz; starting commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac.
Production candidate15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75 unchanged.
Formal1870/2413 and separate E0 CIFAR25/Tiny36 remain unchanged, formal gain0.
No instruction-audit proposal has been applied. All new work is isolated here.

## Mathematical change, same ordinary structural cohort

C55's radix16 has been replaced by ONE common-affine-carrier rule, not a menu.
Inside each source-bound C54 consumer, equal-magnitude long coefficients define
a signed linear form. Its normalization by2^ceil(log2(number_of_terms)) is bounded
by1 on the original continuous box. Identical forms/coefficient magnitudes share
one lift across rows. All coefficients are compared as canonical exact dyadics;
no family, iid, model identity, margin or solver status selects a rule.

For low-to-high digit width w<=min(53,60-k), choose q=max(0,w+k-20).
The ACTUAL equality is
2^q*y_new - 2^(q-w)*y_old - sum(sign*d*2^(q-w-k)*x)=0.
Every literal has at most53 mantissa bits and remains within the ORIGINAL
nonzero[2^-20,2^40] window. The final multiplier2^(L+e+k) is separately checked.
Every auxiliary is a fractional prefix times the same normalized affine form,
hence its[-1,1] box is redundant. Binary factors are neither pivoted nor deleted.

Independent Fraction elimination of actual generated equality rows reconstructs
every auxiliary coordinate, proves its box redundant, and compares every
substituted original predicate/output/RHS with C54's exact source-bound state.
The equations themselves contain the complete auxiliary reconstruction maps;
there is no missing registry or external lift map. The original coordinate
inverse, global/removed/row UIDs, scalar values and source/frame binding remain.

v2 adds ordinary coefficient liveness after the unchanged v1 mathematical lift.
Only exact scalar-table entries still referenced by the complete original
inverse are retained. Scalar IDs are remapped, but their roots and exact values
are compared coordinate-by-coordinate with the ORIGINAL source table. Every
retained pool value must be referenced; original consumer proof uses the original
table, never a remapped inverse-local ID. State schema and report fields remain
the same. No strings, thresholds, workloads or accounting rules were shortened.
All transient v1 construction, liveness scan, repack and inverse remap are measured.

## v1: preserved negative result

25 focused tests passed in1.17s. All actual-row/inverse/box proofs,48 original
feasible vectors and authenticated replay completed.
Auxiliary counts fell from C55's27/7/54 to8/3/8; all three cohorts reduce predicate
nnz, numeric bytes and numeric entries. Fresh Conv combined reported accounting
was128148->128167B, a19B failure. Restored Conv passes, but cannot replace the
required fresh boundary. v1 remains CLOSED/NOT ACCEPTED, worker exit2.

## v2: ALL fixed physical comparisons pass

29 normal focused tests pass (25 mathematical/source guards plus4 inverse-live
pool guards). Each fixed measurement retains the same complete original source
on both sides and compares with the unchanged C52 binary64 reference, NOT an
inflated exact reference or current C31/whole-request native baseline.

| Cohort | Continuous factors | New aux | Predicate nnz | Numeric bytes | Numeric entries | Numeric+reported shallow bytes |
| --- | --- | --- | --- | --- | --- | --- |
| chain | 131 -> 11 | 8 | 280 -> 32 | 12132 -> 8633 | 1240 -> 1120 | 126096 -> 125315 |
| shared Add | 195 -> 6 | 3 | 520 -> 17 | 20900 -> 13452 | 2152 -> 1703 | 188016 -> 182217 |
| Conv/ReLU | 133 -> 13 | 8 | 274 -> 47 | 12052 -> 8809 | 1230 -> 1151 | 128148 -> 127883 |

All four strict comparisons pass for each fresh AND restored cohort.
Full numeric-byte savings are28.84%,35.64%,26.91%; combined reported-accounting
savings are only0.62%,3.08%,0.21%. These are NOT whole-process RSS or verifier
runtime savings. The unchanged shallow ledger is reported accounting and may
overlap separately charged array data; allocator occupancy is not proved by it.

Every original binary remains (1/1/2). All18 original surviving rows and19 new
auxiliary equalities are independently proved, then proved again after export/
decode and fresh C54 regeneration. All48 DISTINCT original feasible vectors
(6/6/36) pass full inverse and exact output checks, then the same48 are replayed.
They are not96 distinct vectors or benchmark counterexamples.
Additional signed-carrier/reuse tests do not change the fixed measurement set.

v2 weak-retirement proves all TEN unshared original C54 numeric arrays dead
per cohort: CSR/RHS/old-EQ-UID5, scalar pool4, inverse1. Original global/removed/
INEQ-UID roots remain explicitly shared. All30 retirements pass; no old pool
is retained in a hidden report. This is not a whole-transport zero-copy claim.

## Runtime and measured resource window

v1 worker4.662353035993874s, supervisor7.489521988667548s,
measured window2.705037225037813s, whole work2892847.
HWM-entryRSS19271680B, traced9432500+metadata928512=10361012B.
v2 worker4.626528928987682s, supervisor7.440425098873675s,
measured window2.6751942671835423s, whole work2943920.
HWM-entryRSS19619840B, traced9432634+metadata929024=10361658B.
Both independent1GiB transient tests pass for both versions.

v2 complete lift/repack branch work25490/28069/27426; actual added radix numeric
entries70/25/86 include CSR data/index, pointer/RHS and new equality UID.
Use these complete counts rather than C55's8-per-aux shortcut as a general
entry bound. No measured C55/C56 cohort approached the131072-entry ceiling.

All CPU1/GPU0,AS16GiB,64M entries,whole256M/nested200M,60s-focused/240s-worker,
16384-auxiliary/131072-added-entry/16M-lift limits remain unchanged.
The measured transaction includes full source/reference/C54/native-coefficient
construction, liveness/remap, actual source/row/box/inverse proofs, points,
retirement, owners, protocol5 export, unchanged C41 authenticated decode, fresh
C54 source regeneration and restored proofs/points/physical/owners.
The exact semantic binding omits only the shared pool's temporal whole_work
counter; it remains charged and logged in the transaction.

v2 fresh full archive58722 numeric bytes/6439 entries/409461 shallow bytes;
fresh+restored117444 numeric bytes/12878 entries/791552 shallow bytes.
Native solver API structures, INPUT/ASSERT frame binding, solver numerical
behavior, actual verification-request LIVE and actual network outcomes remain
UNPROVED. No original real source was regenerated by this prototype.

## Immutable records and next decision

- v1 results/c56_gauged_carrier_20260912_v1/
  result SHA256be7c8d7ad6500b8e4878a0209e6e893ff25cd643ddedff6ea3772a5f858420e7;
  archive132432B SHA2563036caf8c8b05ffce8b699bebb92f7b8f6c4918e016a1ae49d54dccd05d7490a.
- v2 results/c56_inverse_live_20260912_v2/
  result SHA256958c4a6f7353821f273e26dd92c72c4eda1ba8178e0fb3d68da5669138b63f87;
  archive132088B SHA2568a38081c766e63087f9d2a11cce3bebd30313c50aee4c70a381dbf3024f3982a.
  worker and supervisor exit0; all source/provenance drift flags false.

Both runtime jobs are terminal. All802/809 frozen source paths and all logged
artifacts rechecked unchanged. All64 old checkpoint manifests,2259 unique
files/28309192570B pass before and after integrity. No historical result is edited.

The next research step returns to the ACTUAL same-S0 ordinary source: bind
general-scalar chain/consumer applicability, precision and full resource cost.
No extrapolation from the128-copy fixtures, no re-credit of C31's100603 aliases,
no unchanged failed generator/test gate retry. See the dedicated handoff.
