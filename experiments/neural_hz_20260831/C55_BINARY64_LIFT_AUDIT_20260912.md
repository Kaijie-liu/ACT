# C55: exact binary64 coefficients proved; fixed physical representation rejected

Branch redu-hz; starting commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac;
production candidate15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75 unchanged.
Formal1870/2413 and separate E0 CIFAR25/Tiny36 remain unchanged. No new benchmark
run, native solver call, real source generator, default or instruction change.

## Evidence that changes the next action

C55 establishes that ordinary107-417-bit C54 coefficients can be represented
EXACTLY with bounded continuous auxiliaries and binary64 literals in the original
coefficient window. It does not justify an admission: radix16 introduces too many
coordinates and arrays for the fixed chain and Conv/ReLU physical comparisons.
The negative result rules out this unchanged scalar-coordinate radix16 lowering.
Do not select only shared Add, relax the gate or rerun for favorable accounting.

## Mathematical result

For c=m*2^e, L=bit_length(abs(m)), a low-to-high digit recurrence
y'=(y+d*x)/2^w with w<=16 gives y_final=abs(m)*x/2^L.
All prefix linear factors are bounded by |x|, hence all new[-1,1] boxes are redundant.
The consumer receives sign(m)*2^(L+e)*y_final. Original binary phases are untouched;
every original source EQ/INEQ, UID/frame, output and inverse is preserved.
Equal scalar/continuous-coordinate pairs share a lift across consumers.

The independent Fraction audit eliminates ACTUAL emitted equality rows, proves
every new box redundant, and compares all substituted predicates/output/RHS with
the source-bound C54 state. It is not merely a repetition of digit extraction.
The exact-state semantic binding excludes only its shared work pool's cumulative
whole_work counter; every other field stays bound. Actual whole work is measured.

Focused tests:22 passed in1.12s. They cover all three ordinary cohorts, complete
inverse/network-function points, actual-row/RHS/inverse/UID/binary/frame mutation,
box rejection, fail-closed work budget, zero-hit and literal/digit exactness.
All18 original surviving rows and88 new equalities were proved, then re-proved
after protocol5 export and unchanged C41 authenticated restore with fresh C54
source regeneration. All48 DISTINCT original feasible vectors (6/6/36) pass
complete lift, original-coordinate inverse and exact output checks, then the
same48 are replayed. None is a benchmark adversarial example.
This is mathematical coefficient realization, not current native solver-layout
integration, numerical solver soundness, or complete old-suite qualification.

## Complete SAME-source physical results

Original C52 binary64 reference is unchanged. Both sides retain the complete
original source. Candidate retains its full exact inverse/scalar pool, original
global and removed UID data, native coefficient CSR/RHS, new equality UIDs,
lift map, source binding and report. The five unshared C54 CSR/RHS/old-EQ-UID
arrays are weak-retired per cohort; the15 retirements actually pass.

| Cohort | Continuous | Auxiliaries | Predicate nnz | Numeric bytes | Numeric entries | Numeric+reported shallow bytes |
| --- | --- | --- | --- | --- | --- | --- |
| chain | 131 -> 30 | 27 | 280 -> 89 | 12132 -> 9855 | 1240 -> 1324 FAIL | 126096 -> 127806 FAIL |
| shared Add | 195 -> 10 | 7 | 520 -> 29 | 20900 -> 13794 | 2152 -> 1767 | 188016 -> 182948 |
| Conv/ReLU | 133 -> 59 | 54 | 274 -> 177 | 12052 -> 11463 | 1230 -> 1586 FAIL | 128148 -> 133222 FAIL |

All three reduce predicate nnz and numeric bytes. Only shared Add passes ALL four
physical comparisons. The same two cohort failures persist after serialization.
Python shallow accounting is the existing reported ledger, not an exact allocator
or RSS measurement; its array/object conventions may overlap numeric byte charges.
No local-byte decrease overrides the complete-entry or combined gate.

## Terminal execution and resource evidence

One frozen run: results/c55_binary64_lift_20260912_v1/.
Supervisor7.53995200060308s, worker4.738812713883817s,
measured transaction2.779144481755793s, total work3061283.
Lift branch work15888/8432/28256, well below the unchanged16M radix budget.
All CPU1/GPU0,AS16GiB,64M-entry,256M-whole/200M-branch,60s-focused/240s-worker,
16384-auxiliary/131072-added-entry and both1GiB transient caps remain unchanged.
HWM minus entry RSS19484672B; traced peak9449609B+metadata927264B=10376873B.

Fresh full archive62940 numeric bytes/7142 entries and413867 reported shallow
bytes. Fresh+restored125880 numeric bytes/14284 entries and796210 shallow bytes.
The measurement includes original and C54 construction, lift emission and actual
source/row proofs, complete original-vector checks, five-array retirement,
ownership, export, authenticated decode, fresh exact regeneration and restored
proofs/points/physical/owner checks. It is not a whole verification-request time.

Worker exit2 means diagnostic_completed=true, representation_accepted=false.
Tests exit0; source_drift=false and provenance_drift=false. No worker remains live.
Archive136305B, SHA256cdf90931f1ae4981e71ec5512093df229d74a41ec2bdb8e870588011ee6a373b.
Result SHA2565e160292c25a8d40a30d43abe9b85d492fe996fa350e3ee90932726371aae7b0.
Exit SHA2568ebb998cc84785a15cd1e8d94f727f0f824d0ae1663bce5054771d20702fc1b7.
All794 frozen source paths and all logged artifacts rechecked unchanged.
All63 previous checkpoint manifests/2241 unique files/28308887764B pass integrity.

C55v1 is CLOSED/NOT ACCEPTED. The exact lift proof is useful capability evidence,
but formal gain is0. Next work must reduce actual lift cost on the same ordinary
general-scalar consumer structure, not optimize pathological corner cases.
