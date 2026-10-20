# Rational N1/M3 interface control — result and audit finding

**Functional arithmetic/coverage checks passed; clean full-stage acceptance is
withheld because one worker timing field is invalid.** The original online
terminal says PASS and is preserved, not rewritten as a different result.
This is an interface experiment, not A2's strong comparison or a new certificate.

Execution revision: `c1d46f3577061ef98e7baec98dff4b7cb0e989a8`.
[Protocol](margin_product_control_20261021.md),
[freeze](margin_product_freeze_20261021.json),
[lossless archive](margin_product_records_20261021.json).
Freeze SHA256: `f6a8073d0192831b211525a88a38dd2b1db89d980d8c9afb1d4e4a9d06c067f2`.
Original terminal SHA256:
`6baa65c567f2df39d80e87434cbd3cc67fb8107c12214fd43259dce8f9dcb5b0`.
Outer exit0. No native call was repeated after this run.

## Mathematical and execution evidence

Four native calls/four objectives cover two closed cells under N1 and M3.
All original3079 factors, ten-class property coefficients, both router rows and
original constraints remain. Four binary factors are explicitly relaxed.
Checked typed d/−d certificates bind the same source/property/cell; their
zero-dual range here is[-155/256,175/256], not an optimized slice range.
Lambda/w finite boxes and all product/sign rows are reconstructed independently.

| Mode | Cell0 lower | Cell1 and complete partition lower |
| --- | ---: | ---: |
| N1 | −1.240301845115 | −1.393998641944 |
| M3 | −1.236099142203 | −1.388011750991 |

Exact complete lowers are respectively
`−25714736187034769673/18446744073709551616` and
`−6554700170709952551853/4722366482869645213696`.
Their difference is about0.005986891; this is not a fair performance result or
proof of optimality. Both are nonpositive checked dual lower bounds: neither
proves a negative LP optimum, a counterexample, or model unsafety.

Five fixed-candidate auxiliaries cover collapsed gate, zero/positive/negative
constant differences and crossing-zero without a sign premise. Positive local
auxiliary bounds are only first-cell given-source controls, not complete routed
requests. Twenty-three specified mutations and expired continued reception
reject. Worker/controller/relocated standard-library receipts agree. Eight
durable prefixes alternate call intent and checked completion; four calls were
attempted and returned, final pending call is null.

## Cost, and the invalid field

| Independently recorded scope | Seconds |
| --- | ---: |
| Outer entry through terminal-file export | 30.671494 |
| Native owned-session execution | 23.718496 |
| Native owned cleanup | 0.039547 |
| Native imports, nested within execution | 1.483852 |
| Online controller reception | 3.238479 |
| Relocation copying | 0.003260 |
| Portable owned-session execution | 3.434544 |
| Portable owned cleanup | 0.024418 |

Nested scopes must not be added. Pre-entry startup and final stdout/exit tail
are not certified full-cold-start timings. All independently observed sessions,
reception and terminal export were inside the same300s deadline.

**Invalid:** worker `elapsed_before_worker_export` is11386840.625857959 seconds.
In frozen `scripts/margin_product_cases.py:214`, the `mc_sign` fault reassigns
local `start` to a CSR pointer, shadowing the monotonic entry timestamp; line239
then subtracts that index. This field is unusable, not a long-running job.
It does not alter the independently held deadline, exact LPs, duals, Work
counters, controller timer or owned-session clocks. Keep its original value;
do not reconstruct a replacement, silently fix the archive, or rerun the batch.

Work **24,768,815→29,046,959→33,325,103**; each receiver adds4,278,144.
This is approved account-only development Work, not the old proof-policy8M gate.
Run4,836,019B plus archive4,699,393B total9,535,412B, below128MiB.
Per-process AS8GiB; native complete-sample RSS730,738,688B and direct-child
wait4 peak755,687,424B are separate telemetry, not a group simultaneous peak.
FDs4→4; both leaders exit0 and are reaped, no live recorded members.
Receipts explicitly do not establish complete zombie reaping or escaped-session
absence. No trained source, GPU, dependency change or real-development call.

## Independent review and bounded next decision

Saved-only frozen audit passed in3.409256s:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/audit_margin_product_20261021.py
```

Its PASS covers hashes, receipts, arithmetic and continued accounting; **it does
not validate the defective worker subduration**. Use the frozen execution
revision for later source-hash audits; do not weaken identity checks.
Read-only AI agent `product_baseline_math` separately reconstructed9 product
LPs and18 typed range certificates with Fraction from original coordinates,
without importing the production checker or native libraries. All exact rows,
boxes and residuals agree. Agent `product_baseline_integration` independently
checked all27 run files, freeze/archive identities, prefix/fault rosters,
deadlines, Work and valid external timings. Both identified the timing defect.
These are AI reviews, not independent human review or proof of network lowering.

**Stop new execution for this stage; no repair/retry and no new control chain.**
Retain the functional evidence and audit finding separately. Next authorized
work is static preparation of the meaningful A2 comparison: common continuousP,
paid strong ranges, fresh slices/N1/M3 and native LP reference, valid endpoint
fallback, explicit objective/cost accounting and effect/stop gates. The timing
finding must be addressed in a separately frozen future entry, without replaying
this consumed roster. Do not tune this source or reinterpret these numbers as
the scientific comparison. G1–G6 OPEN; real-stage ledger0/0/0.
