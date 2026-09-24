# Full-proof parse reuse comparison: two timeouts, archived and sealed

The user explicitly authorized execution after the freeze. Started clean on
`feat/moe-route-verification` at `5c4fe2fe1203339b0bf5cb01c3d07e460cccfeb0`.
Unchanged config SHA-256:
`0bfe49e812d3eefebce3d48ceb313571e33613baa96a49210e6ff6ccb044c908`.
Both preflight gates passed;515 frozen source files remain unchanged. There
were no retries, extra samples, added time, numerical-gate changes or old-bound
reuse. The raw artifacts and both failures are preserved.

## Main outcome

| Observation | Adapter cache off | Adapter cache on |
| --- | ---: | ---: |
| Terminal | TIMEOUT | TIMEOUT |
| Complete returned-call clock (s) | 298.2577 | 298.3777 |
| Intake process (s) | 8.5295 | 8.3706 |
| Construction process, cutoff included (s) | 289.7230 | 289.9538 |
| Sampled parent+worker peak RSS (GiB) | 5.3875 | 7.2275 |
| Complete construction bundle published | No | No |
| Independent source check completed | No | No |
| Native LP proposals / saved candidates | 0 / 0 | 0 / 0 |
| Independently checked output bounds | 0 / 252 | 0 / 252 |
| New complete positive requests | 0 | 0 |

Both obeyed the original300s request budget with2s publication reserve inside
it. The small overrun of the298s work deadline includes polling/owned-process
cleanup; neither exceeded300s or was accepted late. The sampled8GiB memory gate
did not cause either stop. Returned-call clocks include the final ledger write;
their sum is596.6354s. Batch admission/setup and later audit are separate.

Both intake phases published identical declared-source bytes, SHA-256
`c154dc4419e01f5b2c8d493eca03f186930ff45cd5e07b7ad30207956e7f3854`.
This confirms source-artifact identity, not completed source derivation or output
proof. The mathematical matrix outputs did not get published, so their real-run
equality was **not** independently checked. No missing result is counted equal.

## What the saved trace adds

Both traces completed26 affine lifts and17 ReLU operations. The last expert
affine return was observed at263.2554s without reuse and181.2383s with reuse.
These are per-request elapsed milestones, not complete-request speedups or a
statistically established cache effect. This is one fixed input, one ordered
execution per arm, on shared hardware.

After those layers:

- Without reuse, the first join and pair guard `{0,1}` returned. The cutoff
  occurred in property projection -> state unpack -> CSR parsing. No pair's
  `output_lp` call returned.
- With reuse, `output_lp` returned for `{0,1}` at231.3818s and `{0,2}` at281.1298s.
  The next join returned, then the cutoff occurred inside pair guard `{0,3}` ->
  state unpack -> CSR parsing.

Each `output_lp` call constructs that pair's necessary output LP descriptions;
it is **not** a solver call. Those two returns are saved event observations, not
18 persisted or checked certificates. The in-memory bundle was lost when its
owned worker was killed; this protocol intentionally publishes it only when
all pairs are constructed. Do not manufacture an offline prefix proof from logs.

Closed CSR wrapper calls total290 and368, with recorded inclusive times194.3170s
and202.9392s. These figures cannot be compared as equal workloads: the cached arm
processed more pair work before cutoff. The wrapper also charges current-content
serialization/binding and fresh output containers, not just Fraction parsing.
Hit/miss/eviction statistics are emitted only in the final construction receipt;
neither receipt exists, so real hit rates and the unique cause of remaining
cost are **not established**. Synthetic hit counts cannot fill this gap.

The larger RSS on the cached arm likewise mixes cache retention and additional
constructed pair state. It is not a measurement of cache-only memory overhead.
Nested inclusive event times overlap and must not be summed. The archive keeps
completed-operation totals and right-censored open stacks separately.

## Scientific interpretation and stopping decision

There is an observed local construction-progress signal, but **no complete
request, checked bound or certificate gain in this comparison**. Neither arm
reached the source checker, candidate solver or final aggregation. The evidence
therefore does not diagnose solver limits, LP relaxation precision, the existence
of a valid certificate or model unsafety. It also does not repair the historical
23 source-gap gains or support high-accuracy/native-float/route-change claims.

Seal both executions. Keep reuse opt-in; do not enlarge cache capacity, budget,
sample count or restart this object to obtain a positive result. If further work
is authorized, begin with saved traces and the already frozen constructor to
separate pair-level serialization/parsing/copy costs. Any new representation or
publication strategy requires a separate controlled protocol. No such change or
new real query was made here. Input98 and CIFAR4088 remain sealed.

## Independent saved-only review and retained artifacts

- [Frozen batch audit](scoped_parse_proof_execution_audit_20260924_r1.json):
  PASS/zero issues, both registered terminals retained, complete costs available,
  zero completed constructions compared, zero new solver calls during review.
- [Hash-bound archive](scoped_parse_proof_execution_archive_20260924_r1.json):
  source identity, all phase receipts, nested operation totals/censored stacks,
  evidence absence,28 raw files totaling149,009,613bytes. Raw data remain local.
- [Archive/recheck script](../scripts/archive_scoped_parse_comparison.py): fresh
  `python -S` process with imports of model/solver libraries and external process
  execution forbidden. The saved archive reproduces.12 receipt corruptions and
  four trace corruptions are rejected; two unchanged receipt prefixes and one
  valid nested trace are accepted. These are structural/cost controls, not new
  mathematical proofs or independent human review.

Read-only reproduction:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/archive_scoped_parse_comparison.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/archive_scoped_parse_comparison.py --controls
```

Audit-entry clarification: the frozen batch auditor compares the root string
to the absolute configured output path. Its CLI must receive that **absolute**
path; the relative example in the old freeze note would fail that identity
check. The archive uses the configured absolute path. Frozen execution/audit
code was not changed; this path-interface issue did not affect either run.

An audit PASS here means consistent frozen identities, terminal/cost records
and correctly retained absence of evidence, **not** completed output proof.
