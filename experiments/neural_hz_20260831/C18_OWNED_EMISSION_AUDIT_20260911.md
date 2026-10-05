# C18 actual audit: CLOSED on coupled work, no candidate returned

Formal1870/2413 and separate E061/400 unchanged, gain0. C18 is default-off,
not promoted and cannot be retried or repriced. No production/HyZor/archive
mutation, source drift or provenance drift. All six outputs auto-retained in
results/c18_owned_emission_20260911_v1/; no worker remains running.

798 tests passed in7.12s. Tests exit0, worker exit1, supervisor exit0 (the latter
records the failure; it does not indicate candidate success). Supervisor wall
124.78824401460588s; worker115.42421069182456s. No timeout. Original snapshot,
expression, complete C9 oracle and C10 phase oracle remain unchanged. Production
candidate SHA15198f4ddc40dfa1c37456737b0f2080ddee2c653e2b9d0010cf245b6c5fec75.

The original-expression frontier reached the previous106721 local/100727
eligible/100603 selected aliases and322936 checked products. These are C10
results, not new eliminations or gain. Frontier extra work31243923 at
105.06196171883494 worker seconds. At111.09741357062012s, before the next
ownership update, the shared pool rejected76 work with used35026957 and
capacity35026988. Whole base220973012; branch base124800716. No partial row or
candidate is returned by the public lift.

Consumed extra work:

| Operation | Work |
| --- | ---: |
| MAIN ownership materialization | 729486 |
| physical row UID labels | 246640 |
| radix | 593353 |
| radix ownership relocations | 16 |
| MAIN metadata/frontier | 7781184 |
| complete continuous incidence scan | 11156880 |
| exact cached products | 10333952 |
| known ownership changes | 984880 |
| row order-check/rewrite | 858832 |
| full sort | 2062816 |
| changed-stream checks | 75656 |
| stream merge including any k-sort | 203262 |

Charged whole work255999969; the next76 alone would reach256000045. Mandatory
unrun retirement301809 and final MAIN range check729486 imply a complete lower
bound257031340, exceeding256M by at least1031340 PLUS all other remaining row
work. This is not a45-work overall deficit. The failure happened before the
complete construction returned, so neither full resource acceptance nor final
real-target ownership/equivalence/phase/native/physical gates passed.

Consumed normalization totals2341734, of which2062816 remains full sort. C17
used2339356 full-sort work before its own cutoff. The cutoff prefixes differ;
their difference is NOT an equal-workload speedup/regression measurement.
The new tested normalizer did not pay the complete construction deficit.
Do not infer a complete row distribution from either truncated event prefix.

Absolute worker peak2061092KiB=2110558208bytes includes retained oracles. Since
measured_build raised before completion, this is NOT a passed1GiB construction
growth/tracer gate. No ownership_audit.json, phase_event_audit.json or
owned_hz.pickle exists. No new ReLU, optimizer, terminal verdict or witness.

Artifact SHA256:

- preregistered.json:974d88d013a648365b38254e3c6f0a95de0b8ca917ea74428af3ac253c0b1c16
- events.jsonl:1ebf6fc02550ccdfbf24b9c1d417cd89f76052899f8890aa2d5983b02e02573b
- result.json:e8a12384a17f42cf3829cd4000421872294590394dd2a7af4f4d9a97835eabac
- exit.json:0ae272f39349b4b78c429c7a0731c7ec48ece0fd29e6399d43f80b61ea493ae0
- tests.log:7dec17a163decfd1a125190a78d791da303fa9b2feb9da6cc9ae6900424d9c20
- worker.log:756ca962258472d80064f2b5e8d36afed08cefe6c27c59f5f3934a8e89a4b0b2

Next evidence need: a complete read-only geometry census of ALL37500 C10
rewritten rows from the sealed original C9 and successful C10 lineage, including
changed-column density/order/runs and same-row C17/C18 logical tariff totals.
This must be a separately preregistered diagnostic, not a retry of C18, not
fresh generation from a completed-HZ shortcut, and not a capability result.
Do not select another sorting hypothesis solely from this failed prefix.
