# Repaired source profiler: separate two-call freeze

Implementation `d88e7c9f40906a622c53731b1821d8a85b18282c`.
Configuration `configs/backend_controls/source_cost_supervised_r1.json`, SHA256
`e9f8e267c21c9a36d596ce865a40ef0e3001003ce545b9491131d622c794c3c5`.

180 controls PASS; 593 source/protocol bindings checked, including unchanged
582 historical bindings and the four repaired-interface files. Freeze verifier
PASS: source hashes, control result, environment, ordered two-item inventory,
declared source identities, budget and no-output-proof scope match. No output
directory exists at freeze. This is a new interface-repair follow-up, not a
replacement for the preserved R1 0/2 profiling result.

| Order | Identity | Experts / classes | Width / depth | Seed | Budget |
|---:|---|---:|---:|---:|---:|
|1|repair_small|4 / 3|4 / 1|724|300s|
|2|repair_medium|8 / 10|8 / 2|724|300s|

2 CPU threads, sampled 8GiB, no GPU. The worker and receipt receiver share
the work deadline at 298s; final publication has the remaining 2s, not an
extension. Default R1 streaming stays unchanged. All prior checks remain.
Exactly one attempt per object; no new bound query, extra fixture or retry.

New output: `data/moe/results/source_cost_supervised_20260925_r1`.
After this freeze is committed and pushed, execute once:

```sh
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -m source_cost_supervised.run execute
```

Recompute saved-only cost/receipt checks and source constructions separately;
record that audit's cost. Report all failures and censored durations. Do not
rank bottlenecks using failed interface timings or infer real-MoE latency from
these synthetic objects. The next single-factor proposal must be justified
by valid phase/component data, with any remaining attribution gap explicit.

See [protocol](source_cost_supervised_protocol_20260925_r1.md) and
[controls](source_cost_supervised_controls_20260925_r1.json).
