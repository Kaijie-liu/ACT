# Frozen synthetic study, before execution

Implementation commit `c3b3dd131d6dbd5618873e1f906940076b196732`.
158 controls passed;582 source/protocol bindings independently rehashed.
Config `configs/backend_controls/batched_evidence_study_r1.json`, SHA256
`5288249917407546ad4f76d0e6148bdf3c6fc5d4985e79f5ccbbfd451e41970d`.

Read-only standard-library review checked the exact36-call Cartesian inventory,
unique38 IDs, both separate profiles,30s/300s limits,2s reserve,8GiB/2threads,
source hashes, zero real requests and absent output directory. PASS.

Output: `data/moe/results/batched_evidence_study_20260925_r1`, absent at freeze.
Commit/push this record before one execution. No new samples, retries, extra
time/memory, real4099 reconstruction, mathematical or checker changes.

Two disjoint questions: bounded-array encoding throughput/memory (36 calls),
and unchanged source construction/check component cost (2 calls, OLD R1 writer).
The latter supplies no output lower bounds or full positive proof. Do not pool
traced/untraced times, claim a microbench is an endpoint gain, or change paper
guarantee/competitive claims from this experiment alone.
