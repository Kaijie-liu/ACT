# Read-only source representation: finite cost freeze

Implementation `3f13d07d7`; controls:240PASS in90.181485s,611source bindings.
Config `configs/backend_controls/readonly_source_r1.json`, SHA256
`dbbcc3bfb9b740609aee19d91656ab6f314a0e1ec7147657c2120dccf29dbbf6`.
Protocol: `readonly_source_protocol_20260925_r1.md`.

Exactly18 saved-source checks, not real verification requests. Two already
archived synthetic constructions, each none/copy/readonly x3rotated repeats.
Each300s total including receipt/terminal,2sreserve,8GiB sampled RSS,2threads.
No retries, new propagation/solver queries, additional sizes/repeats or tuning.
Default unchanged; inherited checking predicates and all obligations retained.

Fresh source/certificate files and expected original checker verdict are bound
in each method. Read-only statistics keep legacy names: freeze=seal,copy=borrow.
Report all three arms, checker/whole cost and each paired difference. Import
cost is inside the checker segment for every arm; do not splice old medians.
This is a source-checking segment experiment, not complete MoE timing or SAFE.

Execution is authorized only after this freeze is committed/pushed and clean:

```sh
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -m readonly_source.run verify
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -m readonly_source.run execute
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/audit_readonly_source.py --report docs/readonly_source_audit_20260925_r1.json
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/audit_readonly_source.py --report docs/readonly_source_replay_20260925_r1.json
```

Audit forbids both cached parsers, new view, producer and solver imports; it
replays original source checks separately and validates all18 terminal costs.
No sealed real input reopened, no new lower-bound certificate or historical
source-gap upgrade. A later result report supersedes this NOT RUN status.
