# Four complete upstream integration calls: frozen, not executed

Implementation `f916ac82f`;264controlsPASS/101.810s;619source bindings.
Config `configs/backend_controls/readonly_upstream_r1.json`, SHA256
`325f544fb9cfe3212c4c014d4dcc02a230cf09697df5d6ea8fa229037658d169`.
Protocol: `readonly_upstream_protocol_20260925_r1.md`.

Exactly4 new complete source-generation/propagation/construction/publication/
checking/reception calls. Prior fixed recipes E4/C3/w4/d1 and E8/C10/w8/d2,
seed724, once direct/readonly each; small off/on, medium on/off. No stored
source or matrix can shortcut upstream work. No extra sizes/repeats, real
requests, native output solves, changed serializer or relaxed acceptance gate.
Each300s total,2sreserve,8GiB sampled RSS,2threads. No retry or late rescue.

Timing is descriptive integration evidence, not an estimate of stable speed.
All calls/statuses/phase and outer costs retained. Old microbenchmarks and
real sealed requests remain untouched; zero new output certificate implied.

After this freeze is committed/pushed and resource admission passes:

```sh
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -m readonly_upstream.run verify
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -m readonly_upstream.run execute
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/audit_readonly_upstream.py --report docs/readonly_upstream_audit_20260925_r1.json
PYTHONDONTWRITEBYTECODE=1 /data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/audit_readonly_upstream.py --report docs/readonly_upstream_replay_20260925_r1.json
```

The audit forbids producer/new-parser imports and native solvers, rechecks
every saved construction with the original checker, compares byte identities
between both independently generated arms and closes every terminal cost.
Later result report supersedes this NOT EXECUTED status.
