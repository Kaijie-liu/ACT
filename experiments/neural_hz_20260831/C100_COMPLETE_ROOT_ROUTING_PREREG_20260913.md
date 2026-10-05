# C100v4 — route the same complete proof roots through qualified C78

V3 full117files/2550tests passed40.607s. Complete fresh original-input/affine
proof and runtime251371733/199827989 bound passed, full preparation roots
413197584B/36375547entries. RSS growth1095606272B exceeds unchanged1GiB by
21864448B; traced651712628+130596736 passes. Version CLOSED; no original solver
worker started and no positive/default/score admission.

The stage events locate the excess in complete root traversal: after full
model+native binding HWM1528340480B, after root collection1682337792B. The
current caller used C5's Python integer-set/token traversal although C78's
complete equivalent native integer traversal is already qualified and inherited
in the mandatory suite. V4 reuses unchanged c78_complete_roots_v1.collect for
this one preparation call, with every original root, source payload, full
fingerprint token, unique identity, metadata and numeric owner preserved. Its
extra work and buffers are paid and measured. No C78/compiler/serializer feature
change, omitted checks, prewarming, tracer reset, smaller model or cap expansion.

All other C100v3 semantics and conditions unchanged. All117files/2550tests still
collect+execute<=60s; preparation and fresh worker each240s, ordinary terminal45s
with base feasibility and complete native fidelity. No source, runtime, LIVE,
inverse/witness or promotion gate waived. This is proof-root implementation
routing only, with zero standalone Neural-HZ gain credit. Full research goal
and1870/all13/separate E0 retention remain active and unchanged.
