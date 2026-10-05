# C79: restore original encoded tensor-storage identity, not equal-value merging

C78's complete ledger proves141 extra torch storages/34292008B/4286501entries;
all NumPy numeric content is unchanged apart from the already-accounted1600B
readonly copy. Local torch source and two original payload headers demonstrate
explicit repeated original storage keys in independent legacy save bodies.

Uniform decoder rule, scoped to one externally authenticated complete archive:
parse each torch.storage._load_from_bytes body's five legacy metadata pickles
without constructors, requiring the registered ordinary CPU typed-storage
schema, explicit original storage key, dtype/numel and exact payload extent.
For the first origin key, invoke the original torch loader. For a later equal
origin key, require identical dtype/shape-of-storage/header and every raw byte,
then reuse that original decoded storage. Different keys never merge, even
when all tensor values agree. Mismatches fail closed. Tensor object memo,
shape, strides, storage offsets and all other reducers remain unchanged.

Keep C41 complete input hashes, readonly array owner handling and temporary
retirement; add only this default-off local storage map. There is no global
cache or monkeypatch. All temporary map/cache buffers stay inside the active
tracer/RSS scope, and the entire checkpoint remains strongly held. This is
not a safe loader for untrusted files; only the already hash-anchored original
C77 checkpoint is authorized. No new archive, network rerun or smaller input.

Additional price per storage body before parsing/loading:4096 header work plus
2*ceil(raw_bytes/8) for complete hash/comparison. Original decoder, C78 complete
root tariff and separate source/native authentication accounting are retained.
Record every original storage key/header/hash and aggregate calls/groups/reuses.
Retain all1875 previous tests, add ordinary shared views/offsets/strides and
distinct equal-storage fixtures, and the combined C77 wire/C79 decoder path.
Freeze new sources and exact installed torch serialization/storage code before
the complete qualification; no tuning after the actual target result.

Exclusive results/c79_storage_identity_20260913_v1. Tests60s, full restore60s,
CPU1/GPU0,AS16GiB,256M per diagnostic stage,64M entries,both1GiB unchanged.
The stricter original42948961+copied-entry envelope stays. Full C78 collector
and unchanged numeric owner ledger, C74 source/native proof bindings, original
cache and journal sharing, all199 unit+100965 local equations and11708 original
coordinates must actually pass. Fixed-zero inverse is not a feasible witness.
Automatic events/stacks/results/exits and hashes; no soft qualification limit.

No production/default/history/score change. Formal1870/all13 and E0 CIFAR25/
Tiny36 retained. After complete restoration qualifies, go directly to actual
terminal/witness/shadows/family/full2413 and separate400 under original gates;
do not substitute further wrapper or corner-case projects for the HZ objective.
