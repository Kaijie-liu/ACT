# C77: exact integer memoization without deleting checkpoint content

C76 establishes6749869 integer encodings but only1300550 distinct in-range
values in the complete actual native checkpoint. C77 tests one uniform wire
rule: built-in integer values257<=v<2^24 use a fresh disjoint memo index, first
as original integer value plus explicit PUT and later as GET. Bool, other ints,
floats, array/tensor bytes, mutable objects and all other semantics are unchanged.

The complete source uses16162 original MEMOIZE entries and no explicit PUT.
Transform every original MEMOIZE into an explicit original index; leave every
original GET bound to its original index and require a defined preceding entry.
New integer indices begin after the complete original memo population. Check
all original counts and EOF. Reframe only on whole-opcode boundaries; large byte
payloads stream unchanged outside frames. Full source/output hashes and complete
opcode/payload counts are retained. This changes neither fields nor numeric
sharing, and never executes constructors during conversion.

Implement monotonically increasing integer prefixes in NumPy batches, with a
fixed uint32 memo table of2^24 entries. A non-increasing prefix ends and the same
rule resumes; no value/instance menu, field omission or mutable deduplication.
All eligible scalars follow the same memo rule, including short prefixes. Failed
format/precondition/resource checks produce no qualified checkpoint.

New diagnostic prices: table initialization one per uint32 entry;16 per batch
integer for comparisons, lookup/store, offset and output writes;32 per batch
header;8 per other opcode plus scalar byte width for non-batched payloads;
large byte/string payload read+write charged2*ceil(bytes/8). Small non-batched
eligible integers pay32 including the same memo operation. Full file hashing
is separately reported. Derive the source-bound bill from complete C76 counts
and actual supported run populations; do not retune it after results.

Default-off local codec only; no production, C41, HZ math, source proof, network,
solver or archive mutation. Ordinary tests compare original and transformed
pickle decodes, exact values/types, arrays/read-only flags, object-sharing and
distinct mutable objects, original memo references, frame/payload boundaries,
source/native proof binding and inverse. Retain all1841 previous tests.

Exclusive results/c77_integer_memo_20260913_v1. Freeze sources/provenance and
the directly used pickle/pickletools stdlib sources before all83files/1854tests.
Then a complete preflight with output=None reads every original byte and hashes
the exact would-be output, followed by one complete conversion whose report,
work parts, byte length and SHA must exactly equal that preflight. Finally one
complete decode/source/native/inverse attempt; automatically preserve all
logs/results/exits. CPU1/GPU0,AS16GiB,whole256M per independently bounded stage,
64M entries,both1GiB; tests60s/preflight60s/conversion60s/restore60s. Parent hard
deadlines retain last flushed stage events; no soft-alarm qualification stages.
Hash authentication calls retain the original separate per-call accounting;
report all calls and totals, without an all-CPU256M claim. No warmed/reset entry or tracer
exemption. Every model/cache/caller/bounds/source/phase/proof root is retained.
The complete original archive remains read-only; a transformed file is new
evidence, never a rewritten C74 result. No smaller-body substitution or network
rerun. Full restore must actually pass before terminal/witness/shadow work.
Formal1870/all13 and separate E0 CIFAR25/Tiny36 unchanged. Goal ACTIVE.
