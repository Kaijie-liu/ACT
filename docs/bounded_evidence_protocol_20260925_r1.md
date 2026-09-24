# Bounded-memory canonical evidence publication controls (R1)

Scope: synthetic controls ONLY. Do not rerun sealed4099, recover its unsaved
matrices, change a real request, query count, acceptance policy, time or memory
cap. No production default or frozen implementation is modified. Starting
commit d36683f002c3fd1eaaf60811f05adfe3e2bac201, clean/synchronized.

## Single change and memory boundary

New `bounded_evidence.stream.save` emits the SAME canonical bytes as
`scoped_proof.io.save`: sorted string keys, compact separators, ASCII escapes,
finite floats, no newline; SHA256 and byte count must be identical. It does
not delete/deduplicate source matrices, properties, factors, traces or checks.
String values are escaped in 2048-codepoint fragments; UTF-16 surrogate escape
spelling, signed zero and binary-rational source strings stay unchanged.

Serialization writes blocks of at most65536 bytes with incremental SHA256.
Workspace is bounded independently of total array/string length, but includes
a depth-bounded recursion stack and per-active-dictionary sorted key lists.
Explicit admission limits: depth64,4096 keys/dictionary,4096 chars/key,12000-bit
integer scalar. Exact built-in JSON types only; string keys avoid coercion and
key collisions. Large rationals remain strings. Unsupported objects fail
closed, WITHOUT an automatic fallback to whole-buffer serialization.

This is an ADDITIONAL SERIALIZATION workspace claim, not bounded total request
memory. The full constructed object remains resident. Construction, repeated
source parsing, other whole-buffer identity() calls and downstream load() are
unchanged. No claim that fixing this peak closes a request whose construction
already finished around283s of a300s budget.

## Publication and acceptance

One inherited absolute deadline, never renewed. Exclusive `.partial` creation;
short-write-safe incremental output and hashing; file fsync; exclusive atomic
hard-link to final name, partial cleanup and directory fsync. Existing final or
partial paths, including symlinks, are never overwritten. Killed/failed partials
are retained; final files without a timely bound receipt are not accepted.
Directory/filesystem I/O can block; the EXISTING owned-process watchdog remains
necessary. A late link cannot turn an expired computation into accepted proof.

The control-only worker changes construction publication in the existing
seven-phase synthetic pipeline. Original intake, route/source checks, native
LP candidates, exact output checks, all-original-duty aggregation, receipt
validation and audit run unchanged. Diagnostic serialization metrics are NOT
proof receipts. Their writing is charged inside the construction phase.
No real runner is connected to this control adapter.

## Controls fixed before measured probes

- Exact byte/hash differential: scalars, signed zero, finite extreme floats,
  escaped/unpaired-surrogate Unicode, nested structures, rational strings,
  long strings and CSR-shaped arrays; same synthetic full construction bundle.
- Cycles, nonfinite values, bad types/keys/size limits fail closed. Repeated
  references are serialized twice just as before; no alias compression.
- Deadline before/mid-write/after-link, short/zero writes, disk errors, fsync
  and link failures, preexisting partial/symlink/target race.
- Full synthetic proof under300s, unmodified fresh solver-free checker,
  relocation, deleted obligations, altered factors/coefficients/pairs,
  wrong run/hash and missing completion receipt rejected.
- Real owned-worker cutoff during blocked publication and injected disk error;
  keep partial, no acceptance, cleanup included; shortened8s synthetic fault
  tests only, never a new real request budget.
- Run all121 existing residual/source/checker/watchdog regression tests.

Measured probes (NOT a real model experiment): two predetermined synthetic
sizes2^18 and2^21, two publication implementations, once each. Orders: old/new
then new/old. Every call gets30s and sampled8GiB using the same watchdog; CPU
threads2, no GPU. Payload combines rational CSR-shaped lists with a long escaped
string. Record generation/imports, serialization/fsync, report/publication,
cleanup, full supervisor wall cost, output bytes, exact hash and Python traced
serialization peak (after object generation). Also record OS high-water RSS;
do not confuse that with traced additional memory or an allocation-level profile
of real4099. All outputs/stops retained, no retries or replacement sizes.

Control acceptance: exact bytes/hash match, all old checks pass, partial/late
artifacts rejected, finite closed costs, streaming chunks<=64KiB and traced
serialization peak<=1MiB on these fixed probes. Runtime is recorded, not an
improvement gate. No endpoint/speedup claim follows from passing memory controls.

## Exit

Archive PASS or failure, source identities and full costs. Fresh saved-only
audit checks source bindings, complete output hashes, byte equality, cost and
synthetic full-proof evidence without a model/native solver. Keep frozen source
bindings and historical source-gap/competitive conclusions unchanged. Later
real integration, construction/check cost optimization and any new real freeze
require separately scoped work; no automatic execution is provided here.
