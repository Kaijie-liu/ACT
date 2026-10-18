# Native declared-source generation contract — static, 2026-10-18

Base: `0f2ef5a4b49681f23fc7ddcdc042668940787933`, clean
`feat/moe-route-verification`. This is the next static ledger after
[storage/lifetimes](hz_native_storage_lifetimes_20261018.md), not a new runtime,
resource policy or experiment. The [recomputable record](hz_native_generation_contract_20261018.json)
binds 44 files; the standard-library-only
[analyzer](../scripts/assess_native_generation_contract.py) constructs small
schema templates, not a source forest or a model. The approved finite CPU
[controls](hz_native_controls_results_20261018.md) remain completed, not pending.

## Decision and scope

The registered eight-expert declaration now has a **generated forest-header
bound**, rather than only an acceptance cap. Complete successful archive sizes
are parameterized by the actual source layout and frozen metadata. Large Python
constructor containers have a separate, ABI-bound component ledger.

**This does not close the complete self-owned heap or grant full-shape admission.**
The unchanged 8M Work, membership and candidate-stream refusals still apply.
No quota, acceptance condition, 300-second budget, runtime or production parser
was changed; no native/model/GPU execution occurred. These are supporting
resource-contract results, not new G1 algorithm gains, G2 acceleration, G3 real
certificates or G4 external wins. Historical source gaps remain.

## 1. Generated forest header

For the fixed plan (router four operations, eight experts with six each), use
`N=19496` nodes, `F=32048` factors, `R=277280` references, `H=136232` roots,
82 views and 2176 first-layer rows. These are conservative complete-topology
counts, not observed model sparsity. All compensation factors, nonzero centers,
unequal scales, both endpoint roles and all 252 output obligations remain.

| Generated component | Bound, bytes |
| --- | ---: |
| One factor record | 48 |
| One view record | 116 |
| One alias / sequence span | 28 / 13 |
| Source receipt, including all 53 tensor hashes | 5,059 |
| Complete forest header | **1,652,449** |
| Binary forest payload | 5,329,156 |
| Complete forest packet | **6,981,605** |

The header arithmetic is

```text
1371 - 8 - 471 + 5059
 + (1 + 49*32048) + (1 + 117*82)
 + (1 + 29*2176) + (1 + 14*246) = 1652449.
```

Thus at most **107** 64-KiB pages are generated. The page-index bound used below
is `67+110*107=11837`, conservatively retaining nine-digit byte fields. This
closes this header demand; it does not change other independent limits. The
header carries source/request identities and a receipt, not another copy of the
source manifest, its shape list or optional trust metadata.

## 2. Source grammar and its actual representation

The 53 tensors (52 network tensors plus center) contain **55,715,520 bytes**.
Let `K` be their actual total chunk count. Full 1-MiB packing gives a *minimum*
of 95 chunks, not a promised layout. Singleton center dimensions are permitted:
3072 elements do not by themselves bind the rank to four.

Let `r` be center rank, `Xt`/`Xr` the compact tensor/request shape lengths,
`E4` the total encoded length of the four domain values, and `tau` the encoded
optional trust value. The canonical manifest length is

```text
20427 + Xt + Xr + E4 + 124*K
 + sum(digits(chunk_bytes)) + sum(digits(chunk_offset))
 + delta_model + delta_flatten + optional(9 + tau).
```

The accepted parser uses numerical equality for request shape, model-state
counts and Flatten dimensions. For example, `[true,3.0,32.0,32.0]` can compare
equal to integer shape `[1,3,32,32]`; it is not safe to assume identical JSON
spelling. Retain `Xt<=2r+4`, `Xr<=5r+4`, `0<=delta_model<=4` and
`0<=delta_flatten<=45` (nine dimensions lists). Thus the general accepted-plan
upper is

```text
36876 + 7*r + 138*K + optional(9 + tau).
```

The smaller `36827+4r+138K+optional(9+tau)` only applies to additional strict
integer metadata assumptions; it is **not** the general contract. This finding
changes the static accounting, not the production acceptance grammar.

The on-disk manifest is not required to be canonical. Write its actual raw size
as `Jmanifest<=4194304`. The necessary raw lower bound
`20433+4r+126K` still holds, yielding `K<=33125` and `r<=1040475`
(using `K>=95`). These necessary limits are loose; member admission is tighter.
They are not recommendations to create such layouts.

For successful finite JSON under the bound CPython encoder, canonical text
length `C<=8*Jmanifest`: string escaping expands by at most six, a finite float
token has at least three characters and encodes into at most 24, integers and
punctuation do not expand, and whitespace disappears. A named overlap bound for
raw bytes plus canonical ASCII string plus encoded bytes is therefore
`Jmanifest+2C<=71303168` bytes. **Decoded trees, object headers and encoder
internals are not included.** Source binary plus raw manifest is at most
59,909,824 bytes. The later copy is a private transfer/rename, not a second
simultaneously retained complete private source.

## 3. Complete archive, terminal and caller accounting

Use 84 candidate tiles, 32 result pages, 504 endpoints with both roles,
252 properties and maximum multiplier dimension 2700. The complete candidate
payload is 22,868,160 bytes; this exceeds the old 16-MiB stream cap at the
worst-case shape and does not authorize an increase. Bound code is 32 files /
274,543 bytes. Result pages and index are 1,261,868 and 5,431 bytes.

For `Ns=K+1` source files and 107 forest pages:

```text
package file count nP = K+261
sum of package relative filename lengths LP = 42*K+9264
request files <= K+274; including outside caller receipt <= K+275
```

The four successive census lengths are bounded by
`41349+167K`, `41460+167K`, `41573+167K`, `41682+167K`.
The production package condition is `nP<=4096-O-16`, where `O` counts the bound
outside source members. Do not silently set `O` or share source files for free.
Every individual byte/member/candidate cap also applies.

For field widths only, use the previous conditional complete Work upper
`W=165180263928`, `d=digits(W)=12`, and
`b=digits(134217728*W)=20`. These are **not new quotas**. Phase and progress
metadata bounds are `Phi=17425`, `G=17992`. Actual executable Work stays 8M.

Define all lengths in compact JSON **bytes**, including quotes and escaping:
`Pprofile` is the frozen profile, `Qroot` its encoded root path, `Qmargin`
the property margin, `V` the Torch version string and `Iid` the caller ID.
`Logs` is the aggregate of all four logs; it is not assumed zero.
`p` is PID decimal width, `Z` the encoded zombie-list length, and `rho` the
post-cleanup RSS decimal width. For a complete successful traversal:

```text
COO request <= 87337948 + Jmanifest + 631*K + Logs
              + Pprofile + Qmargin + Qroot + V + Z + p + rho

COO joint   <= 87381544 + Jmanifest + 798*K + Logs
              + Pprofile + Qmargin + Qroot + V + Iid + 2*Z + 2*p + 2*rho
```

The row-block constants are respectively **87,338,052** and **87,381,648**:
104 additional bytes, with the same variable terms. The analyzer independently
assembles actual actor, controller, terminal and caller schemas to obtain these
expressions. The outside caller receipt contains the returned terminal and last
census; its bytes are not inside the request-root census. Its write time was
already charged in the existing implementation: this is not a historical timer
repair.

No-live-process cleanup does not imply no zombies. Given a separately bound
count `z`, `Z<=max(2,1+z*(p+1))`; the current contract does not silently prohibit
all native forks. An OS observation such as `pid_max` is not an immutable
protocol binding. Unpredicted native output is subject to hard rejection, not
a promised complete-success log bound. Consequently these expressions do not
yet constitute one absolute successful archive size or a new admission profile.

## 4. Python constructor components and what is still missing

Bind existing `act-py312` CPython 3.12.12 x86-64: pointer 8, empty list 56,
empty tuple 40, float 24, small index integer 28 and Fraction header 48 bytes.
The JSON record hashes the local integer/dict headers and `fractions.py`.
For the inspected monotonically constructed containers, conservatively sum
old/new resize storage:

```text
A(n) = (n+floor(n/8)+6) & ~3; A(0)=0
LP(n) = 56+16*A(n)
t = ceil_power_of_two(max(8,3*n))
DP(n) = 64+2*(32+index_width(t)*t+24*floor(2*t/3))
SP(n) = 216+32*ceil_power_of_two(max(8,4*n+1))
TP(n) = 2*(40+8*(n+floor(n/4)+16))
```

These follow the fixed-version [list](https://raw.githubusercontent.com/python/cpython/v3.12.12/Objects/listobject.c),
[dict](https://raw.githubusercontent.com/python/cpython/v3.12.12/Objects/dictobject.c),
[set](https://raw.githubusercontent.com/python/cpython/v3.12.12/Objects/setobject.c)
and generator-to-tuple [sequence](https://raw.githubusercontent.com/python/cpython/v3.12.12/Objects/abstract.c)
implementation paths. They are not bounds for arbitrary containers with deletion
history or managed instance dictionaries.

With `D=64`, define the counted large constructor component

```text
H(X,Y) = 3LP(X)+LP(N)+LP(F)+28Y+56N+24(X+F)+DP(D)
         +152D+18R+88R+LP(R)+LP(D)+TP(D)+64D.
```

For COO, `X=6732720,Y=6684672`, giving **749,521,272 bytes**.
For row blocks, `X=48048,Y=0`, plus 729,672 bytes of row/block lookup containers,
giving **41,675,712 bytes**. These are conservative sums of named constructor
components, not simultaneous peak measurements or complete heap/RSS bounds.
They exclude forest objects, tensor payload/wrappers, some fixed instances,
interpreter/allocator storage and native workspace. They must not simply be
added to an unrelated phase maximum and called a measured peak.

In particular, a normalized integer's logical bit length is not necessarily its
allocated digit capacity: [normalization](https://raw.githubusercontent.com/python/cpython/v3.12.12/Objects/longobject.c)
can leave a larger allocation. Retained integer capacity and arithmetic
temporaries need their own lifetime bound; an 8192-bit logical acceptance gate
is not a complete memory contract. A conservative pre-check input parsing ledger
uses 16384-bit rational literals, 17460-bit endpoints and 34921-bit two-operand
intermediates. This does **not** weaken the 8192-bit Work acceptance gate.

Production closes its native operator before the final source seal; replay's
last caller frame can remain live. The seal overlaps new TensorRows metadata /
input Fractions with the checked forest, 504 endpoint receipts and old/new read
chunks. These phase joins, JSON trees, small fixed objects and integer allocation
capacities are still required for complete self-owned peak accounting. They
cannot be relabeled as exempt native overhead.

## Verification and next bounded step

Three separate read-only AI reviews checked source grammar/header, archive
schemas/census, and constructor ABI/components. The source review found the
equal-number/different-JSON-type case above; the general formula was corrected
before archival. These are not independent human review or empirical replication.

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/assess_native_generation_contract.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/assess_native_storage_lifetimes.py --check
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/assess_native_counted_envelope.py --check
git diff --check
```

All three static checks and whitespace checking passed. This verifies the bound
schemas/arithmetic and inherited identities, not a full-shape execution. No
production imports, source construction, native calls or dependency changes were
used to generate this record.

The next static contract should **join the remaining self-owned object lifetimes
and bind the remaining profile/source/cleanup parameters**, explicitly separating
already adopted hard-fail native workspace from self-owned representation. Do
not redo header algebra, repeat small controls, infer a quota from these partial
sums, or retry full shape under old limits. If a bound remains unavailable,
report that exact representation/policy decision. A new absolute resource
policy, implementation or execution still needs its own scope/approval.
