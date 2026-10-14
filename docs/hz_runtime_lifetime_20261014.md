# Runtime and lifetime bounds for the direct evidence proposal

The [unique-root proposal](hz_unique_root_protocol_20261014.md) has a conditional
archive-format envelope, not a complete runtime bound. This follow-up narrows
the remaining gaps using existing source, environment files and four small
binary slices. It adds expression-width and native-loop sub-ledgers; it does
not implement a new checker, change quotas or admit a model/GPU request.

Starting state: clean `feat/moe-route-verification`,
`f88d62478a5ce55b4dd86d87ff1fafc8daf83107`. The [static report](hz_runtime_lifetime_20261014.json)
is reproducible by [its standard-library assessor](../scripts/assess_runtime_lifetime.py).
The original protocol and failed capacity assessments remain unchanged.

## Exact results and temporary arithmetic have different bounds

`Work.exact(q)` checks the reduced result **after** Python evaluates `q`.
The retained 8,192-bit acceptance limit therefore does not bound every temporary.
For numerator/denominator magnitude bounds `(a,b)` and `(c,d)`, use:

| Operation | Conservative expression magnitude bounds |
|---|---|
| Addition or subtraction | `(max(a+d,c+b)+1, b+d)` |
| Multiplication | `(a+c,b+d)` |
| Division, with nonzero divisor separately checked | `(a+d,b+c)` |
| Comparison | Integer cross products bounded by `a+d` and `c+b` |

Absolute value and negation preserve magnitude. Reduction cannot increase these
bounds. Applying the recurrence to already checked B-bit operands, B=8,192:

| Existing expression | Numerator bits | Denominator bits |
|---|---:|---:|
| Binary sum | 16,385 | 16,384 |
| Product | 16,384 | 16,384 |
| Fused `a+b*c` before the outer check | 24,577 | 24,576 |
| Four absolute differences in the inherited ReLU error expression | 65,540 | 65,536 |

Fused operations occur in [scaled affine reconstruction](../scoped_source/check_scaled_references.py:65)
and [reference norm accumulation](../scoped_source/check_source_dag.py:144).
The [ReLU compensation expression](../scoped_source/check_source_dag.py:233)
adds one center difference and at most three local differences with one outer
check. The last row follows that expression tree; it is not permission to
accept a result above 8,192 bits. V3 uses reference guards, so the unrelated V1
expanded-guard sum is not charged as a V3 operation.

These are integer magnitude bounds, not a bound on all simultaneous bigint
objects, gcd/division scratch, allocator-retained pages or elapsed time.
The pinned local `fractions.py` confirms cross-cancellation and gcd calls;
installed integer headers alone do not establish the C implementation's full
workspace bound. The earlier 16,385-bit example was never a whole-program bound.

### Ingress and retained state also need explicit contracts

Two old parsing gates differ. [Candidate rational intake](../scoped_source/hz_request_store.py:63)
parses up to 2,470 characters before checking 4,096 bits; a rejected integer
literal can temporarily reach 8,206 bits. [Request literal intake](../scoped_source/tensor_rows.py:205)
allows up to 4,096 characters, not 4,096 bits: a decimal integer can reach
13,607 bits before downstream domain operations. Clipping can make the eventual
input domain small. The new reader must not silently reject an otherwise allowed
request merely to fit the candidate's bit profile. A field-specific typed
encoding must preserve these distinct domains; the proposal's source-header
reserve is not a substitute for that derivation. No new literal limit is set.

The installed `Fraction.__hash__` uses a global LRU of at most 16,384
numerator/denominator keys. Scale-keyed dictionaries in
[weighted support](../scoped_source/scaled_endpoints.py:38) trigger it.
Returning from a row or support call therefore need not release these integers.
In a **fresh checker process with all hashed operands independently bounded by
8,192 bits**, key magnitudes occupy at most 33,554,432 bytes. Tuple/hash/LRU
objects, integer headers and allocator storage are additional. A long-lived
process with unrelated prior Fraction hashes does not inherit this bound.

The current session also retains roots/objectives, candidate bytes plus decoded
records, alpha/residual dictionaries and receipts; an exception traceback can
retain the local frames. A future fixed-slot layout is not the existing Python
object graph. Cache clearing, a new arithmetic implementation and early bit
rejection have not been introduced by this assessment.

## Native sparse work is now tied to a concrete path

The [current operator](../act/back_end/moe/scaled_forest_endpoints.py:24) prepares
one global COO bank and its transpose. In the pinned upstream implementation,
COO-by-dense multiplication allocates a zero output and a separate result, then
scans the COO entries; it does not coalesce at each multiplication. See
[SparseTensorMath.cpp](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/sparse/SparseTensorMath.cpp).
The explicit double AXPY template uses a dispatch stub and direct loop, rather
than the BLAS overload. See [CPUBlas.h](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/CPUBlas.h)
and [BlasKernel.cpp](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/cpu/BlasKernel.cpp).

Inspected local binary slices support this particular call chain: the double
worker calls the double template, which dispatches to the registered DEFAULT or
AVX2 implementation. The report binds their offsets, lengths and SHA-256 values;
it does not claim a full-library hash or dynamic dispatch proof. Merely finding
MKL symbols elsewhere in the library would not identify this path's workspace.

Let N=19,496, F=32,048, L=6,732,720, R=277,280 and E=504. Width eight has
pair-local groups 8,8,2, so summed widths are E, not `84*8`.

| Conditional native sub-ledger | Width 1 | Width 8 |
|---|---:|---:|
| Tiles T | 504 | 84 |
| Sparse products `257T` | 129,528 | 21,588 |
| COO AXPY calls `257TL` | 872,075,756,160 | 145,345,959,360 |
| Coordinate updates `257EL` | 872,075,756,160 | 872,075,756,160 |
| Logical COO index/value bytes `24*257TL` | 20,929,818,147,840 | 3,488,303,024,640 |
| Cumulative two-output payload `16E(129F+128N)` | 53,461,739,520 | 53,461,739,520 |
| Largest single-call two-output payload `16Fw` | 512,768 | 4,102,144 |
| Per-call select/product payload summed over DAG levels `16Rw` | 4,436,480 | 35,491,840 |

Persistent operator tensor payload is at most `48L+24R+8F=330,081,664` bytes.
These upper bounds use the registered topology envelope, not observed trained
sparsity. Cumulative allocation is not peak RSS, logical reads are not physical
DRAM traffic, and loop counts are not seconds. Batch width reduces sparse-product
and COO traversal call counts without reducing this bound on total coordinate
work; actual overhead or elapsed-time reduction is unmeasured. No native call
was run.

Remaining native items are specific: dispatch registration/overrides;
`index_add_`, boolean indexing and TensorIterator; simultaneous old iterates
during RHS evaluation; Python construction containers; runtime/allocator and
mapped pages. In particular, float64 `isfinite` uses an absolute-value tensor
and several boolean operations, not merely one mask. See
[TensorCompare.cpp](https://github.com/pytorch/pytorch/blob/5811a8d7da873dd699ff6687092c225caffcf1bb/aten/src/ATen/native/TensorCompare.cpp).
The table cannot be substituted for a complete native peak bound.

## Live capture and phase release are separate unresolved interfaces

Historical [training bindings](../act/pipeline/moe/configs/schedule_confirmation_100_selection_r1.json)
identify three checkpoints and their state hashes, 52 tensors and 6,961,368
parameters. The new format has not selected or loaded one of them. Registered
shape evidence is seed1/2 recipe metadata; a matching seed0 parameter count is
not independent topology reconstruction.

The [current factory](../act/back_end/moe/factory.py:80) calls
`torch.load(weights_only=False)`, initializes a new model, copies state and
returns both model and payload. Its constructor probes CUDA and may preserve
device RNG state even with CPU map location; Linear allocation also depends on
the global default dtype. Checkpoint file size does not bound this general
unpickling interface. The [training writer](../act/pipeline/moe/train.py:234)
supplies an intended state/config/metrics schema without optimizer state, but
does not prove the actual loaded object graph has a bounded layout.

[Capture](../scoped_source/capture.py:27) then retains base64 parameter strings
and performs full-tensor identity decoding. Raw promoted parameters occupy
55,690,944 bytes and the source with input center 55,715,520 bytes; these are
payload counts, not a live-capture peak. Input materialization must preserve
the registered float64-before-ToTensor order, not substitute float32 division
followed by a cast. Accepting an already materialized input would not erase its
generation cost from an end-to-end comparison.

The [old supervisor](../scripts/run_scaled_dense_capacity.py:59) samples the
leader's RSS and waits for that leader. It only calls `killpg` while the leader
is live. A surviving descendant is not excluded by these checks. Thus the new
formula `parent + max(phase peaks)` needs a process-tree lifetime premise, not
just sequential calls to this supervisor. This is a static missing guarantee,
not an observation that a historical run leaked a child.

## Bounded next design and stop conditions

The complete-runtime gate remains **NOT CLOSED**. The next design work is now
limited to these concrete interfaces, not another capacity retry:

1. Specify a restricted checkpoint/input adapter: bound metadata, tensor shapes,
   dtype/strides/storage aliases and coefficient bytes; bind the existing artifact
   and state identities; preserve graph and preprocessing semantics. Avoid
   unnecessary random initialization and device probing. Hash checks alone do
   not prove conversion correspondence. Do not implement or select a new live
   request under this static authorization.
2. Specify exact-kernel storage against actual expression widths and ingress
   fields, including hash caches and exception frames. Bind the runtime and
   allocator evidence or propose a separately controlled bounded implementation;
   do not clear a cache or insert stronger acceptance gates invisibly.
3. Close the named native temporary inventory and complete-worker release
   contract before summing phase peaks. Include reception, cleanup and terminal
   publication in the same 300 seconds. Inability to finish remains a timeout,
   not an excuse to omit duties or promise that an RSS stop ensures completion.

No full quota, absolute ceiling or runtime forecast follows from these partial
bounds. All 252 duties, both candidate roles, mathematical checks, original
numerical gates and historical refusals remain. This stage does not change G1–G6
scientific status or justify new samples, real/GPU runs or dependency changes.

## Reproduction and review

Nine new standard-library tests check the arithmetic recurrences, old ingress
distinction, batch accounting, static identities and explicit refusal of a
full-cost claim. The saved report regenerates exactly in `act-py312`, without
importing Torch or loading checkpoint/data arrays. Source and environment files
plus 4,003 binary bytes are read; this is not execution of those kernels.
Three read-only agent reviews examined arithmetic, native work and capture
lifetimes; they are not independent human review or production controls.

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S -c 'import sys,unittest; sys.path.insert(0,"/data1/Kane/MOE/ACT"); unittest.main(module=None,argv=["unittest","scripts.test_runtime_lifetime","scripts.test_unique_root_protocol","scripts.test_wire_capacity","scripts.test_project_layout","scripts.test_review_handoff","scripts.test_manuscript_source_contract"])'
```
