# C82 new NativeState final affine/property proof qualified

V1:1946 tests pass, but complete real preparation rejects its unrounded-real
center check. This new oracle was stronger than the existing native binary64
semantics and started the dot with bias, unlike the actual path. Its failed
source/result/exit remain immutable. V1 result237780b7071bfc250ab400508e94dd0b60d3150e4599e3e1e4e9ea98a4c22972;
exit f1d18de167e1b0a5ae5125545c2e087c679ca133841d4a8178f875a829225931.

V2: independent Fraction multiplication/addition with explicit per-operation
binary64 rounding, original contribution order and bias after the center dot.
No tolerance, changed coefficient, new solver path or weakened unit/local
identity. No unrounded-real affine exactness claim. New ordinary non-dyadic
weights distinguish the two semantics; all24 v2 tests pass1.46s in development.
The frozen run additionally compares final center bytes exactly.

All88 qualification files/1970 exact collected and JUnit nodes pass.
Tests29.10394410043955s including collection; pytest22.78s (see tests.log for
the authoritative display), one unchanged TypedStorage deprecation warning.
Supervisor24163 terminal exit0,60.76841334812343s; source/provenance unchanged.

Actual complete proof preparation30.080989659763873s,145501401work. Complete
C77 checkpoint/C79 original sharing/C78 whole-root ledger and C74 source/native
binding pass. Same input466368992B/42949161entries strongly retained. This stage
depends on C81's successful full inverse and explicitly does NOT claim to rerun
that inverse or requalify it under a split budget.

Original actual ReLU78->DENSE79->ASSERT80 topology,200x200 weight/bias,
all40000 generator products and200 final centers reproduce original binary64
bits. All EQ/INEQ matrices remain shared by identity and both RHS arrays keep
original storage. Original input9408 dimensions, shape[1,3,56,56], all199
TOP1_ROBUST coefficient/threshold rows and polarity match the original output
specification. The final HZ hash is NEW, not the C34 result.

- Final proof file SHA4257d022e5c268d534959b8f6cf084b4ca372d1748504f156f81bf37be30c0c5.
- New final HZ5326a35ead708bdda01559f56ca346b41a8c81a9d63262c5d4c3f7c1c90fae0d.
- Original post HZba3a84a1d508db6330df50e81c688747755134a55a1f097a698a789b927723d1.
- Original input HZf8f15a3928f239d57cedd38d10732183a2cc85de2f98c41e8f3ab6797c883fff.
- Result2d63016795f9f6700fbe4fee748efb96b97b14e4875c56d5eefe38f0a67f47f7.
- Exit2500fda170e49b9ef98c457a474cc4fbfcefd24eb6e94aa1db4caf99c0a59773.

Both1GiB gates: entry578785280B,HWM1573679104B,growth994893824B;
traced713087020B+metadata76729440B=789816460B. Complete source/native
authentication129744022work and final full-HZ hashes44843522work are separate,
both reported; no whole-CPU256M or speed-promotion claim.

The NativeState witness adapter restores every C81 unit/local equation and
independent original row for ordinary nonzero native points, retains original
input coordinates and matches unchanged HZSolver._recover_input bit-for-bit
on fixtures. No ACT numeric/default code was edited. No actual solver point,
concrete network witness, full terminal LIVE gate, target/shadow/family/full
replay or score gain yet. Formal1870/all13 and E0 CIFAR25/Tiny36 unchanged.

Next is the changed fresh original-network terminal, with a source-bound
runtime final checker, complete new live ledger and model/point observer using
the new journal. Do not repeat a serialization campaign or the completed audit.
