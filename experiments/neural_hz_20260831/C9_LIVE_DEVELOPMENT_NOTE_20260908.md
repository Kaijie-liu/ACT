# Pre-target live-interface development checks

Coding of C9_LIVE_RELU_PREREG_20260906 resumed after interruption on 2026-09-08.
No live experiment was running and no c9_live result directory existed on
resumption. The C9_RADIX checkpoint manifest verified successfully.

Focused test development, before any real-target runtime observation:

1. 12 passed / 9 failed. Seven malformed-guard fixtures tried to assign fields
   on the native frozen expression dataclass. Two assertions required Python
   identity of b/ub vectors, whereas SparseHZono preserves their storage via
   reshape views. These are fixture/identity-assertion errors, not target passes.
2. 17 passed / 4 failed. Four deliberately malformed fixtures were rejected by
   the native constructor before reaching the guard under test. Other valid
   structure, Fraction equivalence, native probe/followup and slot tests passed.
3. Malformed guard inputs now use an explicit schema proxy; all equivalence and
   real-consumer tests still use native expression/HZ objects. b/ub assertions
   require both exact values and shared memory; sparse predicate objects retain
   identity. Final focused run: 21 passed in 1.93 s.

No C9 coefficient algorithm, native option, threshold or resource cap was
changed in response to these development tests. Actual native ReLU construction
is now measured under the same 1GiB gate, and all apply arguments and cached
value views are registered in the whole-state numeric ledger. The exclusive
supervisor reruns the frozen 355 prerequisites plus these21 before any target
execution. A target failure closes the runtime version without a parameter retry.
