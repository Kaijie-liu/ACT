# Integrated V2: empty native frontier is not a composed descriptor

Integrated V1 genuinely publishes ReLU20 (11516 continuous/1054 binary slots,
1054 equality/2108 inequality rows), but loses HZ at Conv25 on the unchanged
quarter-product gate. The loop's later stop at ReLU36 is NOT HZ completion.
V1 is closed as a failed ReLU36 candidate; all records/snapshots remain intact.

Read-only reconstruction from sealed ADD24 snapshot
be963b1244123de480af1ba5f088a2e4db8b5b202acf3ac4acc16522e91ab336
establishes ReLU28 has 25088 stable-negative rows, zero P/U and zero requested
native materialization rows. All three source terms have the same two-Conv
shape after appending Conv25, but no composed coefficient is demanded. The
current compiler deliberately reports quarter_product_gate=False for zero
baseline work, so selecting it here was an over-broad dispatcher precondition.

V2 adds ONE uniform pre-selection condition: if the valid requested row mask
is empty, use the original native zero-row materializer directly, without
constructing C5, allocating its budget or changing any quarter-work rule.
The original zero-row composer makes no implicit kernel-row request, retains
all source/frame/continuous/binary predicates and adds the global bias once.
It is not a Zono/interval replacement and not a fallback AFTER a failed C5
selection. A poisoned/previously-selected island still rejects. The C5 compiler,
its thresholds and existing tests remain byte-identical.

Tests must prove zero-row transfer never instantiates the candidate compiler
or queries an implicit Conv row, all full HZ fields match the original native
result, and factors/predicates/bias survive. Existing default-off/unmatched,
successful two-stage and selected-failure-no-rescue tests must pass for V2.

Then one new fresh INPUT-to-ReLU36 prefix under the same qualification,
corrected graph/config, 240s/16 GiB and unused 45s solver cap. Log the explicit
zero-frontier selection and every nonempty C5 call. Treat missing target HZ
as failure even if the analyzer loop reaches layer36. Preserve native ReLU,
publication/GC and all budgets. No selected retry, no gate relaxation, no
formal/default changes. Exclusive results/c5_integrated_prefix_20260905_v2.
