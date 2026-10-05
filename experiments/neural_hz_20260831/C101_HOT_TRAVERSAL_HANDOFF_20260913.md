# Next: remove measured ordinary token allocation overhead, not any checks

C101v1 is CLOSED; all jobs terminal, no solver run/point/score gain. Its real
original-runtime C78 walks take24.3--24.7s EACH under the required tracer, with
6.74M integer visits and about539MB exact token bytes per walk. Preparation's
2.9s number is not a runtime estimate. Do not retry C101 unchanged, reset or
disable measurement, widen time/work/storage caps, or skip any traversal.

The next narrowly scoped implementation hypothesis is supported by the actual
hot population and the unchanged c78_integer_prefix_native_v1.c source: each
integer occurrence allocates a decimal Python string and a pointer-page key,
even when adjacent global IDs reside in the same pointer page. Thus millions
of ordinary per-integer allocations run under tracing, although the actual
token output buffer and identity-page storage are already compact. These are
identities of ordinary neural variables, not an extreme numerical test target.

Try one exact token-loop improvement in NEW versioned files only: direct
machine-word decimal formatting for ordinary built-in integer values (retain
the original interpreter fallback for other existing supported values), and
a bounded last-page lookup cache that never changes pointer-identity bits or
object ownership. No token/path/schema/identity/sizeof/numeric hash may differ
from the original C5 oracle. All heap memory stays visible, stack workspace is
explicit, and no work discount is claimed without an operation argument.
Keep C101's shared256M diagnostic pool and every source generation/native cap.
This is a bounded hot-loop change, NOT a compiler/test-framework feature or
new serialization campaign; reuse the already frozen C78 build workflow.

First require an actual measured improvement on the complete ordinary integer
loop shape under tracing and the inherited full C5-equivalence tests, including
the new circuit-root shapes. Freeze new source/C/binary/build provenance,
inherit ALL2556 old tests and add only relevant checks; collect+execute<=60.
Only then another fresh original network240s / ordinary base+property MILP45s,
with unchanged complete input/proofs, source/native/LIVE/comparator/witness
guards. Add cheap before/after events for the final measurement/reference/walk
boundaries, so a timeout is no longer ambiguously located after root3.
Do not rerun a full model just to collect those extra markers.

If this bounded payment does not have a demonstrated win, do not keep growing
diagnostic wrappers. Preserve C100's qualified source/native/LIVE evidence and
return to the exact repeated affine/predicate generation program; further work
must buy real end-to-end terminal capacity, not merely a passing local helper.

All199 unit/100965 local/7357 circuit inverse relations and1350 binaries survive.
No LP-status/marginal/ray repair, attack/PGD/BaB/split/backward/dual rescue,
binary pivot, convex replacement, identity selector, cap expansion or base
bypass. New files only; all old experiments and /data1/Kane/HyZor read-only.
No production/default/commit/push. The full1870/all13 retention,2413 endpoint,
E0 CIFAR25/Tiny36 and400 replay, later family/generalization/PLDI goals remain
ACTIVE and unchanged. A terminal UNKNOWN is evidence, never score or target
capability admission. If an actual point appears, save then fully reconstruct
and validate it against the original model/property before any credit.
