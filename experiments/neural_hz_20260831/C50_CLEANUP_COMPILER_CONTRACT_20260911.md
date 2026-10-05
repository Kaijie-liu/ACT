# C50 exact test-compiler cleanup contract

This is a verification-execution implementation change, not Neural-HZ
mathematics, source data, a score, a solver rescue or a relaxed test gate.
C49's unchanged full test run timed out before its real generator started.
The new executor must retain every original test in order and all assertions.

## Rejected plain-assert proposal

The initial default-off c50_assert_protocol_v1 canary rejected direct plain
Python assertion mode. On a false chained comparison, ordinary Python visits
a,b while the installed pytest rewriter visits a,b,c. On a successful walrus
expression, ordinary Python invokes the recorded call once while the original
rewriter invokes it twice. Thus equal test IDs and debug assertions alone do
not prove identical test execution. Plain mode is NOT used by C50. This
negative evidence is retained; the canary is expected to continue rejecting it.

## New cleanup transformation

Run the SAME installed pytest assertion rewriter, with the same source,
configuration, conditions, comparisons, evaluation multiplicities, diagnostic
branches and messages. Only its generated function-local private scratch
cleanup is considered. Python source cannot lexically name @py_assert*/
@py_format* variables. Module/class-scope storage is never optimized.

In each statement block, conservatively track definitely assigned private
slots. Direct completed assignments establish binding; deletion removes it.
If branches meet by intersection. Loop bodies start without assumed private
bindings (a previous iteration may delete them). Try/handler/finally/with
blocks also do not inherit speculative bindings, and enclosing states lose
any possibly deleted bindings. Unhandled structures remain unoptimized.
Nested functions are independent scopes. A reached cleanup cannot infer that
a short-circuited operand was evaluated.

At the rewriter's original `private_a = private_b = ... = None` cleanup,
replace a target by DELETE only when its binding is proved on every path
reaching that point. Possibly unassigned targets keep their original None
stores. Preserve left-to-right release order by consecutive same-kind groups.
All original conditions and failure/evaluation branches remain unchanged.
For a bound scratch slot, both operations release the same old reference at
the same point; the only namespace difference is absent versus bound-to-None
for an inaccessible private name after cleanup. Scope admission rejects
direct locals/currentframe/_getframe/f_locals/tb_frame inspection. No user
variable or numeric predicate/owner state is substituted by this rule.

Seventeen canaries compare complete outcomes, failure strings (only process
addresses normalized), evaluation traces, repeated short circuits and object
release order against the ORIGINAL rewriter. They retain the chained-compare
and double-walrus behavior above. The loop/temporary-owner canary releases
left then right before the explicit after event in both versions. Additional
tests cover branches, loops, try/finally, with and nested functions, module/
class exclusion, optimization rejection, wrong source/loaded code and omitted
asserts. The initial implementation's global AST location fill was rejected
by compilation and fixed before qualification; replacement nodes now copy
the original locations without modifying the rewriter's other metadata.

## Actual loaded-code authority, no cached test success

The new runner keeps assertmode=rewrite, normal plugins, debug mode and
Python optimization0. The guard uses a forced failure, not a removable assert,
and compiled candidates whose assertions were removed are rejected too.
Only the complete frozen test-path list gets the new compilation hook. Old
rewritten pyc files are not accepted for those NEW code objects; every one is
compiled from SHA-checked source. PYTHONDONTWRITEBYTECODE remains set, so no
old source cache or installed package is modified. Hooks are process-local
and restored; timeouts kill only that child process.

At collection, independently rebuild expected cleanup bytecode and compare
every loaded selected test and same-module function/helper/fixture against
it, including nested code/constants, flags, arguments, variables, free/cell
variables, line tables and exception tables. The immutable source hash and
actual code must both match. No old success flag, test name, pyc metadata or
source-only check can authorize an omitted assertion.

The complete original2104 ordered test IDs must match C49's independently
collected hash143258636d9b6840970829d30810ae55649671a30f0e40a43a63cd7302797c0e.
No skip/skipif/xfail marker is admitted. All original items and the16 new
compiler tests execute normally. Success requires2120 ordered call reports
and6360 successful setup/call/teardown reports with no skip/xfail. All phase
times and outcomes are saved incrementally for diagnosis if the60s gate
fails. These checks and compiler/auditor costs are INSIDE that same timer.

## Work evidence and limits

Full original75-source compile:1502 assertions retained;1498 cleanup groups,
6324 definitely assigned deletes,1143 targets still cleared to None.
Total rewritten code1482820->1471320 bytes. The fixed7-round alternating
100000-iteration passing-assert probe has identical checksums and median
0.00748717226->0.00732806046s (1.02171x). This is small, compiler-only evidence,
not a full-suite speed bound or any HZ/verification runtime payment. No claim
is made that it guarantees the remaining90 tests fit; the new complete run
must itself meet the unchanged gate once, without retry or threshold tuning.
