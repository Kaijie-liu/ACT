# Finite mature-generator comparison V1 — protocol, not execution

Date: 2026-09-20. PI requested original-four-LP sparse import fidelity followed
by a finite comparison freeze. Starting ACT HEAD
`b3d91358e94313f4d84288fd6edbd9dc45405fd7`, clean feature branch.

## Scientific question and scope

Can one fixed SoPlex exact-mode call per original rational LP produce a point
whose feasibility/objective the unchanged independent checker accepts? This is
a **full-LP candidate-generation** diagnostic, not a fixed-basis arithmetic
comparison. SoPlex may select its own basis, presolve and scaling under fixed
defaults/options. No self-written elimination, alternative bases, fallback
algorithm, parameter search, new samples, retraining or changed relaxation.

The comparison is against preserved historical outcomes (zero checked feasible
U), not a new paired speed experiment. Old 20M Python visit limits and all old
failures remain unchanged; they are not a fair unit for measuring C++ work.
Historical full-network/HZ/F0 generation is supplied and excluded explicitly.

## Immutable subjects

Use the four complete original LPs, original coefficient semantics, statements,
properties and order from `docs/lp_diagnostic_v1_freeze.json`. The jobs also
match the later frozen modular roster exactly:

| Job | Pair | Property | Variables | A rows | E rows | Nonzero coefficients |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| input220_p0 | {1,2} | 0 | 7,397 | 2,890 | 1,441 | 246,558 |
| input222_p1 | {0,1} | 1 | 7,682 | 3,080 | 1,536 | 236,502 |
| input230_p2 | {0,3} | 2 | 9,482 | 4,280 | 2,136 | 348,915 |
| input232_p0 | {0,1} | 0 | 9,095 | 4,022 | 2,007 | 329,664 |

No difficulty-based replacement, partial roster, extra property or sample.
The machine freeze binds every original export, source/LP/statement identity,
reader and SoPlex binary, exact settings, unchanged checker and this protocol.
It reserves a new result directory; no R1 directory is resumed or overwritten.

## Exact input gate

The separate reader **has no optimize/basis/candidate call**. It parses LP text
with rational READMODE=1 / SYNCMODE=1 and reads every original column, row side
and sparse matrix entry through rational APIs. Names, rather than incidental
native ordering, identify coordinates. Variable bounds and the objective vector
must agree exactly. Inequality lower sides must remain unbounded; equalities
retain both sides. Sparse zero omission is allowed only for exactly zero values.

Export converts stored binary floats to their exact rational values and emits
fraction tokens without decimal rounding, coefficient pruning or scaling.
Physical text lines are wrapped at 4096 characters. The native objective omits
**only** the exact constant offset; original objective evaluation re-adds it in
the unchanged LP checker. The offset and translation are explicitly recorded.

Readback must pass before accepting any candidate. Future timed requests must
re-load, re-export, re-import and compare for themselves; preparation's files
and answers are not free inputs. CLI `--writefile` is not this fidelity gate:
the pinned source's implementation writes its floating-point model.

## Fixed external configuration

SoPlex8.0.3, upstream `13e2ab2467e0016d02116802ac4dc7a89560dbc1`, isolated
installation from `docs/soplex_compat_installation_v1.json`; no environment or
dependency changes. Exact settings are the hash-bound upstream
`settings/exact.set`: READMODE=1, SYNCMODE=1, SOLVEMODE=2, CHECKMODE=2,
FEASTOL=0, OPTTOL=0, ratrec_freq=1.2. Other parameters remain pinned-version
defaults. PaPILO is unavailable in this build; no fallback to another build.

Future candidate command: the pinned `soplex` binary, fixed `--loadset`, one
`-t<remaining_proposal_seconds>`, `-X=<new rational-point file>`, and this
request's exact exported LP. Rational solution parsing must require a complete
solution header/footer, legal distinct names and the declared nonzero count;
omitted coordinates are zero **only** under that complete exact-output contract.
Native status, printed objective and native infeasibility claims are untrusted.
No retries, basis reconstruction, point repair or decimal-to-rational guessing.

The old ten-second HiGHS call proposed a basis before Python arithmetic. This
new one-call SoPlex proposer replaces that **whole** proposal procedure; it may
use the remaining proposal window, not ten seconds plus a new free budget.
This deliberate difference must be reported, not called a same-algorithm test.

## One full supplied-LP budget and finite limits

One worker/thread, nice10, no GPU. Original clock starts before file loading.

| Boundary | Fixed limit |
| --- | --- |
| Load, validation, exact export/import/readback, native solve and candidate capture | absolute 218 s |
| Packaging, isolated original-LP check, result review | absolute 298 s |
| Terminal publication | absolute 300 s |
| Address-space cap | 8 GiB per worker/native/checker process |
| Native solver attempts | one per job; internal iterations are not extra attempts |
| Native/check log or other individual file | 128 MiB |
| Rational point file | 64 MiB |
| Total per-job output | 512 MiB |
| Serialized evidence numerator/denominator | at most 4096 bits each |
| Schema capacity | 16,384 variables / 16,384 rows / 1M stored matrix entries |

The output bit limit is an admission/serialization limit, **not** a claim that
SoPlex's internal arithmetic obeys the old Python bit/visit limits. It must be
checked before unbounded parsing; original checker's limits also remain intact.
Memory limits concern address space; measured RSS must be reported separately.
No clock reset between processes or phases. Owned descendant cleanup only.
Timeouts, memory/file/bit limits and partial artifacts are retained. An expired
publication deadline invalidates any late positive/feasible record.

Resource gate before each timed job: >=16GiB available RAM, >=5GiB disk,
load/core<=0.5, polling30s up to24h. Resource waiting is outside request time
and recorded separately, not silently counted as solver time. Do not interrupt
other jobs. Gates do not authorize extending a request's 300 seconds.

Report whole request and nested load/export/import/readback/solve/capture/
package/check/publication costs, observed RSS, output bytes and maximum rational
bit sizes. Missing phases are null/censored, not zero. Do not add nested times
twice. Batch waiting and post-terminal archival review are separately charged.

## Acceptance and stopping

Always check candidates against the original rational LP and original statement
using the unchanged relocated `lp_sandwich/check.py` under `python -I -S`.
All inequalities/equalities/box constraints and the offset-containing objective
are mandatory, with no feasibility tolerance. The proposed point is not proof.
This protocol exports a primal witness only; it does not claim new independently
checked optimality or a dual bound. Old checked L values are labeled historical
context, not newly measured or combined into an optimality certificate here.

- Checked feasible `U<=0`: this supplied relaxation cannot have a strictly
  positive minimum. **Not** a full-network UNSAFE witness.
- Checked feasible `U>0`: feasible upper bound obtained, safety remains unresolved.
- Inexact candidate or no complete candidate: unresolved, not LP infeasibility.
- TIMEOUT/LIMIT: preserve missing/censored evidence, continue the fixed roster.
- ERROR (identity/readback mismatch, malformed complete output, unexpected I/O):
  stop batch; keep later slots as NOT_RUN_AFTER_ERROR in the denominator of four.

No outcome-driven rerun, time increase, range strengthening or additional LP.
After all four outcomes (or fail-stop), archive and close this study. If still
unresolved, retain that limitation; do not restart a custom-arithmetic loop or
block writing the method paper. No current production acceptance gate changes.

## Freeze versus execution readiness

This turn freezes **subjects, question, tool/options, budgets and interpretation**
after the real sparse input checks. It does **not** claim a tested external
execution supervisor: rational CLI-output admission, combined deadline/memory/
file controls, partial terminal publication and batch audit for this exact
SoPlex path are not yet implemented/validated. Thus machine status is
`PROTOCOL_FROZEN_NOT_EXECUTED`, `execution_ready=false`.

Before launch, bind a separately reviewed execution addendum implementing this
unchanged protocol, with deadline/error/partial-evidence/cost controls. Bind its
source/binary/control hashes without editing this frozen protocol or roster;
then obtain the separate explicit execution decision. If implementing the
contract requires changing a scientific setting, expose it as a new protocol,
not a silent amendment. The four native imports already performed are
preparation, not four solver runs or evidence of candidate-generation efficacy.
