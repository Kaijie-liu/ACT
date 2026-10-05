# D144: Where does the saved pilot contain joint nonlinearity?

Registered before executing the audit, 2026-10-04 Australia/Sydney.
Branch `redu-hz`, HEAD `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Tracked diff SHA256 `29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5`.

This is an evidence-only structural screen, not a Neural-HZ candidate,
implementation qualification, fresh bound query, or verification gain. Its sole
purpose is to decide whether the already studied common-anchor groups actually
contain multiple unresolved nonlinear gates. The main research objective remains
a stronger nonconvex domain definition, not optimizing this audit.

## Frozen population and operation

Read all three complete D120 JSON artifacts, in order 0, 1, 2, without filtering
by a successful row, model result, margin, or label. Each contains 1600 distinct
source forms and 325 pre-existing groups over five saved spatial windows. This
is not a full-spatial or full-network census. The complete population is 4800
forms, 975 groups and 1600 downstream rows (320, 640, 640).

Use `audit.jq` with jq 1.7, under a 60-second command timeout. Check input SHA256
before and after execution. Preserve every per-file histogram, including zero
and no-op categories. A missing reference, wrong population or malformed bound
stops the audit rather than producing a qualified result.

The saved rational numerators/denominators fit the old 512-bit contract. The
script compares their signs/zero only: no division, cross multiplication,
magnitude comparison or arithmetic on large endpoints. jq's binary floating
representation does not preserve their integer values, so no exact rational
equality or ordering is claimed. Denominators must be positive and endpoints
finite. Coordinate references and endpoint sign classes are checked; exact
rational validity/order are inherited from the authenticated D120 artifact,
not newly established here.

Report strict-active (`lower>0`), strict-inactive (`upper<0`), outer-crossing
(`lower<0<upper`), and zero-touch classes. Outer-crossing means stability was
not established by that interval; it does not prove actual two-sided reachability.
Report group histograms by crossing/zero-touch counts, anchor classes, and
residual interval sign/zero classes. Do not equate a nonzero residual with lack
of approximate alignment; its magnitude is not measured here.

## Preregistered interpretation

For a group with no zero-touch gate and at most one outer-crossing gate, the
complete source-labelled group hull is an affine lifting of the remaining
single-gate source-labelled hull (or of the source set if all gates are strict
stable). This uses the full shared source, not a scalar triangle. Such groups
cannot demonstrate an additional multi-hinge hull relation over that strong
reference. This standard fact is not a novelty claim.

At least two unresolved gates is only a necessary screen for this kind of
joint relation, not proof of gain. Groups touching zero stay separate because
their original labels need not be fixed. The screen does not rule out improved
source correlation, cross-group relations, or subsequent nonlinear propagation.

If there are no multi-cross groups, do not implement a common-anchor joint-hull
candidate on this pilot. If there are any, report their complete count, but do
not promote the old common-anchor formula into a new domain: a compositional
nonconvex rule, a strong comparison and complete cost remain required.

## Authority and qualification boundary

No model is loaded, no candidate is imported/compiled/executed, no solver runs,
and no new source bound is computed. This audit does not rerun or reduce the
latest D136 population (3965 tests / 208 files), nor inherit a new component,
native-source, GPU, full-cost or replay qualification. Any future numerical
candidate retains all existing preregistration and qualification requirements.

Formal baseline stays 1870/2413; separate E0 stays CIFAR100 25 + TinyImageNet 36
=61/400. All frozen files and history remain read-only. New files are confined
to this isolated directory and a new resume note. No default change, commit,
push, background job, attack, split or rescue is authorized by this audit.

Input identities are recorded in `INPUT_SHA256SUMS`; preregistration and script
are hashed into `FROZEN_SHA256SUMS` before the one audit execution.
