# Common curvature mathematical component contract

This default-off component implements the source-bound ReLU rule proved in [the preceding theory](../d118_shared_curvature_transfer_20261002/THEORY.md). It preserves the nonconvex HZ mathematical System, its continuous and original signed binary factors, original EQ/LE, shared frame and input interpretation. Its goal is to verify an actual forward relation generator and bounded residual compensation, not to present a classical interpolation inequality as a completed new domain.

## Authenticated original gates

An immutable Gate contains g,q,bit,lo,hi. g is a canonical rational affine readout, q is an existing unit nonbinary output column, and bit is an existing signed binary column. The active indicator is (1+signed_bit)/2. All gates in a group use one frame, distinct output columns and distinct original bit columns. The compiler authenticates the four literal rows

```text
-q <= 0
g-q <= 0
q-max(0,hi)*active <= 0
q-g+min(0,lo)*(1-active) <= 0.
```

lo/hi must enclose the exact rational box range of g from the retained System bounds. No caller metadata token substitutes for the actual rows, bounds or column identities. These rows imply q=ReLU(g) at every original integer state, including both legal labels at g=0. Stable gates remain valid and their original factors are not removed. This authenticates a mathematical System; it is not original ONNX floating-point or native SparseHZono qualification.

## Uniform group transform and residual proof

The gate tuple contains the existing f endpoint, m interior gates and the existing h endpoint, in strictly increasing fixed knots 0=t_0<...<t_(m+1)=1. No endpoint gate or bit is synthesized. For each interior gate form the actual source residual r_i=g_i-((1-t_i)f+t_i*h). Keep it on the same source coordinates.

For exact matching the m convex interpolation rows and the curvature budget are

```text
-(t_(i+1)-t_i)*q_(i-1) + (t_(i+1)-t_(i-1))*q_i
    -(t_i-t_(i-1))*q_(i+1) <= 0
(p-q_1)/(t_1-t_0) + (q-q_m)/(t_(m+1)-t_m)
    -(2*p+2*q-f-h) <= 0.
```

For each row write its interior output coefficients as c_i. Let eta_i=ReLU(g_i^0+r_i)-ReLU(g_i^0). Since c_i*eta_i<=ReLU(c_i*r_i), the actual row is valid after subtracting the sum of certified secant upper bounds for ReLU(c_i*r_i). Compute each secant from the exact box of its own affine argument and THEN merge the affine secants on the common coordinates. In general ReLU(sum arguments) would be unsound here and is not used. Shared residual cancellation is not asserted without this proof. The original rows and all factors stay unchanged.

The reference hull theorem and support routine concern the declared standalone tent family, not the full source hull. The support helper computes the minimum and maximum of the mixed kernel on the fixed coefficient knots using prefix sums; it performs no input/phase search or optimization against a solver state. Its pure algebraic use is distinct from the opt-in System mutation entry points.

The group adds m+1 LE and no columns. At zero residual and m>=2, if every referenced readout is already a physical column, the local coefficient count is at most 3m+6. The implementation reports actual merged row nnz. Dense source readouts and residual secants can increase that count and are not hidden.

## Actual downstream use

The child transform authenticates an existing original child gate and requires the given budget row to occur literally in System.le. It derives a positive multiplier from the child preactivation and budget coefficients on the actual group output columns. Every group output coefficient must cancel in V=g_child-multiplier*budget_row. The exact rational box lower bound of V must be nonnegative. Then g_child<=V and ReLU(g_child)<=V, so adding q_child-V<=0 is sound. All original child rows and its original bit remain.

This is a fixed algebraic cancellation rule, not a choice among LP outcomes or a free caller-supplied upper bound. An unsupported cancellation or sign premise is rejected with no partially updated System. General arbitrary-depth closure, online native allocation and all network graph patterns remain unproved.

## Registered exact controls

Use x,y in [-1,1], f=x+y/5+1/10, h=-x/4+y+1/5, p=ReLU(f), q=ReLU(h), and t=(1/4,1/2,3/4). The next original gate is

```text
Y=(p+q)/2+(f+h)/4-q_1-q_3
r=ReLU(Y+(f+h)/4+1/2).
```

The exact structure yields Y<=0 and r<=(f+h)/4+1/2. The registered false tuple has f=h=0, p=q=1/5, interior outputs 1/20, five active means 1/2 and r=3/5 with its active bit 1. Verify all ten D118 two-point source-labelled pair-hull witnesses by exact fractions; do not manufacture a shared five-gate witness. The tuple passes the independent tent caps and concavity but violates the child row by 1/10.

The second registered structure has a third source z in [-1,1], residuals +(1/20)z and -(1/20)z on the first and last interior gates. Per-term secants merge to a 1/20 allowance in Y, leaving a 1/20 child exclusion at z=0. The same complete pair witnesses at z=0 remain valid. The two structures, all witnesses and all query kernels are fixed before execution. No model, dataset, label, attack or numeric point search is involved.

The true source point (f,h)=(-2/5,6/5) gives child 7/10, while x=y=-1 gives child preactivation -13/40. These establish that an exact global scalar child interval cannot itself eliminate the false 3/5. Explicit zero-gate fixtures retain both legal labels. More generally exact canonical fixtures test all added rows without replacing the mathematical proof.

## Resource and qualification scope

All coefficients use exact Fraction arithmetic with the inherited 512-bit cap. Disabled System calls inspect no task arguments. Bad flags, noncanonical readouts, frame/bit/guard mismatches, unordered knots, unsupported child premises and insufficient retained-entry budget fail closed. Check a conservative final retained-entry bound before constructing appended rows, and verify actual entries afterwards; max_entries may not exceed 64000000. This is NOT a whole-pipeline arithmetic or peak-memory certificate. All source scans, original H, normalization, future inequalities-to-slacks, source expansion, evidence, terminal conversion, host/device copies and concrete decoder obligations still count.

The original default, source provenance, full historical tests and resource gates remain. No solver is called by the new compiler or four new tests; unchanged inherited tests include their frozen terminal LP diagnostics. No trained-model, GPU, physical admission, shadow or full replay is registered here. Formal 1870/2413 and separate E0 61/400 cannot change from this run.

Provenance: branch redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac; tracked binary diff 29baf0c0fcc19070a97a5ddf0fea3d1591ff937d00abca28689e9a871a530bc5. Reuse the authenticated D112 mathematical Form/System library; no new runtime dependency is added. New files and evidence are isolated; previous code and consumed runs are read-only.
