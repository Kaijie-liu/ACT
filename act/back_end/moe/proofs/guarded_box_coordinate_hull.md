# Coordinate hull of a box intersected with one route halfspace

## Scope

This note characterizes exactly when a single affine route guard changes a
coordinate bound. It explains why replacing a retained route polytope by its
coordinate hull can erase almost all guard information in a high-dimensional
input, but it does **not** claim that every collection of route halfspaces has
an unchanged hull. Multiple halfspaces require a joint feasibility problem.

## Proposition 1: exact coordinate bounds for one halfspace

Let

\[
B=[\ell,u]\subset\mathbb R^n,
\qquad
H=\{x:a^\top x\le \beta\},
\]

and assume \(B\cap H\ne\varnothing\). For coordinate \(d\), define the best
compensation available from all other coordinates by

\[
m_{-d}=\sum_{j\ne d}\min(a_j\ell_j,a_j u_j).
\]

The coordinate-hull bounds of \(B\cap H\) are

\[
\overline x_d=
\begin{cases}
u_d, & a_d\le 0,\\
\min\!\left(u_d,\dfrac{\beta-m_{-d}}{a_d}\right), & a_d>0,
\end{cases}
\]

and

\[
\underline x_d=
\begin{cases}
\ell_d, & a_d\ge 0,\\
\max\!\left(\ell_d,\dfrac{\beta-m_{-d}}{a_d}\right), & a_d<0.
\end{cases}
\]

Consequently, the upper face survives unchanged exactly when

\[
a_d\le0
\quad\text{or}\quad
a_d u_d+m_{-d}\le\beta,
\]

and the lower face survives unchanged exactly when

\[
a_d\ge0
\quad\text{or}\quad
a_d\ell_d+m_{-d}\le\beta.
\]

### Proof

Fix \(x_d=t\). Minimizing the halfspace left-hand side over the remaining box
coordinates is separable:

\[
\min_{x_{-d}\in[\ell_{-d},u_{-d}]}a^\top x
=a_dt+m_{-d}.
\]

Therefore a value \(t\in[\ell_d,u_d]\) occurs in \(B\cap H\) if and only if

\[
a_dt+m_{-d}\le\beta.
\]

If \(a_d>0\), this inequality supplies an upper bound on \(t\), and if
\(a_d<0\), it supplies a lower bound. If \(a_d=0\), feasibility is independent
of \(t\); nonemptiness of \(B\cap H\) then makes the entire coordinate interval
feasible. Intersecting the resulting half-line with \([\ell_d,u_d]\) gives the
two formulas. Substituting \(u_d\) and \(\ell_d\) gives the face-survival
conditions. \(\square\)

An independent executable differential check sampled 100 nonempty random
eight-dimensional box--halfspace intersections and compared both bounds for
all eight coordinates against SciPy/HiGHS. All 1,600 coordinate bounds agreed
exactly in the recorded float64 run (maximum absolute difference 0). This check
guards the implementation transcription; the proof above is the soundness
argument.

## Corollary 2: why coordinate boxing can discard a route guard

For a hard top-1 branch \(i\), each membership guard has the affine form

\[
r_j(x)-r_i(x)\le0.
\]

After folding the input normalization, Proposition 1 applies with
\(a=w_j-w_i\) and \(\beta=b_i-b_j\). A coordinate side changes only when that
side's adverse router coefficient cannot be compensated by the other
\(n-1\) coordinates. When many coordinates provide compensation, a route
halfspace can remove a substantial oblique part of the input box while leaving
most or all coordinate extrema unchanged. Passing only the coordinate hull to
another verifier then loses precisely that oblique dependence.

This is a geometric explanation, not a distribution-free claim that the hull
must remain unchanged in high dimension. The frozen P0b experiment supplies
the empirical observation for the evaluated CIFAR-10 cohort: guarded-box and
original-box CROWN certify the same pairs, and their property bounds differ
only at the recorded micro scale.

## Multiple route halfspaces

For

\[
P=B\cap\bigcap_{q=1}^{Q}\{x:a_q^\top x\le\beta_q\},
\]

the exact coordinate upper bound is the linear program

\[
\max_{x\in P}x_d,
\]

and the lower bound is the corresponding minimization. Testing every
halfspace separately is necessary but not sufficient for a face to survive:
different halfspaces may require incompatible compensating assignments to
\(x_{-d}\). ACT therefore computes the multi-guard coordinate hull jointly;
it does not compose the single-halfspace closed form as though the constraints
were independent.

## Claim boundary

- The proposition concerns coordinate-hull construction, not downstream
  certificate ordering or runtime.
- An unchanged coordinate hull does not mean the guard is redundant. The
  retained polytope can be strictly smaller than its hull.
- The P0b micro-scale CROWN differences are not outward-rounded certificate
  margins and are reported only as an engineering observation.
- The proposition does not replace guard-aware support. Its point is that the
  retained constraints contain information that coordinate boxing can erase.
