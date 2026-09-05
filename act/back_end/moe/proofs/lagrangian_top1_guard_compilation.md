# Lagrangian hard-top1 guard compilation

## Statement

Let hard top-1 branch \(i\) be legal under tie-inclusive semantics when

\[
m_{ij}(x)=r_i(x)-r_j(x)\ge 0\qquad\text{for every }j\ne i.
\]

For one safety row \(s_\ell(x)\ge0\), choose any finite multipliers
\(\mu_{\ell j}\ge0\) and define

\[
\phi_\ell(x)=s_\ell(x)-\sum_{j\ne i}\mu_{\ell j}m_{ij}(x).
\]

If a sound verifier proves \(\phi_\ell(x)\ge0\) for every \(x\) in the
original input set and for every safety row, then the expert property holds on
every input for which branch \(i\) is legal.

## Proof

For a legal branch, every selected margin is nonnegative. Nonnegative
multipliers therefore give

\[
\sum_{j\ne i}\mu_{\ell j}m_{ij}(x)\ge0
\quad\Longrightarrow\quad
\phi_\ell(x)\le s_\ell(x).
\]

The premise \(\phi_\ell(x)\ge0\) then implies
\(s_\ell(x)\ge\phi_\ell(x)\ge0\). This holds independently for every safety
row. A tied competitor has exactly zero margin, so its multiplier term cannot
discharge or weaken that tied branch obligation. Margins to other, strictly
lower competitors may remain positive, so this statement is tie-safe rather
than a claim that the whole reduction is exact whenever any tie occurs. QED.

## Scope

This is a standard Lagrangian sufficient reduction, not an exact encoding of
the guarded optimization problem. A failed compiled bound is UNKNOWN. The
multipliers may differ by property row, but they must be fixed independently
of the input and nonnegative. Selecting the best sound lower bound from a
frozen finite multiplier grid remains sound row by row.

The reduction preserves router/expert input correlation only if both subnetworks
are lowered into one shared-input graph. Passing separately computed scalar
intervals to the formula would lose this benefit. Numerical soundness is a
separate layer: in the current CROWN environment positive bounds remain filters
until an outward-rounding or independently validated bound contract exists.
