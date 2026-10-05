# C119 exact denominator-bearing HZ theorem and its limits

This is a representation result, not a solver, resource or benchmark permit.
The F(4,3) transforms are from Lavin and Gray, section4.3 equation15:
https://arxiv.org/pdf/1509.09308 . C119 independently checks all72 coefficients
of the corresponding one-dimensional bilinear identity, not a sampled image.

Write G=D^-1 GNUM, with D=(4,6,6,24,24,1). For each original source-box point,
V=BT*d*BT^T and M=sum_c (G*g_c*G^T) .* V_c are uniquely determined. Tensorizing
the complete1D identity proves AT*M*AT^T is exactly the ordinary3x3 convolution.
Masks substitute literal zeros; coincident source IDs substitute the SAME
latent coordinate, not independent copies. Neither operation changes identity.

Choose semantic V unit2^eV above the exact coalesced source-form L1 norm. Its
normalized coordinate v then lies in[-1,1] for every old box point, including
all correlated/nonconvex feasible old points. For M[t,u], choose semantic
unit2^eM above the channel-form L1 norm divided by D[t]*D[u]. Emit

`D[t]*D[u]*2^eM*m = sum_c N[c,t,u]*2^eV*v_c`,

where N=GNUM*g*GNUM^T is computed exactly. Thus m has a redundant box and a
unique extension. The output equation uses2^eM, NOT the positive defining
pivot D[t]*D[u]*2^eM. This distinction is essential: treating the pivot as the
semantic unit would multiply output contributions by spurious odd factors.
Original output coefficients remain2^eY before their whole-row dyadic gauges.

All N and original coefficients are dyadic; the integer denominators are on
new pivots. Whole-row power-two gauges therefore permit exact native literals
when the unchanged representability/window tests pass. No rounded thirds,
epsilon equalities, rational solver coefficients or changed original box are
introduced. A rejected native window is a rejection, not an approximate proof.

Retain all old continuous/binary coordinates and every external predicate.
Replacing only authenticated original convolution definitions with the exact
topological circuit gives a unique box-admissible extension for every original
feasible point. Conversely, every new feasible point satisfies the original
output equations after eliminating the new definitions. Projection onto old
coordinates is therefore exactly equal, including both binary branches and
all old EQ/INEQ correlations. Dropping only new variables and composing the
unchanged old scalar inverse recovers the original input coordinates.

The oracle proves EVERY actual V/M/output row, support, native coefficient,
positive pivot, gauge and redundant-box bound, then connects their semantic
composition with the complete basis identity. The fixture separately binds
EVERY original actual selected equation to the mapped direct convolution and
C91 independently audits every retained/replaced row, map and owner delta.
Small tests additionally expand actual coefficients without using the basis
proof. This is stronger than a floating numerical convolution comparison.

The completed ordinary states show lower complete numeric storage for both
fixed guards. The aggregate C119 run nevertheless FAILED its256M work cap
before the final combined ledger. This theorem does not complete that gate,
authenticate a fresh original network, establish runtime payment, obtain a
witness/verdict, preserve all13 families by replay or authorize promotion.
