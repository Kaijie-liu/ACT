# TLL width-adaptive dead-graph family gate v2

The frozen authority remains the read-only 13-family HyZor overlay at
`1,870 / 2,413` (`1,063 CERT + 807 validated ADV`). This document records an
opt-in candidate family gate only; it does not change the formal headline.
All new JSON records are exclusive-create files under this experiment tree.

## Exact rule

Trial 4 first partitions unstable affine rows by byte-exact equality up to
sign and represents each class with one compact signed ReLU graph, preserving
the original affine latent row and using `ReLU(-x) = ReLU(x) - x`.

Trial 5 observes the unique successor Dense layer. For a signed class, the
shared nonlinear term is dead exactly when every successor output row has
exact-real zero weight sum over that class. Zero is proved over the integer
ratios of the stored binary64 values, not by rounded summation. After the
Dense, the candidate removes exactly the class's three local inequalities and
private continuous/binary columns. Any frame mismatch, unexpected equality,
or retained-inequality coupling makes the transform return the original HZ.

The final lift budget is local and width-adaptive:

- delete every proven dead class of cardinality two;
- also delete cardinality-four classes only when that ReLU source has at least
  8,192 scalar rows;
- retain every larger class as a redundant aggregate lift.

The selector cannot observe the instance id, property verdict, solver state,
or future layer. It preserves continuous and binary factors, equality and
inequality predicates, the shared frame identity, and concrete input
reconstruction. It performs no attack, PGD, split, BaB, backward, or dual
rescue. All flags remain default-off.

## Negative boundaries retained in the evidence

Exactness alone did not guarantee solver monotonicity:

- deleting all dead graphs solved iid 24 but regressed ADV iid 21/27/29;
- deleting only cardinality-two graphs recovered those ADV and iid 24 but
  regressed iid 31;
- deleting cardinality-two-and-four graphs recovered iid 31 but regressed iid
  10/26.

Every regression was fail-closed UNKNOWN, never a wrong CERT or invalid ADV.
The width-adaptive rule is the first single rule to retain the complete solved
set. These failed arms remain documented and unpromoted.

## Controlling family gate

Candidate hash:
`0591e23a3e9bae368d9192a06783dbdad2777b068080ab71b632773190d1770d`.

The complete replay retains all 28 previously solved/candidate cases:

- 11/11 certificates retained;
- 17/17 ADV reconstructed and concrete-PyTorch valid;
- invalid ADV: zero;
- ERROR: zero.

The remaining four UNKNOWNs were then tried once at 45 seconds. iid 8 became
CERT twice, at about 15.9 seconds, with a lowered model of
`282c/280b/840 rows/3646 nnz` after 243 exact eliminations. iid 9/15/16 remain
UNKNOWN.

| iid set | Candidate result | Gate role |
|:---|:---:|:---|
| 0,1,2,4,6 | CERT | all frozen CERT retained |
| 3,5,10,11,13,17,20,23,25,27,29,30 | ADV | all frozen ADV retained, concrete-valid |
| 7,12,14,18,22 | CERT | Trial 4 new CERT retained |
| 19,21,26,28,31 | ADV | Trial 4 new ADV retained, concrete-valid + prior ORT |
| 24 | CERT | Trial 5 new CERT, reproduced |
| 8 | CERT | Trial 5 new CERT, reproduced |
| 9,15,16 | UNKNOWN | closed at this 45-second gate |

Family totals:

- frozen: `5 CERT + 12 ADV = 17/32`;
- final candidate: `12 CERT + 17 validated ADV = 29/32`;
- net: `+7 CERT +5 validated ADV = +12`;
- retained old solved: `17/17`;
- invalid candidate ADV: zero.

The provisional cross-family union is therefore `1,882 / 2,413`. Promotion
still requires representative non-TLL structural shadows, per-family retained
replay, and ultimately the complete 2,413-case gate.
