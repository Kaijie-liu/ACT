# TLL compact signed-share family gate v1

Baseline authority is the frozen 13-family overlay under the read-only HyZor
archive. Candidate records are exclusive-create JSON files under this
experiment directory. The formal cross-family headline remains 1,870 / 2,413;
this file records a candidate-only family gate.

The final candidate is exact signed ReLU sharing with the compact graph,
enabled explicitly by `HybridZConfig(signed_relu_sharing=True,
signed_relu_compact=True)`. It groups only byte-exact affine copies or
negations and uses `ReLU(-x) = ReLU(x) - x`. Every FALSIFIED result below has a
concrete in-box witness checked by the converted PyTorch model. New iid 19, 21,
26, 28, and 31 witnesses were additionally checked with original ONNX Runtime
against the original VNNLIB direction.

The final guarded selector uses compact form only in layers containing a
proven negative member. All ten new results and all five old certificates were
rerun after this correction; the table below reflects those guarded records.

| iid | Frozen result | Candidate result | Gate role |
|---:|:---:|:---:|:---|
| 0 | CERT | CERT | retained |
| 1 | CERT | CERT | retained |
| 2 | CERT | CERT | retained |
| 3 | ADV | ADV | retained, concrete |
| 4 | CERT | CERT | retained |
| 5 | ADV | ADV | retained, concrete |
| 6 | CERT | CERT | retained |
| 7 | UNKNOWN | CERT | new, reproduced |
| 8 | UNKNOWN | UNKNOWN | closed at 60 s |
| 9 | UNKNOWN | UNKNOWN | closed at 30/60 s |
| 10 | ADV | ADV | retained, concrete |
| 11 | ADV | ADV | retained, concrete |
| 12 | UNKNOWN | CERT | new, reproduced |
| 13 | ADV | ADV | retained, concrete |
| 14 | UNKNOWN | CERT | new, reproduced |
| 15 | UNKNOWN | UNKNOWN | closed at 30/60 s |
| 16 | UNKNOWN | UNKNOWN | closed at 30/60 s |
| 17 | ADV | ADV | retained, concrete |
| 18 | UNKNOWN | CERT | new, reproduced |
| 19 | UNKNOWN | ADV | new, reproduced + ORT |
| 20 | ADV | ADV | retained, concrete |
| 21 | UNKNOWN | ADV | new, reproduced + ORT |
| 22 | UNKNOWN | CERT | new, reproduced |
| 23 | ADV | ADV | retained, concrete |
| 24 | UNKNOWN | UNKNOWN | closed at 30 s |
| 25 | ADV | ADV | retained, concrete |
| 26 | UNKNOWN | ADV | new, reproduced + ORT |
| 27 | ADV | ADV | retained, concrete |
| 28 | UNKNOWN | ADV | new, reproduced + ORT |
| 29 | ADV | ADV | retained, concrete |
| 30 | ADV | ADV | retained, concrete |
| 31 | UNKNOWN | ADV | new, reproduced + ORT |

Family totals:

- frozen: `5 CERT + 12 ADV = 17/32`;
- candidate: `10 CERT + 17 ADV = 27/32`;
- net: `+5 CERT +5 validated ADV = +10`;
- invalid candidate ADV: zero;
- retained old solved: `17/17`.

The provisional cross-family union is therefore 1,880 / 2,413. It is not a
formal score update until the remaining family-level and complete 2,413-case
promotion gates finish.
