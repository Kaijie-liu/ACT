# C6 V1 actual-state diagnostic: closed, incomplete certificate

160 tests passed (31 new, 129 inherited). Worker exit 1, supervisor wall
12.119224023073912 seconds; source and provenance drift false. The integer
support engine hit its frozen 256M edge-visit ceiling while processing term
11, after complete records for terms 0 through 10. Last completed count was
242,230,256 visits. No result.json, numerical contraction, live publication,
terminal run or verdict was produced. Raw logs/events/exit hashes are retained
in results/c6_support_affine_plan_20260905_v1/. Do not retry this source hash.

The partial certificate already proves all five zero-source terms have zero
value-composition work without discarding predicates. It does NOT prove the
whole suffix admissible. Nonzero term upper bounds include 46,046,000 (term3),
261,963,200 (term5), 823,627,200 (term6), 178,584,000 (term7), 23,014,400
(term8), and 95,436,800 (term9). These are conservative structural bounds,
not measured numerical work and not impossibility lower bounds. In particular,
term6's inner Conv alone accounts for an upper bound of 766,745,600.

The next diagnostic V2 keeps the SAME support identity, multiplicity/work
certificate, target, scalar realization and all ceilings. Its only algorithm
change is in certificate construction: integer Conv transport skips known
zero selected channels and masked output coordinates BEFORE the small matrix
product, rather than multiplying a dense channel rectangle containing zeros.
Complete all 14 terms to identify the dominant remaining structure; a negative
admission result remains negative and does not authorize a terminal run.

Formal 1870/2413 and E0 61/400 are unchanged; candidate remains default-off.
