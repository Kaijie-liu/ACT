# Unimplemented next hypothesis: preserve the existing scalar association

C5-v2 preserves C5-v1's scalar matrices bitwise but both reorder the original
unfused spatial composition: middle-channel contractions occur before spatial
tap accumulation, and output BN scale is applied afterward. The real shadow
therefore has small but nonzero coefficient differences. These are a general
arithmetic-association issue, not an iid-specific near-boundary witness repair.

An admissible successor hypothesis is to key demanded coefficients by output
channel, input channel, displacement and the ORDERED valid spatial-pair pattern,
and compute them in the original left-composition middle-channel/spatial order.
Output and middle diagonals must be multiplied in the original scalar order,
not merely in an algebraically equivalent order. Reusing an identical ordered
coefficient demand across spatial occurrences may retain the work reduction
while making every emitted CSR coefficient match the existing row oracle.

This is NOT implemented, preregistered or proved. It must first inspect the
actual `_lazy_left_compose`/CSR ordering, explicit zeros and cancellation rules,
derive a complete key including masks and diagonal payloads, and recount actual
work and storage for all branches. It may fail a resource gate; no gate changes
or promotion is implied. A future preregistration must bind those facts before
any real successor result. The full same-frame materialization and whole-state
gates remain required regardless of scalar equality.
