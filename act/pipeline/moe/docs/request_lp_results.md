# Request-level supplied-HZ LP control R1 result

Frozen execution `5c98e5399`, separate protocol `request_lp_r1.md`.
Seed0, old CIFAR10 index3000,2/255; one request, not a sampled performance
experiment. Runtime58.63seconds; normal exit. Compact independent recheck:
`../results/request_lp_review_20260914_r1.json`. Raw exports/checkpoint/input
remain local;78 file hashes bind the complete record.

All28 unordered route queries completed: feasible{2,4},{4,5},26 excluded
under the existing trusted router solver policy. All18 pair/classification
obligations were inventoried before output LP solving. Membership experts
2,4,5 yield27 LPs; three residual properties need6 difference LPs and3 F0
output LPs. All36 stored exports and rational dual bounds were independently
rechecked without solving again, including negative bounds.27 positive LP
bounds are NOT27 output certificates: several are only auxiliary endpoints.

| Request obligations | Result |
|---|---:|
| Positive via two scoped membership facts |15|
| Positive via residual F0 LP |0|
| Residual LP lower bound nonpositive |3|
| Missing/unexecuted required obligations |0|

The three residuals are pair{2,4},competitors2,3,4 (clean class5), with checked
lower bounds approximately -0.03526004,-0.05940228,-0.01215695. Full request
status **UNKNOWN**, not UNSAFE. No replacement sample, larger budget or altered
gate envelope is run to obtain a positive result. The universal lambda[0,1]
envelope is specific to this proof infrastructure control; it is not a change
to the production F0 gate range, so this is not a performance ablation.

Maximum exported factor count3173; CSR avoids dense constraint export. The
independent chain now reaches ALL necessary output obligations of one real
request, but does not close them all positively. The trusted base still includes
network/input-to-HZ, guard lowering, router infeasibility exclusions and the
F0 outer-HZ construction/floating coefficients. Exact rational downstream
checking does NOT establish these translations or native floating execution.
