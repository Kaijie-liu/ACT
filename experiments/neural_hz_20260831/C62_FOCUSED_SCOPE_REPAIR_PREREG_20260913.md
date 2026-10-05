# C62 v2: repair the focused Conv precondition expectation

C62 v1 is closed:73 pass,1 fail,74 collected; no actual worker ran.
The original merging Conv fixture hits the explicitly preregistered no-coalescence
guard. C61's independent fixture test already establishes this fixture is outside
that theorem's complete support-disjoint class. C62 v1 incorrectly asserted
positive construction for it. This is a test expectation defect, not positive
evidence about the representation and not authority to implement coalescence.

Keep v1 source, test, preregistration, logs and exit unchanged. In v2, independently
derive its coalesced-term count using the C58 inverse/C59 consumer profile, require
the exact rejection, and check both original and C31 HZ inputs remain unchanged.
Also positively exercise an ordinary pointwise Conv/diagonal/Conv chain, with
the same full original-to-new Fraction residual and independent owner-incidence
checks used for chain/shared cases. No skipped/xfail check or failure swallowing.
Retain all61 preceding focused checks. The corrected C62 set has14 checks;
the complete v2 inventory is75 unique tests, all required to pass.

The production candidate, C62 planner, physical writer, compact inverse,
measurement, all arithmetic and resource thresholds are IDENTICAL to v1.
Only a new worker entrypoint routes the immutable implementation to the unique
v2 output. No target result has been seen for either C62 version. The complete
real C9/C31/C61 inputs, optimum100965, exchange and physical/native restrictions
are inherited from C62_PHYSICAL_BOUNDARY_PREREG_20260913.md without weakening.

CPU1/GPU0,AS16GiB,64M entries,256M whole/200M nested,60s focused/240s worker,
both1GiB transient, native16384/131072/16M unchanged. Freeze all old/new sources;
automatically retain tests/events/result/exit under the exclusive
results/c62_physical_boundary_20260913_v2/. Any hard failure closes v2.

Formal1870/2413,13-family and per-case retention, E0 CIFAR25/Tiny36 unchanged.
No solver/native/generator execution, default change, historical mutation,
commit/push, forbidden rescue, binary removal or convex-domain substitution.
Full research goal remains ACTIVE. This correction itself earns no research gain.

