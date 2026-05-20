"""ACT HZ algorithm library.

Shared algorithm primitives for HZ-domain solvers. Future HZ solvers
(HyZor-v2, simple-HZ baseline, etc.) compose these primitives instead
of reimplementing each algorithm.

Modules:
  sgm           -- Shared Generator Merge (shares_generator + hz_sgm_add)
  (eq_elim)     -- planned P2-Step-4: QR-based equality elimination
  (binary_probe)-- planned P2-Step-6: RIIM + LP probing
  (lp_verify)   -- planned P2-Step-7: final HiGHS LP for unsafe-set
                   feasibility + witness extraction
  (cascade)     -- planned P2-Step-8: per-layer encoding scheduler
                   (triangle for first N layers + eq_lagr for last K)
  (representations) -- planned P2-Step-5: BoxHZ / LazyChainHZ /
                       SparseGcZ Phase-1-3 wrappers
"""
