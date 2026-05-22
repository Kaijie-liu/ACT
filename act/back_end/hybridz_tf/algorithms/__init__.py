#===- act/back_end/hybridz_tf/algorithms/__init__.py - HZ TF Algorithms -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   TF-internal HZ algorithms (HZ → HZ transformations only). Pure
#   construction-side helpers used by tf_mlp / hz_routing during forward
#   propagation. Solver-side LP utilities live in ``act.back_end.solver``
#   (hz_lp_verify, hz_strict_replay).
#
#===---------------------------------------------------------------------===#

"""ACT HZ TF-internal algorithm helpers.

Construction-side primitives (HZ → HZ): SGM merge, ReLU encoding
alternatives, eq-elim QR projection, binary probing, mem-aware
dispatch, bounds tightening cascade. Used internally by hz_routing.py
during forward propagation.

Solver-side concerns (HZ + spec → verdict) live in ``act.back_end.solver``:
  - hz_lp_verify       : unsafe-set LP feasibility + witness extraction
  - hz_strict_replay   : zero-tolerance witness replay against ONNX

Modules in this package:
  sgm             -- Shared Generator Merge (shares_generator + hz_sgm_add)
  relu_methods    -- triangle / compact / bigM ReLU encodings
  v8_memaware     -- memory-aware encoding dispatcher
  bounds_tighten  -- 3-tier bound cascade (UNC / dual / eq_elim LP)
  eq_elim         -- QR-based equality elimination
  binary_probe    -- RIIM + single-LP binary fix-up
"""
