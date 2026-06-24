# ===- act/back_end/hybridz_tf/algorithms/__init__.py -------------------===#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
"""Self-contained HZono algebra algorithms.

The strict product path uses exact shared-generator add (``sgm``) and exact
redundancy removal.  Lossy helpers in ``order_reduce`` are explicit
audit/ablation tools only; they are not called by strict HybridZ propagation.
Import submodules directly.
"""
