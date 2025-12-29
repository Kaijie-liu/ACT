#===- act/back_end/serialization/__init__.py - Serialization Package ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Serialization package for ACT neural networks providing JSON serialization,
#   network analysis, validation, and utility functions for model persistence.
#
#===---------------------------------------------------------------------====#

from act.back_end import BACK_END_IMPORT_ERROR  # type: ignore

TensorEncoder = LayerSerializer = NetSerializer = None
save_net_to_file = load_net_from_file = save_net_to_string = load_net_from_string = validate_json_schema = None

if BACK_END_IMPORT_ERROR is None:
    try:
        from .serialization import (
            TensorEncoder,
            LayerSerializer, 
            NetSerializer,
            save_net_to_file,
            load_net_from_file,
            save_net_to_string,
            load_net_from_string,
            validate_json_schema
        )
    except Exception:  # pragma: no cover
        BACK_END_IMPORT_ERROR = True

__all__ = [
    # Core serialization
    'TensorEncoder',
    'LayerSerializer',
    'NetSerializer', 
    'save_net_to_file',
    'load_net_from_file',
    'save_net_to_string',
    'load_net_from_string',
    'validate_json_schema'
]
