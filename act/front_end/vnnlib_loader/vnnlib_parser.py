#===- act/front_end/vnnlib_loader/vnnlib_parser.py - VNNLIB Parser ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Parse VNNLIB SMT-LIB format files to extract input tensors and constraints.
#   Converts VNNLIB specifications to InputSpec and OutputSpec objects.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import logging
import math
import torch
import re

from act.front_end.specs import InputSpec, OutputSpec, InKind, OutKind

if TYPE_CHECKING:
    from act.front_end.spec_creator_base import LabeledInputTensor

logger = logging.getLogger(__name__)


class VNNLibParseError(Exception):
    """Exception raised when VNNLIB parsing fails."""
    pass


class UnsupportedSpecError(Exception):
    """Exception raised for soundly unsupported VNNLIB features."""
    pass


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------


def evaluate_vnnlib_2_concrete(
    vnnlib_path: Path,
    x: Any,
    y: Any,
    tol: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate a raw VNNLIB 2.0 unsafe set on one concrete ``(x, y)``.

    Legacy flat VNNLIB is deliberately rejected by this version-specific API;
    strict replay uses :func:`evaluate_vnnlib_concrete`, which recognizes both
    audited dialects without a permissive fallback.
    """

    return _evaluate_vnnlib_concrete(
        vnnlib_path, x, y, tol, required_dialect="vnnlib-2.0"
    )


def evaluate_vnnlib_concrete(
    vnnlib_path: Path,
    x: Any,
    y: Any,
    tol: float = 0.0,
) -> Dict[str, Any]:
    """Evaluate raw VNNLIB 1.0 or 2.0 assertions on concrete ``(x, y)``.

    This is deliberately independent of query materialisation.  It preserves
    top-level assertion conjunction, nested ``and``/``or`` structure, coupled-X
    rows, mixed X/Y affine atoms, and strict comparisons.  Dialect recognition
    is exact: mixed or unknown declaration syntax fails closed.
    """

    return _evaluate_vnnlib_concrete(
        vnnlib_path, x, y, tol, required_dialect=None
    )


def extract_vnnlib_concrete_layout(vnnlib_path: Path) -> Dict[str, Any]:
    """Return the strictly recognized concrete-evaluation layout.

    Unlike the evaluator, this inspection helper raises on malformed or unknown
    syntax.  It is used before ONNX Runtime execution to validate witness and
    model tensor sizes.
    """

    path = Path(vnnlib_path)
    content = path.read_text(encoding="utf-8")
    layout = _prepare_concrete_vnnlib(content, required_dialect=None)
    return {
        "dialect": layout["dialect"],
        "vnnlib_version": layout["vnnlib_version"],
        "num_inputs": layout["num_inputs"],
        "num_outputs": layout["num_outputs"],
        "input_shape": list(layout["input_shape"]),
        "output_shape": list(layout["output_shape"]),
    }


def _evaluate_vnnlib_concrete(
    vnnlib_path: Path,
    x: Any,
    y: Any,
    tol: float,
    *,
    required_dialect: Optional[str],
) -> Dict[str, Any]:
    """Fail-closed implementation shared by the public dialect entry points.

    The returned mapping contains only JSON-serializable values.  Any read,
    parse, shape, numeric, or unsupported-expression error fails closed with
    ``holds=False`` and ``evaluated=False`` rather than raising.
    """

    path = Path(vnnlib_path)
    base: Dict[str, Any] = {
        "schema_version": 1,
        "vnnlib_path": str(path),
        "dialect": None,
        "vnnlib_version": None,
        "tolerance": None,
        "evaluated": False,
        "holds": False,
        "num_inputs": None,
        "num_outputs": None,
        "input_shape": None,
        "output_shape": None,
        "assertions": [],
        "atoms": [],
        "error": None,
    }
    try:
        tolerance = float(tol)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise VNNLibParseError(
                f"concrete evaluation tolerance must be finite and nonnegative, got {tol!r}"
            )
        base["tolerance"] = tolerance

        content = path.read_text(encoding="utf-8")
        layout = _prepare_concrete_vnnlib(
            content, required_dialect=required_dialect
        )
        input_shape = layout["input_shape"]
        output_shape = layout["output_shape"]
        num_inputs = layout["num_inputs"]
        num_outputs = layout["num_outputs"]
        x_values = _concrete_flat_values(x, "x", num_inputs)
        y_values = _concrete_flat_values(y, "y", num_outputs)

        forms = _parse_all_forms(layout["rewritten"])
        assert_forms = [
            form
            for form in forms
            if isinstance(form, list) and form and form[0] == "assert"
        ]
        if not assert_forms:
            raise VNNLibParseError(
                "concrete VNNLIB evaluation requires at least one assertion"
            )
        for form in assert_forms:
            if len(form) != 2:
                raise VNNLibParseError(
                    f"assert expects exactly one body, got {len(form) - 1}"
                )

        atom_log: List[Dict[str, Any]] = []
        assertion_log: List[Dict[str, Any]] = []
        all_hold = True
        for index, form in enumerate(assert_forms):
            atom_start = len(atom_log)
            holds, tree = _evaluate_concrete_boolean(
                form[1],
                x_values,
                y_values,
                tolerance,
                atom_log,
                num_inputs,
                num_outputs,
            )
            assertion_log.append({
                "index": index,
                "holds": bool(holds),
                "atom_indices": list(range(atom_start, len(atom_log))),
                "tree": tree,
            })
            all_hold = all_hold and holds

        base.update({
            "evaluated": True,
            "holds": bool(all_hold),
            "dialect": layout["dialect"],
            "vnnlib_version": layout["vnnlib_version"],
            "num_inputs": num_inputs,
            "num_outputs": num_outputs,
            "input_shape": list(input_shape),
            "output_shape": list(output_shape),
            "assertions": assertion_log,
            "atoms": atom_log,
        })
        return base
    except Exception as exc:
        base["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        return base


def parse_vnnlib_to_tensors(
    vnnlib_path: Path,
    input_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Parse a VNNLIB file to extract input tensor and metadata.
    
    The input tensor represents the center of the constrained input region.
    For box constraints with bounds [lb, ub], the center is (lb + ub) / 2.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        input_shape: Expected input shape INCLUDING batch dimension (e.g., (1, 3, 32, 32))
                    If None, will be inferred or use flat shape
        
    Returns:
        Tuple of (input_tensor, metadata_dict) where:
        - input_tensor: torch.Tensor with batch dimension (e.g., shape (1, 3, 32, 32))
        - metadata_dict: Contains 'input_bounds', 'num_outputs', 'property_type'
        
    Raises:
        VNNLibParseError: If parsing fails
    """
    if not vnnlib_path.exists():
        raise VNNLibParseError(f"VNNLIB file not found: {vnnlib_path}")
    
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
        
        # VNNLIB 2.0 (declare-network) uses bracket vars X[i,..]/Y[j,..]; rewrite
        # them to flat X_n/Y_n so the shared bound-extraction logic below applies.
        if "(vnnlib-version" in content or "(declare-network" in content:
            if len(re.findall(r"\(\s*declare-network\b", content)) >= 2:
                if "isomorphic-to" not in content:
                    raise UnsupportedSpecError("multi-network (equal-to/monotonic) not yet supported")
                content, num_inputs, num_outputs, _f_in_shape = _isomorphic_multinet_rewrite(content)
            else:
                in_name, _in_dtype, in_shape = _extract_vnnlib_2_decl(content, "input")
                out_name, _out_dtype, out_shape = _extract_vnnlib_2_decl(content, "output")
                num_inputs = _numel(in_shape)
                num_outputs = _numel(out_shape)
                content = _rewrite_vnnlib_2_bracket_vars(content, in_name, in_shape, out_name, out_shape)
        else:
            layout = _prepare_concrete_vnnlib(
                content, required_dialect="vnnlib-1.0-flat"
            )
            content = layout["rewritten"]
            num_inputs = int(layout["num_inputs"])
            num_outputs = int(layout["num_outputs"])
            _validate_legacy_query_materialization(
                content, num_inputs, num_outputs
            )

        # Extract input bounds from top-level simple X-bound asserts only.
        # Constraints inside (or ...) branches must not be intersected here —
        # that would produce empty boxes for disjunctive-input properties
        # (e.g. ACAS Xu prop_5..10).
        try:
            forms = _parse_all_forms(content)
            simple_bodies = [
                f[1] for f in forms
                if isinstance(f, list) and len(f) >= 2 and f[0] == "assert"
                and _is_simple_x_bound(f[1])
            ]
        except Exception:
            simple_bodies = []
        input_bounds = _extract_input_bounds(simple_bodies, num_inputs)
        
        # Create input tensor from bounds center
        input_values = []
        for i in range(num_inputs):
            if i in input_bounds:
                lb, ub = input_bounds[i]
                center = (lb + ub) / 2.0
            else:
                # Default to 0 if no constraint
                center = 0.0
            input_values.append(center)
        
        input_tensor = torch.tensor(input_values)
        
        # Reshape if shape is provided; the shape already includes a leading batch dimension.
        if input_shape is not None:
            expected_numel = 1
            for dim in input_shape:
                expected_numel *= dim
            if input_tensor.numel() != expected_numel:
                raise VNNLibParseError(
                    f"Input size mismatch: got {input_tensor.numel()} "
                    f"values but expected {expected_numel} from shape {input_shape}"
                )
            # Reshape directly - input_shape already includes batch dimension
            input_tensor = input_tensor.view(*input_shape)
        
        # Infer property type
        property_type = _infer_property_type(content, num_outputs)
        
        metadata = {
            'input_bounds': input_bounds,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'property_type': property_type,
            'vnnlib_path': str(vnnlib_path),
            'dialect': (
                'vnnlib-2.0'
                if "(vnnlib-version" in content or "(declare-network" in content
                else 'vnnlib-1.0-flat'
            ),
        }
        
        logger.info(
            f"Parsed VNNLIB: {num_inputs} inputs, {num_outputs} outputs, "
            f"type={property_type}"
        )
        
        return input_tensor, metadata

    except (VNNLibParseError, UnsupportedSpecError):
        raise
    except Exception as e:
        raise VNNLibParseError(f"Failed to parse {vnnlib_path}: {str(e)}")


def parse_vnnlib_queries(
    vnnlib_path: Path,
    labeled_tensor: Optional['LabeledInputTensor'] = None
) -> List[Tuple[InputSpec, OutputSpec]]:
    """
    Parse an audited VNNLIB 1.0-flat or VNNLIB 2.0 file into queries.

    The legacy path is intentionally narrow.  It accepts complete finite
    per-coordinate input boxes plus affine output-only Boolean assertions.
    Coupled/non-rectangular inputs, branch-local input constraints, mixed X/Y
    rows, strict inequalities and equalities are rejected instead of being
    silently approximated.  Raw concrete replay remains able to evaluate those
    constructs without materialising them.

    Semantics:
      - Multiple top-level ``(assert ...)`` forms are conjunctive (implicit AND).
      - ``(or ...)`` inside an assert expands to multiple queries
        (Cartesian product across all asserts).
      - Inequalities involving only X are folded into the input BOX.
      - Inequalities involving Y become rows of an UNSAFE_LINEAR OutputSpec.
      - When ``labeled_tensor.label`` is provided and queries match the
        classification pattern (Y_j - Y_true <= 0 for all j != true), the
        result collapses to a single TOP1_ROBUST OutputSpec.

    Raises:
        UnsupportedSpecError: If the property cannot be materialised exactly.
        VNNLibParseError: If the file is missing or unparseable.
    """
    if not vnnlib_path.exists():
        raise VNNLibParseError(f"VNNLIB file not found: {vnnlib_path}")
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
    except Exception as e:
        raise VNNLibParseError(f"Failed to read {vnnlib_path}: {e}") from e

    semantic_content = _strip_smt_comments(content)
    if (
        re.search(r"\(\s*vnnlib-version\b", semantic_content)
        or re.search(r"\(\s*declare-network\b", semantic_content)
    ):
        return parse_vnnlib_2_0(vnnlib_path, labeled_tensor=labeled_tensor)

    layout = _prepare_concrete_vnnlib(
        content, required_dialect="vnnlib-1.0-flat"
    )
    num_inputs = int(layout["num_inputs"])
    num_outputs = int(layout["num_outputs"])
    _validate_legacy_query_materialization(
        layout["rewritten"], num_inputs, num_outputs
    )

    tensor_shape = (
        tuple(labeled_tensor.tensor.shape)
        if labeled_tensor is not None
        else (num_inputs,)
    )
    if _numel(tensor_shape) != num_inputs:
        raise VNNLibParseError(
            f"legacy VNNLIB declares {num_inputs} inputs, but model/sample "
            f"shape {tensor_shape} has {_numel(tensor_shape)} elements"
        )
    true_label = (
        labeled_tensor.label if labeled_tensor is not None else None
    )
    if true_label is None:
        true_label = extract_label_from_vnnlib(vnnlib_path)
    true_label = _validate_true_label(true_label, num_outputs)
    return _queries_from_rewritten(
        layout["rewritten"],
        num_inputs,
        num_outputs,
        tensor_shape,
        true_label,
        vnnlib_path.name,
        dialect_name="vnnlib 1.0 flat",
    )


def validate_vnnlib_file(vnnlib_path: Path) -> bool:
    """
    Validate that a VNNLIB file is parseable.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        parse_vnnlib_to_tensors(vnnlib_path)
        return True
    except (VNNLibParseError, UnsupportedSpecError) as e:
        logger.error(f"VNNLIB validation failed: {e}")
        return False


def extract_label_from_vnnlib(vnnlib_path: Path) -> Optional[int]:
    """
    Extract ground truth label from VNNLIB file comment.
    
    Many VNNLIB files (e.g., CIFAR-100) include ground truth labels in comments:
    ; CIFAR100 property with label: 14.
    
    Args:
        vnnlib_path: Path to .vnnlib file
        
    Returns:
        Ground truth label as integer, or None if not found
        
    Example:
        >>> label = extract_label_from_vnnlib(Path("spec.vnnlib"))
        >>> print(label)
        14
    """
    try:
        with open(vnnlib_path, 'r') as f:
            # Read first few lines (label is typically in first comment)
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                
                # Match patterns like: ; CIFAR100 property with label: 14.
                match = re.search(r'label:\s*(\d+)', line, re.IGNORECASE)
                if match:
                    return int(match.group(1))
        
        return None
    except Exception as e:
        logger.debug(f"Failed to extract label from {vnnlib_path}: {e}")
        return None


# -------------------------------------------------------------------------
# Module-level regex patterns and type aliases
# -------------------------------------------------------------------------


_X_RE = re.compile(r"X_(\d+)")
_Y_RE = re.compile(r"Y_(\d+)")
_Ineq = Tuple[List[float], List[float], float]
_Query = List[_Ineq]
_MAX_LEGACY_DNF_QUERIES = 10_000


# -------------------------------------------------------------------------
# VNNLIB 2.0 parsing and legacy-token rewrite
# -------------------------------------------------------------------------


def parse_vnnlib_2_0(
    vnnlib_path: Path,
    labeled_tensor: Optional['LabeledInputTensor'] = None,
) -> List[Tuple[InputSpec, OutputSpec]]:
    """Parse VNNLIB 2.0 by ravel-rewriting bracket variables to legacy names."""
    if not vnnlib_path.exists():
        raise VNNLibParseError(f"VNNLIB file not found: {vnnlib_path}")
    try:
        with open(vnnlib_path, 'r') as f:
            content = f.read()
    except Exception as e:
        raise VNNLibParseError(f"Failed to read {vnnlib_path}: {e}") from e

    if len(re.findall(r"\(\s*declare-network\b", content)) >= 2:
        if "isomorphic-to" in content:
            return _parse_vnnlib_2_0_isomorphic(content, labeled_tensor)
        raise UnsupportedSpecError("multi-network (equal-to/monotonic) not yet supported")

    input_name, _input_dtype, input_shape = _extract_vnnlib_2_decl(content, "input")
    output_name, _output_dtype, output_shape = _extract_vnnlib_2_decl(content, "output")
    num_inputs = _numel(input_shape)
    num_outputs = _numel(output_shape)
    tensor_shape = tuple(labeled_tensor.tensor.shape) if labeled_tensor is not None else tuple(input_shape)
    tensor_numel = _numel(tensor_shape)
    if tensor_numel != num_inputs:
        raise VNNLibParseError(
            f"VNNLIB 2.0 declared input shape {input_shape} has {num_inputs} elements, "
            f"but model/sample input shape {tensor_shape} has {tensor_numel}"
        )
    true_label = _validate_true_label(
        labeled_tensor.label if labeled_tensor is not None else None,
        num_outputs,
    )

    rewritten = _rewrite_vnnlib_2_bracket_vars(
        content,
        input_name=input_name,
        input_shape=input_shape,
        output_name=output_name,
        output_shape=output_shape,
    )
    return _queries_from_rewritten(
        rewritten, num_inputs, num_outputs, tensor_shape, true_label, vnnlib_path.name
    )


def _queries_from_rewritten(
    rewritten: str,
    num_inputs: int,
    num_outputs: int,
    tensor_shape: Tuple[int, ...],
    true_label,
    name: str,
    *,
    dialect_name: str = "vnnlib 2.0",
) -> List[Tuple[InputSpec, OutputSpec]]:
    """Shared core: turn a flat-name-rewritten 2.0 body into (InputSpec, OutputSpec)
    queries. Used by both single-network and isomorphic dual-network 2.0 parsing."""
    try:
        forms = _parse_all_forms(rewritten)
    except VNNLibParseError:
        raise
    except Exception as e:
        raise VNNLibParseError(f"S-expression parse failed: {e}") from e

    asserts = [f for f in forms if isinstance(f, list) and len(f) >= 2 and f[0] == "assert"]
    assert_bodies = [_normalize_vnnlib_2_body(f[1]) for f in asserts]
    for body in assert_bodies:
        if _contains_non_affine(body, num_inputs, num_outputs):
            raise UnsupportedSpecError("nonlinear VNNLIB 2.0 assertion not supported")

    simple_assert_bodies = [f for f in assert_bodies if _is_simple_x_bound(f)]
    complex_assert_bodies = [f for f in assert_bodies if not _is_simple_x_bound(f)]
    bounds_dict = _extract_input_bounds(simple_assert_bodies, num_inputs)
    base_in_spec = _build_input_spec(num_inputs, tensor_shape, bounds_dict, [])

    if not complex_assert_bodies:
        out_spec = _build_output_spec([], num_outputs, true_label)
        logger.info(
            f"Parsed {name}: 1 query(ies) [{dialect_name} input-only]"
        )
        return [(base_in_spec, out_spec)]

    per_assert: List[List[_Query]] = []
    for body in complex_assert_bodies:
        qs = _process_body(body, num_inputs, num_outputs)
        if qs is None:
            raise UnsupportedSpecError(f"unsupported VNNLIB 2.0 assertion: {body}")
        per_assert.append(qs)

    complex_queries = _combine_conjunctive_queries(per_assert)
    results: List[Tuple[InputSpec, OutputSpec]] = []
    for q in complex_queries:
        x_ineqs: _Query = []
        y_ineqs: _Query = []
        skip = False
        for xc, yc, d in q:
            if any(v != 0 for v in yc):
                y_ineqs.append((xc, yc, d))
            elif any(v != 0 for v in xc):
                x_ineqs.append((xc, yc, d))
            elif d < 0:
                logger.debug(f"Infeasible constant constraint: 0 <= {d}")
                skip = True
                break
        if skip:
            continue
        in_spec = _build_input_spec(num_inputs, tensor_shape, bounds_dict, x_ineqs) if x_ineqs else base_in_spec
        out_spec = _build_output_spec(y_ineqs, num_outputs, true_label)
        results.append((in_spec, out_spec))

    if true_label is not None:
        promoted = _try_promote_to_top1(results, num_outputs, true_label)
        if promoted is not None:
            results = [promoted]

    if len(results) > 1:
        promoted = _try_promote_to_top1_unlabeled(results, num_outputs)
        if promoted is not None:
            results = [promoted]

    logger.info(
        f"Parsed {name}: {len(results)} query(ies) [{dialect_name}]"
    )
    return results


def _extract_all_vnnlib_2_decls(content: str, io_kind: str) -> List[Tuple[str, Tuple[int, ...]]]:
    """All declare-input/output entries (multi-network files have one per network)."""
    pattern = re.compile(
        rf"\(\s*declare-{io_kind}\s+([A-Za-z_]\w*)\s+\S+\s+\[([^\]]+)\]\s*\)",
        re.MULTILINE,
    )
    decls: List[Tuple[str, Tuple[int, ...]]] = []
    for m in pattern.finditer(content):
        dims = tuple(int(p.strip()) for p in m.group(2).split(",") if p.strip())
        if not dims or any(d <= 0 for d in dims):
            raise VNNLibParseError(f"Invalid VNNLIB 2.0 declare-{io_kind} shape: {m.group(2)}")
        decls.append((m.group(1), dims))
    if not decls:
        raise VNNLibParseError(f"VNNLIB 2.0 missing declare-{io_kind}")
    return decls


def _isomorphic_multinet_rewrite(content: str) -> Tuple[str, int, int, Tuple[int, ...]]:
    """Flatten an isomorphic (f,g) dual-network file to one variable namespace.

    Inputs X_f/X_g are SHARED (tied by ``(== X_f[i] X_g[i])``) so both map to the
    same X_<flat>; outputs concatenate as [Y_f ; Y_g], i.e. Y_f[j]->Y_<flat>,
    Y_g[j]->Y_<numel(Y_f)+flat>. The self-equality link asserts collapse to the
    trivial ``0<=0`` and are ignored downstream. Returns
    (rewritten, num_inputs, num_outputs_concat, f_input_shape).
    """
    inputs = _extract_all_vnnlib_2_decls(content, "input")
    outputs = _extract_all_vnnlib_2_decls(content, "output")
    if len(inputs) < 2 or len(outputs) < 2:
        raise UnsupportedSpecError("isomorphic spec requires two networks")
    (f_in_name, f_in_shape), (g_in_name, g_in_shape) = inputs[0], inputs[1]
    (f_out_name, f_out_shape), (g_out_name, g_out_shape) = outputs[0], outputs[1]
    f_out_numel = _numel(f_out_shape)
    num_inputs = _numel(f_in_shape)
    num_outputs = f_out_numel + _numel(g_out_shape)
    name_map = {
        f_in_name: ("X", f_in_shape, 0),
        g_in_name: ("X", g_in_shape, 0),
        f_out_name: ("Y", f_out_shape, 0),
        g_out_name: ("Y", g_out_shape, f_out_numel),
    }
    var_re = re.compile(r"\b([A-Za-z_]\w*)\s*\[([^\]]+)\]")

    def repl(m: "re.Match[str]") -> str:
        info = name_map.get(m.group(1))
        if info is None:
            return m.group(0)
        prefix, shape, base = info
        idx = tuple(int(p.strip()) for p in m.group(2).split(",") if p.strip())
        return f"{prefix}_{base + _ravel_c_order(idx, shape)}"

    return var_re.sub(repl, content), num_inputs, num_outputs, f_in_shape


def _parse_vnnlib_2_0_isomorphic(
    content: str,
    labeled_tensor: Optional['LabeledInputTensor'] = None,
) -> List[Tuple[InputSpec, OutputSpec]]:
    """Isomorphic equivalence: verify f and g (shared input) agree; specs are
    built over the concatenated output [Y_f ; Y_g] of the combined model."""
    rewritten, num_inputs, num_outputs, f_in_shape = _isomorphic_multinet_rewrite(content)
    tensor_shape = tuple(labeled_tensor.tensor.shape) if labeled_tensor is not None else tuple(f_in_shape)
    if _numel(tensor_shape) != num_inputs:
        raise VNNLibParseError(
            f"isomorphic input shape {f_in_shape} ({num_inputs} elems) != sample {tensor_shape}"
        )
    return _queries_from_rewritten(rewritten, num_inputs, num_outputs, tensor_shape, None, "isomorphic")


def _extract_vnnlib_2_decl(content: str, io_kind: str) -> Tuple[str, str, Tuple[int, ...]]:
    """Return ``(var_name, dtype, shape)``; dtype feeds the 2.0 witness header."""
    pattern = re.compile(
        rf"\(\s*declare-{io_kind}\s+([A-Za-z_]\w*)\s+(\S+)\s+\[([^\]]+)\]\s*\)",
        re.MULTILINE,
    )
    match = pattern.search(content)
    if match is None:
        raise VNNLibParseError(f"VNNLIB 2.0 missing declare-{io_kind}")
    dims = tuple(int(part.strip()) for part in match.group(3).split(",") if part.strip())
    if not dims or any(dim <= 0 for dim in dims):
        raise VNNLibParseError(f"Invalid VNNLIB 2.0 declare-{io_kind} shape: {match.group(3)}")
    return match.group(1), match.group(2), dims


def _numel(shape: Tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def _ravel_c_order(indices: Tuple[int, ...], shape: Tuple[int, ...]) -> int:
    if len(indices) != len(shape):
        raise VNNLibParseError(f"Index rank {indices} does not match declared shape {shape}")
    flat = 0
    for idx, dim in zip(indices, shape):
        if idx < 0 or idx >= dim:
            raise VNNLibParseError(f"Index {indices} out of bounds for declared shape {shape}")
        flat = flat * dim + idx
    return flat


def _rewrite_vnnlib_2_bracket_vars(
    content: str,
    input_name: str,
    input_shape: Tuple[int, ...],
    output_name: str,
    output_shape: Tuple[int, ...],
) -> str:
    var_re = re.compile(r"\b([A-Za-z_]\w*)\s*\[([^\]]+)\]")

    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in (input_name, output_name):
            return match.group(0)
        indices = tuple(int(part.strip()) for part in match.group(2).split(",") if part.strip())
        if name == input_name:
            return f"X_{_ravel_c_order(indices, input_shape)}"
        return f"Y_{_ravel_c_order(indices, output_shape)}"

    return var_re.sub(repl, content)


def _legacy_flat_declared_size(
    forms: List[Any],
    prefix: str,
) -> int:
    declared: List[int] = []
    seen = set()
    token_re = re.compile(r"([XY])_(\d+)")
    for form in forms:
        if not (
            isinstance(form, list)
            and form
            and form[0] == "declare-const"
        ):
            continue
        if len(form) != 3:
            raise VNNLibParseError(
                f"legacy declare-const expects name and sort: {form!r}"
            )
        name, sort = form[1], form[2]
        match = token_re.fullmatch(name) if isinstance(name, str) else None
        if match is None:
            raise UnsupportedSpecError(
                f"unsupported legacy declaration {form!r}"
            )
        if match.group(1) != prefix:
            continue
        if sort != "Real":
            raise UnsupportedSpecError(
                f"legacy concrete evaluator supports Real declarations, got {sort!r}"
            )
        index = int(match.group(2))
        if index in seen:
            raise VNNLibParseError(
                f"duplicate legacy declaration {prefix}_{index}"
            )
        seen.add(index)
        declared.append(index)
    if not declared:
        raise VNNLibParseError(
            f"legacy VNNLIB missing declare-const {prefix}_i"
        )
    expected = set(range(max(declared) + 1))
    if seen != expected:
        missing = sorted(expected - seen)
        raise VNNLibParseError(
            f"legacy {prefix} declarations are not contiguous; missing {missing[:8]}"
        )
    return len(expected)


def _strip_smt_comments(content: str) -> str:
    """Remove SMT-LIB line comments before dialect recognition."""

    return "\n".join(line.split(";", 1)[0] for line in content.splitlines())


def _prepare_concrete_vnnlib(
    content: str,
    *,
    required_dialect: Optional[str],
) -> Dict[str, Any]:
    semantic_content = _strip_smt_comments(content)
    has_v2 = bool(
        re.search(r"\(\s*vnnlib-version\b", semantic_content)
        or re.search(r"\(\s*declare-network\b", semantic_content)
        or re.search(r"\(\s*declare-input\b", semantic_content)
        or re.search(r"\(\s*declare-output\b", semantic_content)
    )
    has_legacy = bool(
        re.search(
            r"\(\s*declare-const\s+[XY]_\d+\s+Real\s*\)",
            semantic_content,
        )
    )
    if has_v2 and has_legacy:
        raise UnsupportedSpecError("mixed VNNLIB 1.0/2.0 declarations")
    if not has_v2 and not has_legacy:
        raise UnsupportedSpecError("unrecognized VNNLIB declaration dialect")

    if has_v2:
        dialect = "vnnlib-2.0"
        if len(re.findall(r"\(\s*declare-network\b", semantic_content)) > 1:
            raise UnsupportedSpecError(
                "concrete evaluator requires one network/output tensor"
            )
        input_name, _input_dtype, input_shape = _extract_vnnlib_2_decl(
            semantic_content, "input"
        )
        output_name, _output_dtype, output_shape = _extract_vnnlib_2_decl(
            semantic_content, "output"
        )
        rewritten = _rewrite_vnnlib_2_bracket_vars(
            semantic_content,
            input_name=input_name,
            input_shape=input_shape,
            output_name=output_name,
            output_shape=output_shape,
        )
    else:
        dialect = "vnnlib-1.0-flat"
        forms = _parse_all_forms(semantic_content)
        num_inputs = _legacy_flat_declared_size(forms, "X")
        num_outputs = _legacy_flat_declared_size(forms, "Y")
        input_shape = (num_inputs,)
        output_shape = (num_outputs,)
        rewritten = semantic_content

        declared_x = set(range(num_inputs))
        declared_y = set(range(num_outputs))
        referenced_x = {int(item) for item in _X_RE.findall(semantic_content)}
        referenced_y = {int(item) for item in _Y_RE.findall(semantic_content)}
        if not referenced_x.issubset(declared_x):
            raise VNNLibParseError(
                f"legacy assertion references undeclared X indices "
                f"{sorted(referenced_x - declared_x)[:8]}"
            )
        if not referenced_y.issubset(declared_y):
            raise VNNLibParseError(
                f"legacy assertion references undeclared Y indices "
                f"{sorted(referenced_y - declared_y)[:8]}"
            )

    if required_dialect is not None and dialect != required_dialect:
        raise UnsupportedSpecError(
            f"expected {required_dialect}, recognized {dialect}"
        )
    return {
        "dialect": dialect,
        "vnnlib_version": "2.0" if has_v2 else "1.0",
        "input_shape": tuple(input_shape),
        "output_shape": tuple(output_shape),
        "num_inputs": _numel(tuple(input_shape)),
        "num_outputs": _numel(tuple(output_shape)),
        "rewritten": rewritten,
    }


def _validate_legacy_output_boolean(
    body: Any,
    num_inputs: int,
    num_outputs: int,
) -> None:
    """Validate one exactly materialisable legacy output Boolean tree."""

    if not isinstance(body, list) or not body:
        raise UnsupportedSpecError(
            f"legacy output assertion is not a Boolean form: {body!r}"
        )
    op = body[0]
    if op in ("and", "or"):
        if len(body) < 2:
            raise UnsupportedSpecError(
                f"zero-arity legacy {op} is not materialised"
            )
        for child in body[1:]:
            _validate_legacy_output_boolean(
                child, num_inputs, num_outputs
            )
        return

    if op in ("<", ">", "=", "=="):
        raise UnsupportedSpecError(
            "strict inequalities/equalities are supported by raw replay but "
            "are not materialised by the legacy query frontend"
        )
    if op not in ("<=", ">=") or len(body) != 3:
        raise UnsupportedSpecError(
            f"unsupported legacy Boolean/output form: {body!r}"
        )
    inequality = _parse_inequality(
        op, body[1], body[2], num_inputs, num_outputs
    )
    if inequality is None:
        raise UnsupportedSpecError(
            f"legacy output comparison is non-affine or malformed: {body!r}"
        )
    x_coeffs, y_coeffs, constant = inequality
    if any(value != 0.0 for value in x_coeffs):
        raise UnsupportedSpecError(
            "legacy coupled/non-rectangular or mixed X/Y assertion cannot be "
            f"materialised as BOX + output rows: {body!r}"
        )
    if not any(value != 0.0 for value in y_coeffs):
        raise UnsupportedSpecError(
            f"legacy non-input assertion must constrain at least one Y: {body!r}"
        )
    if not all(
        math.isfinite(float(value))
        for value in [*y_coeffs, constant]
    ):
        raise VNNLibParseError(
            f"legacy output assertion has non-finite coefficients: {body!r}"
        )


def _legacy_dnf_branch_count(body: Any, cap: int) -> int:
    """Count materialised DNF branches without constructing the product."""

    if not isinstance(body, list) or not body:
        return 1
    op = body[0]
    if op == "or":
        count = 0
        for child in body[1:]:
            count += _legacy_dnf_branch_count(child, cap)
            if count > cap:
                return count
        return count
    if op == "and":
        count = 1
        for child in body[1:]:
            count *= _legacy_dnf_branch_count(child, cap)
            if count > cap:
                return count
        return count
    return 1


def _validate_legacy_query_materialization(
    content: str,
    num_inputs: int,
    num_outputs: int,
) -> None:
    """Fail closed unless legacy syntax maps exactly to ACT's query objects."""

    forms = _parse_all_forms(content)
    simple_input_bodies: List[Any] = []
    output_assertions = 0
    projected_queries = 1
    for form in forms:
        if not isinstance(form, list) or not form:
            raise UnsupportedSpecError(
                f"unsupported top-level legacy form: {form!r}"
            )
        command = form[0]
        if command == "declare-const":
            continue
        if command == "set-logic":
            if len(form) != 2:
                raise VNNLibParseError(
                    f"malformed legacy set-logic: {form!r}"
                )
            continue
        if command != "assert":
            raise UnsupportedSpecError(
                f"unsupported top-level legacy command {command!r}"
            )
        if len(form) != 2:
            raise VNNLibParseError(
                f"assert expects exactly one body: {form!r}"
            )
        body = form[1]
        if _is_simple_x_bound(body):
            simple_input_bodies.append(body)
            continue
        _validate_legacy_output_boolean(
            body, num_inputs, num_outputs
        )
        projected_queries *= _legacy_dnf_branch_count(
            body, _MAX_LEGACY_DNF_QUERIES
        )
        if projected_queries > _MAX_LEGACY_DNF_QUERIES:
            raise UnsupportedSpecError(
                "legacy Boolean expansion would create "
                f"{projected_queries} queries; limit is "
                f"{_MAX_LEGACY_DNF_QUERIES}"
            )
        output_assertions += 1

    bounds = _extract_input_bounds(simple_input_bodies, num_inputs)
    if len(bounds) != num_inputs:
        missing = [index for index in range(num_inputs) if index not in bounds]
        raise UnsupportedSpecError(
            "legacy query frontend requires a bound for every input; "
            f"missing indices {missing[:8]}"
        )
    for index in range(num_inputs):
        lower, upper = bounds[index]
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise UnsupportedSpecError(
                "legacy query frontend requires finite lower and upper bounds "
                f"for X_{index}, got [{lower}, {upper}]"
            )
        if lower > upper:
            raise VNNLibParseError(
                f"legacy input box is empty at X_{index}: [{lower}, {upper}]"
            )
    if output_assertions == 0:
        raise UnsupportedSpecError(
            "legacy query frontend requires at least one output assertion"
        )


def _concrete_flat_values(value: Any, name: str, expected: int) -> List[float]:
    try:
        tensor = (
            value.detach().cpu()
            if isinstance(value, torch.Tensor)
            else torch.as_tensor(value)
        )
    except Exception as exc:
        raise VNNLibParseError(
            f"concrete {name} cannot be converted to a numeric tensor"
        ) from exc
    if tensor.is_complex():
        raise VNNLibParseError(f"concrete {name} must be real-valued")
    flat = tensor.reshape(-1)
    if int(flat.numel()) != int(expected):
        raise VNNLibParseError(
            f"concrete {name} has {flat.numel()} elements, expected {expected}"
        )
    try:
        finite = bool(torch.isfinite(flat).all().item())
    except (RuntimeError, TypeError) as exc:
        raise VNNLibParseError(f"concrete {name} must be numeric") from exc
    if not finite:
        raise VNNLibParseError(f"concrete {name} contains NaN or infinity")
    return [float(item) for item in flat.tolist()]


def _concrete_linear_value(
    expr: Any,
    x: List[float],
    y: List[float],
    num_inputs: int,
    num_outputs: int,
) -> float:
    parsed = _parse_linear_expr(expr, num_inputs, num_outputs)
    if parsed is None:
        raise UnsupportedSpecError(
            f"unsupported or non-affine concrete expression: {expr!r}"
        )
    x_coeffs, y_coeffs, constant = parsed
    terms = [float(constant)]
    terms.extend(float(coef) * value for coef, value in zip(x_coeffs, x))
    terms.extend(float(coef) * value for coef, value in zip(y_coeffs, y))
    if not all(math.isfinite(term) for term in terms):
        raise VNNLibParseError(
            f"non-finite coefficient or value in concrete expression: {expr!r}"
        )
    try:
        result = math.fsum(terms)
    except OverflowError as exc:
        raise VNNLibParseError(
            f"overflow while evaluating concrete expression: {expr!r}"
        ) from exc
    if not math.isfinite(result):
        raise VNNLibParseError(
            f"non-finite result for concrete expression: {expr!r}"
        )
    return result


def _evaluate_concrete_boolean(
    body: Any,
    x: List[float],
    y: List[float],
    tol: float,
    atom_log: List[Dict[str, Any]],
    num_inputs: int,
    num_outputs: int,
) -> Tuple[bool, Dict[str, Any]]:
    if not isinstance(body, list) or not body:
        raise UnsupportedSpecError(
            f"unsupported concrete Boolean expression: {body!r}"
        )
    op = body[0]
    if op in ("and", "or"):
        children = [
            _evaluate_concrete_boolean(
                child,
                x,
                y,
                tol,
                atom_log,
                num_inputs,
                num_outputs,
            )
            for child in body[1:]
        ]
        child_holds = [holds for holds, _tree in children]
        holds = all(child_holds) if op == "and" else any(child_holds)
        return bool(holds), {
            "kind": op,
            "holds": bool(holds),
            "children": [tree for _holds, tree in children],
        }

    if op not in ("<", ">", "<=", ">=", "=", "==") or len(body) != 3:
        raise UnsupportedSpecError(
            f"unsupported concrete Boolean operator/form: {body!r}"
        )
    lhs = _concrete_linear_value(
        body[1], x, y, num_inputs, num_outputs
    )
    rhs = _concrete_linear_value(
        body[2], x, y, num_inputs, num_outputs
    )
    residual = lhs - rhs
    if not math.isfinite(residual):
        raise VNNLibParseError(
            f"non-finite comparison residual: lhs={lhs!r}, rhs={rhs!r}"
        )

    if op == "<=":
        holds = residual <= tol
        slack = -residual
    elif op == ">=":
        holds = residual >= -tol
        slack = residual
    elif op == "<":
        holds = residual < tol
        slack = -residual
    elif op == ">":
        holds = residual > -tol
        slack = residual
    else:
        holds = abs(residual) <= tol
        slack = -abs(residual)

    atom_index = len(atom_log)
    atom_log.append({
        "index": atom_index,
        "op": op,
        "lhs": body[1],
        "rhs": body[2],
        "lhs_value": float(lhs),
        "rhs_value": float(rhs),
        "residual_lhs_minus_rhs": float(residual),
        "slack": float(slack),
        "tolerance": float(tol),
        "holds": bool(holds),
    })
    return bool(holds), {
        "kind": "atom",
        "holds": bool(holds),
        "atom_index": atom_index,
    }


def _normalize_vnnlib_2_body(body: Any) -> Any:
    if not isinstance(body, list) or not body:
        return body
    op = body[0]
    children = [_normalize_vnnlib_2_body(child) for child in body[1:]]
    if op == "<" and len(children) == 2:
        return ["<=", children[0], children[1]]
    if op == ">" and len(children) == 2:
        return [">=", children[0], children[1]]
    if op in ("=", "==") and len(children) == 2:
        return ["and", ["<=", children[0], children[1]], [">=", children[0], children[1]]]
    return [op] + children


def _contains_non_affine(expr: Any, num_inputs: int, num_outputs: int) -> bool:
    if not isinstance(expr, list) or not expr:
        return False
    op = expr[0]
    if op == "*":
        non_const = sum(
            1 for sub in expr[1:]
            if not _is_constant_linear_expr(sub, num_inputs, num_outputs)
        )
        if non_const >= 2:
            return True
    if op == "/" and len(expr) >= 3:
        for denom in expr[2:]:
            if not _is_constant_linear_expr(denom, num_inputs, num_outputs):
                return True
    if op in ("^", "pow") and len(expr) == 3:
        base_const = _is_constant_linear_expr(expr[1], num_inputs, num_outputs)
        try:
            exponent = float(expr[2])
        except (TypeError, ValueError):
            exponent = None
        if not base_const and exponent != 1.0:
            return True
    return any(_contains_non_affine(sub, num_inputs, num_outputs) for sub in expr[1:])


def _is_constant_linear_expr(expr: Any, num_inputs: int, num_outputs: int) -> bool:
    parsed = _parse_linear_expr(expr, num_inputs, num_outputs)
    if parsed is None:
        return False
    xc, yc, _d = parsed
    return all(v == 0 for v in xc) and all(v == 0 for v in yc)


# -------------------------------------------------------------------------
# Shared bound/property extractors (operate on flat X_n/Y_n post-rewrite)
# -------------------------------------------------------------------------


def _extract_input_bounds(
    simple_bodies: List[Any],
    num_inputs: int,
) -> Dict[int, Tuple[float, float]]:
    """Extract per-variable [lb, ub] from top-level simple X-bound asserts.

    ``simple_bodies`` is the list of assertion bodies that already passed
    ``_is_simple_x_bound``. Constraints inside ``(or ...)`` branches must be
    excluded by the caller — intersecting them here would produce empty boxes
    for disjunctive-input properties (ACAS Xu prop_5..10).
    """
    bounds = {i: [float('-inf'), float('inf')] for i in range(num_inputs)}
    for body in simple_bodies:
        if not (isinstance(body, list) and len(body) == 3):
            continue
        op, left, right = body
        if op not in ("<=", ">="):
            continue
        x_is_left = isinstance(left, str) and bool(_X_RE.fullmatch(left))
        x_is_right = isinstance(right, str) and bool(_X_RE.fullmatch(right))
        if x_is_left == x_is_right:
            continue
        x_tok = left if x_is_left else right
        lit_tok = right if x_is_left else left
        try:
            val = float(lit_tok)
        except (TypeError, ValueError):
            continue
        x_match = _X_RE.fullmatch(x_tok)
        if x_match is None:
            continue
        idx = int(x_match.group(1))
        if idx >= num_inputs:
            continue
        # (<= X val)  -> X <= val   (upper)
        # (>= X val)  -> X >= val   (lower)
        # (<= val X)  -> val <= X   -> X >= val (lower)
        # (>= val X)  -> val >= X   -> X <= val (upper)
        tighten_upper = (op == "<=" and x_is_left) or (op == ">=" and not x_is_left)
        if tighten_upper:
            bounds[idx][1] = min(bounds[idx][1], val)
        else:
            bounds[idx][0] = max(bounds[idx][0], val)
    return {
        i: (lb, ub)
        for i, (lb, ub) in bounds.items()
        if lb != float('-inf') or ub != float('inf')
    }


def _infer_property_type(content: str, num_outputs: int) -> str:
    """
    Infer the property type from VNNLIB content.
    
    Returns:
        One of: 'classification', 'safety', 'unknown'
    """
    content_lower = content.lower()
    
    # Classification properties often involve comparisons between outputs
    if 'y_' in content_lower and num_outputs > 1:
        # Check for patterns like Y_i - Y_j > 0 (classification margin)
        if re.search(r'y_\d+\s*[-]\s*y_\d+', content_lower):
            return 'classification'
    
    # Safety properties typically have output range constraints
    if num_outputs == 1 or 'range' in content_lower:
        return 'safety'
    
    return 'unknown'


# -------------------------------------------------------------------------
# Helper: label coercion
# -------------------------------------------------------------------------


def _validate_true_label(
    true_label: Any,
    num_outputs: int,
) -> Optional[int]:
    """Return one in-range integer class label or reject explicitly."""

    if true_label is None:
        return None
    if isinstance(true_label, torch.Tensor):
        if true_label.numel() != 1:
            raise VNNLibParseError(
                "classification label must contain exactly one element, "
                f"got shape {tuple(true_label.shape)}"
            )
        value = true_label.detach().cpu().reshape(-1)[0].item()
    else:
        value = true_label
    if isinstance(value, bool):
        raise VNNLibParseError("classification label cannot be Boolean")
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VNNLibParseError(
            f"classification label must be an integer, got {value!r}"
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer():
        raise VNNLibParseError(
            f"classification label must be a finite integer, got {value!r}"
        )
    label = int(numeric)
    if label < 0 or label >= int(num_outputs):
        raise VNNLibParseError(
            f"classification label {label} is outside [0, {num_outputs})"
        )
    return label


def _coerce_label_to_tensor(true_label: Any) -> torch.Tensor:
    """Coerce int/tensor label to 1-D int64 tensor (defensive)."""
    if isinstance(true_label, torch.Tensor):
        return true_label.clone()
    return torch.tensor([int(true_label)], dtype=torch.int64)


# -------------------------------------------------------------------------
# Step 1: S-expression tokenization and AST parsing
# -------------------------------------------------------------------------


def _tokenize_sexpr(text: str) -> List[str]:
    lines = []
    for raw in text.split("\n"):
        idx = raw.find(";")
        if idx >= 0:
            raw = raw[:idx]
        lines.append(raw)
    text = " ".join(lines)
    text = text.replace("(", " ( ").replace(")", " ) ")
    return [str(tok) for tok in text.split()]


def _parse_sexpr(tokens: List[str], pos: int) -> Tuple[Any, int]:
    if pos >= len(tokens):
        raise VNNLibParseError("Unexpected EOF in S-expression")
    tok = tokens[pos]
    if tok == "(":
        result = []
        pos += 1
        while pos < len(tokens) and tokens[pos] != ")":
            item, pos = _parse_sexpr(tokens, pos)
            result.append(item)
        if pos >= len(tokens):
            raise VNNLibParseError("Unbalanced '('")
        return result, pos + 1
    if tok == ")":
        raise VNNLibParseError(f"Unexpected ')' at pos {pos}")
    return tok, pos + 1


def _parse_all_forms(text: str) -> List[Any]:
    tokens = _tokenize_sexpr(text)
    out = []
    pos = 0
    while pos < len(tokens):
        form, pos = _parse_sexpr(tokens, pos)
        out.append(form)
    return out


# -------------------------------------------------------------------------
# Step 2: Linear algebra on S-expressions
# -------------------------------------------------------------------------


def _parse_linear_atom(expr: Any, num_inputs: int, num_outputs: int) -> Optional[_Ineq]:
    if not isinstance(expr, str):
        return None
    xm = _X_RE.fullmatch(expr)
    if xm:
        idx = int(xm.group(1))
        if idx >= num_inputs:
            return None
        xc = [0.0] * num_inputs
        xc[idx] = 1.0
        return xc, [0.0] * num_outputs, 0.0
    ym = _Y_RE.fullmatch(expr)
    if ym:
        idx = int(ym.group(1))
        if idx >= num_outputs:
            return None
        yc = [0.0] * num_outputs
        yc[idx] = 1.0
        return [0.0] * num_inputs, yc, 0.0
    try:
        return [0.0] * num_inputs, [0.0] * num_outputs, float(expr)
    except (ValueError, TypeError):
        return None


def _parse_linear_expr(expr: Any, num_inputs: int, num_outputs: int) -> Optional[_Ineq]:
    atom = _parse_linear_atom(expr, num_inputs, num_outputs)
    if atom is not None:
        return atom
    if not isinstance(expr, list) or not expr:
        return None
    op = expr[0]
    if op == "+":
        xc = [0.0] * num_inputs
        yc = [0.0] * num_outputs
        const = 0.0
        for sub in expr[1:]:
            r = _parse_linear_expr(sub, num_inputs, num_outputs)
            if r is None:
                return None
            sxc, syc, sd = r
            xc = [a + b for a, b in zip(xc, sxc)]
            yc = [a + b for a, b in zip(yc, syc)]
            const += sd
        return xc, yc, const
    if op == "-":
        if len(expr) == 2:
            r = _parse_linear_expr(expr[1], num_inputs, num_outputs)
            if r is None:
                return None
            sxc, syc, sd = r
            return [-a for a in sxc], [-a for a in syc], -sd
        if len(expr) == 3:
            a = _parse_linear_expr(expr[1], num_inputs, num_outputs)
            b = _parse_linear_expr(expr[2], num_inputs, num_outputs)
            if a is None or b is None:
                return None
            axc, ayc, ad = a
            bxc, byc, bd = b
            return (
                [x - y for x, y in zip(axc, bxc)],
                [x - y for x, y in zip(ayc, byc)],
                ad - bd,
            )
    if op == "*" and len(expr) == 3:
        a = _parse_linear_expr(expr[1], num_inputs, num_outputs)
        b = _parse_linear_expr(expr[2], num_inputs, num_outputs)
        if a is None or b is None:
            return None
        axc, ayc, ad = a
        bxc, byc, bd = b
        a_is_const = all(v == 0 for v in axc) and all(v == 0 for v in ayc)
        b_is_const = all(v == 0 for v in bxc) and all(v == 0 for v in byc)
        if a_is_const:
            return [ad * v for v in bxc], [ad * v for v in byc], ad * bd
        if b_is_const:
            return [bd * v for v in axc], [bd * v for v in ayc], ad * bd
    return None


def _parse_inequality(op: str, lhs: Any, rhs: Any, num_inputs: int, num_outputs: int) -> Optional[_Ineq]:
    l = _parse_linear_expr(lhs, num_inputs, num_outputs)
    r = _parse_linear_expr(rhs, num_inputs, num_outputs)
    if l is None or r is None:
        return None
    lxc, lyc, ld = l
    rxc, ryc, rd = r
    xc = [a - b for a, b in zip(lxc, rxc)]
    yc = [a - b for a, b in zip(lyc, ryc)]
    d = rd - ld
    if op == "<=":
        return xc, yc, d
    if op == ">=":
        return [-v for v in xc], [-v for v in yc], -d
    return None


# -------------------------------------------------------------------------
# Step 3: AND/OR composition and query assembly
# -------------------------------------------------------------------------


def _is_simple_x_bound(body: Any) -> bool:
    """Fast check: is body a simple single-variable X bound (op X_i val) / (op val X_i)?

    These asserts are fully handled by the regex-based :func:`_extract_input_bounds`
    and do NOT need S-expression processing; skipping them avoids quadratic blow-up
    from the Cartesian product when many X-bound asserts coexist with a large
    disjunctive output (common in CIFAR-100-style classification files).
    """
    if not (isinstance(body, list) and len(body) == 3):
        return False
    op = body[0]
    if op not in ("<=", ">="):
        return False
    left, right = body[1], body[2]
    if not (isinstance(left, str) and isinstance(right, str)):
        return False
    left_is_x = bool(_X_RE.fullmatch(left))
    right_is_x = bool(_X_RE.fullmatch(right))
    if left_is_x == right_is_x:
        return False
    other = right if left_is_x else left
    try:
        float(other)
        return True
    except (ValueError, TypeError):
        return False


def _process_body(body: Any, num_inputs: int, num_outputs: int) -> Optional[List[_Query]]:
    if not (isinstance(body, list) and body):
        return None
    op = body[0]
    if op == "and":
        subs = [_process_body(c, num_inputs, num_outputs) for c in body[1:]]
        subs = [s for s in subs if s is not None]
        if not subs:
            return [[]]
        combined = [[]]
        for sq in subs:
            new_combined = []
            for base in combined:
                for q in sq:
                    new_combined.append(base + q)
            combined = new_combined
        return combined
    if op == "or":
        all_q = []
        for d in body[1:]:
            sub = _process_body(d, num_inputs, num_outputs)
            if sub is None:
                continue
            all_q.extend(sub)
        return all_q if all_q else None
    if op in ("<=", ">=") and len(body) == 3:
        ineq = _parse_inequality(op, body[1], body[2], num_inputs, num_outputs)
        if ineq is None:
            return None
        return [[ineq]]
    return None


def _combine_conjunctive_queries(qs: List[List[_Query]]) -> List[_Query]:
    combined = [[]]
    for q_list in qs:
        new_combined = []
        for base in combined:
            for q in q_list:
                new_combined.append(base + q)
        combined = new_combined
    return combined


# -------------------------------------------------------------------------
# Step 4 (+ Step 5 TOP1 promotion): Spec builders
# -------------------------------------------------------------------------


def _build_input_spec(
    num_inputs: int,
    input_shape: Optional[Tuple[int, ...]],
    bounds_dict: Dict[int, Tuple[float, float]],
    extra_x_ineqs: _Query,
) -> InputSpec:
    """Build an InputSpec BOX from regex-extracted base bounds + optional tightening.

    ``bounds_dict`` is the output of :func:`_extract_input_bounds` — fast regex
    per-variable (lb, ub) pairs. ``extra_x_ineqs`` is a (typically empty) list of
    single-variable X inequalities from complex asserts (e.g. disjunctive X ranges)
    that further refine the BOX for one particular query. Multi-variable X
    constraints cannot be represented as a BOX and are silently ignored.
    """
    lb_vals = [bounds_dict.get(i, (float('-inf'), float('inf')))[0] for i in range(num_inputs)]
    ub_vals = [bounds_dict.get(i, (float('-inf'), float('inf')))[1] for i in range(num_inputs)]
    lb_tensor = torch.tensor(lb_vals)
    ub_tensor = torch.tensor(ub_vals)
    for xc, _yc, d in extra_x_ineqs:
        nonzero = [(i, v) for i, v in enumerate(xc) if v != 0]
        if len(nonzero) == 1:
            idx, coef = nonzero[0]
            bound = d / coef
            if coef > 0 and bound < ub_tensor[idx].item():
                ub_tensor[idx] = bound
            elif coef < 0 and bound > lb_tensor[idx].item():
                lb_tensor[idx] = bound
    if input_shape is not None:
        lb_tensor = lb_tensor.view(*input_shape)
        ub_tensor = ub_tensor.view(*input_shape)
    return InputSpec(kind=InKind.BOX, lb=lb_tensor, ub=ub_tensor)


def _build_output_spec(
    output_ineqs: _Query,
    num_outputs: int,
    true_label,
) -> OutputSpec:
    if not output_ineqs:
        if true_label is not None:
            y_true = _coerce_label_to_tensor(true_label)
            return OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=y_true)
        return OutputSpec(
            kind=OutKind.RANGE,
            lb=torch.tensor([float('-inf')] * num_outputs),
            ub=torch.tensor([float('inf')] * num_outputs),
        )
    rows_c = [list(yc) for _xc, yc, _d in output_ineqs]
    rows_d = [float(d) for _xc, _yc, d in output_ineqs]
    return OutputSpec(
        kind=OutKind.UNSAFE_LINEAR,
        c=torch.tensor(rows_c),
        d=torch.tensor(rows_d),
    )


def _try_promote_to_top1(
    queries: List[Tuple[InputSpec, OutputSpec]],
    num_outputs: int,
    true_label,
) -> Optional[Tuple[InputSpec, OutputSpec]]:
    """Collapse N UNSAFE_LINEAR queries into a single TOP1_ROBUST when possible.

    Structural requirements:
      - All queries share identical input BOX bounds.
      - Every query has exactly ONE output inequality with coefficients +1 / -1,
        RHS = 0, involving ``Y_true`` and one ``Y_other``.
      - The set of ``other`` indices covers every class except ``true_label``.

    Orientation (CRITICAL): only ``Y_true - Y_other <= 0`` (= ``Y_other >= Y_true``,
    ``pos==t_idx``) is the standard top-1 violation and may collapse to TOP1_ROBUST
    (Y_true MAXIMAL); CIFAR/standard classification uses this. The opposite
    ``Y_other - Y_true <= 0`` (``neg==t_idx``) means "Y_true MINIMAL" -- collapsing it
    would flip non-classification specs, so it is left as N UNSAFE_LINEAR disjuncts
    (which encode the OR-of-``<=`` exactly).
    """
    if not queries:
        return None
    t_idx = (int(true_label.item()) if isinstance(true_label, torch.Tensor)
             else int(true_label))
    first_in = queries[0][0]
    if first_in.lb is None or first_in.ub is None:
        return None
    for in_spec, _ in queries[1:]:
        if in_spec.lb is None or in_spec.ub is None:
            return None
        if not torch.equal(first_in.lb, in_spec.lb) or not torch.equal(first_in.ub, in_spec.ub):
            return None
    expected = {j for j in range(num_outputs) if j != t_idx}
    covered = set()
    for _, out_spec in queries:
        if out_spec.kind != OutKind.UNSAFE_LINEAR:
            return None
        c_mat = out_spec.c
        d_vec = out_spec.d
        if c_mat is None or d_vec is None:
            return None
        if c_mat.dim() == 1:
            c_mat = c_mat.unsqueeze(0)
        if c_mat.shape[0] != 1 or d_vec.reshape(-1).shape[0] != 1:
            return None
        row = c_mat[0].tolist()
        d_val = float(d_vec.reshape(-1)[0].item())
        if abs(d_val) > 1e-6:
            return None
        nz = [(i, v) for i, v in enumerate(row) if abs(v) > 1e-9]
        if len(nz) != 2:
            return None
        pos = [i for i, v in nz if v > 0]
        neg = [i for i, v in nz if v < 0]
        if len(pos) != 1 or len(neg) != 1:
            return None
        val_pos = [v for _, v in nz if v > 0][0]
        val_neg = [v for _, v in nz if v < 0][0]
        if abs(val_pos - 1.0) > 1e-6 or abs(val_neg + 1.0) > 1e-6:
            return None
        if pos[0] == t_idx:
            covered.add(neg[0])
        else:
            return None   # neg==t_idx (Y_true MINIMAL) flips TOP1_ROBUST -- see docstring
    if covered != expected:
        return None
    y_true = _coerce_label_to_tensor(true_label)
    return queries[0][0], OutputSpec(kind=OutKind.TOP1_ROBUST, y_true=y_true)


def _try_promote_to_top1_unlabeled(
    queries: List[Tuple[InputSpec, OutputSpec]],
    num_outputs: int,
) -> Optional[Tuple[InputSpec, OutputSpec]]:
    """Label-agnostic TOP1 recognition for 2.0 files that omit the label comment.

    Soundness invariant: the shared class ``t`` must be one of the two nonzero
    indices of the first canonicalised row ``e_t - e_j``, so only those two are
    trialled; the full structural + full-coverage check is delegated to
    :func:`_try_promote_to_top1`, which rejects any non-top-1 OR (mixed labels,
    extra conjuncts, nonzero RHS) — so no unsound collapse is possible.
    """
    if len(queries) < 2:
        return None
    first_out = queries[0][1]
    if first_out.kind != OutKind.UNSAFE_LINEAR or first_out.c is None:
        return None
    c0 = first_out.c
    if c0.dim() == 1:
        c0 = c0.unsqueeze(0)
    if c0.shape[0] != 1:
        return None
    candidates = [i for i, v in enumerate(c0[0].tolist()) if abs(v) > 1e-9]
    if len(candidates) != 2:
        return None
    for cand in candidates:
        promoted = _try_promote_to_top1(queries, num_outputs, cand)
        if promoted is not None:
            return promoted
    return None


def extract_vnnlib_2_io_decls(vnnlib_path):
    """``((in_name, in_dtype, in_shape), (out_name, out_dtype, out_shape))`` for a
    single-network VNNLIB 2.0 file, else None (legacy 1.0, or multi-network): the
    caller then has no single-tensor witness header and degrades ``sat`` to
    ``unknown``."""
    try:
        with open(vnnlib_path, "r") as f:
            content = f.read()
    except OSError:
        return None
    if "(vnnlib-version" not in content and "(declare-network" not in content:
        return None
    if len(re.findall(r"\(\s*declare-network\b", content)) >= 2:
        return None
    try:
        return (_extract_vnnlib_2_decl(content, "input"),
                _extract_vnnlib_2_decl(content, "output"))
    except VNNLibParseError:
        return None


def write_vnncomp_result(out_path, token: str, *, x=None, y=None,
                         in_decl=None, out_decl=None) -> None:
    """Emit the VNN-COMP result token; for ``sat`` append the counterexample as a
    VNNLIB 2.0 command-line assignment (VNNLIB-Standard §5.3): per variable a
    header ``<name> <dtype> [d0,d1,..]`` then its values one-per-line in
    row-major (C) order, input decl then output decl. This is what the 2026
    checker (``counterexamples_v2.py::parse_text_assignment``) parses — not the
    legacy flat ``((X_0 v)...)`` pairs. ``in_decl``/``out_decl`` are
    ``(name, dtype, shape)`` triples; if either is None the token is written
    witness-less so the caller can degrade to ``unknown``."""
    def _assignment(name, dtype, shape, values):
        dims = ",".join(str(int(d)) for d in shape)
        return [f"{name} {dtype} [{dims}]"] + [f"{v:.16g}" for v in values]

    with open(out_path, "w") as f:
        if (token != "sat" or x is None or y is None
                or in_decl is None or out_decl is None):
            f.write(token + "\n")
            return
        xf = x.detach().cpu().flatten().tolist()
        yf = y.detach().cpu().flatten().tolist()
        lines = ["sat"]
        lines += _assignment(in_decl[0], in_decl[1], in_decl[2], xf)
        lines += _assignment(out_decl[0], out_decl[1], out_decl[2], yf)
        f.write("\n".join(lines) + "\n")
