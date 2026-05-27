#===- act/front_end/vnnlib/onnx_converter.py - ONNX to PyTorch -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Convert ONNX models to PyTorch nn.Module for unified verification interface.
#   Supports model validation and shape inference.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ONNXConversionError(Exception):
    """Exception raised when ONNX conversion fails."""
    pass


def _preprocess_onnx_for_onnx2torch(onnx_model):
    """Workarounds for onnx2torch quirks. Called on both main and retry paths.

    1. Symbolic batch dim (vit_2023): set first ``dim_value=0`` → 1. Must
       ``ClearField('dim_param')`` first since ``dim_value`` / ``dim_param``
       are a protobuf oneof. Only normalise the first dim; leave variable
       spatial dims alone so we don't mask real shape-concreteness errors.
    2. Empty Clip max input (cctsdb_yolo_2023): trim trailing empty input
       slots so onnx2torch's clip.py doesn't see them as present.
    """
    for inp in onnx_model.graph.input:
        dims = list(inp.type.tensor_type.shape.dim)
        if dims and dims[0].dim_value == 0:
            dims[0].ClearField('dim_param')
            dims[0].dim_value = 1
    for node in onnx_model.graph.node:
        if node.op_type == 'Clip':
            while len(node.input) > 1 and not node.input[-1]:
                del node.input[-1]
    return onnx_model


def convert_onnx_to_pytorch(
    onnx_path: Path,
    simplify: bool = True
) -> nn.Module:
    """
    Convert ONNX model to PyTorch nn.Module.
    
    Args:
        onnx_path: Path to .onnx file
        simplify: Whether to simplify ONNX model before conversion
        
    Returns:
        PyTorch nn.Module equivalent to ONNX model
        
    Raises:
        ONNXConversionError: If conversion fails
    """
    if not onnx_path.exists():
        # The runner may have substituted a sibling .onnx.gz when the .onnx
        # is a broken symlink (nn4sys mscn_2048d_*). Try the .gz sibling
        # before giving up.
        gz_sibling = onnx_path.parent / f"{onnx_path.name}.gz"
        if gz_sibling.exists():
            onnx_path = gz_sibling
        else:
            raise ONNXConversionError(f"ONNX file not found: {onnx_path}")

    try:
        # Import here to avoid requiring onnx for non-VNNLIB workflows
        import onnx
        from onnx2torch import convert

        # Auto-decompress .onnx.gz to a tempfile before parsing. onnx2torch
        # cannot read gzipped protobuf directly, so we materialise the raw
        # .onnx into a NamedTemporaryFile and re-target the conversion path.
        if str(onnx_path).endswith(".onnx.gz"):
            import gzip, tempfile
            with gzip.open(str(onnx_path), "rb") as f_in:
                tmp = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False)
                try:
                    while True:
                        chunk = f_in.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        tmp.write(chunk)
                finally:
                    tmp.close()
            logger.info(f"Decompressed {onnx_path.name} → {tmp.name}")
            onnx_path = Path(tmp.name)

        # Load ONNX model
        logger.info(f"Loading ONNX model from {onnx_path}")
        onnx_model = onnx.load(str(onnx_path))
        current_opset = max((op.version for op in onnx_model.opset_import if not op.domain or op.domain == 'ai.onnx'), default=0)

        # nn4sys mscn_*.onnx (opset 10) breaks under opset→13 upgrade + onnxsim
        # (Slice_9 KeyError from onnx2torch). The raw graph converts fine on its
        # own. Try the cheap raw path first; only fall back to the preprocessing
        # ladder when the raw convert fails for a reason that preprocessing has
        # been observed to fix.
        def _convert_raw():
            raw_model = onnx.load(str(onnx_path))
            raw_model = _preprocess_onnx_for_onnx2torch(raw_model)
            try:
                from onnx import shape_inference
                raw_model = shape_inference.infer_shapes(raw_model)
            except Exception as e:
                logger.debug("raw shape inference suppressed: %s", e)
            return convert(raw_model)

        def _convert_with_full_pipeline():
            m = onnx.load(str(onnx_path))
            try:
                from onnx import version_converter
                if 0 < current_opset < 13:
                    logger.info(f"Upgrading ONNX opset {current_opset} → 13")
                    m = version_converter.convert_version(m, 13)
            except Exception as e:
                logger.warning(f"Opset upgrade failed ({e}), proceeding with original opset")
            m = _preprocess_onnx_for_onnx2torch(m)
            if simplify:
                try:
                    import onnxsim
                    logger.info("Simplifying ONNX model")
                    m_sim, check = onnxsim.simplify(m)
                    if check:
                        m = m_sim
                    else:
                        logger.warning("ONNX simplification check failed, using original model")
                except ImportError:
                    logger.warning("onnxsim not available, skipping simplification")
                except Exception as e:
                    logger.warning(f"ONNX simplification failed: {e}, using original model")
            try:
                from onnx import shape_inference
                m = shape_inference.infer_shapes(m)
            except Exception as e:
                logger.warning(f"ONNX shape inference failed ({e}); proceeding without it")
            return convert(m)

        # Order: try the historically-working simplify+upgrade pipeline FIRST
        # (used to clean up shape annotations + dead nodes; ml4acopf 14_ieee_*
        # and other opset-14 models rely on its constant folding). Only fall
        # back to the raw path when the full pipeline fails — that covers
        # nn4sys mscn_*.onnx where opset 10→13 + onnxsim rename a node
        # (Slice_9 KeyError from onnx2torch).
        logger.info("Converting ONNX to PyTorch (simplify-first; raw fallback)")
        try:
            pytorch_model = _convert_with_full_pipeline()
        except Exception as full_err:
            logger.warning(
                f"Full pipeline failed ({type(full_err).__name__}: {str(full_err)[:120]}); "
                f"retrying with raw graph (no opset upgrade, no simplify)"
            )
            try:
                pytorch_model = _convert_raw()
            except Exception as raw_err:
                # Both paths failed — re-raise the simpler raw error which is
                # usually a clearer signal of the actual unsupported op.
                raise raw_err
        pytorch_model.eval()
        
        # Convert model to match device_manager settings
        try:
            from act.util.device_manager import get_default_device, get_default_dtype
            target_device = get_default_device()
            target_dtype = get_default_dtype()
            
            # Move model to target device and dtype
            pytorch_model = pytorch_model.to(dtype=target_dtype, device=target_device)
            logger.info(f"Converted model to device={target_device}, dtype={target_dtype}")
        except Exception as e:
            logger.warning(f"Could not apply device_manager settings: {e}")
        
        logger.info(f"Successfully converted ONNX model: {onnx_path.name}")
        return pytorch_model
        
    except ImportError as e:
        raise ONNXConversionError(
            f"Missing dependency for ONNX conversion: {e}\n"
            "Install with: pip install onnx onnx2torch onnx-simplifier"
        )
    except Exception as e:
        raise ONNXConversionError(f"Failed to convert {onnx_path}: {str(e)}")


def get_onnx_input_shape(onnx_path: Path) -> Tuple[int, ...]:
    """
    Extract input shape from ONNX model.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        Input shape tuple WITH batch=1 (normalized to (1, C, H, W) format)
        
    Raises:
        ONNXConversionError: If shape extraction fails
    """
    try:
        import onnx

        # Honour .gz fallback when the .onnx symlink is broken (nn4sys mscn_2048d_*).
        actual = onnx_path
        if not actual.exists():
            gz_sibling = onnx_path.parent / f"{onnx_path.name}.gz"
            if gz_sibling.exists():
                actual = gz_sibling
        if str(actual).endswith(".onnx.gz"):
            import gzip, io
            with gzip.open(str(actual), "rb") as f:
                onnx_model = onnx.load(io.BytesIO(f.read()))
        else:
            onnx_model = onnx.load(str(actual))
        graph = onnx_model.graph

        if not graph.input:
            raise ONNXConversionError("ONNX model has no inputs")

        # Filter out initializers — older exporters list weights/biases in
        # graph.input alongside the actual model placeholders, so naively
        # taking graph.input[0] can return e.g. conv_1_W instead of the
        # imageinput. collins_rul_cnn_2022 NN_rul_full_window_40 hits this.
        initializer_names = {init.name for init in graph.initializer}
        model_inputs = [t for t in graph.input if t.name not in initializer_names]
        if not model_inputs:
            raise ONNXConversionError(
                "ONNX model has no non-initializer inputs (all graph.input "
                "entries are weights/biases)"
            )
        input_tensor = model_inputs[0]
        shape = _extract_shape_from_tensor(input_tensor)
        
        # Handle batch dimension - keep original, but normalize dynamic batch
        if not shape:
            raise ONNXConversionError("Failed to extract valid shape from ONNX model")
        
        if shape[0] == -1:
            # Dynamic batch: normalize to 1 for verification (requires concrete shape)
            shape = (1,) + tuple(shape[1:])
            logger.info(f"Normalized dynamic batch to 1: {shape}")
        else:
            # Keep original batch dimension (whether 1, 32, etc.)
            logger.info(f"Extracted input shape: {shape}")
            if shape[0] != 1:
                logger.warning(
                    f"ONNX model has batch size {shape[0]}, but verification "
                    f"assumes batch=1. Results may be incorrect."
                )
        
        return tuple(shape)
        
    except ImportError:
        raise ONNXConversionError("onnx library not installed")
    except Exception as e:
        raise ONNXConversionError(f"Failed to extract shape from {onnx_path}: {str(e)}")


def get_onnx_output_shape(onnx_path: Path) -> Tuple[int, ...]:
    """
    Extract output shape from ONNX model.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        Output shape tuple WITH batch=1 (normalized to (1, num_classes) format)
        
    Raises:
        ONNXConversionError: If shape extraction fails
    """
    try:
        import onnx
        
        onnx_model = onnx.load(str(onnx_path))
        graph = onnx_model.graph
        
        if not graph.output:
            raise ONNXConversionError("ONNX model has no outputs")
        
        # Get first output tensor
        output_tensor = graph.output[0]
        shape = _extract_shape_from_tensor(output_tensor)
        
        # Handle batch dimension - keep original, but normalize dynamic batch
        if not shape:
            raise ONNXConversionError("Failed to extract valid shape from ONNX model")
        
        if shape[0] == -1:
            # Dynamic batch: normalize to 1 for verification (requires concrete shape)
            shape = (1,) + tuple(shape[1:])
            logger.info(f"Normalized dynamic batch to 1: {shape}")
        else:
            # Keep original batch dimension
            logger.info(f"Extracted output shape: {shape}")
            if shape[0] != 1:
                logger.warning(
                    f"ONNX model has output batch size {shape[0]}, but verification "
                    f"assumes batch=1. Results may be incorrect."
                )
        
        return tuple(shape)
        
    except ImportError:
        raise ONNXConversionError("onnx library not installed")
    except Exception as e:
        raise ONNXConversionError(f"Failed to extract output shape from {onnx_path}: {str(e)}")


def _extract_shape_from_tensor(tensor) -> list:
    """
    Extract shape from ONNX tensor proto.
    
    Args:
        tensor: ONNX tensor (ValueInfoProto)
        
    Returns:
        List of dimension sizes (-1 for dynamic dimensions)
    """
    shape = []
    
    if hasattr(tensor, 'type') and hasattr(tensor.type, 'tensor_type'):
        tensor_type = tensor.type.tensor_type
        if hasattr(tensor_type, 'shape'):
            for dim in tensor_type.shape.dim:
                if hasattr(dim, 'dim_value'):
                    shape.append(dim.dim_value if dim.dim_value > 0 else -1)
                elif hasattr(dim, 'dim_param'):
                    # Dynamic dimension
                    shape.append(-1)
    
    return shape


def test_onnx_conversion(
    onnx_path: Path,
    input_shape: Optional[Tuple[int, ...]] = None,
    batch_size: int = 1
) -> bool:
    """
    Test ONNX to PyTorch conversion with dummy input.
    
    Args:
        onnx_path: Path to .onnx file
        input_shape: Input shape (inferred from model if not provided)
        batch_size: Batch size for test input
        
    Returns:
        True if conversion successful and model runs, False otherwise
    """
    try:
        # Convert model
        pytorch_model = convert_onnx_to_pytorch(onnx_path)
        
        # Get input shape if not provided
        if input_shape is None:
            input_shape = get_onnx_input_shape(onnx_path)
        
        # Create dummy input
        dummy_input = torch.randn(batch_size, *input_shape)
        
        # Run forward pass
        with torch.no_grad():
            output = pytorch_model(dummy_input)
        
        logger.info(
            f"ONNX conversion test passed: "
            f"input {dummy_input.shape} -> output {output.shape}"
        )
        return True
        
    except Exception as e:
        logger.error(f"ONNX conversion test failed: {e}")
        return False


def get_onnx_metadata(onnx_path: Path) -> dict:
    """
    Extract metadata from ONNX model.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        Dict with model metadata (producer, version, shapes, etc.)
    """
    try:
        import onnx
        
        onnx_model = onnx.load(str(onnx_path))
        
        metadata = {
            'producer_name': onnx_model.producer_name,
            'producer_version': onnx_model.producer_version,
            'ir_version': onnx_model.ir_version,
            'opset_version': None,
            'input_shapes': [],
            'output_shapes': []
        }
        
        # Get opset version
        if onnx_model.opset_import:
            metadata['opset_version'] = onnx_model.opset_import[0].version
        
        # Get input/output shapes
        graph = onnx_model.graph
        
        for inp in graph.input:
            shape = _extract_shape_from_tensor(inp)
            metadata['input_shapes'].append({
                'name': inp.name,
                'shape': shape
            })
        
        for out in graph.output:
            shape = _extract_shape_from_tensor(out)
            metadata['output_shapes'].append({
                'name': out.name,
                'shape': shape
            })
        
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to extract ONNX metadata: {e}")
        return {}


def validate_onnx_file(onnx_path: Path) -> bool:
    """
    Validate that an ONNX file is well-formed.
    
    Args:
        onnx_path: Path to .onnx file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        import onnx
        
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info(f"ONNX model validated: {onnx_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"ONNX validation failed: {e}")
        return False


def convert_and_save_pytorch(
    onnx_path: Path,
    output_path: Optional[Path] = None,
    simplify: bool = True
) -> Path:
    """
    Convert ONNX model to PyTorch and save as .pt file.
    
    Args:
        onnx_path: Path to .onnx file
        output_path: Path for .pt file (defaults to same dir as ONNX)
        simplify: Whether to simplify ONNX before conversion
        
    Returns:
        Path to saved .pt file
        
    Raises:
        ONNXConversionError: If conversion or saving fails
    """
    try:
        # Convert to PyTorch
        pytorch_model = convert_onnx_to_pytorch(onnx_path, simplify=simplify)
        
        # Determine output path
        if output_path is None:
            output_path = onnx_path.with_suffix('.pt')
        
        # Save model
        torch.save(pytorch_model.state_dict(), output_path)
        logger.info(f"Saved PyTorch model to {output_path}")
        
        return output_path
        
    except Exception as e:
        raise ONNXConversionError(f"Failed to convert and save: {str(e)}")
