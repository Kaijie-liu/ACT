"""Identity records for requested and numerically represented input sets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib

import torch


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class RepresentedSetIdentity:
    requested_radius: float
    representation_dtype: str
    shape: tuple[int, ...]
    center_sha256: str
    lower_sha256: str
    upper_sha256: str
    lower_delta_sha256: str
    upper_delta_sha256: str
    box_width_sha256: str
    effective_lower_linf: float
    effective_upper_linf: float
    minimum_box_width: float
    maximum_box_width: float
    zero_box_width_coordinates: int
    unchanged_lower_coordinates: int
    unchanged_upper_coordinates: int
    coordinate_count: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def represented_linf_box(
    center: torch.Tensor,
    requested_radius: float,
    *,
    domain_lower: float = 0.0,
    domain_upper: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, RepresentedSetIdentity]:
    """Materialize the exact box passed by a floating-point frontend.

    This function records the represented tensor set; it does not assert that
    the set contains the requested real-valued L-infinity ball.  In particular,
    a positive radius may round to a zero-width box at some or all coordinates.
    """

    if not center.is_floating_point():
        raise TypeError("represented input center must have a floating dtype")
    radius = float(requested_radius)
    if not torch.isfinite(torch.tensor(radius, dtype=torch.float64)) or radius < 0:
        raise ValueError("requested_radius must be finite and non-negative")
    if not domain_lower <= domain_upper:
        raise ValueError("domain_lower must not exceed domain_upper")
    radius_tensor = torch.as_tensor(radius, dtype=center.dtype, device=center.device)
    lower = torch.clamp(center - radius_tensor, domain_lower, domain_upper)
    upper = torch.clamp(center + radius_tensor, domain_lower, domain_upper)
    lower_delta = center - lower
    upper_delta = upper - center
    width = upper - lower
    coordinate_count = int(center.numel())
    identity = RepresentedSetIdentity(
        requested_radius=radius,
        representation_dtype=str(center.dtype),
        shape=tuple(int(value) for value in center.shape),
        center_sha256=_tensor_sha256(center),
        lower_sha256=_tensor_sha256(lower),
        upper_sha256=_tensor_sha256(upper),
        lower_delta_sha256=_tensor_sha256(lower_delta),
        upper_delta_sha256=_tensor_sha256(upper_delta),
        box_width_sha256=_tensor_sha256(width),
        effective_lower_linf=float(lower_delta.abs().max().item()),
        effective_upper_linf=float(upper_delta.abs().max().item()),
        minimum_box_width=float(width.min().item()),
        maximum_box_width=float(width.max().item()),
        zero_box_width_coordinates=int(torch.count_nonzero(width == 0).item()),
        unchanged_lower_coordinates=int(torch.count_nonzero(lower == center).item()),
        unchanged_upper_coordinates=int(torch.count_nonzero(upper == center).item()),
        coordinate_count=coordinate_count,
    )
    return lower, upper, identity
