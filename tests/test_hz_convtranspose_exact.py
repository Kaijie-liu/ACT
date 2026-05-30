import torch
import torch.nn.functional as F

from act.back_end.solver.solver_hz import HZono, _hz_convtranspose2d_native


def test_convtranspose2d_hz_transfer_is_exact_per_generator():
    dtype = torch.float64
    device = torch.device("cpu")
    in_shape = (1, 1, 2, 2)
    weight = torch.tensor([[[[1.0, -2.0], [0.5, 3.0]]]], dtype=dtype)
    bias = torch.tensor([0.25], dtype=dtype)
    c = torch.tensor([[0.2], [-0.3], [0.4], [0.1]], dtype=dtype)
    Gc = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [-1.0, 0.5], [0.25, -0.5]],
        dtype=dtype,
    )
    Gb = torch.tensor([[0.5], [-0.25], [0.0], [1.0]], dtype=dtype)
    hz = HZono(
        c=c,
        Gc=Gc,
        Gb=Gb,
        Ac=torch.zeros((0, 2), dtype=dtype, device=device),
        Ab=torch.zeros((0, 1), dtype=dtype, device=device),
        b=torch.zeros((0, 1), dtype=dtype, device=device),
        eq_mask=None,
    )
    out_shape = tuple(F.conv_transpose2d(c.view(*in_shape), weight, bias).shape)
    out = _hz_convtranspose2d_native(
        hz,
        {
            "weight": weight,
            "b": bias,
            "input_shape": in_shape,
            "output_shape": out_shape,
            "conv_params": {
                "stride": 1,
                "padding": 0,
                "output_padding": 0,
                "dilation": 1,
                "groups": 1,
            },
        },
    )

    expected_c = F.conv_transpose2d(c.view(*in_shape), weight, bias).reshape(-1, 1)
    expected_gc = []
    for j in range(Gc.shape[1]):
        expected_gc.append(
            F.conv_transpose2d(Gc[:, j].view(*in_shape), weight, None).reshape(-1)
        )
    expected_gb = F.conv_transpose2d(Gb[:, 0].view(*in_shape), weight, None).reshape(-1, 1)

    assert torch.allclose(out.c, expected_c)
    assert torch.allclose(out.Gc, torch.stack(expected_gc, dim=1))
    assert torch.allclose(out.Gb, expected_gb)
    assert out.Ac.shape == hz.Ac.shape
    assert out.Ab.shape == hz.Ab.shape
