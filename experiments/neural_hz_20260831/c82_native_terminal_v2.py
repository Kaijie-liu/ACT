"""Original binary64 affine realization and exact local-journal witness extension."""
from fractions import Fraction as F
import math
import numpy as np
import torch

from act.back_end.solver.solver_hz import _HZMILP
from experiments.neural_hz_20260831.c74_native_binding_v1 import NativeState
from experiments.neural_hz_20260831.c34_terminal_binding_v1 import suffix, array, property_signature
from experiments.neural_hz_20260831.c81_binned_inverse_v1 import reconstruct, matrix_row
from experiments.neural_hz_20260831.c26_tagged_transplant_v1 import Plan
from experiments.neural_hz_20260831.c70_native_proof_v1 import entries
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest
from experiments.neural_hz_20260831.c6_support_affine_plan_v1 import digest_arrays
import json


def check_input(native, input_hz, shape):
    if (type(native) is not NativeState or not shape or shape[0] != 1
            or math.prod(shape) != input_hz.n_out or input_hz.frame_id != native.hz.frame_id
            or input_hz.n_cont > native.source.old_n_cont
            or input_hz.n_bin > native.source.old_n_bin):
        raise ValueError('original input shape/global continuous and binary frame differ')


def affine(native, net, producer, output, input_hz, input_shape, out_spec, *, pool, enabled=False):
    """Reproduce each original binary64 multiply/add using rounded Fraction steps."""
    if not enabled:
        return None
    if type(native) is not NativeState:
        raise ValueError('new immutable local-journal NativeState required')
    native.validate()
    pool.charge('c82_actual_source_input_publication', 1024)
    check_input(native, input_hz, input_shape)
    dense, assertion, signature = suffix(net, producer, pool=pool)
    hz = native.hz
    weight = array(dense.params['weight'])
    bias = dense.params.get('bias')
    bias = np.zeros(weight.shape[0]) if bias is None else array(bias).reshape(-1)
    if (weight.shape != (output.n_out, hz.n_out)
            or (output.n_cont, output.n_bin, output.frame_id) != (hz.n_cont, hz.n_bin, hz.frame_id)
            or not output.exact):
        raise ValueError('final affine dimensions or shared latent frame differ')
    for name in ('Ac', 'Ab', 'Auc', 'Aub'):
        if getattr(output, name) is not getattr(hz, name):
            raise ValueError('final affine copied or replaced original predicates')
    for name in ('b', 'ub'):
        a, b = getattr(output, name), getattr(hz, name)
        if (a.shape != b.shape or a.strides != b.strides
                or a.__array_interface__['data'][0] != b.__array_interface__['data'][0]):
            raise ValueError('final affine replaced original RHS storage')
    encoded = out_spec.encode_linear(B=1, n_out=output.n_out,
                                     device=torch.device('cpu'), dtype=torch.float64)
    if property_signature(encoded) != signature['assertion']:
        raise ValueError('original ASSERT property or polarity changed')
    count = np.count_nonzero(weight, axis=0)
    products = int(np.sum(count * (np.diff(hz.Gc.indptr) + np.diff(hz.Gb.indptr))))
    charge = 128 * (products + weight.size + output.Gc.nnz + output.Gb.nnz + output.n_out)
    pool.charge('c82_complete_independent_binary64_affine', int(charge))
    for row in range(output.n_out):
        active = np.flatnonzero(weight[row])
        center = F(0)
        expected = {'Gc': {}, 'Gb': {}}
        for index in active:
            w = F(float(weight[row, index]))
            product = F(float(w * F(float(hz.c[index]))))
            center = F(float(center + product))
            for name in expected:
                cc, cv = matrix_row(getattr(hz, name), int(index))
                dest = expected[name]
                for col, value in zip(cc, cv):
                    k = int(col)
                    product = F(float(w * F(float(value))))
                    dest[k] = F(float(dest.get(k, F(0)) + product))
        center = F(float(center + F(float(bias[row]))))
        if np.float64(float(center)).tobytes() != np.float64(output.c[row]).tobytes():
            raise ValueError('final center differs from original binary64 affine equation')
        for name, expected_row in expected.items():
            cc, cv = matrix_row(getattr(output, name), row)
            actual = {int(k): F(float(v)) for k, v in zip(cc, cv) if v != 0}
            if len(actual) != np.count_nonzero(cv) or actual != {k: v for k, v in expected_row.items() if v}:
                raise ValueError('final generator differs from original binary64 affine equation')
    # Full hashes are source-authentication work, separately disclosed.
    return dict(schema='c82_native_local_final_affine_v2', completed=True,
        all_final_affine_bits_match_original_binary64=True,
        unrounded_real_affine_exact_claim=False, original_rounding_order_preserved=True, all_predicates_shared_by_identity=True,
        original_suffix_signature=signature, input_shape=list(input_shape),
        post_HZ_sha256=source_digest(hz), final_HZ_sha256=source_digest(output),
        input_HZ_sha256=source_digest(input_hz),
        source_identity=native.source.validate()['identity'],
        underlying_native_transfer_sha256=native.expected_transfer_sha256,
        additional_complete_HZ_hash_work=entries(hz) + entries(output) + entries(input_hz),
        exact_affine_products=products, diagnostic_work=pool.used,
        diagnostic_work_parts=dict(pool.parts), terminal_solve_executed=False,
        concrete_witness=False, formal_gain=0)


def recover(native, model, x, input_hz, input_shape, lane, original, *, pool, enabled=False):
    """Restore exact original factors; concrete network/property replay still required."""
    if not enabled:
        return None
    if type(native) is not NativeState or type(model) is not _HZMILP:
        raise ValueError('new bound NativeState and actual native model required')
    native.validate()
    check_input(native, input_hz, input_shape)
    hz = native.hz
    pool.charge('c82_native_witness_maps', 8 * (hz.n_cont + model.n_var) + 512)
    x = np.asarray(x)
    if (x.dtype != np.float64 or x.shape != (model.n_var,) or not np.isfinite(x).all()
            or model.cont_eliminations or model.bin_fixes or lane != 0):
        raise ValueError('native point, lane or forbidden reduction differs')
    for indices, n, limit in ((model.cont_source, model.n_cont, hz.n_cont),
                               (model.bin_source, model.n_bin, hz.n_bin)):
        if (type(indices) is not np.ndarray or indices.dtype != np.dtype(np.int64)
                or indices.shape != (n,) or np.any(indices < 0) or np.any(indices >= limit)
                or np.any(indices[1:] <= indices[:-1])):
            raise ValueError('native retained global coordinate map differs')
    z = x[model.n_cont:]
    if np.any((z != 0.) & (z != 1.)):
        raise ValueError('native binary phases are not integral')
    continuous = np.zeros(hz.n_cont)
    continuous[model.cont_source] = x[:model.n_cont]
    proof = json.loads(native.transfer_proof_bytes)
    # Actual C74 binds plans in the complete proof; small native fixtures bind
    # them in the authenticated construction report. Neither is a solver repair.
    raw_plans = (proof['complete_component_proof']['plans'] if 'complete_component_proof' in proof
                 else native.construction_report['plans'])
    plans = [Plan(**{**p, 'tail': tuple(p['tail'])}) for p in raw_plans]
    full, report = reconstruct(native.source, hz, native.lineage, plans, continuous,
                               pool=pool, enabled=True)
    if any(full[i] != F(float(continuous[i])) for i in range(input_hz.n_cont)):
        raise ValueError('unit/local extension changed original input coordinates')
    pool.charge('c82_original_input_reconstruction',
                64 * (input_hz.Gc.nnz + input_hz.Gb.nnz + input_hz.n_out))
    xi = np.asarray([float(v) for v in full[:input_hz.n_cont]])
    binary = -np.ones(input_hz.n_bin)
    select = model.bin_source < input_hz.n_bin
    binary[model.bin_source[select]] = 2 * z[select] - 1
    value = input_hz.c.copy()
    if input_hz.n_cont:
        value += np.asarray(input_hz.Gc @ xi).reshape(-1)
    if input_hz.n_bin:
        value += np.asarray(input_hz.Gb @ binary).reshape(-1)
    expected = torch.from_numpy(value.reshape(input_shape).copy())[lane].clone()
    actual = original(model, x, input_hz, input_shape, lane)
    if actual is None or not torch.isfinite(actual).all() or not torch.equal(actual, expected):
        raise ValueError('native input recovery disagrees with exact local-journal extension')
    return actual, dict(inverse=report, original_input_latents_unchanged=True,
        native_input_recovery_bitwise_equal=True,
        native_point_coordinate_sha256=digest_arrays(x, model.cont_source, model.bin_source),
        diagnostic_work=pool.used, diagnostic_work_parts=dict(pool.parts),
        concrete_network_validation_still_required=True, formal_gain=0)
