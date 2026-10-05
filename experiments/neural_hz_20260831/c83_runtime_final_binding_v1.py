"""Bind the actual new NativeState terminal to the independently proved C82 image."""
import hashlib
import json
import time
import torch
from experiments.neural_hz_20260831.c74_native_binding_v1 import NativeState
from experiments.neural_hz_20260831.c34_terminal_binding_v1 import suffix, property_signature
from experiments.neural_hz_20260831.c82_native_terminal_v2 import check_input
from experiments.neural_hz_20260831.c70_native_proof_v1 import entries
from experiments.neural_hz_20260831.c5_live_value_contraction_v1 import source_digest


def load(raw, expected_sha):
    if type(raw) is not bytes or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('independent new final affine proof missing or changed')
    proof = json.loads(raw)
    if (proof.get('schema') != 'c82_native_local_final_affine_v2'
            or proof.get('completed') is not True
            or proof.get('all_final_affine_bits_match_original_binary64') is not True
            or proof.get('original_rounding_order_preserved') is not True
            or proof.get('all_predicates_shared_by_identity') is not True
            or proof.get('unrounded_real_affine_exact_claim') is not False):
        raise ValueError('incomplete original binary64 final proof')
    return proof


def final_extra(layer_count):
    if type(layer_count) is not int or layer_count <= 0:
        raise ValueError('complete positive network layer count required')
    return 512 + 1024 + 256 + 64 * layer_count


def verify(state, output, input_hz, out_spec, kwargs, raw, expected_sha, *, pool, enabled=False):
    if not enabled:
        return None
    started = time.monotonic()
    proof = load(raw, expected_sha)
    new, tf = state['lifted'], state['tf']
    if type(new) is not NativeState:
        raise ValueError('actual new local-journal NativeState required')
    pool.charge('c83_final_actual_source_publication', 1024)
    check_input(new, input_hz, kwargs.get('input_shape', ()))
    native = new.validate()
    source = new.source.validate()
    dense, _, signature = suffix(tf._net, state['layer'].id, pool=pool)
    if (signature != proof['original_suffix_signature'] or kwargs.get('batch_size') != 1
            or kwargs.get('n_out') != output.n_out
            or list(kwargs.get('input_shape', ())) != proof['input_shape']
            or kwargs.get('timelimit') != 45.
            or tf._sparse_hz_cache.get(dense.id) is not output
            or dense.id in tf._sparse_affine_expr_cache or state['native_block_calls'] != 1):
        raise ValueError('actual terminal cache, shape, original suffix or budget differs')
    if (new.expected_transfer_sha256 != proof['underlying_native_transfer_sha256']
            or source['identity'] != proof['source_identity']
            or source_digest(new.hz) != proof['post_HZ_sha256']
            or source_digest(output) != proof['final_HZ_sha256']
            or source_digest(input_hz) != proof['input_HZ_sha256']
            or output.frame_id != new.hz.frame_id or not output.exact
            or (output.n_cont, output.n_bin) != (new.hz.n_cont, new.hz.n_bin)):
        raise ValueError('full new final/source/input HZ or latent identity differs')
    for name in ('Ac', 'Ab', 'Auc', 'Aub'):
        if getattr(output, name) is not getattr(new.hz, name):
            raise ValueError('final affine changed actual predicate identity')
    for name in ('b', 'ub'):
        a, b = getattr(output, name), getattr(new.hz, name)
        if (a.shape != b.shape or a.strides != b.strides
                or a.__array_interface__['data'][0] != b.__array_interface__['data'][0]):
            raise ValueError('final affine changed actual RHS storage')
    encoded = out_spec.encode_linear(B=1, n_out=output.n_out,
                                     device=torch.device('cpu'), dtype=torch.float64)
    if property_signature(encoded) != signature['assertion']:
        raise ValueError('actual original property/polarity differs')
    return dict(final_HZ_sha256=proof['final_HZ_sha256'], input_HZ_sha256=proof['input_HZ_sha256'],
        complete_native_binding=native, complete_source_binding=source,
        source_hash_authentication_elapsed_s=time.monotonic() - started,
        additional_complete_HZ_hash_work=entries(new.hz) + entries(output) + entries(input_hz),
        hash_authentication_separate_from_construction=True,
        original_suffix_signature=signature, actual_predicates_shared_by_identity=True,
        reconstructable_native_local_state_preserved=True, base_feasibility_shortcut=False,
        formal_gain=0)
