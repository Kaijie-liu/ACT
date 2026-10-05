"""Five explicit delta tests; execute only through the frozen full v2 gate."""
import hashlib
import pytest
from experiments.neural_hz_20260831.definition_first_20260928.d015_batch_binding_20260928_v2 import source_binding_v2 as v2
from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928.test_source_packet_v1 import _base
from experiments.neural_hz_20260831.definition_first_20260928.d015_source_shielding_20260928 import census_worker_v1 as worker


def test_default_disabled_without_parsing_or_binding():
    assert v2.extract_model(object()) is None
    assert v2.extract_model(b'invalid', enabled='true', input_batch=1) is None
    with pytest.raises(v2.SourcePacketError):
        v2.extract_model(b'invalid', enabled=True)


def test_static_batch_matches_unchanged_parser():
    _, model = _base()
    raw = model.SerializeToString()
    old = v2.base.extract_model(raw, enabled=True)
    new = v2.extract_model(raw, enabled=True, input_batch=1)
    declared = new.pop('original_declared_input_dimensions')
    binding = new.pop('input_batch_binding')
    assert new == old and declared[0] == dict(kind='dim_value', value=1, symbol='')
    assert binding['symbols'] == {} and model.SerializeToString() == raw


def test_symbolic_batch_has_same_local_coefficients_and_retains_original_bytes():
    _, model = _base()
    reference = v2.base.extract_model(model.SerializeToString(), enabled=True)
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'batch_size'
    raw = model.SerializeToString()
    new = v2.extract_model(raw, enabled=True, input_batch=1)
    declared = new.pop('original_declared_input_dimensions')
    binding = new.pop('input_batch_binding')
    assert declared[0] == dict(kind='dim_param', value=0, symbol='batch_size')
    assert binding['symbols'] == {'batch_size': 1}
    assert new.pop('raw_model_sha256') == hashlib.sha256(raw).hexdigest()
    reference.pop('raw_model_sha256')
    assert new == reference and model.SerializeToString() == raw


def test_nonbatch_unknown_and_conflicting_or_implicit_batch_reject():
    for axis, kind, value in ((2, 'dim_param', 'height'), (0, 'dim_value', 2),
                              (0, 'dim_value', 0), (0, 'unset', None)):
        _, model = _base()
        dim = model.graph.input[0].type.tensor_type.shape.dim[axis]
        if kind == 'unset':
            dim.ClearField('dim_value')
        else:
            setattr(dim, kind, value)
        with pytest.raises(v2.SourcePacketError):
            v2.extract_model(model.SerializeToString(), enabled=True, input_batch=1)
    _, model = _base()
    with pytest.raises(v2.SourcePacketError):
        v2.extract_model(model.SerializeToString(), enabled=True, input_batch=True)


def test_bound_shape_still_requires_exact_original_spec_population():
    _, model = _base()
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'N'
    packet = v2.extract_model(model.SerializeToString(), enabled=True, input_batch=1)
    raw = b'(declare-const X_0 Real)(assert (>= X_0 0))(assert (<= X_0 1))'
    with pytest.raises(ValueError):
        worker.input_box(raw, packet['input_shape'])
    claims = ''.join('(declare-const X_%d Real)(assert (>= X_%d 0))(assert (<= X_%d 1))' %
                     (i, i, i) for i in range(48)).encode()
    assert len(worker.input_box(claims, packet['input_shape'])) == 48
