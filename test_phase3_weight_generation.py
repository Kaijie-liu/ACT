#!/usr/bin/env python3
"""
Phase 3 Test: RNN Weight Generation in NetFactory

Tests the RNN weight generation functionality added to NetFactory.
Verifies correct weight shapes, naming conventions, and initialization.
"""

import sys
import random
import torch
from pathlib import Path

# Add ACT to path
act_path = Path(__file__).parent
sys.path.insert(0, str(act_path))

from act.back_end.net_factory.factory import NetFactory


def test_lstm_single_layer():
    """Test LSTM weight generation: single layer, unidirectional."""
    print("\n[TEST 1] LSTM: Single Layer, Unidirectional")

    factory = NetFactory()
    rng = random.Random(42)

    meta = {
        'input_size': 10,
        'hidden_size': 20,
        'num_layers': 1,
        'bidirectional': False,
        'proj_size': 0,
    }

    params = factory.generate_rnn_params('LSTM', meta, rng)

    # Expected weights: l0 only
    expected_keys = [
        'weight_ih_l0', 'weight_hh_l0',
        'bias_ih_l0', 'bias_hh_l0'
    ]

    # Verify all expected weights present
    for key in expected_keys:
        if key not in params:
            print(f"  ✗ FAILED: Missing weight '{key}'")
            return False

    # Verify no extra weights
    extra_keys = set(params.keys()) - set(expected_keys)
    if extra_keys:
        print(f"  ✗ FAILED: Unexpected weights: {extra_keys}")
        return False

    # Verify shapes
    if params['weight_ih_l0'].shape != (80, 10):  # 4*hidden_size, input_size
        print(f"  ✗ FAILED: weight_ih_l0 shape {params['weight_ih_l0'].shape} != (80, 10)")
        return False

    if params['weight_hh_l0'].shape != (80, 20):  # 4*hidden_size, hidden_size
        print(f"  ✗ FAILED: weight_hh_l0 shape {params['weight_hh_l0'].shape} != (80, 20)")
        return False

    if params['bias_ih_l0'].shape != (80,):
        print(f"  ✗ FAILED: bias_ih_l0 shape {params['bias_ih_l0'].shape} != (80,)")
        return False

    print(f"  ✓ PASSED: Correct weights and shapes")
    print(f"    Generated {len(params)} weight tensors")
    return True


def test_lstm_multi_layer():
    """Test LSTM weight generation: multi-layer."""
    print("\n[TEST 2] LSTM: Multi-Layer (3 layers)")

    factory = NetFactory()
    rng = random.Random(123)

    meta = {
        'input_size': 8,
        'hidden_size': 16,
        'num_layers': 3,
        'bidirectional': False,
        'proj_size': 0,
    }

    params = factory.generate_rnn_params('LSTM', meta, rng)

    # Expected: l0, l1, l2
    expected_count = 3 * 4  # 3 layers * 4 weights per layer

    if len(params) != expected_count:
        print(f"  ✗ FAILED: Expected {expected_count} weights, got {len(params)}")
        return False

    # Verify layer 0 input size
    if params['weight_ih_l0'].shape != (64, 8):  # 4*16, input_size
        print(f"  ✗ FAILED: Layer 0 input size incorrect")
        return False

    # Verify layer 1 input size (should be hidden_size from previous layer)
    if params['weight_ih_l1'].shape != (64, 16):  # 4*16, hidden_size
        print(f"  ✗ FAILED: Layer 1 input size incorrect")
        return False

    # Verify layer 2 input size
    if params['weight_ih_l2'].shape != (64, 16):  # 4*16, hidden_size
        print(f"  ✗ FAILED: Layer 2 input size incorrect")
        return False

    print(f"  ✓ PASSED: Multi-layer dimensions correct")
    print(f"    Generated {len(params)} weight tensors")
    return True


def test_lstm_bidirectional():
    """Test LSTM weight generation: bidirectional."""
    print("\n[TEST 3] LSTM: Bidirectional")

    factory = NetFactory()
    rng = random.Random(456)

    meta = {
        'input_size': 5,
        'hidden_size': 10,
        'num_layers': 1,
        'bidirectional': True,
        'proj_size': 0,
    }

    params = factory.generate_rnn_params('LSTM', meta, rng)

    # Expected: l0 forward + l0 reverse
    expected_keys = [
        'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0',
        'weight_ih_l0_reverse', 'weight_hh_l0_reverse',
        'bias_ih_l0_reverse', 'bias_hh_l0_reverse'
    ]

    if len(params) != len(expected_keys):
        print(f"  ✗ FAILED: Expected {len(expected_keys)} weights, got {len(params)}")
        return False

    # Verify forward direction
    if params['weight_ih_l0'].shape != (40, 5):  # 4*10, 5
        print(f"  ✗ FAILED: Forward weight_ih_l0 shape incorrect")
        return False

    # Verify reverse direction
    if params['weight_ih_l0_reverse'].shape != (40, 5):  # 4*10, 5
        print(f"  ✗ FAILED: Reverse weight_ih_l0_reverse shape incorrect")
        return False

    print(f"  ✓ PASSED: Bidirectional weights correct")
    print(f"    Generated {len(params)} weight tensors")
    return True


def test_lstm_projection():
    """Test LSTM weight generation: with projection."""
    print("\n[TEST 4] LSTM: With Projection")

    factory = NetFactory()
    rng = random.Random(789)

    meta = {
        'input_size': 12,
        'hidden_size': 24,
        'num_layers': 1,
        'bidirectional': False,
        'proj_size': 8,  # Project to 8 dimensions
    }

    params = factory.generate_rnn_params('LSTM', meta, rng)

    # Expected: standard weights + projection weight
    expected_keys = [
        'weight_ih_l0', 'weight_hh_l0', 'bias_ih_l0', 'bias_hh_l0',
        'weight_hr_l0'  # Projection weight
    ]

    if set(params.keys()) != set(expected_keys):
        print(f"  ✗ FAILED: Unexpected weight keys")
        print(f"    Expected: {expected_keys}")
        print(f"    Got: {list(params.keys())}")
        return False

    # Verify projection weight shape
    if params['weight_hr_l0'].shape != (8, 24):  # proj_size, hidden_size
        print(f"  ✗ FAILED: Projection weight shape {params['weight_hr_l0'].shape} != (8, 24)")
        return False

    print(f"  ✓ PASSED: LSTM projection weights correct")
    print(f"    Generated {len(params)} weight tensors")
    return True


def test_lstm_multi_layer_bidirectional():
    """Test LSTM weight generation: multi-layer + bidirectional."""
    print("\n[TEST 5] LSTM: Multi-Layer + Bidirectional")

    factory = NetFactory()
    rng = random.Random(999)

    meta = {
        'input_size': 6,
        'hidden_size': 8,
        'num_layers': 2,
        'bidirectional': True,
        'proj_size': 0,
    }

    params = factory.generate_rnn_params('LSTM', meta, rng)

    # Expected: 2 layers * (forward + reverse) * 4 weights = 16 weights
    expected_count = 16

    if len(params) != expected_count:
        print(f"  ✗ FAILED: Expected {expected_count} weights, got {len(params)}")
        return False

    # Layer 0: input_size = 6
    if params['weight_ih_l0'].shape != (32, 6):  # 4*8, 6
        print(f"  ✗ FAILED: Layer 0 input size incorrect")
        return False

    # Layer 1: input_size = hidden_size * 2 (bidirectional)
    if params['weight_ih_l1'].shape != (32, 16):  # 4*8, 8*2
        print(f"  ✗ FAILED: Layer 1 input size {params['weight_ih_l1'].shape} != (32, 16)")
        print(f"    (should be hidden_size * 2 for bidirectional)")
        return False

    print(f"  ✓ PASSED: Multi-layer bidirectional dimensions correct")
    print(f"    Generated {len(params)} weight tensors")
    return True


def test_gru_weights():
    """Test GRU weight generation."""
    print("\n[TEST 6] GRU: Single Layer")

    factory = NetFactory()
    rng = random.Random(111)

    meta = {
        'input_size': 10,
        'hidden_size': 15,
        'num_layers': 1,
        'bidirectional': False,
    }

    params = factory.generate_rnn_params('GRU', meta, rng)

    # Verify weight shapes (GRU has 3 gates)
    if params['weight_ih_l0'].shape != (45, 10):  # 3*15, 10
        print(f"  ✗ FAILED: GRU weight_ih_l0 shape incorrect")
        return False

    if params['weight_hh_l0'].shape != (45, 15):  # 3*15, 15
        print(f"  ✗ FAILED: GRU weight_hh_l0 shape incorrect")
        return False

    print(f"  ✓ PASSED: GRU weights correct (3 gates)")
    return True


def test_rnn_weights():
    """Test RNN weight generation."""
    print("\n[TEST 7] RNN: Single Layer")

    factory = NetFactory()
    rng = random.Random(222)

    meta = {
        'input_size': 8,
        'hidden_size': 12,
        'num_layers': 1,
        'bidirectional': False,
        'nonlinearity': 'tanh',
    }

    params = factory.generate_rnn_params('RNN', meta, rng)

    # Verify weight shapes (RNN has 1 gate)
    if params['weight_ih_l0'].shape != (12, 8):  # 1*12, 8
        print(f"  ✗ FAILED: RNN weight_ih_l0 shape incorrect")
        return False

    if params['weight_hh_l0'].shape != (12, 12):  # 1*12, 12
        print(f"  ✗ FAILED: RNN weight_hh_l0 shape incorrect")
        return False

    print(f"  ✓ PASSED: RNN weights correct (1 gate)")
    return True


def test_unlimited_layers():
    """Test unlimited layer support (pattern-based schema validation)."""
    print("\n[TEST 8] Unlimited Layers (10 layers)")

    factory = NetFactory()
    rng = random.Random(333)

    meta = {
        'input_size': 4,
        'hidden_size': 5,
        'num_layers': 10,  # Pattern-based schema supports unlimited layers
        'bidirectional': False,
        'proj_size': 0,
    }

    params = factory.generate_rnn_params('LSTM', meta, rng)

    # Expected: 10 layers * 4 weights = 40 weights
    expected_count = 40

    if len(params) != expected_count:
        print(f"  ✗ FAILED: Expected {expected_count} weights, got {len(params)}")
        return False

    # Verify all layers have correct naming
    for layer_idx in range(10):
        for weight_type in ['weight_ih', 'weight_hh', 'bias_ih', 'bias_hh']:
            key = f'{weight_type}_l{layer_idx}'
            if key not in params:
                print(f"  ✗ FAILED: Missing weight '{key}'")
                return False

    print(f"  ✓ PASSED: 10-layer LSTM weights generated correctly")
    print(f"    Pattern-based schema supports unlimited layers!")
    return True


def main():
    """Run all Phase 3 weight generation tests."""
    print("=" * 70)
    print("Phase 3: RNN Weight Generation Tests")
    print("=" * 70)

    tests = [
        ("LSTM Single Layer", test_lstm_single_layer),
        ("LSTM Multi-Layer", test_lstm_multi_layer),
        ("LSTM Bidirectional", test_lstm_bidirectional),
        ("LSTM Projection", test_lstm_projection),
        ("LSTM Multi+Bidir", test_lstm_multi_layer_bidirectional),
        ("GRU Weights", test_gru_weights),
        ("RNN Weights", test_rnn_weights),
        ("Unlimited Layers (10)", test_unlimited_layers),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ Phase 3: RNN weight generation complete!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
