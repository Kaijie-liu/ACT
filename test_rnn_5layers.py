#!/usr/bin/env python3
"""
Test for pattern-based schema validation: 5-layer LSTM
This verifies that the schema now supports truly unlimited num_layers.
"""

import torch
import torch.nn as nn
from act.back_end.core import Layer, Bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.interval_tf.tf_rnn import tf_lstm
from act.back_end.layer_util import validate_layer

def test_lstm_5_layers():
    """Test 5-layer LSTM (should pass with pattern-based validation)"""
    print("=" * 70)
    print("Testing 5-Layer LSTM with Pattern-Based Validation")
    print("=" * 70)

    # Create PyTorch 5-layer LSTM
    input_size = 4
    hidden_size = 3
    num_layers = 5
    seq_len = 2
    batch_size = 1

    print(f"\nConfiguration:")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")
    print(f"  bidirectional: False")

    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=False,
        batch_first=True
    )

    # Extract all weights for all 5 layers
    params = {}
    for layer_idx in range(num_layers):
        params[f'weight_ih_l{layer_idx}'] = getattr(lstm, f'weight_ih_l{layer_idx}')
        params[f'weight_hh_l{layer_idx}'] = getattr(lstm, f'weight_hh_l{layer_idx}')
        params[f'bias_ih_l{layer_idx}'] = getattr(lstm, f'bias_ih_l{layer_idx}')
        params[f'bias_hh_l{layer_idx}'] = getattr(lstm, f'bias_hh_l{layer_idx}')

    print(f"\nExtracted weights: {len(params)} tensors")
    print(f"  Layers: l0, l1, l2, l3, l4")

    # Create ACT Layer
    layer = Layer(
        id=0,
        kind=LayerKind.LSTM.value,
        params=params,
        meta={
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'bidirectional': False,
            'batch_first': True,
            'proj_size': 0,
            'input_shape': (batch_size, seq_len, input_size),
            'output_shape': (batch_size, seq_len, hidden_size),
        },
        in_vars=list(range(batch_size * seq_len * input_size)),
        out_vars=list(range(1000, 1000 + batch_size * seq_len * hidden_size))
    )

    # Validate schema (this would fail with old enumeration-based schema)
    print("\n[STEP 1] Validating layer schema...")
    try:
        validate_layer(layer)
        print("  ✓ Schema validation PASSED (pattern-based validation works!)")
    except ValueError as e:
        print(f"  ✗ Schema validation FAILED: {e}")
        return False

    # Generate bounds
    print("\n[STEP 2] Computing interval bounds...")
    input_bounds = Bounds(
        lb=torch.zeros(batch_size, seq_len, input_size),
        ub=torch.ones(batch_size, seq_len, input_size)
    )

    try:
        fact = tf_lstm(layer, input_bounds)
        print(f"  ✓ Bounds computation PASSED")
        print(f"    Output shape: {fact.bounds.lb.shape}")
    except Exception as e:
        print(f"  ✗ Bounds computation FAILED: {e}")
        return False

    # Soundness verification
    print("\n[STEP 3] Verifying soundness (100 random samples)...")
    n_samples = 100
    violations = 0

    for i in range(n_samples):
        # Generate random input within bounds
        x = torch.rand(batch_size, seq_len, input_size)

        # PyTorch forward pass
        with torch.no_grad():
            y_pytorch, _ = lstm(x)

        # Check if output is within computed bounds
        lb_flat = fact.bounds.lb.reshape(-1)
        ub_flat = fact.bounds.ub.reshape(-1)
        y_flat = y_pytorch.reshape(-1)

        if not torch.all((y_flat >= lb_flat - 1e-5) & (y_flat <= ub_flat + 1e-5)):
            violations += 1

    if violations == 0:
        print(f"  ✓ Soundness PASSED: {n_samples}/{n_samples} samples within bounds")
        return True
    else:
        print(f"  ✗ Soundness FAILED: {violations}/{n_samples} samples violated bounds")
        return False

def test_lstm_10_layers():
    """Test 10-layer LSTM to verify truly unlimited support"""
    print("\n" + "=" * 70)
    print("Testing 10-Layer LSTM (Stress Test)")
    print("=" * 70)

    num_layers = 10
    input_size = 3
    hidden_size = 2

    print(f"\nConfiguration:")
    print(f"  input_size: {input_size}")
    print(f"  hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers}")

    lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_first=True
    )

    # Extract weights
    params = {}
    for layer_idx in range(num_layers):
        params[f'weight_ih_l{layer_idx}'] = getattr(lstm, f'weight_ih_l{layer_idx}')
        params[f'weight_hh_l{layer_idx}'] = getattr(lstm, f'weight_hh_l{layer_idx}')
        params[f'bias_ih_l{layer_idx}'] = getattr(lstm, f'bias_ih_l{layer_idx}')
        params[f'bias_hh_l{layer_idx}'] = getattr(lstm, f'bias_hh_l{layer_idx}')

    print(f"\nExtracted weights: {len(params)} tensors")
    print(f"  Layers: l0, l1, ..., l9")

    layer = Layer(
        id=0,
        kind=LayerKind.LSTM.value,
        params=params,
        meta={
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'bidirectional': False,
            'batch_first': True,
            'proj_size': 0,
            'input_shape': (1, 2, input_size),
            'output_shape': (1, 2, hidden_size),
        },
        in_vars=list(range(6)),
        out_vars=list(range(1000, 1004))
    )

    print("\n[VALIDATION] Testing schema validation...")
    try:
        validate_layer(layer)
        print("  ✓ 10-layer LSTM schema validation PASSED!")
        print("  → Pattern-based validation supports truly unlimited layers")
        return True
    except ValueError as e:
        print(f"  ✗ Schema validation FAILED: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RNN Pattern-Based Schema Validation Test")
    print("=" * 70)
    print("\nPurpose: Verify schema supports unlimited num_layers")
    print("Previous limit: 4 layers (l0-l3 enumerated)")
    print("New approach: Pattern-based validation (unlimited)")

    results = []

    # Test 5 layers
    results.append(("5-layer LSTM", test_lstm_5_layers()))

    # Test 10 layers
    results.append(("10-layer LSTM", test_lstm_10_layers()))

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
        print("\n✓ Pattern-based validation successfully supports unlimited layers!")
    else:
        print("\n✗ Some tests failed")
        exit(1)
