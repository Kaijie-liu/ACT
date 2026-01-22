#!/usr/bin/env python3
"""
Phase 2 Test: End-to-End RNN Network Generation

Tests the complete RNN generation pipeline:
1. Config parsing
2. Layer graph construction (build_rnn_layers)
3. Weight generation (generate_rnn_params)
4. Network serialization

This validates integration between Phase 1 (Config), Phase 2 (Builder), and Phase 3 (Weights).
"""

import sys
import json
from pathlib import Path

# Add ACT to path
act_path = Path(__file__).parent
sys.path.insert(0, str(act_path))

from act.back_end.net_factory.factory import NetFactory
from act.back_end.serialization.serialization import NetSerializer


def test_rnn_generation_basic():
    """Test basic RNN network generation."""
    print("\n[TEST 1] Basic RNN Network Generation")

    # Create minimal RNN config
    config = {
        "common": {
            "num_instances": 1,
            "base_seed": 42,
            "name_prefix": "test_rnn",
            "output_dir": "/tmp/act_test_rnn",
            "write_manifest": False,
            "dtype": "torch.float64"
        },
        "input_spec": {
            "kind": {"weighted": {"BOX": 1.0}},
            "value_range": {"choice": [[0.0, 1.0]]},
            "box_shrink_range": [0.0, 0.0]
        },
        "output_spec": {
            "kind": {"weighted": {"TOP1_ROBUST": 1.0}},
            "margin": {"choice": [0.0]}
        },
        "family_selection": {
            "weighted": {"rnn": 1.0}
        },
        "families": {
            "rnn": {
                "input_shape": {"choice": [[1, 8, 10]]},
                "cell_kind": {"choice": ["LSTM"]},
                "hidden_size": {"choice": [16]},
                "num_layers": {"range": [1, 1]},
                "bidirectional": {"probability": 0.0},
                "proj_size": {"weighted": {0: 1.0}},
                "nonlinearity": {"choice": ["tanh"]},
                "return_sequence": {"const": False},
                "head_mode": {"choice": ["last"]},
                "num_classes": {"choice": [10]}
            }
        }
    }

    # Save config
    config_path = Path("/tmp/act_test_rnn_config.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)

    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    # Generate network
    factory = NetFactory(gen_config_path=str(config_path))
    names = factory.generate()

    if len(names) != 1:
        print(f"  ✗ FAILED: Expected 1 network, got {len(names)}")
        return False

    # Load generated network
    net_path = factory.output_dir / f"{names[0]}.json"
    if not net_path.exists():
        print(f"  ✗ FAILED: Network file not found: {net_path}")
        return False

    with open(net_path) as f:
        net_dict = json.load(f)

    # Validate network structure
    layers = net_dict["act_net"]["layers"]

    # Expected layer sequence: INPUT, INPUT_SPEC, LSTM, SLICE, FLATTEN, DENSE, ASSERT
    expected_kinds = ["INPUT", "INPUT_SPEC", "LSTM", "SLICE", "FLATTEN", "DENSE", "ASSERT"]
    actual_kinds = [layer["kind"] for layer in layers]

    if actual_kinds != expected_kinds:
        print(f"  ✗ FAILED: Unexpected layer sequence")
        print(f"    Expected: {expected_kinds}")
        print(f"    Got: {actual_kinds}")
        return False

    # Validate LSTM layer
    lstm_layer = layers[2]
    if lstm_layer["kind"] != "LSTM":
        print(f"  ✗ FAILED: LSTM layer not found")
        return False

    # Check metadata
    lstm_meta = lstm_layer["meta"]
    if lstm_meta["input_size"] != 10:
        print(f"  ✗ FAILED: LSTM input_size {lstm_meta['input_size']} != 10")
        return False

    if lstm_meta["hidden_size"] != 16:
        print(f"  ✗ FAILED: LSTM hidden_size {lstm_meta['hidden_size']} != 16")
        return False

    # Check weights
    lstm_params = lstm_layer["params"]
    expected_weights = ["weight_ih_l0", "weight_hh_l0", "bias_ih_l0", "bias_hh_l0"]

    for weight_name in expected_weights:
        if weight_name not in lstm_params:
            print(f"  ✗ FAILED: Missing weight '{weight_name}'")
            return False

    print(f"  ✓ PASSED: Generated valid LSTM network")
    print(f"    Layers: {len(layers)}")
    print(f"    LSTM weights: {len(lstm_params)}")
    return True


def test_rnn_generation_complex():
    """Test complex RNN configuration (multi-layer, bidirectional)."""
    print("\n[TEST 2] Complex RNN Network (Multi-layer + Bidirectional)")

    config = {
        "common": {
            "num_instances": 1,
            "base_seed": 123,
            "name_prefix": "test_complex_rnn",
            "output_dir": "/tmp/act_test_rnn",
            "write_manifest": False,
            "dtype": "torch.float64"
        },
        "input_spec": {
            "kind": {"weighted": {"BOX": 1.0}},
            "value_range": {"choice": [[0.0, 1.0]]},
            "box_shrink_range": [0.0, 0.0]
        },
        "output_spec": {
            "kind": {"weighted": {"TOP1_ROBUST": 1.0}},
            "margin": {"choice": [0.0]}
        },
        "family_selection": {
            "weighted": {"rnn": 1.0}
        },
        "families": {
            "rnn": {
                "input_shape": {"choice": [[1, 12, 8]]},
                "cell_kind": {"choice": ["LSTM"]},
                "hidden_size": {"choice": [20]},
                "num_layers": {"range": [2, 2]},  # 2 layers
                "bidirectional": {"probability": 1.0},  # Always bidirectional
                "proj_size": {"weighted": {0: 1.0}},
                "nonlinearity": {"choice": ["tanh"]},
                "return_sequence": {"const": False},
                "head_mode": {"choice": ["last"]},
                "num_classes": {"choice": [5]}
            }
        }
    }

    config_path = Path("/tmp/act_test_complex_rnn_config.yaml")
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    factory = NetFactory(gen_config_path=str(config_path))
    names = factory.generate()

    net_path = factory.output_dir / f"{names[0]}.json"
    with open(net_path) as f:
        net_dict = json.load(f)

    layers = net_dict["act_net"]["layers"]
    lstm_layer = [l for l in layers if l["kind"] == "LSTM"][0]

    # Validate configuration
    lstm_meta = lstm_layer["meta"]
    if lstm_meta["num_layers"] != 2:
        print(f"  ✗ FAILED: num_layers {lstm_meta['num_layers']} != 2")
        return False

    if not lstm_meta["bidirectional"]:
        print(f"  ✗ FAILED: bidirectional should be True")
        return False

    # Check weight count: 2 layers × (forward + reverse) × 4 weights = 16
    lstm_params = lstm_layer["params"]
    expected_count = 16

    if len(lstm_params) != expected_count:
        print(f"  ✗ FAILED: Expected {expected_count} weights, got {len(lstm_params)}")
        return False

    # Verify all layer weights present
    for layer_idx in range(2):
        for direction in ["", "_reverse"]:
            for weight_type in ["weight_ih", "weight_hh", "bias_ih", "bias_hh"]:
                key = f"{weight_type}_l{layer_idx}{direction}"
                if key not in lstm_params:
                    print(f"  ✗ FAILED: Missing weight '{key}'")
                    return False

    print(f"  ✓ PASSED: Generated complex LSTM network")
    print(f"    Config: 2 layers, bidirectional")
    print(f"    Weights: {len(lstm_params)}")
    return True


def test_gru_generation():
    """Test GRU network generation."""
    print("\n[TEST 3] GRU Network Generation")

    config = {
        "common": {
            "num_instances": 1,
            "base_seed": 456,
            "name_prefix": "test_gru",
            "output_dir": "/tmp/act_test_rnn",
            "write_manifest": False,
            "dtype": "torch.float64"
        },
        "input_spec": {
            "kind": {"weighted": {"BOX": 1.0}},
            "value_range": {"choice": [[0.0, 1.0]]},
            "box_shrink_range": [0.0, 0.0]
        },
        "output_spec": {
            "kind": {"weighted": {"MARGIN_ROBUST": 1.0}},
            "margin": {"choice": [0.1]}
        },
        "family_selection": {
            "weighted": {"rnn": 1.0}
        },
        "families": {
            "rnn": {
                "input_shape": {"choice": [[1, 10, 6]]},
                "cell_kind": {"choice": ["GRU"]},
                "hidden_size": {"choice": [12]},
                "num_layers": {"range": [1, 1]},
                "bidirectional": {"probability": 0.0},
                "proj_size": {"weighted": {0: 1.0}},
                "nonlinearity": {"choice": ["tanh"]},
                "return_sequence": {"const": False},
                "head_mode": {"choice": ["last"]},
                "num_classes": {"choice": [3]}
            }
        }
    }

    config_path = Path("/tmp/act_test_gru_config.yaml")
    import yaml
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    factory = NetFactory(gen_config_path=str(config_path))
    names = factory.generate()

    net_path = factory.output_dir / f"{names[0]}.json"
    with open(net_path) as f:
        net_dict = json.load(f)

    layers = net_dict["act_net"]["layers"]
    gru_layer = [l for l in layers if l["kind"] == "GRU"][0]

    # Validate GRU
    if gru_layer["kind"] != "GRU":
        print(f"  ✗ FAILED: Expected GRU layer, got {gru_layer['kind']}")
        return False

    # Check weight shapes (GRU has 3 gates)
    gru_params = gru_layer["params"]
    if "weight_ih_l0" not in gru_params:
        print(f"  ✗ FAILED: Missing GRU weights")
        return False

    print(f"  ✓ PASSED: Generated valid GRU network")
    print(f"    Weights: {len(gru_params)}")
    return True


def main():
    """Run all Phase 2 end-to-end tests."""
    print("=" * 70)
    print("Phase 2: End-to-End RNN Generation Tests")
    print("=" * 70)

    tests = [
        ("Basic LSTM Generation", test_rnn_generation_basic),
        ("Complex LSTM (Multi+Bidir)", test_rnn_generation_complex),
        ("GRU Generation", test_gru_generation),
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
        print("\n✓ Phase 2: RNN layer builder complete!")
        print("✓ Phase 1: Config YAML complete!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
