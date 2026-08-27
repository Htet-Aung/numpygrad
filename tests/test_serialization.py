"""
Comprehensive unit tests for NumPyGrad model serialization and deserialization.
"""

import os
import tempfile
import zipfile
import numpy as np
import pytest

from numpygrad.core.tensor import Tensor
import numpygrad.nn as nn
from numpygrad.serialization import save_model, load_model, get_layer_config, build_layer_from_config


def test_linear_roundtrip_serialization():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "linear_model.ng")

        # 1. Create and customize linear layer
        model = nn.Linear(in_features=6, out_features=3, bias=True)
        model.weight.data = np.random.randn(6, 3).astype(np.float32)
        model.bias.data = np.random.randn(3).astype(np.float32)

        # 2. Save via save_model
        save_model(model, filepath)
        assert os.path.exists(filepath)
        assert zipfile.is_zipfile(filepath)

        # 3. Load via load_model
        loaded_model = load_model(filepath)
        assert isinstance(loaded_model, nn.Linear)
        assert loaded_model.in_features == 6
        assert loaded_model.out_features == 3
        assert loaded_model.bias is not None

        # 4. Numerical verification on identical inputs
        X = Tensor(np.random.randn(10, 6).astype(np.float32))
        orig_out = model(X).data
        loaded_out = loaded_model(X).data
        np.testing.assert_allclose(orig_out, loaded_out, atol=1e-6)

        # 5. Exact parameter array match
        np.testing.assert_array_equal(model.weight.data, loaded_model.weight.data)
        np.testing.assert_array_equal(model.bias.data, loaded_model.bias.data)


def test_linear_no_bias_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "linear_nobias.ng")

        model = nn.Linear(in_features=4, out_features=2, bias=False)
        model.save(filepath)

        loaded = load_model(filepath)
        assert isinstance(loaded, nn.Linear)
        assert loaded.bias is None

        X = Tensor(np.random.randn(5, 4).astype(np.float32))
        np.testing.assert_allclose(model(X).data, loaded(X).data, atol=1e-6)


def test_sequential_deep_network_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "deep_mlp.ng")

        # 1. Build complex sequential model
        model = nn.Sequential(
            nn.Linear(8, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 4),
            nn.GELU(),
            nn.Linear(4, 2),
            nn.Sigmoid(),
        )

        # Simulate training step to populate batchnorm running statistics
        model.train()
        x_train = Tensor(np.random.randn(30, 8).astype(np.float32) * 3.0 + 1.0)
        _ = model(x_train)

        # Switch to eval mode before saving
        model.eval()

        # 2. Save model
        model.save(filepath)

        # 3. Load model
        loaded = load_model(filepath)
        assert isinstance(loaded, nn.Sequential)
        assert len(loaded) == 10
        assert loaded.training is False

        # 4. Check layer types
        assert isinstance(loaded[0], nn.Linear)
        assert isinstance(loaded[1], nn.BatchNorm1d)
        assert isinstance(loaded[2], nn.ReLU)
        assert isinstance(loaded[3], nn.Dropout)
        assert loaded[3].p == 0.3
        assert isinstance(loaded[4], nn.Linear)
        assert isinstance(loaded[5], nn.Tanh)
        assert isinstance(loaded[6], nn.Linear)
        assert isinstance(loaded[7], nn.GELU)
        assert isinstance(loaded[8], nn.Linear)
        assert isinstance(loaded[9], nn.Sigmoid)

        # 5. Check BatchNorm running stats preservation
        np.testing.assert_array_equal(model[1].running_mean, loaded[1].running_mean)
        np.testing.assert_array_equal(model[1].running_var, loaded[1].running_var)
        np.testing.assert_array_equal(model[1].weight.data, loaded[1].weight.data)
        np.testing.assert_array_equal(model[1].bias.data, loaded[1].bias.data)

        # 6. Verify exact numerical equivalence on inference test set
        X_test = Tensor(np.random.randn(25, 8).astype(np.float32))
        orig_preds = model(X_test).data
        loaded_preds = loaded(X_test).data
        np.testing.assert_allclose(orig_preds, loaded_preds, atol=1e-6)


def test_custom_composite_module_serialization():
    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(4, 4)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "composite.ng")

        model = Block()
        model.save(filepath)

        # 1. Load with custom_objects dict
        loaded = load_model(filepath, custom_objects={"Block": Block})
        assert isinstance(loaded, Block)
        assert hasattr(loaded, "fc1")
        assert hasattr(loaded, "act")
        assert hasattr(loaded, "fc2")

        X = Tensor(np.random.randn(5, 4).astype(np.float32))
        orig_out = model(X).data
        loaded_out = loaded(X).data
        np.testing.assert_allclose(orig_out, loaded_out, atol=1e-6)



def test_serialization_error_handling_and_corrupted_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Non-existent file
        with pytest.raises(FileNotFoundError):
            load_model(os.path.join(tmpdir, "does_not_exist.ng"))

        # 2. Corrupted file (not a zip)
        corrupted_file = os.path.join(tmpdir, "corrupted.ng")
        with open(corrupted_file, "w") as f:
            f.write("This is not a valid zip archive")

        with pytest.raises(ValueError, match="not a valid zip container"):
            load_model(corrupted_file)

        # 3. Zip missing architecture.json
        missing_arch_file = os.path.join(tmpdir, "missing_arch.ng")
        with zipfile.ZipFile(missing_arch_file, "w") as zf:
            zf.writestr("weights.npz", b"dummy")

        with pytest.raises(ValueError, match="missing 'architecture.json'"):
            load_model(missing_arch_file)

        # 4. Zip missing weights.npz
        missing_weights_file = os.path.join(tmpdir, "missing_weights.ng")
        with zipfile.ZipFile(missing_weights_file, "w") as zf:
            zf.writestr("architecture.json", b'{"architecture": {"type": "ReLU"}}')

        with pytest.raises(ValueError, match="missing 'weights.npz'"):
            load_model(missing_weights_file)

        # 5. Invalid layer type in config
        with pytest.raises(ValueError, match="Unknown or unsupported layer type"):
            build_layer_from_config({"type": "NonExistentLayer", "config": {}})

        # 6. Non-module save error
        with pytest.raises(TypeError, match="expected a Module"):
            save_model("not_a_module", os.path.join(tmpdir, "out.ng"))


def test_load_state_dict_validation():
    model = nn.Linear(4, 2)

    # Shape mismatch
    invalid_state = {
        "weight": np.zeros((5, 2), dtype=np.float32),
        "bias": np.zeros(2, dtype=np.float32),
    }
    with pytest.raises(ValueError, match="Shape mismatch"):
        model.load_state_dict(invalid_state)

    # Missing key with strict=True
    missing_key_state = {
        "weight": np.zeros((4, 2), dtype=np.float32),
    }
    with pytest.raises(KeyError, match="Missing parameter key"):
        model.load_state_dict(missing_key_state, strict=True)
