"""
Comprehensive test suite for NumPyGrad Training Infrastructure:
Dataset, DataLoader, Module train/eval modes, and no_grad context manager.
"""

import numpy as np
import pytest
from numpygrad.core.tensor import (
    Tensor,
    no_grad,
    enable_grad,
    is_grad_enabled,
    set_grad_enabled,
)
from numpygrad.data.dataset import Dataset, TensorDataset
from numpygrad.data.dataloader import DataLoader
import numpygrad.nn as nn
import numpygrad.optim as optim


# =============================================================================
# 1. Dataset & TensorDataset Unit Tests
# =============================================================================

def test_dataset_direct_instantiation_features_and_labels():
    X = np.random.randn(50, 4).astype(np.float32)
    y = np.random.randint(0, 2, size=(50,)).astype(np.int64)

    dataset = Dataset(features=X, labels=y)
    assert len(dataset) == 50

    # Test single sample indexing
    sample_x, sample_y = dataset[0]
    assert np.allclose(sample_x, X[0])
    assert sample_y == y[0]

    # Test negative indexing
    sample_last_x, sample_last_y = dataset[-1]
    assert np.allclose(sample_last_x, X[-1])
    assert sample_last_y == y[-1]

    # Test slicing
    slice_x, slice_y = dataset[0:5]
    assert slice_x.shape == (5, 4)
    assert slice_y.shape == (5,)


def test_dataset_tensor_instantiation():
    X_t = Tensor(np.random.randn(20, 3).astype(np.float32), requires_grad=False)
    y_t = Tensor(np.random.randint(0, 3, size=(20, 1)).astype(np.float32), requires_grad=False)

    dataset = Dataset(features=X_t, labels=y_t)
    assert len(dataset) == 20

    item_x, item_y = dataset[5]
    assert isinstance(item_x, Tensor)
    assert isinstance(item_y, Tensor)
    assert item_x.shape == (3,)
    assert item_y.shape == (1,)
    assert np.allclose(item_x.data, X_t.data[5])


def test_dataset_features_only():
    X = np.random.randn(30, 8).astype(np.float32)
    dataset = Dataset(features=X)
    assert len(dataset) == 30

    item = dataset[10]
    assert np.allclose(item, X[10])


def test_dataset_length_mismatch_raises_error():
    X = np.random.randn(20, 4)
    y = np.random.randn(25, 1)

    with pytest.raises(ValueError, match="length mismatch"):
        Dataset(features=X, labels=y)


def test_dataset_out_of_bounds_index():
    X = np.random.randn(10, 2)
    dataset = Dataset(features=X)

    with pytest.raises(IndexError, match="out of bounds"):
        _ = dataset[10]

    with pytest.raises(IndexError, match="out of bounds"):
        _ = dataset[-11]


def test_dataset_custom_subclass():
    class CustomSquaresDataset(Dataset):
        def __init__(self, size: int):
            super().__init__()
            self.size = size

        def __len__(self) -> int:
            return self.size

        def __getitem__(self, idx: int):
            if idx < 0 or idx >= self.size:
                raise IndexError("Out of bounds")
            return idx, idx ** 2

    ds = CustomSquaresDataset(10)
    assert len(ds) == 10
    assert ds[3] == (3, 9)
    assert ds[7] == (7, 49)


def test_dataset_unimplemented_base_methods():
    ds = Dataset()
    with pytest.raises(NotImplementedError):
        len(ds)
    with pytest.raises(NotImplementedError):
        _ = ds[0]


def test_tensor_dataset_multi_tensors():
    t1 = Tensor(np.random.randn(40, 5).astype(np.float32))
    t2 = Tensor(np.random.randn(40, 2).astype(np.float32))
    t3 = np.random.randint(0, 10, size=(40,))

    tds = TensorDataset(t1, t2, t3)
    assert len(tds) == 40

    out1, out2, out3 = tds[0]
    assert isinstance(out1, Tensor)
    assert isinstance(out2, Tensor)
    assert np.allclose(out1.data, t1.data[0])
    assert np.allclose(out2.data, t2.data[0])
    assert out3 == t3[0]


def test_tensor_dataset_validation_errors():
    with pytest.raises(ValueError, match="at least one tensor"):
        TensorDataset()

    t1 = np.ones((10, 2))
    t2 = np.ones((12, 2))
    with pytest.raises(ValueError, match="Size mismatch"):
        TensorDataset(t1, t2)


# =============================================================================
# 2. DataLoader Unit Tests
# =============================================================================

def test_dataloader_invalid_batch_size():
    ds = Dataset(features=np.zeros((10, 2)))
    with pytest.raises(ValueError, match="positive integer"):
        DataLoader(ds, batch_size=0)
    with pytest.raises(ValueError, match="positive integer"):
        DataLoader(ds, batch_size=-5)


def test_dataloader_exact_batching():
    X = np.arange(100).reshape(100, 1).astype(np.float32)
    y = np.arange(100).astype(np.int64)
    ds = Dataset(X, y)
    loader = DataLoader(ds, batch_size=20, shuffle=False)

    assert len(loader) == 5
    batches = list(loader)
    assert len(batches) == 5

    for i, (bx, by) in enumerate(batches):
        assert isinstance(bx, Tensor)
        assert isinstance(by, Tensor)
        assert bx.shape == (20, 1)
        assert by.shape == (20,)
        np.testing.assert_allclose(bx.data, X[i * 20 : (i + 1) * 20])
        np.testing.assert_allclose(by.data, y[i * 20 : (i + 1) * 20])


def test_dataloader_remainder_drop_last_false():
    X = np.arange(50).reshape(50, 1).astype(np.float32)
    ds = Dataset(X)
    loader = DataLoader(ds, batch_size=16, shuffle=False, drop_last=False)

    # 50 samples with batch_size 16 -> 4 batches (16, 16, 16, 2)
    assert len(loader) == 4
    batches = list(loader)
    assert len(batches) == 4
    assert batches[0].shape == (16, 1)
    assert batches[1].shape == (16, 1)
    assert batches[2].shape == (16, 1)
    assert batches[3].shape == (2, 1)


def test_dataloader_remainder_drop_last_true():
    X = np.arange(50).reshape(50, 1).astype(np.float32)
    ds = Dataset(X)
    loader = DataLoader(ds, batch_size=16, shuffle=False, drop_last=True)

    # 50 samples with batch_size 16 -> 3 batches (16, 16, 16), dropping remainder 2
    assert len(loader) == 3
    batches = list(loader)
    assert len(batches) == 3
    for b in batches:
        assert b.shape == (16, 1)


def test_dataloader_deterministic_shuffling_with_seed():
    X = np.arange(60).reshape(60, 1).astype(np.float32)
    ds = Dataset(X)

    loader1 = DataLoader(ds, batch_size=10, shuffle=True, seed=42)
    loader2 = DataLoader(ds, batch_size=10, shuffle=True, seed=42)

    batches1 = [b.data for b in loader1]
    batches2 = [b.data for b in loader2]

    # Same seed must yield identical batch order
    for b1, b2 in zip(batches1, batches2):
        np.testing.assert_array_equal(b1, b2)

    # Different seed should yield different order
    loader3 = DataLoader(ds, batch_size=10, shuffle=True, seed=999)
    batches3 = [b.data for b in loader3]
    all_equal = all(np.array_equal(b1, b3) for b1, b3 in zip(batches1, batches3))
    assert not all_equal


def test_dataloader_custom_collate_fn():
    X = np.arange(20)
    ds = Dataset(X)

    def custom_collate(samples):
        # Custom dict collation
        return {"data": np.array(samples), "count": len(samples)}

    loader = DataLoader(ds, batch_size=5, collate_fn=custom_collate)
    for batch in loader:
        assert isinstance(batch, dict)
        assert "data" in batch
        assert batch["count"] == 5


def test_dataloader_empty_dataset():
    class EmptyDataset(Dataset):
        def __len__(self):
            return 0
        def __getitem__(self, idx):
            raise IndexError

    loader = DataLoader(EmptyDataset(), batch_size=4)
    assert len(loader) == 0
    assert list(loader) == []


# =============================================================================
# 3. Module Execution Modes (train / eval) Unit Tests
# =============================================================================

def test_module_recursive_mode_propagation():
    class SubBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.drop = nn.Dropout(p=0.3)
            self.bn = nn.BatchNorm1d(8)

        def forward(self, x):
            return self.bn(self.drop(x))

    class NestedNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.block1 = SubBlock()
            self.seq = nn.Sequential(
                nn.Linear(8, 8),
                SubBlock(),
            )

        def forward(self, x):
            return self.seq(self.block1(x))

    net = NestedNetwork()

    # Initial state must be training
    assert net.training is True
    assert net.block1.training is True
    assert net.block1.drop.training is True
    assert net.block1.bn.training is True
    assert net.seq.training is True
    assert net.seq[1].training is True
    assert net.seq[1].drop.training is True
    assert net.seq[1].bn.training is True

    # Switch to eval mode
    net.eval()
    assert net.training is False
    assert net.block1.training is False
    assert net.block1.drop.training is False
    assert net.block1.bn.training is False
    assert net.seq.training is False
    assert net.seq[1].training is False
    assert net.seq[1].drop.training is False
    assert net.seq[1].bn.training is False

    # Switch back to train mode
    net.train()
    assert net.training is True
    assert net.block1.training is True
    assert net.block1.drop.training is True
    assert net.block1.bn.training is True
    assert net.seq[1].drop.training is True


def test_dropout_mode_mechanics():
    np.random.seed(123)
    drop = nn.Dropout(p=0.5)

    x = Tensor(np.ones((200, 200), dtype=np.float32))

    # In eval mode: exact identity
    drop.eval()
    out_eval = drop(x)
    np.testing.assert_array_equal(out_eval.data, x.data)

    # In train mode: inverted dropout
    drop.train()
    out_train = drop(x)
    zero_ratio = np.mean(out_train.data == 0.0)
    assert 0.4 < zero_ratio < 0.6
    non_zeros = out_train.data[out_train.data != 0.0]
    np.testing.assert_allclose(non_zeros, 2.0)  # 1 / (1 - 0.5) = 2.0


def test_batchnorm1d_mode_mechanics():
    bn = nn.BatchNorm1d(4, momentum=0.1)

    initial_mean = bn.running_mean.copy()
    initial_var = bn.running_var.copy()

    # Training mode: running stats MUST update
    bn.train()
    x_train = Tensor(np.random.randn(30, 4).astype(np.float32) * 5.0 + 10.0)
    _ = bn(x_train)

    assert not np.array_equal(bn.running_mean, initial_mean)
    assert not np.array_equal(bn.running_var, initial_var)

    # Freeze running stats
    frozen_mean = bn.running_mean.copy()
    frozen_var = bn.running_var.copy()

    # Evaluation mode: running stats MUST NOT update
    bn.eval()
    x_eval = Tensor(np.random.randn(15, 4).astype(np.float32) * 2.0 - 3.0)
    _ = bn(x_eval)

    np.testing.assert_array_equal(bn.running_mean, frozen_mean)
    np.testing.assert_array_equal(bn.running_var, frozen_var)


# =============================================================================
# 4. Inference Context Manager (no_grad & enable_grad) Unit Tests
# =============================================================================

def test_is_grad_enabled_and_set_grad_enabled():
    assert is_grad_enabled() is True
    set_grad_enabled(False)
    assert is_grad_enabled() is False
    set_grad_enabled(True)
    assert is_grad_enabled() is True


def test_no_grad_context_manager_disables_dag_and_tracking():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)

    with no_grad():
        c = a * b + 10.0
        d = c.relu()
        loss = d.sum()

        assert is_grad_enabled() is False
        assert c.requires_grad is False
        assert d.requires_grad is False
        assert loss.requires_grad is False
        assert len(c._prev) == 0
        assert len(d._prev) == 0
        assert len(loss._prev) == 0

    # Exiting context manager restores gradient tracking
    assert is_grad_enabled() is True
    e = a * b
    assert e.requires_grad is True
    assert len(e._prev) == 2


def test_no_grad_function_decorator():
    @no_grad()
    def compute_inference(x_t: Tensor) -> Tensor:
        assert is_grad_enabled() is False
        return (x_t ** 2.0).sum()

    x = Tensor([3.0, 4.0], requires_grad=True)
    out = compute_inference(x)

    assert is_grad_enabled() is True
    assert out.requires_grad is False
    assert len(out._prev) == 0
    np.testing.assert_allclose(out.data, 25.0)


def test_nested_no_grad_and_enable_grad():
    x = Tensor([2.0], requires_grad=True)

    assert is_grad_enabled() is True
    with no_grad():
        assert is_grad_enabled() is False
        y1 = x * 2.0
        assert y1.requires_grad is False

        with enable_grad():
            assert is_grad_enabled() is True
            y2 = x * 3.0
            assert y2.requires_grad is True
            assert len(y2._prev) > 0

        assert is_grad_enabled() is False
        y3 = x * 4.0
        assert y3.requires_grad is False

    assert is_grad_enabled() is True


def test_no_grad_restores_on_exception():
    assert is_grad_enabled() is True
    try:
        with no_grad():
            assert is_grad_enabled() is False
            raise RuntimeError("Test error inside no_grad")
    except RuntimeError:
        pass

    # Must be safely restored back to True
    assert is_grad_enabled() is True


# =============================================================================
# 5. Canonical End-to-End Training & Evaluation Loop Integration Test
# =============================================================================

def test_canonical_training_and_evaluation_pipeline():
    np.random.seed(42)

    # 1. Generate synthetic binary classification dataset
    num_samples = 120
    X_data = np.random.randn(num_samples, 6).astype(np.float32)
    # Simple non-linear target
    y_data = ((X_data[:, 0] * X_data[:, 1] + X_data[:, 2]) > 0.0).astype(np.int64)

    # Train / Val split
    train_ds = Dataset(features=X_data[:90], labels=y_data[:90])
    val_ds = Dataset(features=X_data[90:], labels=y_data[90:])

    train_loader = DataLoader(train_ds, batch_size=18, shuffle=True, seed=123)
    val_loader = DataLoader(val_ds, batch_size=15, shuffle=False)

    # 2. Build model architecture
    model = nn.Sequential(
        nn.Linear(6, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(p=0.1),
        nn.Linear(16, 2),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.03, weight_decay=1e-4)

    initial_loss = None
    final_loss = None

    # 3. Training Loop
    model.train()
    for epoch in range(15):
        epoch_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.data)

        avg_epoch_loss = epoch_loss / len(train_loader)
        if epoch == 0:
            initial_loss = avg_epoch_loss
        final_loss = avg_epoch_loss

    assert final_loss < initial_loss

    # 4. Evaluation Loop under eval mode & no_grad()
    model.eval()
    val_loss_total = 0.0
    correct_predictions = 0
    total_val_samples = len(val_ds)

    with no_grad():
        for val_x, val_y in val_loader:
            # Under no_grad, model outputs have requires_grad=False
            val_logits = model(val_x)
            assert val_logits.requires_grad is False
            assert len(val_logits._prev) == 0

            loss = criterion(val_logits, val_y)
            assert loss.requires_grad is False
            val_loss_total += float(loss.data)

            preds = np.argmax(val_logits.data, axis=-1)
            correct_predictions += int(np.sum(preds == val_y.data))

    avg_val_loss = val_loss_total / len(val_loader)
    val_accuracy = correct_predictions / total_val_samples

    assert not np.isnan(avg_val_loss)
    assert 0.0 <= val_accuracy <= 1.0
