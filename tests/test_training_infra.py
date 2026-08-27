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


# =============================================================================
# 7. Studio 2D Interactive Inference Pipeline Unit Tests
# =============================================================================

def test_studio_2d_dataset_generator():
    from app.app import generate_dataset
    for dataset_name in ["Two Moons", "Concentric Circles", "Spirals"]:
        X, y = generate_dataset(dataset_name, n_samples=200, noise=0.1, random_state=42)
        assert X.shape == (200, 2)
        assert y.shape == (200,)
        assert X.dtype == np.float32
        assert np.isin(y, [0, 1]).all()


def test_studio_build_model_and_summary():
    from app.app import build_model, get_architecture_summary
    model1 = build_model(num_layers=1, hidden_dim=16, activation_name="ReLU")
    assert get_architecture_summary(model1) == "2 → 16 → 2"

    model2 = build_model(num_layers=3, hidden_dim=64, activation_name="Tanh", use_batchnorm=True)
    assert get_architecture_summary(model2) == "2 → 64 → 64 → 64 → 2"

    model3 = build_model(num_layers=2, hidden_dim=32, activation_name="GELU", dropout_p=0.2)
    assert get_architecture_summary(model3) == "2 → 32 → 32 → 2"


def test_studio_predict_point_inference():
    from app.app import build_model, predict_point
    model = build_model(num_layers=2, hidden_dim=16, activation_name="ReLU")
    pred_class, conf, probs = predict_point(model, 0.5, -0.5)

    assert pred_class in (0, 1)
    assert 0.0 <= conf <= 1.0
    assert len(probs) == 2
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    assert np.isclose(probs[pred_class], conf)
    assert not model.training


def test_studio_predict_point_batchnorm():
    from app.app import build_model, predict_point
    model = build_model(num_layers=2, hidden_dim=16, activation_name="ReLU", use_batchnorm=True)
    
    # Populate batchnorm running statistics with training step
    model.train()
    dummy_input = Tensor(np.random.randn(10, 2).astype(np.float32))
    _ = model(dummy_input)

    # Evaluate point
    pred_class, conf, probs = predict_point(model, 1.2, -0.8)
    assert pred_class in (0, 1)
    assert 0.0 <= conf <= 1.0
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)


def test_studio_trace_forward_pass():
    from app.app import build_model, trace_forward_pass
    model = build_model(num_layers=2, hidden_dim=16, activation_name="ReLU")
    steps, softmax_details = trace_forward_pass(model, 0.75, -0.25)

    # Step 0 is input, followed by 2 Linear layers + 2 ReLU layers (or Head Linear) = total steps
    assert len(steps) >= 3
    assert steps[0]["Step"] == 0
    assert steps[0]["Output Shape"] == "[1, 2]"
    assert steps[-1]["Output Shape"] == "[1, 2]"

    # Softmax breakdown verification
    assert "raw_logits" in softmax_details
    assert "shifted_logits" in softmax_details
    assert "probabilities" in softmax_details
    assert len(softmax_details["probabilities"]) == 2
    assert np.isclose(sum(softmax_details["probabilities"]), 1.0, atol=1e-5)


def test_studio_parameter_diagnostics():
    from app.app import build_model, get_parameter_diagnostics, get_layer_raw_weights
    model = build_model(num_layers=2, hidden_dim=32, activation_name="Tanh")
    diagnostics = get_parameter_diagnostics(model)

    assert len(diagnostics) > 0
    for diag in diagnostics:
        assert "Layer" in diag
        assert "Param" in diag
        assert "Shape" in diag
        assert isinstance(diag["Shape"], str)
        assert "Count" in diag
        assert diag["Count"] > 0
        assert isinstance(diag["Count"], int)
        assert "L2 Norm" in diag
        assert isinstance(diag["L2 Norm"], float)
        assert "Mean" in diag
        assert isinstance(diag["Mean"], float)
        assert "Std" in diag
        assert isinstance(diag["Std"], float)
        assert "Sparsity" in diag
        assert isinstance(diag["Sparsity"], str)
        # Ensure no raw ndarrays or non-serializable objects in tabular diagnostics
        assert "Array" not in diag
        assert "Values" not in diag

    raw_weights = get_layer_raw_weights(model)
    assert len(raw_weights) > 0
    for k, v in raw_weights.items():
        assert isinstance(k, str)
        assert isinstance(v, np.ndarray)


def test_studio_comparison_dashboard_figures():
    from app.app import build_model, generate_dataset, plot_comparison_dashboard_figures
    X, y = generate_dataset("Spirals", n_samples=100, noise=0.1, random_state=42)
    model_a = build_model(num_layers=1, hidden_dim=4, activation_name="Tanh")
    model_b = build_model(num_layers=2, hidden_dim=16, activation_name="ReLU")

    fig_a, fig_b, fig_c = plot_comparison_dashboard_figures(
        model_a, model_b, X, y,
        loss_hist_a=[0.69, 0.65], acc_hist_a=[50.0, 60.0],
        loss_hist_b=[0.69, 0.40], acc_hist_b=[50.0, 85.0],
        test_point=(0.5, -0.5),
        title_a="Model A Test",
        title_b="Model B Test",
    )
    assert fig_a is not None
    assert fig_b is not None
    assert fig_c is not None


def test_studio_dual_model_training():
    from app.app import build_model, generate_dataset, predict_point
    X, y = generate_dataset("Two Moons", n_samples=80, noise=0.1, random_state=42)
    model_a = build_model(num_layers=1, hidden_dim=4, activation_name="Tanh")
    model_b = build_model(num_layers=2, hidden_dim=16, activation_name="ReLU")

    crit_a = nn.CrossEntropyLoss()
    crit_b = nn.CrossEntropyLoss()
    opt_a = optim.SGD(model_a.parameters(), lr=0.1)
    opt_b = optim.AdamW(model_b.parameters(), lr=0.05)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=40, shuffle=True)

    # Train for 2 epochs
    for _ in range(2):
        for bx, by in loader:
            opt_a.zero_grad()
            l_a = crit_a(model_a(bx), by)
            l_a.backward()
            opt_a.step()

            opt_b.zero_grad()
            l_b = crit_b(model_b(bx), by)
            l_b.backward()
            opt_b.step()

    # Inference comparison
    pred_a, conf_a, probs_a = predict_point(model_a, 0.0, 0.0)
    pred_b, conf_b, probs_b = predict_point(model_b, 0.0, 0.0)

    assert pred_a in (0, 1)
    assert pred_b in (0, 1)
    assert 0.0 <= conf_a <= 1.0
    assert 0.0 <= conf_b <= 1.0


def test_gradient_flow_telemetry():
    from app.app import build_model, generate_dataset, plot_gradient_norms, plot_plotly_gradient_norms, get_model_gradient_norms, HAS_PLOTLY
    X, y = generate_dataset("Two Moons", n_samples=50, noise=0.1, random_state=42)
    model = build_model(num_layers=2, hidden_dim=8, activation_name="ReLU")
    crit = nn.CrossEntropyLoss()

    # Forward + Backward to populate grads
    loss = crit(model(Tensor(X)), y)
    loss.backward()

    grad_norms = get_model_gradient_norms(model)
    assert len(grad_norms) > 0
    for k, v in grad_norms.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
        assert v >= 0.0

    fig_mpl = plot_gradient_norms(grad_norms)
    assert fig_mpl is not None

    if HAS_PLOTLY:
        fig_plotly = plot_plotly_gradient_norms(grad_norms)
        assert fig_plotly is not None


def test_multi_digit_segmentation():
    from app.app import segment_and_preprocess_digits, preprocess_canvas_image
    from types import SimpleNamespace

    # Mock blank canvas
    blank_canvas = SimpleNamespace(image_data=np.zeros((280, 280, 4), dtype=np.uint8))
    assert segment_and_preprocess_digits(blank_canvas) == []
    assert preprocess_canvas_image(blank_canvas) is None

    # Mock single digit 1
    single_canvas = SimpleNamespace(image_data=np.zeros((280, 280, 4), dtype=np.uint8))
    single_canvas.image_data[50:180, 135:145, 0] = 255
    digits_1 = segment_and_preprocess_digits(single_canvas)
    assert len(digits_1) == 1
    assert digits_1[0].shape == (28, 28)
    assert 0.0 <= digits_1[0].min() and digits_1[0].max() <= 1.0

    # Mock multi-digit 42 (two separated components)
    multi_canvas = SimpleNamespace(image_data=np.zeros((280, 280, 4), dtype=np.uint8))
    # Left digit: '4' (vertical + crossbar)
    multi_canvas.image_data[60:160, 60:68, 0] = 255
    multi_canvas.image_data[110:118, 40:80, 0] = 255
    # Right digit: '2'
    multi_canvas.image_data[60:160, 180:190, 0] = 255

    digits_2 = segment_and_preprocess_digits(multi_canvas)
    assert len(digits_2) == 2
    assert digits_2[0].shape == (28, 28)
    assert digits_2[1].shape == (28, 28)


def test_simulate_rover_path():
    from app.app import build_model, generate_dataset, plot_plotly_rover_path, plot_rover_path_mpl, HAS_PLOTLY
    from numpygrad.utils.pathfinding import simulate_rover_path

    X, y = generate_dataset("Two Moons", n_samples=60, noise=0.1, random_state=42)
    model = build_model(num_layers=2, hidden_dim=8, activation_name="Tanh")

    sim_res = simulate_rover_path(
        model=model,
        start_pos=(-1.5, 1.0),
        target_pos=(1.5, -1.0),
        max_steps=10,
        step_size=0.15,
        num_rays=5,
        ray_len=0.3,
        avoidance_weight=2.0,
    )

    assert "trajectory" in sim_res
    assert "ray_history" in sim_res
    assert "hazard_history" in sim_res
    assert "success" in sim_res
    assert "collisions" in sim_res
    assert "steps_taken" in sim_res
    assert "final_distance" in sim_res

    assert len(sim_res["trajectory"]) >= 2
    assert isinstance(sim_res["success"], bool)
    assert isinstance(sim_res["collisions"], int)
    assert sim_res["final_distance"] >= 0.0

    fig_mpl = plot_rover_path_mpl(model, X, y, sim_res["trajectory"], (-1.5, 1.0), (1.5, -1.0))
    assert fig_mpl is not None

    if HAS_PLOTLY:
        fig_plotly = plot_plotly_rover_path(
            model=model,
            X=X,
            y=y,
            trajectory=sim_res["trajectory"],
            ray_history=sim_res["ray_history"],
            start_pos=(-1.5, 1.0),
            target_pos=(1.5, -1.0),
            step_index=2,
        )
        assert fig_plotly is not None


def test_dual_model_navigation_race():
    from app.app import build_model, generate_dataset, plot_plotly_rover_path, HAS_PLOTLY
    from numpygrad.utils.pathfinding import simulate_rover_path

    X, y = generate_dataset("Spirals", n_samples=80, noise=0.1, random_state=42)
    model_a = build_model(num_layers=1, hidden_dim=4, activation_name="Tanh")
    model_b = build_model(num_layers=3, hidden_dim=16, activation_name="ReLU")

    sim_a = simulate_rover_path(model=model_a, start_pos=(-2.0, -2.0), target_pos=(2.0, 2.0), max_steps=12)
    sim_b = simulate_rover_path(model=model_b, start_pos=(-2.0, -2.0), target_pos=(2.0, 2.0), max_steps=12)

    assert "trajectory" in sim_a and "trajectory" in sim_b
    assert len(sim_a["trajectory"]) > 1
    assert len(sim_b["trajectory"]) > 1

    if HAS_PLOTLY:
        fig_a = plot_plotly_rover_path(model_a, X, y, sim_a["trajectory"], sim_a["ray_history"], (-2.0, -2.0), (2.0, 2.0), step_index=1)
        fig_b = plot_plotly_rover_path(model_b, X, y, sim_b["trajectory"], sim_b["ray_history"], (-2.0, -2.0), (2.0, 2.0), step_index=1)
        assert fig_a is not None and fig_b is not None



