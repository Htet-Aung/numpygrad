"""
Unit tests for model inspection and architecture summary utilities.
"""

import numpy as np
import pytest

import numpygrad as ng
import numpygrad.nn as nn
from numpygrad.nn.summary import summary


def test_sequential_mlp_summary_with_input_shape():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    report = model.summary(input_shape=(1, 28, 28), verbose=False)

    # Check key content in summary table
    assert "Flatten-1" in report
    assert "Linear-2" in report
    assert "ReLU-3" in report
    assert "Linear-4" in report
    assert "Total params: 109,386" in report
    assert "Trainable params: 109,386" in report
    assert "Non-trainable params: 0" in report
    assert "Estimated Total Size (MB):" in report


def test_cnn_summary_with_input_shape():
    model = nn.Sequential(
        nn.Conv2D(1, 8, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2D(2),
        nn.Conv2D(8, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2D(2),
        nn.Flatten(),
        nn.Linear(16 * 7 * 7, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )

    report = summary(model, input_shape=(1, 1, 28, 28), verbose=False)

    assert "Conv2D-1" in report
    assert "MaxPool2D-3" in report
    assert "Conv2D-4" in report
    assert "Total params: 52,138" in report
    assert "Trainable params: 52,138" in report
    assert "[1, 8, 28, 28]" in report
    assert "[1, 8, 14, 14]" in report
    assert "[1, 16, 7, 7]" in report


def test_summary_without_input_shape():
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 2),
    )

    report = model.summary(input_shape=None, verbose=False)

    assert "Linear-1" in report
    assert "Linear-3" in report
    assert "Total params: 262" in report
    # Output shape should be placeholder when input_shape is omitted
    assert "--" in report


def test_summary_with_batchnorm_buffers():
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )

    report = model.summary(input_shape=(4, 16), verbose=False)

    assert "BatchNorm1d-2" in report
    # BatchNorm1d has 32 gamma + 32 beta = 64 params, and 32 mean + 32 var = 64 buffer elements
    assert "Total buffer elements: 64" in report
    assert "Trainable params: 674" in report  # (16*32+32) + 64 + (32*2+2) = 544 + 64 + 66 = 674


def test_single_module_summary():
    layer = nn.Linear(32, 8)
    report = layer.summary(input_shape=(4, 32), verbose=False)

    assert "Linear-1" in report
    assert "Total params: 264" in report
