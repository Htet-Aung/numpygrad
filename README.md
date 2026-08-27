# NumPyGrad: Deep Learning & Autograd Engine from Scratch

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![No DL Frameworks](https://img.shields.io/badge/Dependencies-Pure%20NumPy-brightgreen.svg)](#)

A modular, educational deep learning library and dynamic tensor autograd engine built entirely from scratch using only Python and NumPy.

## ✨ Highlights
- **Dynamic Computational DAG:** Reverse-mode automatic differentiation with topological backpropagation.
- **Broadcasting-Aware Calculus:** Exact gradient accumulation across arbitrary batch and feature dimensions.
- **PyTorch-Style API:** Modular `nn.Module`, `Linear`, `BatchNorm1d`, `Dropout`, and `CrossEntropyLoss`.
- **Custom Optimizers:** Vectorized `SGD` (with Momentum) and `AdamW` with decoupled weight decay.
- **Strict Verification:** Built-in numerical gradient checker (`gradcheck`) using finite-difference approximations.
