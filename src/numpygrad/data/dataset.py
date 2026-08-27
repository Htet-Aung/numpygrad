"""
Dataset abstractions for managing samples, features, and targets in NumPyGrad.
"""

from __future__ import annotations
from typing import Sequence, Tuple, Union, Any, Optional
import numpy as np
from numpygrad.core.tensor import Tensor


class Dataset:
    """
    Abstract base and container class representing a dataset.

    Can be used in two ways:
    1. Subclassed by implementing custom `__len__()` and `__getitem__()` methods.
    2. Instantiated directly with `features` and optional `labels` arrays/tensors.

    Parameters
    ----------
    features : Optional[Union[Tensor, np.ndarray, Sequence]], optional
        Feature array, tensor, or sequence of shape (N, ...).
    labels : Optional[Union[Tensor, np.ndarray, Sequence]], optional
        Target labels array, tensor, or sequence of shape (N, ...).
    """

    def __init__(
        self,
        features: Optional[Union[Tensor, np.ndarray, Sequence]] = None,
        labels: Optional[Union[Tensor, np.ndarray, Sequence]] = None,
    ) -> None:
        self.features: Optional[Union[Tensor, np.ndarray]] = None
        self.labels: Optional[Union[Tensor, np.ndarray]] = None

        if features is not None:
            self.features = features if isinstance(features, (Tensor, np.ndarray)) else np.asarray(features)
            if labels is not None:
                self.labels = labels if isinstance(labels, (Tensor, np.ndarray)) else np.asarray(labels)
                # Validate length alignment
                if len(self.features) != len(self.labels):
                    raise ValueError(
                        f"Features and labels length mismatch: {len(self.features)} vs {len(self.labels)}"
                    )

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        if self.features is not None:
            return len(self.features)
        raise NotImplementedError("Subclasses must implement __len__ or pass features to Dataset")

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> Any:
        """
        Retrieves sample(s) at the specified index.

        Parameters
        ----------
        index : Union[int, slice, np.ndarray]
            Index, slice, or integer array to retrieve.

        Returns
        -------
        Any
            Tuple of (feature, label) if labels exist, otherwise feature sample.
        """
        if self.features is not None:
            total_len = len(self.features)
            if isinstance(index, (int, np.integer)):
                idx = int(index)
                if idx < -total_len or idx >= total_len:
                    raise IndexError(f"Index {idx} is out of bounds for dataset of length {total_len}")

            feat = (
                self.features[index]
                if isinstance(self.features, np.ndarray)
                else Tensor(
                    self.features.data[index],
                    requires_grad=self.features.requires_grad,
                    dtype=self.features.dtype,
                )
            )

            if self.labels is not None:
                lbl = (
                    self.labels[index]
                    if isinstance(self.labels, np.ndarray)
                    else Tensor(
                        self.labels.data[index],
                        requires_grad=self.labels.requires_grad,
                        dtype=self.labels.dtype,
                    )
                )
                return feat, lbl
            return feat

        raise NotImplementedError("Subclasses must implement __getitem__ or pass features to Dataset")


class TensorDataset(Dataset):
    """
    Dataset wrapping one or more tensors or NumPy ndarrays.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Parameters
    ----------
    *tensors : Union[Tensor, np.ndarray, Sequence]
        Tensors or ndarrays with identical size along the first dimension (N).
    """

    def __init__(self, *tensors: Union[Tensor, np.ndarray, Sequence]) -> None:
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor argument")

        processed = []
        for t in tensors:
            if not isinstance(t, (Tensor, np.ndarray)):
                t = np.asarray(t)
            processed.append(t)

        first_len = len(processed[0])
        for idx, t in enumerate(processed[1:], start=1):
            if len(t) != first_len:
                raise ValueError(
                    f"Size mismatch among tensors: tensor 0 has len {first_len}, but tensor {idx} has len {len(t)}"
                )

        self.tensors: Tuple[Union[Tensor, np.ndarray], ...] = tuple(processed)
        if len(self.tensors) >= 2:
            super().__init__(features=self.tensors[0], labels=self.tensors[1])
        else:
            super().__init__(features=self.tensors[0])

    def __len__(self) -> int:
        return len(self.tensors[0])

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> Union[Tuple[Any, ...], Any]:
        total_len = len(self)
        if isinstance(index, (int, np.integer)):
            idx = int(index)
            if idx < -total_len or idx >= total_len:
                raise IndexError(f"Index {idx} is out of bounds for TensorDataset of length {total_len}")

        items = tuple(
            t[index]
            if isinstance(t, np.ndarray)
            else Tensor(t.data[index], requires_grad=t.requires_grad, dtype=t.dtype)
            for t in self.tensors
        )
        return items if len(items) > 1 else items[0]
