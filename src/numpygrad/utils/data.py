"""
Dataset and DataLoader utilities for mini-batching, shuffling, and data pipelines.
"""

from __future__ import annotations
from typing import Sequence, Iterator, Tuple, Union, List, Any
import numpy as np
from numpygrad.core.tensor import Tensor


class Dataset:
    """
    An abstract class representing a Dataset.
    All datasets that represent a map from keys to data samples should subclass it.
    """

    def __getitem__(self, index: int) -> Any:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError


class TensorDataset(Dataset):
    """
    Dataset wrapping tensors or numpy arrays.
    Each sample will be retrieved by indexing tensors along the first dimension.
    """

    def __init__(self, *tensors: Union[Tensor, np.ndarray]) -> None:
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor argument")

        first_len = len(tensors[0])
        for t in tensors:
            if len(t) != first_len:
                raise ValueError(f"Size mismatch among tensors: {first_len} vs {len(t)}")

        self.tensors: Tuple[Union[Tensor, np.ndarray], ...] = tensors

    def __getitem__(self, index: Union[int, slice, np.ndarray]) -> Tuple[Any, ...]:
        return tuple(
            t[index] if isinstance(t, np.ndarray) else Tensor(t.data[index], requires_grad=t.requires_grad, dtype=t.dtype)
            for t in self.tensors
        )

    def __len__(self) -> int:
        return len(self.tensors[0])


class DataLoader:
    """
    Data loader providing mini-batch iteration, optional dataset shuffling,
    and batch collation.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.drop_last: bool = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        else:
            return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        n = len(self.dataset)
        indices = np.arange(n)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n, self.batch_size):
            end_idx = start_idx + self.batch_size
            if end_idx > n:
                if self.drop_last:
                    break
                end_idx = n

            batch_indices = indices[start_idx:end_idx]

            # Fetch samples
            samples = [self.dataset[i] for i in batch_indices]

            # Collate batch
            if isinstance(samples[0], tuple):
                # Tuple of items -> stack each element across samples
                num_elements = len(samples[0])
                collated = []
                for elem_idx in range(num_elements):
                    elem_list = [s[elem_idx] for s in samples]
                    if isinstance(elem_list[0], Tensor):
                        stacked_data = np.stack([e.data for e in elem_list], axis=0)
                        collated.append(Tensor(stacked_data, requires_grad=elem_list[0].requires_grad, dtype=elem_list[0].dtype))
                    elif isinstance(elem_list[0], np.ndarray):
                        collated.append(Tensor(np.stack(elem_list, axis=0), requires_grad=False))
                    elif isinstance(elem_list[0], (int, float, np.number)):
                        collated.append(Tensor(np.array(elem_list), requires_grad=False))
                    else:
                        collated.append(elem_list)
                yield tuple(collated)
            else:
                if isinstance(samples[0], Tensor):
                    stacked_data = np.stack([s.data for s in samples], axis=0)
                    yield Tensor(stacked_data, requires_grad=samples[0].requires_grad, dtype=samples[0].dtype)
                elif isinstance(samples[0], np.ndarray):
                    yield Tensor(np.stack(samples, axis=0), requires_grad=False)
                elif isinstance(samples[0], (int, float, np.number)):
                    yield Tensor(np.array(samples), requires_grad=False)
                else:
                    yield samples
