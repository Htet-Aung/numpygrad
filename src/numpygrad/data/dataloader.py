"""
DataLoader pipeline for mini-batching, deterministic shuffling, and collation.
"""

from __future__ import annotations
from typing import Iterator, Union, List, Any, Optional, Callable, Sequence
import numpy as np
from numpygrad.core.tensor import Tensor
from numpygrad.data.dataset import Dataset


class DataLoader:
    """
    Data loader providing mini-batch iteration, optional dataset shuffling,
    deterministic seeding, dropping or yielding remainder batches, and batch collation.

    Parameters
    ----------
    dataset : Dataset
        The dataset from which to load the data.
    batch_size : int, optional
        Number of samples per batch (default: 1).
    shuffle : bool, optional
        Whether to shuffle data at each epoch (default: False).
    drop_last : bool, optional
        Whether to drop the last incomplete batch if the dataset size is not
        divisible by `batch_size` (default: False).
    seed : Optional[int], optional
        Random seed for deterministic shuffling across iterations/epochs (default: None).
    generator : Optional[np.random.Generator], optional
        Custom NumPy Generator for random permutations (default: None).
    collate_fn : Optional[Callable[[List[Any]], Any]], optional
        Custom callable that merges a list of samples into a mini-batch (default: None).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: Optional[int] = None,
        generator: Optional[np.random.Generator] = None,
        collate_fn: Optional[Callable[[List[Any]], Any]] = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}")

        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.drop_last: bool = drop_last
        self.seed: Optional[int] = seed
        self.generator: Optional[np.random.Generator] = generator
        self.collate_fn: Optional[Callable[[List[Any]], Any]] = collate_fn
        self._epoch: int = 0

    def reset_seed(self) -> None:
        """Resets the internal epoch counter used for deterministic seed sequences."""
        self._epoch = 0

    def __len__(self) -> int:
        n = len(self.dataset)
        if n == 0:
            return 0
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        n = len(self.dataset)
        if n == 0:
            return

        if self.shuffle:
            if self.generator is not None:
                indices = self.generator.permutation(n)
            elif self.seed is not None:
                rng = np.random.default_rng(self.seed + self._epoch)
                self._epoch += 1
                indices = rng.permutation(n)
            else:
                indices = np.random.permutation(n)
        else:
            indices = np.arange(n)

        for start_idx in range(0, n, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n)
            if self.drop_last and (end_idx - start_idx) < self.batch_size:
                break

            batch_indices = indices[start_idx:end_idx]
            samples = [self.dataset[i] for i in batch_indices]

            if self.collate_fn is not None:
                yield self.collate_fn(samples)
            else:
                yield self._default_collate(samples)

    @staticmethod
    def _default_collate(samples: List[Any]) -> Any:
        """Default batch collator that stacks elements into Tensor instances."""
        if not samples:
            return ()

        first_sample = samples[0]

        if isinstance(first_sample, (tuple, list)):
            num_elements = len(first_sample)
            collated = []
            for elem_idx in range(num_elements):
                elem_list = [s[elem_idx] for s in samples]
                first_elem = elem_list[0]
                if isinstance(first_elem, Tensor):
                    stacked_data = np.stack([e.data for e in elem_list], axis=0)
                    collated.append(
                        Tensor(
                            stacked_data,
                            requires_grad=first_elem.requires_grad,
                            dtype=first_elem.dtype,
                        )
                    )
                elif isinstance(first_elem, np.ndarray):
                    collated.append(
                        Tensor(
                            np.stack(elem_list, axis=0),
                            requires_grad=False,
                            dtype=first_elem.dtype,
                        )
                    )
                elif isinstance(first_elem, (int, float, np.number, bool)):
                    collated.append(
                        Tensor(np.array(elem_list), requires_grad=False)
                    )
                else:
                    collated.append(elem_list)
            return tuple(collated)

        elif isinstance(first_sample, Tensor):
            stacked_data = np.stack([s.data for s in samples], axis=0)
            return Tensor(
                stacked_data,
                requires_grad=first_sample.requires_grad,
                dtype=first_sample.dtype,
            )

        elif isinstance(first_sample, np.ndarray):
            return Tensor(
                np.stack(samples, axis=0),
                requires_grad=False,
                dtype=first_sample.dtype,
            )

        elif isinstance(first_sample, (int, float, np.number, bool)):
            return Tensor(np.array(samples), requires_grad=False)

        return samples
