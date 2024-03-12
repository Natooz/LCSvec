"""Testing LCCS extension on CPU."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from lcsvec import lccs, lccs_length
from torch import IntTensor, LongTensor, arange

if TYPE_CHECKING:
    from numpy.typing import NDArray


TEST_CASES = [
    (range(12), [8, 0, 1, 2, 8, 2, 3, 8, 4, 0], range(3)),
    (range(12), [8, 0, 9, 2, 8, 2, 7, 3, 4, 5], range(3, 6)),
    (range(12), [0, 1, 2, 3, 8, 9, 2, 3, 4, 5], range(4)),
    (range(-2, 10), [9, -1, 0, 1, 2, 9, 2, 4, 4, 5], range(-1, 3)),
]


def _test_lccs(
    seq1: NDArray | IntTensor | LongTensor,
    seq2: NDArray | IntTensor | LongTensor,
    ref: list[int],
) -> True:
    lccs_ = lccs(seq1, seq2)
    lccs_len = lccs_length(seq1, seq2)

    assert lccs_len == len(ref)
    assert lccs_ == ref
    return True


@pytest.mark.parametrize("sequences", TEST_CASES)
def test_lccs_numpy(
    sequences: tuple[list[int] | range, list[int] | range, list[int] | range],
) -> None:
    r"""Test the LCCS methods with numpy."""
    seq1, seq2, ref = sequences
    seq1 = (
        np.arange(seq1.start, seq1.stop)
        if isinstance(seq1, range)
        else np.array(seq1, dtype=np.int64)
    )
    seq2 = (
        np.arange(seq2.start, seq2.stop)
        if isinstance(seq2, range)
        else np.array(seq2, dtype=np.int64)
    )
    ref = (
        np.arange(ref.start, ref.stop).tolist()
        if isinstance(ref, range)
        else np.array(ref, dtype=np.int64).tolist()
    )

    assert _test_lccs(seq1, seq2, ref)


@pytest.mark.parametrize("sequences", TEST_CASES)
def test_lccs_torch(
    sequences: tuple[list[int] | range, list[int] | range, list[int] | range],
) -> None:
    r"""Test the LCCS methods with pytorch."""
    seq1, seq2, ref = sequences
    seq1 = (
        arange(seq1.start, seq1.stop) if isinstance(seq1, range) else LongTensor(seq1)
    )
    seq2 = (
        arange(seq2.start, seq2.stop) if isinstance(seq2, range) else LongTensor(seq2)
    )
    ref = arange(ref.start, ref.stop) if isinstance(ref, range) else LongTensor(ref)
    ref = ref.tolist()

    assert _test_lccs(seq1, seq2, ref)
