"""Testing LCS extension on CPU."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lcsvec import lcs, lcs_length, lcs_table
from torch import IntTensor, LongTensor, arange

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _test_lcs(
    seq1: NDArray | IntTensor | LongTensor,
    seq2: NDArray | IntTensor | LongTensor,
    ref: list[int],
) -> bool:
    lcs_ = lcs(seq1, seq2)
    lcs_table_ = lcs_table(seq1, seq2)
    lcs_len = lcs_length(seq1, seq2)

    assert lcs_len == lcs_table_[-1][-1] == len(ref)
    assert lcs_ == ref
    return True


def test_lcs_numpy() -> None:
    r"""Test the LCS methods with numpy."""
    seq1 = np.arange(0, 12)
    seq2 = np.array([8, 0, 1, 2, 8, 2, 3, 4, 5, 6], dtype=np.int64)
    ref = np.arange(0, 7).tolist()

    assert _test_lcs(seq1, seq2, ref)


def test_lcs_torch() -> None:
    r"""Test the LCS methods with pytorch."""
    seq1 = arange(0, 12)
    seq2 = LongTensor([8, 0, 1, 2, 8, 2, 3, 4, 5, 6])
    ref = arange(0, 7).tolist()

    assert _test_lcs(seq1, seq2, ref)
