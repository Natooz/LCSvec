"""Testing LCCS extension on CPU."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from lcsvec import lccs_length
from torch import IntTensor, LongTensor, arange

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _test_lccs(
    seq1: NDArray | IntTensor | LongTensor,
    seq2: NDArray | IntTensor | LongTensor,
    ref: list[int],
) -> None:
    lcs_len = lccs_length(seq1, seq2)
    assert lcs_len == len(ref)


def test_lccs_numpy() -> None:
    r"""Test the LCCS methods with numpy."""
    seq1 = np.arange(0, 12)
    seq2 = np.array([8, 0, 1, 2, 8, 2, 3, 8, 4, 0], dtype=np.int64)
    ref = np.arange(0, 3).tolist()

    lcs_len = lccs_length(seq1, seq2)
    assert lcs_len == len(ref)


def test_lccs_torch() -> None:
    r"""Test the LCCS methods with pytorch."""
    seq1 = arange(0, 12)
    seq2 = LongTensor([8, 0, 1, 2, 8, 2, 3, 8, 4, 0])
    ref = arange(0, 3).tolist()

    lcs_len = lccs_length(seq1, seq2)
    assert lcs_len == len(ref)
