"""LCS implementation in Python, for benchmark purposes."""

import numpy as np
import numpy.typing as npt


def longest_common_subsequence(x: npt.NDArray, y: npt.NDArray) -> npt.NDArray:
    """
    Dynamically retrieve the longest common subsequence between two sequences.

    This works with PyTorch tensors.
    """
    # generate matrix for subsequences of both sequences
    lengths = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            if xi == yi:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    # read subsequences from the matrix
    j = len(y)
    result = [
        x[i - 1].reshape(1)
        for i in range(1, len(x) + 1)
        if lengths[i][j] != lengths[i - 1][j]
    ]
    return np.concatenate(result, dtype=np.int64)


def test_lcs_simple() -> None:
    r"""Test the Python LCS method."""
    seq1 = np.arange(0, 12)
    seq2 = np.array([8, 0, 1, 2, 8, 2, 3, 4, 5, 6], dtype=np.int64)
    ref = np.arange(0, 7)

    lcs_ = longest_common_subsequence(seq1, seq2)
    assert np.all(np.array(lcs_, dtype=np.int64) == ref)
