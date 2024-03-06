"""
Python implementation of the lcs algorithm using PyTorch.
"""

from torch import LongTensor, cat


def longest_common_subsequence(x: LongTensor, y: LongTensor) -> LongTensor:
    """
    Dynamically retrieve the longest common subsequence between two sequences.

    This works with PyTorch tensors.
    """
    # generate matrix of length of longest common subsequence for subsequences of both sequences
    lengths = [[0] * (len(y) + 1) for _ in range(len(x) + 1)]
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            if xi == yi:
                lengths[i + 1][j + 1] = lengths[i][j] + 1
            else:
                lengths[i + 1][j + 1] = max(lengths[i + 1][j], lengths[i][j + 1])

    # read subsequences from the matrix
    result = []
    j = len(y)
    for i in range(1, len(x) + 1):
        if lengths[i][j] != lengths[i - 1][j]:
            result.append(x[i - 1].reshape(1))

    return cat(result).long()
