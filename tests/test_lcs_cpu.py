"""Testing LCS extension on CPU."""

import numpy as np
from lcspy import lcs, lcs_length, lcs_table


def test_lcs_simple() -> None:
    r"""
    Test the LCS method with a simple case.

    TODO test with numpy, torch, tensorflow and jax
    """
    seq1 = np.arange(0, 12)
    seq2 = np.array([8, 0, 1, 2, 8, 2, 3, 4, 5, 6], dtype=np.int64)
    ref = np.arange(0, 7)
    lcs_ = lcs(seq1, seq2)
    lcs_table_ = lcs_table(seq1, seq2)
    lcs_len = lcs_length(seq1, seq2)
    assert lcs_len == lcs_table_[-1][-1] == len(ref)
    assert np.all(np.array(lcs_, dtype=np.int64) == ref)


if __name__ == "main":
    test_lcs_simple()
