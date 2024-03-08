"""Testing LCS extension on CPU."""

import numpy as np

from .lcs_py import longest_common_subsequence


def test_lcs_simple() -> None:
    r"""Tokenize and decode a MIDI back to make sure the possible I/O format are ok."""
    seq1 = np.arange(0, 12)
    seq2 = np.array([8, 0, 1, 2, 8, 2, 3, 4, 5, 6], dtype=np.int64)
    ref = np.arange(0, 7)
    lcs_ = longest_common_subsequence(seq1, seq2)
    assert np.all(np.array(lcs_, dtype=np.int64) == ref)
