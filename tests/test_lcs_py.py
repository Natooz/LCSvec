"""Testing LCS extension on CPU."""

from torch import LongTensor, all, arange

from .lcs_py import longest_common_subsequence


def test_lcs_simple() -> None:
    r"""Tokenize and decode a MIDI back to make sure the possible I/O format are ok."""
    seq1 = arange(0, 12)
    seq2 = LongTensor([4, 0, 1, 2, 8, 2, 3, 4, 5, 6])
    lcs_ = longest_common_subsequence(seq1, seq2)
    assert all(lcs_ == seq2[1:])
