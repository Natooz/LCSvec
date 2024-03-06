"""
Testing LCS extension on CPU.
"""

from torch import LongTensor, arange, all

from .lcs_py import longest_common_subsequence


def test_lcs_simple():
    r"""
    Tokenize and decode a MIDI back to make sure the possible I/O format are ok.
    """
    seq1 = arange(0, 12)
    seq2 = LongTensor([8, 2, 3, 4, 6])
    lcs_ = longest_common_subsequence(seq1, seq2)
    assert all(lcs_ == seq2[1:])
