"""
Testing LCS extension on CPU.
"""

from torch import LongTensor, arange, all

from lcstorch import lcs  # TODO make it appear


def test_lcs_simple():
    r"""
    Tokenize and decode a MIDI back to make sure the possible I/O format are ok.
    """
    seq1 = arange(0, 12)
    seq2 = LongTensor([0, 1, 2, 8, 2, 3, 4, 5, 6])
    lcs_ = lcs(seq1, seq2)
    assert all(lcs_ == seq2[4:])
