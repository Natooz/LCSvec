"""
Testing LCS extension on CPU.
"""

from torch import LongTensor, arange, all

from lcstorch import lcs, lcs_length, lcs_table  # TODO make methods appear


def test_lcs_simple():
    r"""
    Tokenize and decode a MIDI back to make sure the possible I/O format are ok.
    """
    seq1 = arange(0, 12)
    seq2 = LongTensor([8, 0, 1, 2, 8, 2, 3, 4, 5, 6])
    ref = arange(0, 7)
    lcs_ = lcs(seq1, seq2)
    lcs_table_ = lcs_table(seq1, seq2)
    lcs_len = lcs_length(seq1, seq2)
    assert lcs_len == lcs_table_[-1][-1] == len(ref)
    assert all(LongTensor(lcs_) == ref)


if __name__ == "main":
    test_lcs_simple()
