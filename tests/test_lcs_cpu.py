"""
Testing LCS extension on CPU.
"""

from numpy import all, arange, array

from lcspy import lcs, lcs_length, lcs_table


def test_lcs_simple():
    r"""
    Test the LCS method with a simple case
    """
    seq1 = arange(0, 12).tolist()
    seq2 = array([8, 0, 1, 2, 8, 2, 3, 4, 5, 6]).tolist()
    ref = arange(0, 7).tolist()
    lcs_ = lcs(seq1, seq2)
    lcs_table_ = lcs_table(seq1, seq2)
    lcs_len = lcs_length(seq1, seq2)
    assert lcs_len == lcs_table_[-1][-1] == len(ref)
    assert all(array(lcs_) == ref)


if __name__ == "main":
    test_lcs_simple()
