"""Testing LCCS extension on CPU."""

import numpy as np
from lcspy import lccs_length


def test_lccs_simple() -> None:
    r"""
    Test the LCCS method with a simple case.

    TODO test with numpy, torch, tensorflow and jax
    """
    seq1 = np.arange(0, 12)
    seq2 = np.array([8, 0, 1, 2, 8, 2, 3, 8, 4, 0], dtype=np.int64)
    ref = np.arange(0, 3)

    lcs_len = lccs_length(seq1, seq2)
    assert lcs_len == len(ref)


if __name__ == "main":
    test_lccs_simple()
