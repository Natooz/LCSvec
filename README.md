[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/)
[![GitHub CI](https://github.com/Natooz/LCSpy/actions/workflows/tests.yml/badge.svg)](https://github.com/Natooz/LCSpy/actions/workflows/tests.yml)
[![GitHub license](https://img.shields.io/github/license/Natooz/LCSpy.svg)](https://github.com/Natooz/LCSpy/blob/main/LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

# LCSpy

Longest Common Subsequence (LCS) extension for numpy arrays

## Why LCSpy

While looking for fast implementations of the [Longest Common Subsequence](https://wikipedia.org/wiki/Longest_common_subsequence) and [Longest Common Substring (i.e. Contiguous Subsequence)](https://wikipedia.org/wiki/Longest_common_substring) (LCCS) problems, I only found pieces of code for strings, while I needed something working for **vectors of integers**.
Yet, string LCS implementations 1) work at the character level, which might not be suitable for certain use-cases where one wants to work at word or sentence level; 2) only work with strings, other modalities will not work and might not be designed to be converted to bytes.

LCSpy aims to solve this gap by providing user-friendly and fast implementations of the LCS and LCCS problems for numpy arrays.
The code is written in C++ and the methods are bind with Python with [nanobind](https://github.com/wjakob/nanobind) for optimal performances.

## Example

You can install the package with pip: `pip install lcspy`.

```Python
from lcspy import lcs, lcs_length, lccs_length
import numpy as np

seq1 = np.arange(0, 12)
seq2 = np.array([8, 0, 1, 2, 8, 2, 3, 4, 5, 6], dtype=np.int64)

lcs_ = lcs(seq1, seq2)  # [0, 1, 2, 3, 4, 5, 6]
lcs_len = lcs_length(seq1, seq2)  # 7, more efficient than calling len(lcs(seq1, seq2))

lccs_len = lccs_length(seq1, seq2)  # 5, [2, 3, 4, 5, 6]
```

## TODOs

* batch methods, i.e. supporting 2D arrays;
* batch methods with any number of dimensions (nD array) and add a `dim` argument;
* make it work with an unlimited number of sequences, and set `dim` and `pad_token` as kwargs only;
