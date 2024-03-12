#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;


// Returns the length of the longest common subsequence and the idx of its end in s1
std::vector<int> lccsLengthIdx(
    const nb::ndarray<double, nb::ndim<1>>& s1,
    const nb::ndarray<double, nb::ndim<1>>& s2
) {
    // Sequence lengths
    const int s1Len = s1.shape(0);
    const int s2Len = s2.shape(0);

    // Create views for arrays
    auto v1 = s1.view();
    auto v2 = s2.view();

    std::vector<std::vector<int>> table(s1Len + 1, std::vector<int>(s2Len + 1, 0));
    int max_length = 0;
    int imax = 0;  // ending idx of the lccs
    for (int i = 0; i < s1Len; ++i) {
        for (int j = 0; j < s2Len; ++j) {
            if (v1(i) == v2(j)) {
                table[i + 1][j + 1] = table[i][j] + 1;
                if (table[i + 1][j + 1] > max_length) {
                    imax = i;
                    max_length = table[i + 1][j + 1];
                }
            }
        }
    }
    std::vector<int> lccsLenIdx = {max_length, imax};
    return lccsLenIdx;
}


// Calculating the length of the longest common contiguous subsequence with dynamic programming
int lccsLength(
    const nb::ndarray<double, nb::ndim<1>>& s1,
    const nb::ndarray<double, nb::ndim<1>>& s2
) {
    std::vector<int> lccsLenIdx = lccsLengthIdx(s1, s2);
    return lccsLenIdx[0];
}


// Calculating the longest common contiguous subsequence with dynamic programming
std::vector<int> lccs(
    const nb::ndarray<double, nb::ndim<1>>& s1,
    const nb::ndarray<double, nb::ndim<1>>& s2
) {
    std::vector<int> lccsLenIdx = lccsLengthIdx(s1, s2);

    // Extract the longest common substring from s1
    std::vector<int> longestSubseq(lccsLenIdx[0]);
    int idx = 0;
    for (int i = lccsLenIdx[1] - lccsLenIdx[0] + 1; i <= lccsLenIdx[1]; ++i)
        longestSubseq[idx++] = s1(i);

    return longestSubseq;
}
