#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;


// Calculating the length of the longest common contiguous subsequence with dynamic programming
int lccs_length(
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
    for (int i = 0; i < s1Len; ++i) {
        for (int j = 0; j < s2Len; ++j) {
            if (v1(i) == v2(j)) {
                table[i + 1][j + 1] = table[i][j] + 1;
                if (table[i + 1][j + 1] > max_length) {
                    max_length = table[i + 1][j + 1];
                }
            }
        }
    }

    return max_length;
}
