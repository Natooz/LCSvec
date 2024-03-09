#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;


// Create the LCS table
std::vector<std::vector<int>> createLCSTable(
    const nb::ndarray<double, nb::ndim<1>>& s1,
    const nb::ndarray<double, nb::ndim<1>>& s2
) {
    // Sequence lengths
    const int s1Len = s1.shape(0);
    const int s2Len = s2.shape(0);

    // Building the matrix
    auto v1 = s1.view();
    auto v2 = s2.view();
    std::vector<std::vector<int>> lcsTable(s1Len + 1, std::vector<int>(s2Len + 1, 0));
    for (int i = 1; i <= s1Len; i++) {
        for (int j = 1; j <= s2Len; j++) {
            if (v1(i - 1) == v2(j - 1))
                lcsTable[i][j] = lcsTable[i - 1][j - 1] + 1;
            else
                lcsTable[i][j] = std::max(lcsTable[i - 1][j], lcsTable[i][j - 1]);
        }
    }

    return lcsTable;
}


// Return the length of the sequence from the table
int lcsLength(
    const nb::ndarray<double, nb::ndim<1>>& s1,
    const nb::ndarray<double, nb::ndim<1>>& s2
) {
    auto lcsTable = createLCSTable(s1, s2);
    return lcsTable[std::size(s1)][std::size(s2)];
}


// Return the longest common subsequence by parsing the table
std::vector<int> lcs(
    const nb::ndarray<double, nb::ndim<1>>& s1,
    const nb::ndarray<double, nb::ndim<1>>& s2
) {
    // Sequence lengths
    const int s1Len = s1.shape(0);
    const int s2Len = s2.shape(0);

    // Zero length
    if (s1Len == 0 || s2Len == 0)
        return std::vector<int>();

    // Building the matrix
    std::vector<std::vector<int>> lcsTable = createLCSTable(s1, s2);

    int index = lcsTable[s1Len][s2Len];
    std::vector<int> lcsArr(index);
    int i = s1Len, j = s2Len;
    while (i > 0 && j > 0) {
        if (s1(i - 1) == s2(j - 1)) {
            lcsArr[index - 1] = s1(i - 1);
            i--;
            j--;
            index--;
        }
        else if (lcsTable[i - 1][j] > lcsTable[i][j - 1])
            i--;
        else
            j--;
    }

    return lcsArr;
}
