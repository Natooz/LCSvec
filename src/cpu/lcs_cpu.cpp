#include <nanobind/nanobind.h>
#include <vector>


// Create the LCS table
std::vector<std::vector<int>> createLCSTable(
    const std::vector<int>& s1,
    const std::vector<int>& s2
) {
    // Sequence lengths
    const int s1Len = std::size(s1);
    const int s2Len = std::size(s2);

    // Building the matrix
    std::vector<std::vector<int>> lcsTable(s1Len + 1, std::vector<int>(s2Len + 1, 0));
    for (int i = 1; i <= s1Len; i++) {
        for (int j = 1; j <= s2Len; j++) {
            if (s1[i - 1] == s2[j - 1])
                lcsTable[i][j] = lcsTable[i - 1][j - 1] + 1;
            else
                lcsTable[i][j] = std::max(lcsTable[i - 1][j], lcsTable[i][j - 1]);
        }
    }

    return lcsTable;
}


// Return the length of the sequence from the table
int lcsLength(
    const std::vector<int>& s1,
    const std::vector<int>& s2
) {
    std::vector<std::vector<int>> lcsTable = createLCSTable(s1, s2);
    return lcsTable[std::size(s1)][std::size(s2)];
}


// Return the longest common subsequence by parsing the table
std::vector<int> lcs(
    const std::vector<int>& s1,
    const std::vector<int>& s2
) {
    // Sequence lengths
    const int s1Len = std::size(s1);
    const int s2Len = std::size(s2);

    // Zero length
    if (s1Len == 0 || s2Len == 0)
        return std::vector<int>();

    // Building the matrix
    std::vector<std::vector<int>> lcsTable = createLCSTable(s1, s2);

    int index = lcsTable[s1Len][s2Len];
    std::vector<int> lcsArr(index);
    int i = s1Len, j = s2Len;
    while (i > 0 && j > 0) {
        if (s1[i - 1] == s2[j - 1]) {
            lcsArr[index - 1] = s1[i - 1];
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


namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(lcstorch, m) {
    m.doc() = "A python extension for fast Longest Common Subsequence (LCS) calculation on scalar vectors;";

    m.def("lcs", &lcs, "seq1"_a, "seq2"_a,
          "Returns the longest common subsequence (lcs) from `seq1` and `seq2`.");
    m.def("lcs_length", &lcsLength, "seq1"_a, "seq2"_a,
          "Returns the length longest common subsequence (lcs) from `seq1` and `seq2`. If you only need to get the length of the lcs, calling this method will be more efficient than calling `lcs()`.");
    m.def("lcs_table", &createLCSTable, "seq1"_a, "seq2"_a,
          "Returns the longest common subsequence (lcs) table from `seq1` and `seq2`.");
    // TODO longest unique subsequence
}
