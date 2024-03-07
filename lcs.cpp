#include <torch/extension.h>
#include <vector>


std::vector<std::vector<int>> createLCSTable(
    const torch::Tensor& s1,
    const torch::Tensor& s2
) {
    // Sequence lengths
    const int s1Len = s1.size(0);
    const int s2Len = s2.size(0);

    // TODO longest unique subsequence
    // Building the matrix
    std::vector<std::vector<int>> lcsTable(s1Len + 1, std::vector<int>(s2Len + 1, 0));
    for (int i = 1; i <= s1Len; i++) {
        for (int j = 1; j <= s2Len; j++) {
            if (s1[i - 1].equal(s2[j - 1]))
                lcsTable[i][j] = lcsTable[i - 1][j - 1] + 1;
            else
                lcsTable[i][j] = std::max(lcsTable[i - 1][j], lcsTable[i][j - 1]);
        }
    }

    return lcsTable;
}


int lcsLength(
    const torch::Tensor& s1,
    const torch::Tensor& s2
) {
    std::vector<std::vector<int>> lcsTable = createLCSTable(s1, s2);
    return lcsTable[s1.size(0)][s2.size(0)];
}


std::vector<int> lcs(
    const torch::Tensor& s1,
    const torch::Tensor& s2
) {
    // Sequence lengths
    auto s1Len = s1.size(0);
    auto s2Len = s2.size(0);

    // Zero length
    if (s1Len == 0 || s2Len == 0)
        // return torch::empty({0}, {torch::kInt64});
        return std::vector<int>();

    // Building the matrix
    std::vector<std::vector<int>> lcsTable = createLCSTable(s1, s2);

    int index = lcsTable[s1Len][s2Len];
    std::vector<int> lcsArr(index + 1);
    int i = s1Len, j = s2Len;
    while (i > 0 && j > 0) {
        if (s1[i - 1].equal(s2[j - 1])) {
            lcsArr[index - 1] = s1[i - 1].item<int>();
            i--;
            j--;
            index--;
        }
        else if (lcsTable[i - 1][j] > lcsTable[i][j - 1])
            i--;
        else
            j--;
    }

    // TODO handle pointer memory assignment
    // return torch::tensor(lcsArr.data(), {torch::kInt64});
    return lcsArr;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lcs", lcs, "lcs forward");
    m.def("lcs_length", lcsLength, "lcs length");
    m.def("lcs_table", createLCSTable, "lcs table");
}
