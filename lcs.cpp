#include <torch/extension.h>


torch::Tensor lcs(
    torch::Tensor const s1,
    torch::Tensor const s2
) {
    // Checks
    int64_t s1Dims = s1.ndimension();
    int64_t s2Dims = s2.ndimension();
    TORCH_CHECK(s1Dims == 1,
                "lcs: Expect 1D Tensor, got: ",
                s1.sizes());

    TORCH_CHECK(s2Dims == 1,
                "lcs: Expect 1D Tensor, got: ",
                s2.sizes());

    TORCH_CHECK(s1.device() == s2.device(),
                "lcs: tensors must be on the same device, got ",
                "s1 on device ", s1.device(),
                " and s2 on device ", s2.device());

    // Sequence lengths
    auto s1Len = s1.size(0);
    auto s2Len = s2.size(0);

    // Zero length
    if (s1Len == 0 || s2Len == 0)
        return torch::empty({0}, {torch::kInt64});

    // Building the matrix
    int lcsTable[s1Len + 1][s2Len + 1];
    for (int i = 0; i < s1Len; i++) {
        for (int j = 0; j < s2Len; j++) {
            if (i == 0 || j == 0)
                lcsTable[i][j] = 0;
            else if (s1[i].equal(s2[j]))
                lcsTable[i + 1][j + 1] = lcsTable[i][j] + 1;
            else
                lcsTable[i + 1][j + 1] = std::max(lcsTable[i][j + 1], lcsTable[i + 1][j]);
        }
    }

    int index = lcsTable[s1Len][s2Len];
    int lcsArr[index + 1];
    int i = s1Len, j = s2Len;
    while (i > 0 && j > 0) {
        if (s1[i - 1].equal(s2[j - 1])) {
            lcsArr[index - 1] = s1[i - 1].item<int>();  // TODO detect indexes?
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
    return torch::tensor(lcsArr, {torch::kInt64});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lcs", lcs, "lcs forward");
}
