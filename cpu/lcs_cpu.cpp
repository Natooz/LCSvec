#include <torch/extension.h>


namespace {

template <typename scalar_t>
int64_t handlePadLen(scalar_t* str, int64_t strLen, int64_t padToken) {
    for (int i=0; i < strLen; i++)
	    if (str[i] == padToken) return i;

    return strLen;
}


template <typename scalar_t>
torch::Tensor lcsSinglePairSeqs(
    const torch::Tensor& s1,
    const torch::Tensor& s2,
    int64_t s1Len,
    int64_t s2Len,
    int64_t padToken)
{
    // Handle padding
    s1Len = handlePadLen(s1, s1Len, padToken);
    s2Len = handlePadLen(s2, s2Len, padToken);

    // Filling 0's in the matrix
    auto lcs_table = torch::zeros(s1Len, s2Len);

    // Building the matrix in bottom-up way
    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (s1[i - 1] == s2[j - 1]) {
                lcs_table[i][j] = lcs_table[i - 1][j - 1] + 1;
            } else if (lcs_table[i - 1][j] >= lcs_table[i][j - 1]) {
                lcs_table[i][j] = lcs_table[i - 1][j];
            } else {
                lcs_table[i][j] = lcs_table[i][j - 1];
            }
        }
    }

    int index = lcs_table[m][n]; // TODO fix here
    char lcs_[index + 1];
    lcs_[index] = '\0';

    int i = m, j = n;
    while (i > 0 && j > 0) {
    if (s1[i - 1] == s2[j - 1]) {
        lcs_[index - 1] = s1[i - 1];
        i--;
        j--;
        index--;
    }

    else if (lcs_table[i - 1][j] > lcs_table[i][j - 1])
        i--;
    else
        j--;
    }

    return lcs_
}


template <typename scalar_t>
static void lcsParallel(
    scalar_t* const s1,
    scalar_t* const s2,
    int32_t* result,
    int64_t s1Len,
    int64_t s2Len,
    int64_t numBatch,
    int64_t padToken) {

    at::parallel_for(0, numBatch, 0, [&](int64_t start, int64_t end) {
        for (const auto batch : c10::irange(start, end)) {
            lcsSinglePairSeqs<scalar_t>(
                s1 + batch * s1Len,
                s2 + batch * s2Len,
                result + batch,
                s1Len,
                s2Len,
		        padToken
            );
        }
    });
}
}


torch::Tensor lcs_cpu(
    const torch::Tensor& s1,
    const torch::Tensor& s2,
    int64_t padToken){

    auto numBatch = s1.size(0);
    auto s1Len = s1.size(1);
    auto s2Len = s2.size(1);

    at::TensorOptions options(s1.device());
    options = options.dtype(at::ScalarType::Int);
    auto result = at::empty({numBatch, 1}, options);

    AT_DISPATCH_ALL_TYPES(
        s1.scalar_type(),
        "lcs_cpu",
        [&] {
            lcsParallel<scalar_t>(
            s1.data_ptr<scalar_t>(),
            s2.data_ptr<scalar_t>(),
            result.data_ptr<int32_t>(),
            s1Len,
            s2Len,
            numBatch,
	        padToken
          );
        }
    );

    return result;
}



TORCH_LIBRARY_IMPL(lcs, CPU, m) {
    m.impl("lcs", lcs_cpu);
}
