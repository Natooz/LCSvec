#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAFunctions.h>

namespace {

// TODO: this can probabliy be parallelized using a gpu
template <typename scalar_t>
__device__ int64_t handlePadLen(scalar_t* str, int64_t strLen, int64_t padToken) {
    for (int i=0; i < strLen; i++)
	    if (str[i] == padToken) return i;
    return strLen;
}


template <typename scalar_t>
__global__ void lcs_cuda_kernel(
    scalar_t* const __restrict__ s1,
    scalar_t* const __restrict__ s2,
    int* __restrict__ result,
    int64_t s1Len,
    int64_t s2Len,
    int64_t padToken
) {

    const int batch = blockIdx.x;

    auto s1Batch = s1 + batch * s1Len;
    auto s2Batch = s2 + batch * s2Len;
    auto result_ = result + batch;

    // Handle padding
    s1Len = handlePadLen(s1Batch, s1Len, padToken);
    s2Len = handlePadLen(s2Batch, s2Len, padToken);

    // Zero length
    if (s1Len == 0 || s2Len == 0) {
        *result = torch::empty({0}, {torch::kInt64});
        return;
    }

    // Filling 0's in the matrix
    auto lcsTable = torch::zeros({s1Len, s2Len});

    // Building the matrix in bottom-up way
    for (int i = 1; i <= s1Len; i++) {
        for (int j = 1; j <= s2Len; j++) {
            if (s1[i - 1].equal(s2[j - 1])) {
                lcsTable[i][j] = lcsTable[i - 1][j - 1] + 1;
            } else if ((lcsTable[i - 1][j] >= lcsTable[i][j - 1]).item<bool>()) {
                lcsTable[i][j] = lcsTable[i - 1][j];
            } else {
                lcsTable[i][j] = lcsTable[i][j - 1];
            }
        }
    }

    int index = lcsTable[s1Len][s2Len].item<int>();
    int lcsArr[index + 1];
    int i = s1Len, j = s2Len;
    while (i > 0 && j > 0) {
        if (s1[i - 1].equal(s2[j - 1])) {
            lcsArr[index - 1] = s1[i - 1];
            i--;
            j--;
            index--;
        }
        else if ((lcsTable[i - 1][j] > lcsTable[i][j - 1]).item<bool>())
            i--;
        else
            j--;
    }

    *result = torch::tensor(lcsArr, {torch::kInt64});
    delete(lcsArr);
    delete(lcsTable);
}

}


torch::Tensor lcs_cuda_kernel(
    const torch::Tensor& s1,
    const torch::Tensor& s2,
    torch::Tensor& result,
    int64_t padToken) {

    const auto numBatch = s1.size(0);
    const auto s1Len = s1.size(1);
    const auto s2Len = s2.size(1);

    const int threads = 1;
    const dim3 blocks(numBatch);

    // see https://github.com/pytorch/pytorch/issues/21819
    // to avoid random errors when executing on cuda:1 we need to set the device manually
    c10::cuda::set_device(static_cast<c10::DeviceIndex>(s1.device().index()));

    AT_DISPATCH_ALL_TYPES(
        s1.scalar_type(),
        "lcs_cuda",
        [&] {
             lcs_cuda_kernel<scalar_t><<<numBatch, threads>>>(
                 s1.data<scalar_t>(),
                 s2.data<scalar_t>(),
                 result.data<int>(),
                 s1Len,
                 s2Len,
	             padToken
	         );
        }
    );

    return result;
}
