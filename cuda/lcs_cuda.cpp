#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor lcs_cuda_kernel(
    const torch::Tensor& src,
	const torch::Tensor& trg,
	torch::Tensor& result,
	int64_t padToken
);

torch::Tensor lcs_cuda(
    const torch::Tensor& s1,
    const torch::Tensor& s2,
    int64_t padToken
) {

    CHECK_INPUT(s1);
    CHECK_INPUT(s2);

    auto numBatch = s1.size(0);
    at::TensorOptions options(s1.device());
    options = options.dtype(at::ScalarType::Int);
    auto result = at::empty({numBatch, 1}, options);

    return lcs_cuda_kernel(s1, s2, result, padToken);
}

TORCH_LIBRARY_IMPL(lcs, CUDA, m) {
    m.impl("lcs", lcs_cuda);
}