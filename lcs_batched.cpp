
#include "lcs.h"

torch::Tensor lcs(
    const torch::Tensor& s1,
    const torch::Tensor& s2,
    int64_t padToken)
{
    int64_t s1Dims = s1.ndimension();
    int64_t s2Dims = s2.ndimension();

    // TODO make it work on any number of dimensions and add a `dim` argument
    // TODO make it work with an unlimited number of sequences, and `pad_token` as kwarg
    TORCH_CHECK(s1Dims == 2 || s1Dims == 1,
                "lcs: Expect 1D or 2D Tensor, got: ",
                s1.sizes());

    TORCH_CHECK(s2Dims == 2 || s2Dims == 1,
                "lcs: Expect 1D or 2D Tensor, got: ",
                s2.sizes());

    TORCH_CHECK(s1Dims == s2Dims,
                "lcs: Expect tensors to have the same number of dimensions");

    TORCH_CHECK(s1.device() == s2.device(),
                "tensors must be on the same device, got ",
                "s1 on device ", s1.device(),
                " and s2 on device ", s2.device());

    auto s1_ = s1;
    auto s2_ = s2;
    if (s1Dims == 1)
    {
        s1_ = s1_.reshape({1, s1_.size(0)});
        s2_ = s2_.reshape({1, s2_.size(0)});
    }

    // must have the same shapes except for the processed dimension
    TORCH_CHECK(s1_.size(0) == s2_.size(0),
	        "lcs: expected tensors to have same batch size");

    // dispatch
    static auto op = torch::Dispatcher::singleton()
        .findSchemaOrThrow("lcs::lcs", "")
        .typed<decltype(lcs)>();
    return op.call(s1_, s2_, padToken);
}


TORCH_LIBRARY(lcs, m) {
    m.def("lcs(Tensor seq1, Tensor seq2, int padToken) -> Tensor");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lcs", &lcs, "lcs forward");
}
