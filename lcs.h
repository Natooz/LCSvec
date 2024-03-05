#pragma once
#include <torch/extension.h>

torch::Tensor lcs(
    const torch::Tensor& src,
    const torch::Tensor& trg,
    int64_t padToken
);