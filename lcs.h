#pragma once
#include <torch/extension.h>

torch::Tensor lcs(
    const torch::Tensor& seq1,
    const torch::Tensor& seq2,
    int64_t padToken
);