#include <torch/extension.h>
#include "copyif_pos.h"

// #define CHECK_CUDA(x) \
//     TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
// #define CHECK_CONTIGUOUS(x) \
//     TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
// #define CHECK_INPUT(x) \
//     CHECK_CUDA(x);       \
//     CHECK_CONTIGUOUS(x)

int64_t copyif_forward(torch::Tensor dst,  const torch::Tensor src,
                    int64_t N, int64_t gridSize){
    // CHECK_INPUT(src);
    return launch_copyif_forward((float *)dst.data_ptr(), (const float *)src.data_ptr(),
                                    N, gridSize);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def(
        "copyif_forward",
        &copyif_forward,
        " add copyif forward kernel function"
    );
}

TORCH_LIBRARY(copyif_pos, m) {
    m.def("copyif_forward", copyif_forward);
}