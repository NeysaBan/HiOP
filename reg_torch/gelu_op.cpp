#include <torch/extension.h>
#include "gelu.h"


void gelu_forward(torch::Tensor &dst,
                    const torch::Tensor &src,
                    int64_t N, int64_t gridSize){
    launch_gelu_forward((half *)dst.data_ptr(), 
                        (const half *)src.data_ptr(), 
                        N, gridSize);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def(
        "gelu_forward",
        &gelu_forward,
        " add reduce forward kernel function"
    );
}

TORCH_LIBRARY(gelu, m) {
    m.def("gelu_forward", gelu_forward);
}