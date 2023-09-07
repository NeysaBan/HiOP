#include <torch/extension.h>
#include "softmax.h"

void softmax_forward(   torch::Tensor &dst, 
                        const torch::Tensor &src, 
                        int64_t row, int64_t col,
                        int64_t block_num){
    launch_softmax_forward((float *)dst.data_ptr(), 
                            (const float *)src.data_ptr(), 
                            row, col,
                            block_num);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def(
        "softmax_forward",
        &softmax_forward,
        " add reduce forward kernel function"
    );
}

TORCH_LIBRARY(softmax, m) {
    m.def("softmax_forward", softmax_forward);
}