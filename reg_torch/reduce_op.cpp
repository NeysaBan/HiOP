#include <torch/extension.h>
#include "reduce_arr.h"

#include <iostream>
using namespace std;

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);       \
    CHECK_CONTIGUOUS(x)

void reduce_forward(torch::Tensor &output,
                        const torch::Tensor &input,
                        int64_t N, int64_t gridSize){
    CHECK_INPUT(input);
    launch_reduce_forward((float *)output.data_ptr(), 
                            (const float *)input.data_ptr(), 
                            N, gridSize);
}

void reduce_backward(torch::Tensor &grad_output,
                        int64_t N, int64_t gridSize){
    launch_reduce_backward((float *)grad_output.data_ptr(), N, gridSize);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def(
        "reduce_forward",
        &reduce_forward,
        " add reduce forward kernel function"
    );
    m.def(
        "reduce_backward",
        &reduce_backward,
        " add reduce backward kernel function"
    );

}

TORCH_LIBRARY(reduce_arr, m) {
    m.def("reduce_forward", reduce_forward);
    m.def("reduce_backward", reduce_backward);
}