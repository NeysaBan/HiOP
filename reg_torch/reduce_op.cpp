#include <torch/extension.h>
#include "reduce_arr.h"

void torch_launch_reduce(torch::Tensor &output,
                        const torch::Tensor &input,
                        int64_t N){ // HACK
    launch_reduce((float *)output.data_ptr(), (const float *)input.data_ptr(), N);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def(
        "torch_launch_reduce",
        &torch_launch_reduce,
        " add reduce kernel function"
    );
}

TORCH_LIBRARY(reduce_arr, m) { // HACK 
    m.def("torch_launch_reduce", torch_launch_reduce);
}