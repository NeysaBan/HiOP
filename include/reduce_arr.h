#include "cuda_config.h"

void launch_reduce_forward(T *output, const T *input, int N);
void launch_reduce_backward(T *grad_output, int N);