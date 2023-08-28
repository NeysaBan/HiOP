#include "cuda_config.h"

void launch_reduce_forward(T *output, const T *input, int N, int gridSize, bool isBlock);
void launch_reduce_backward(T *grad_output, int N, int gridSize);