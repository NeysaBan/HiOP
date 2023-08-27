#include <cuda_fp16.h>

void launch_gelu_forward(half *dst, const half *src, int N, int gridSize);