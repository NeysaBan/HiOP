#include "cuda_config.h"

int64_t launch_copyif_forward(T *dst, const T *src, int N, int gridSize);