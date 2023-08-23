#include "cuda_config.h"

__device__ void warp_sharedMem(volatile T *smem, int tid){
    T tmp = smem[tid];

    /* TODO 这里warpSize一般是32, 所以就先都写死是32, 
            不然warpSize >>= 1, 应该会对性能造成一些损耗
    */

    tmp += smem[tid + 32];
    __syncwarp();
    smem[tid] = tmp;
    __syncwarp();

    tmp += smem[tid + 16];
    __syncwarp();
    smem[tid] = tmp;
    __syncwarp();

    tmp += smem[tid + 8];
    __syncwarp();
    smem[tid] = tmp;
    __syncwarp();

    tmp += smem[tid + 4];
    __syncwarp();
    smem[tid] = tmp;
    __syncwarp();

    tmp += smem[tid + 2];
    __syncwarp();
    smem[tid] = tmp;
    __syncwarp();

    tmp += smem[tid + 1];
    __syncwarp();
    smem[tid] = tmp;
    __syncwarp();
}


__global__ void reduce_kernel(T *output, const T *input, int N){
    // 一个block可以负责2xthread_num(block)的数据
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int startIdx = blockIdx.x * (2 * blockDim.x) + threadIdx.x;

    __shared__ T smem[blockSize];
    // 这一步就是把这个block所能处理的所有数据读到shared mem了
    // 索引是 0 ~ blockSize - 1
    smem[tid] = input[startIdx] + input[startIdx + blockSize];
    __syncthreads();

    for(unsigned int i = blockSize / 2 ; i > warpSize ; i >>= 1){
        if(tid < i)
            smem[tid] += smem[tid + i];
        __syncthreads();
    }

    // 只剩一个warp在干活时,不用__syncthreads()
    if(tid < warpSize)
        warp_sharedMem(smem, tid);

    if(tid == 0)
        output[blockIdx.x] = smem[0];
}


void launch_reduce(T *output, const T *input, int N){
    int gridSize = (blockSize + N - 1) / blockSize;
    dim3 grid(gridSize), block(blockSize);
    reduce_kernel<<<grid, block>>>(output, input, N);
}