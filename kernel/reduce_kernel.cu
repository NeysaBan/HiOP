#include "cuda_config.h"
#include <iostream>
using namespace std;

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


__global__ void reduce_forward_kernel(T *output, const T *input, int N, int gridSize){
    // 一个block可以负责2xthread_num(block)的数据
    int tid = threadIdx.x;
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

void launch_reduce_forward(T *output, const T *input, int N, int gridSize){
    dim3 grid(gridSize), block(blockSize);
    reduce_forward_kernel<<<grid, block>>>(output, input, N, gridSize);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
    } 
    // cout<<"forward: "<<*(output)<<endl; // BUG 这里越界访问可能是因为,output是在显存上的,所以在内存上读不到
}

__global__ void reduce_backward_kernel(T *grad_output, int N, int gridSize){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads_num = blockDim.x * gridDim.x;

    for(int i = tid ; i < N ; i += total_threads_num){
        if(i < N)
            grad_output[i] = 1.0;
    }
}

void launch_reduce_backward(T *grad_output, int N, int gridSize){
    dim3 grid(gridSize), block(blockSize);
    reduce_backward_kernel<<<grid, block>>>(grad_output, N, gridSize);
}