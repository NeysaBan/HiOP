#include <iostream>
#include "cuda_config.h"
#include <iostream>
using namespace std;
/*
    @nRes 有几个大于0的数字
*/
__global__ void copyif_forward_kernel(T *dst, const T *src, int *nRes, int N){
    // 一个线程，判断一个数字
    int tid = threadIdx.x;
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    int dataPerIterPerBlock = gridDim.x * blockDim.x;

    // 使用一个变量来记录本次iter这个block中处理的所有数据，满足要求的有多少
    // 这样就防止了每个线程都去访问global memory的nRes（线程争用 && 访问延迟）
    __shared__ int resPerIterThisBlock;

    for(int i = gtid ; i < N ; i += dataPerIterPerBlock){
        if(tid == 0)
            resPerIterThisBlock = 0;
        __syncthreads();

        T num;
        int pos;

        // if(i < N){
            num = src[i];
            if(num > 0)
                pos = atomicAdd(&resPerIterThisBlock, 1);
        // }
        __syncthreads();

        // 走到这里，以block的层面来看，所有线程已经全部判断完自己读到的数字是否符合要求
        if(tid == 0)
            resPerIterThisBlock = atomicAdd(nRes, resPerIterThisBlock); // 现在resPerIterThisBlock存储的是这个block向global memory存储符合条件的数字时的起始位置
        __syncthreads();

        if(num > 0){
            pos += resPerIterThisBlock; // 这个线程的数字存储到global memory的哪个位置
            dst[pos] = num;
        }
        __syncthreads();
    }
}

int64_t launch_copyif_forward(T *dst, const T *src, int N, int gridSize){
    int *dnRes, nRes = 0; // 这里nRes如果定义为int类型，就不能用指针，因为返回的时候隐式类型转换为int64_t，会报越界访问
    cudaMalloc((void **)&dnRes, sizeof(int));
    dim3 grid(gridSize), block(blockSize);
    copyif_forward_kernel<<<grid, block>>>(dst, src, dnRes, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
    } 
    else{
        cudaMemcpy(&nRes, dnRes, sizeof(int), cudaMemcpyDeviceToHost);
    }
    return nRes;
}