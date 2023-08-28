#include <iostream>
#include "cuda_config.h"

// TODO  Warp层面编程

__device__ void warp_sharedMem(volatile T *smem, int tid){
    T tmp = smem[tid];

    /* TODO 
            这里warpSize一般是32, 所以就先都写死是32, 
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

__global__ void reduce_forward_kernel(T *output, const T *input, int N){
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

// __global__ void reduce_forward_kernel(T *output, const T *input, int N){
//     // 一个block可以负责2xthread_num(block)的数据
//     int tid = threadIdx.x;
//     int dataPerBlock = 2 * blockDim.x;
//     int startIdx = blockIdx.x * dataPerBlock + threadIdx.x;
//     int dataCover = gridDim.x * dataPerBlock; // 221184
//     int blockStratIdx = blockIdx.x * dataPerBlock;

//     __shared__ T smem[blockSize]; // 这一句一定要放在外面，否则会反复分配shared mem，会导致内存访问错误
//     // 当前block处理的数据应该是off + blockIdx.x * (2 * blockDim.x) + 2 * blockSize
//     // for(int off = 0 ; off + startIdx < N ; off += dataCover){
//     for(int off = 0 ; off + blockStratIdx + dataPerBlock <= N ; off += dataCover){
//         startIdx += off;

//         smem[tid] = 0;
//         __syncthreads();

//         // 这一步就是把这个block所能处理的所有数据读到shared mem了
//         // 索引是 0 ~ blockSize - 1
//         smem[tid] = input[startIdx] + input[startIdx + blockSize];
//         __syncthreads();

//         for(unsigned int i = blockSize / 2 ; i > warpSize ; i >>= 1){
//             if(tid < i)
//                 smem[tid] += smem[tid + i];
//             __syncthreads();
//         }

//         // 只剩一个warp在干活时,不用__syncthreads()
//         if(tid < warpSize)
//             warp_sharedMem(smem, tid);

//         if(tid == 0)
//             output[blockIdx.x] += smem[0];
//     }
// }

__device__ int WarpShuffle(int sum, int threadNum){
    //__shfl_down_sync：前面的thread向后面的thread要数据
    //__shfl_up_sync: 后面的thread向前面的thread要数据
    //返回前面的thread向后面的thread要的数据，比如__shfl_down_sync(0xffffffff, sum, 16)那就是返回16号线程，17号线程的数据
    //warp内的数据交换不会出现warp在shared memory上交换数据时的不一致现象，无需syncwarp
    if (threadNum >= 32)sum += __shfl_down_sync(0xffffffff, sum, 16); // 0-16, 1-17, 2-18, etc.
    if (threadNum >= 16)sum += __shfl_down_sync(0xffffffff, sum, 8);// 0-8, 1-9, 2-10, etc.
    if (threadNum >= 8)sum += __shfl_down_sync(0xffffffff, sum, 4);// 0-4, 1-5, 2-6, etc.
    if (threadNum >= 4)sum += __shfl_down_sync(0xffffffff, sum, 2);// 0-2, 1-3, 4-6, 5-7, etc.
    if (threadNum >= 2)sum += __shfl_down_sync(0xffffffff, sum, 1);// 0-1, 2-3, 4-5, etc.
    return sum;
}

__global__ void reduce_forward_kernel_warp(T *output, const T *input, int N){
    // 一个block可以负责2xthread_num(block)的数据
    int tid = threadIdx.x;
    int startIdx = blockIdx.x * (2 * blockDim.x) + threadIdx.x;
    int total_thread_num = blockDim.x * gridDim.x;
    // printf("ok-1\n");
    int sum = 0; // 当前线程的私有寄存器,每个线程1个sum寄存器
    for(int i = startIdx ; i < N ; i += total_thread_num)
        sum += input[i];

    __shared__ float warpSums[blockSize / warpSize];
    const int laneIdx = tid % warpSize;
    const int warpIdx = tid / warpSize;
    sum = WarpShuffle(sum, blockSize);
    if(laneIdx == 0)
        warpSums[warpIdx] = sum;
    __syncthreads();

    //至此，得到了每个warp的reduce sum结果
    //接下来，再使用第一个warp(laneId=0-31)对每个warp的reduce sum结果求和
    //首先，把warpsums存入前blockDim.x / WarpSize个线程的sum寄存器中
    //接着，继续warpshuffle
    sum = (tid < blockSize / warpSize) ? warpSums[laneIdx] : 0;
    // Final reduce using first warp
    if (warpIdx == 0) {
        sum = WarpShuffle(sum, blockSize/warpSize); 
    }
    // write result for this block to global mem
    if (tid == 0)
        output[blockIdx.x] = sum;
}

void launch_reduce_forward(T *output, const T *input, int N, int gridSize, bool isBlock){
    dim3 grid(gridSize), block(blockSize);
    if(isBlock)
        reduce_forward_kernel<<<grid, block>>>(output, input, N);
    else{
        reduce_forward_kernel_warp<<<grid, block>>>(output, input, N);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        // Possibly: exit(-1) if program cannot continue....
        } 
    }
        
    // cout<<"forward: "<<*(output)<<endl; // NOTE 这里越界访问可能是因为,output是在显存上的,所以在内存上读不到
}

__global__ void reduce_backward_kernel(T *grad_output, int N){
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int total_threads_num = blockDim.x * gridDim.x;

    for(int i = tid ; i < N ; i += total_threads_num){
        if(i < N)
            grad_output[i] = 1.0;
    }
}

void launch_reduce_backward(T *grad_output, int N, int gridSize){
    dim3 grid(gridSize), block(blockSize);
    reduce_backward_kernel<<<grid, block>>>(grad_output, N);
}