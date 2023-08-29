#include "cuda_config.h"
#include <iostream>
#include <cmath>
#include <cuda_fp16.h>

__global__ void gelu_forward_kernel(half *dst, const half *src, int N){
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int dataPerIter = gridDim.x * blockDim.x;

    /*
    * 使用__float2half_rn得到的
    * alpha =  __float2half_rn(0.7978845608028654),
    * beta = __float2half_rn(0.044714998453855515);
    */
    __half alpha = 0.797852, beta = 0.0447083, hl = 0.5, one = 1.0; 
    // __half alpha = 0.7978845, beta = 0.0447150, hl = 0.5, one = 1.0; // 自己近似的
    for(int i = gtid ; i < N ; i += dataPerIter){
        __half num = src[i];
        const float tanh_res = tanhf(__half2float(alpha * 
                                                (num + beta * num * num * num)
                                            ));
        num = hl * num * (one + __float2half(tanh_res));
        __stcg(dst + i, num);
    }
}

void launch_gelu_forward(half *dst, const half *src, int N, int gridSize){
    dim3 grid(gridSize), block(blockSize);
    gelu_forward_kernel<<<grid, block>>>(dst, src, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
    } 
}

__global__ void gelu_backward_kernel(half *grad_src, const half* grad_out, const half *src, int N){
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int dataPerIter = gridDim.x * blockDim.x;

    __half alpha = 0.797852, beta = 0.0447083, hl = 0.5, one = 1.0, gamma = 0.1341249;
    for(int i = gtid ; i < N ; i += dataPerIter){
        __half num = src[i];
        const float tanh_res = tanhf(__half2float(alpha * 
                                                (num + beta * num * num * num)
                                                            )) ;
        __half half_tanh_res = __float2half(tanh_res);
        __half tmp1 = hl * (one + half_tanh_res);
        __half tmp2 = one - half_tanh_res;
        num = tmp1 * ( one + num * tmp2 * (alpha + (one + gamma * num * num)));
        __stcg(grad_src + i, grad_out[i] * num);
        // __stcg(grad_src + i, num);
    }
}


void launch_gelu_backward(half *grad_src, const half* grad_out, const half *src, int N, int gridSize){
    dim3 grid(gridSize), block(blockSize);
    gelu_backward_kernel<<<grid, block>>>(grad_src, grad_out, src, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
    } 
}


/************************
*   近似计算(用sigmoid)的方式过不了测试
*************************/

// __device__ __forceinline__ float sigmoid(float x){
//     return 1.0 / (1 + expf(-x));
// }

// __global__ void gelu_close_forward_kernel(half *dst, const half *src, int N){
//     int gtid = blockIdx.x * blockDim.x + threadIdx.x;
//     int dataPerIter = gridDim.x * blockDim.x;

//     for(int i = gtid ; i < N ; i += dataPerIter){
//         __half num = src[i];
//         num = num * __float2half_rn(sigmoid(1.702 * __half2float(num)));
//         __stcg(dst + i, num);
//     }
// }

// void launch_gelu_forward(half *dst, const half *src, int N, int gridSize){
//     dim3 grid(gridSize), block(blockSize);
//     gelu_close_forward_kernel<<<grid, block>>>(dst, src, N);
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         printf("CUDA Error: %s\n", cudaGetErrorString(err));
//     // Possibly: exit(-1) if program cannot continue....
//     } 
// }
