#include "cuda_config.h"
#include <iostream>
#include <cmath>
#include <cuda_fp16.h>


__global__ void gelu_forward_kernel(half *dst, const half *src, int N, 
                                    const half *hl, const half *one, 
                                    const half *alpha, const half *beta){

    // __half sAlpha = __ldca(alpha), sBeta = __ldca(beta),
    //     sHl = __ldca(hl), sOne = __ldca(one);
    __half sAlpha = *(alpha), sBeta = *(beta),
        sHl = *(hl), sOne = *(one);

    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int dataPerIter = gridDim.x * blockDim.x;

    for(int i = gtid ; i < N ; i += dataPerIter){
        __half num = src[i];
        // const float tanhParam = __half2float(sAlpha * 
        //                                         (num + sBeta * num * num * num)
        //                                     );
        // const float tanh_res = tanhf(tanhParam);
        const float tanh_res = tanhf(__half2float(sAlpha * 
                                                (num + sBeta * num * num * num)
                                            ));
        num = sHl * num * (sOne + __float2half(tanh_res));
        __stcg(dst + i, num);
    }
}



void launch_gelu_forward(half *dst, const half *src, int N, int gridSize){
    half *dhl, *done, *dalpha, *dbeta;

    int halfBytes = sizeof(half), floatBytes = sizeof(float);

    cudaMalloc((void**)&dhl, halfBytes);
    cudaMalloc((void**)&done, halfBytes);
    cudaMalloc((void**)&dalpha, floatBytes);
    cudaMalloc((void**)&dbeta, floatBytes);

    const half  hl = __float2half_rn(0.5f),
                one =  __float2half_rn(1.0f),
                // alpha =  __float2half_rn(0.7978845608028654),
                // beta = __float2half_rn(0.044714998453855515);
                alpha =  __float2half_rn(0.7978845608028654),
                beta = __float2half_rn(0.044714998453855515);


    cudaMemcpy(dhl, &hl, halfBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(done, &one, halfBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dalpha, &alpha, halfBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dbeta, &beta, halfBytes, cudaMemcpyHostToDevice);

    dim3 grid(gridSize), block(blockSize);
    gelu_forward_kernel<<<grid, block>>>(dst, src, N,
                                                dhl, done, dalpha, dbeta);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Possibly: exit(-1) if program cannot continue....
    } 
}