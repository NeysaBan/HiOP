#include "cuda_config.h"

template <int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
    T val[VecSize];
};

__inline__ __device__ T Exp(T x);
__inline__ __device__ T Inf();
__inline__ __device__ T Div(T a, T b);

__inline__ __device__ float Exp(float x) {
  //return __expf(x);//fast math
    return exp(x);
}

__inline__ __device__ float Inf() {
    return 10000000000;
}

__inline__ __device__ float Div(float a, float b) {
  //return __fdividef(a, b);//fast math
    return a / b;
}

template<int VecSize>
__device__ void load(const float* src, float* dst, int row, const int row_size, const int col) {
    using VecType = VectorType<VecSize>;
    const int offset = (row * row_size + col) / VecSize;
    *reinterpret_cast<VecType*>(dst) = *(reinterpret_cast<VecType*>(const_cast<float*>(src)) + offset);
}


template<int VecSize>
__device__ void store(float* dst, float* src, int row, const int row_size, const int col) {
    using VecType = VectorType<VecSize>;
    const int offset = (row * row_size + col) / VecSize;
    *(reinterpret_cast<VecType*>(dst) + offset) = *reinterpret_cast<VecType*>(src);
}

struct SumOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

struct MaxOp {
    __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<class ReductionOp, int warp_width>
__inline__ __device__ T WarpReduce(T val) {
    for (int mask = warp_width / 2; mask > 0; mask /= 2) {
        // you can change L61 with __shfl_down_sync like 6_warp_level_reduce and see performance change
        val = ReductionOp()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<int pack_size, int cols_per_thread,
        int warp_width, int rows_per_thread>
__global__ void softmax_forward_kernel(float* dst, const float* src, const int rows, const int cols) {
    
    constexpr int num_packs = cols_per_thread / pack_size; // 把线程处理的向量分成几个块
    assert(cols <= cols_per_thread * warp_width);
    float buf[rows_per_thread][cols_per_thread];
    const int global_warp_id = blockIdx.y * blockDim.y + threadIdx.y; // 一行是一个warp，threadIdx.y： 行号
    const int num_global_warp = gridDim.y * blockDim.y;
    const int lane_id = threadIdx.x;
    const int step = num_global_warp * rows_per_thread;

    /*
    *   warp level ： 一个warp干多行的活
    */
    for (int row = global_warp_id * rows_per_thread; row < rows; row += step) {
        float thread_max[rows_per_thread];

        /*
        * thread level 一个线程一次处理 1 * (1024 / 32) 的数据
        */
        for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
            thread_max[row_id] = -Inf();
            float* row_buf = buf[row_id];

            /*
            * local vector level
            */
            for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
                const int pack_offset = pack_id * pack_size;
                const int col = (pack_id * warp_width + lane_id) * pack_size;
                if (col < cols) {
                // load (row+row_id, col) data from src to reg row_buf
                    load<pack_size>(src, row_buf + pack_offset, row + row_id, rows, col);

                    for (int i = 0; i < pack_size; ++i) { // 不断去比较最大值
                        thread_max[row_id] = max(thread_max[row_id], row_buf[pack_offset + i]);
                    }
                } else {
                    for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = -Inf(); }
                }
            }
        }

        /*
        * 求warp层面最大值
        */
        float warp_max[rows_per_thread];

        for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
            warp_max[row_id] = WarpReduce<MaxOp, warp_width>(thread_max[row_id]);
        }

        /*
        * thread 求和自己负责数据区域的每一行的sum 和 exp
        */
        float thread_sum[rows_per_thread];

        for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
            thread_sum[row_id] = 0;
            float* row_buf = buf[row_id];

            /*
            * 求指数
            */
            for (int i = 0; i < cols_per_thread; ++i) {
                row_buf[i] = Exp(row_buf[i] - warp_max[row_id]);
                thread_sum[row_id] += row_buf[i];
            }
        }

        /*
        * warp 求和自己负责的每一行的sum
        */
        float warp_sum[rows_per_thread]; // 寄存器

        for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
            warp_sum[row_id] = WarpReduce<SumOp, warp_width>(thread_sum[row_id]);
        }

        /*
        *   thread：求出自己负责区域的数据的最后结果
        */
        for (int row_id = 0; row_id < rows_per_thread; ++row_id) {
            float* row_buf = buf[row_id];

            for (int i = 0; i < cols_per_thread; ++i) {
                row_buf[i] = Div(row_buf[i], warp_sum[row_id]);
            }

            /*
            * 存入
            */
            for (int i = 0; i < num_packs; ++i) {
                const int col = (i * warp_width + lane_id) * pack_size;
                if (col < cols) {
                    store<pack_size>(dst, row_buf + i * pack_size, row + row_id, rows, col);
                }
            }
        }
    }
}


void launch_softmax_forward(float *dst, const float *src, 
                            int row, int col,
                            int block_num){
    dim3 Grid(1, block_num), Block(thread_x_cnt, thread_y_cnt);
    softmax_forward_kernel<1, 1024 / 32, 32, 1><<<Grid, Block>>>(   dst, src, 
                                                                    row, col);
}