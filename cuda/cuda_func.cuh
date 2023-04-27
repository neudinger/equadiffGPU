#pragma once

#if not defined(_CUDA_FUNC_HH)
#define _CUDA_FUNC_HH

// https://forums.developer.nvidia.com/t/how-to-use-thrust-for-each-with-cuda-streams/177797/9
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include <cstdlib>
#include <algorithm>
#include <time.h>
#include <limits.h>
/*  */
#include <fstream> // std::ofstream, std::ostream_iterator
#include <string>
/*  */
#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/async/for_each.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define not_limit_left (idx % nbr_of_col)
#define not_limit_right ((idx + 1) % nbr_of_col)
#define ROW_MAJOR_IDX(i, j) ((i) * (nbr_of_col)) + (j)

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
inline void checkLast(const char *const file,
                      const int line)
{
    cudaError_t cudaStatus{cudaGetLastError()};
    if (cudaStatus not_eq cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << "\n"
                  << cudaGetErrorString(cudaStatus) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_RT_CALL(val) check((val), #val, __FILE__, __LINE__, __FUNCTION__)
template <typename T = cudaError_t>
inline void check(T cudaStatus,
                  const char *const func,
                  const char *const file,
                  const int line,
                  const char *const fn_name)
{
    if (cudaStatus not_eq cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << "\n"
                  << "ERROR Nb "
                  << "( " << cudaStatus << " )"
                  << " at " << fn_name
                  << cudaGetErrorString(cudaStatus) << " " << func << std::endl;
        exit(cudaStatus);
    }
}

// https://matthewmcgonagle.github.io/blog/2019/01/25/CUDAJacobi

// https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html
// https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
// https://stackoverflow.com/questions/6404992/cuda-block-synchronization
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#grid-synchronization-cg
// https://on-demand.gputechconf.com/gtc/2017/presentation/s7622-Kyrylo-perelygin-robust-and-scalable-cuda.pdf

// https://stackoverflow.com/questions/48547409/can-i-launch-a-cooperative-kernel-without-passing-an-array-of-pointers
// https://gist.github.com/aerosayan/efb994c8538e6c2a18b71f4dbe1ba86a

// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/conjugateGradientMultiBlockCG/conjugateGradientMultiBlockCG.cu
// https://github.com/NVIDIA/cuda-samples/blob/master/Samples/4_CUDA_Libraries/conjugateGradientMultiDeviceCG/conjugateGradientMultiDeviceCG.cu

#define up_p2p (in_top and in not_eq in_top ? in_top[(size_of_chunk - nbr_of_col) + idx] : in[idx - nbr_of_col])
#define down_p2p (in_bottom ? in_bottom[nbr_of_col - (size_of_chunk - idx)] : in[idx + nbr_of_col])
#define not_limit_left_cg (idx % nbr_of_col)
#define not_limit_right_cg ((idx + 1) % nbr_of_col)

__global__ void stencil_cg(double *__restrict__ in,
                           double *__restrict__ out,
                           const size_t nbr_of_col,
                           const size_t offset_end,
                           const size_t domain_size,
                           const size_t iter_max,
                           const size_t size_of_chunk,
                           const double *__restrict__ in_top,
                           const double *__restrict__ in_bottom)

{
    cg::grid_group grid = cg::this_grid();
    auto idx = grid.thread_rank();

    for (size_t it = 0; it < iter_max; it++)
    {
        // Stride Loop
        for (auto idx = grid.thread_rank();
             idx < domain_size;
             idx += grid.size())
        {
            if (idx > nbr_of_col and
                idx < offset_end and
                not_limit_left_cg and
                not_limit_right_cg)
            {
                out[idx] = 0.125 * (up_p2p +
                                    in[idx - 1] + in[idx] + in[idx + 1] +
                                    down_p2p);
            }
        }
        thrust::swap(in, out);
        cg::sync(grid);
    }
}

__global__ void stencil_host_cg(double *__restrict__ in,
                           double *__restrict__ out,
                           const size_t nbr_of_col,
                           const size_t offset_end,
                           const size_t domain_size,
                           const size_t iter_max,
                           const size_t size_of_chunk,
                           const double *__restrict__ in_top,
                           const double *__restrict__ in_bottom)

{
    cg::grid_group grid = cg::this_grid();
    auto idx = grid.thread_rank();

    // Stride Loop
    for (auto idx = grid.thread_rank();
         idx < domain_size;
         idx += grid.size())
    {
        if (idx > nbr_of_col and
            idx < offset_end and
            not_limit_left_cg and
            not_limit_right_cg)
        {
            out[idx] = 0.125 * (up_p2p +
                                in[idx - 1] + in[idx] + in[idx + 1] +
                                down_p2p);
        }
    }
}

// https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
__global__ void stencil(const double *__restrict__ in,
                        double *__restrict__ out,
                        const size_t nbr_of_col,
                        const size_t offset_end,
                        const size_t domain_size,
                        const size_t iter_max,
                        const size_t size_of_chunk = 0,
                        const double *__restrict__ in_top = nullptr,
                        const double *__restrict__ in_bottom = nullptr)

{
    // Maybe change to
    // http://harrism.github.io/hemi/
    // https://developer.nvidia.com/blog/simple-portable-parallel-c-hemi-2/
    const size_t threadRowID = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t threadColId = blockIdx.y * blockDim.y + threadIdx.y;

#define thisSMIdxPosition (threadColId * nbr_of_col + threadRowID)
    // const size_t idx = threadColId * nbr_of_col + threadRowID;

    /* total number of threads in the grid.
    grid_stride_range
    (blockDim.x * gridDim.x) * (blockDim.y * gridDim.y) */
    const size_t thisSMblockSize = (blockDim.x * gridDim.x) *
                                   (blockDim.y * gridDim.y);
    // Stride Loop
    for (size_t idx = thisSMIdxPosition;
         idx < domain_size;
         idx += thisSMblockSize)
    {
        if (idx > nbr_of_col and
            idx < offset_end and
            not_limit_left and
            not_limit_right)
        {
            out[idx] = 0.125 * (up_p2p +
                                in[idx - 1] + in[idx] + in[idx + 1] +
                                down_p2p);
        }
    }
}


// Optimisation
// https://stackoverflow.com/questions/24850442/strided-reduction-by-cuda-thrust
// 
struct stencil_stuct
{
    double *__restrict__ in;
    double *__restrict__ out;
    const size_t nbr_of_col;
    const size_t size_of_chunk = 0;
    const double *__restrict__ in_top = nullptr;
    const double *__restrict__ in_bottom = nullptr;

    const size_t iter_max;

    stencil_stuct(double *__restrict__ in,
                  double *__restrict__ out,
                  const size_t nbr_of_col,
                  const size_t iter_max,
                  const size_t size_of_chunk = 0,
                  const double *__restrict__ in_top = nullptr,
                  const double *__restrict__ in_bottom = nullptr) : in(in),
                                                                    out(out),
                                                                    size_of_chunk(size_of_chunk),
                                                                    in_top(in_top),
                                                                    in_bottom(in_bottom),
                                                                    iter_max(iter_max),
                                                                    nbr_of_col(nbr_of_col){};
    template <typename T>
    __host__ __device__ void operator()(T &idx)
    {
        if (not_limit_left and not_limit_right)
            out[idx] = 0.125 * (up_p2p +
                                in[idx - 1] + in[idx] + in[idx + 1] +
                                down_p2p);
    }
};

#undef not_limit_left
#undef not_limit_right
#undef up_p2p
#undef down_p2p

#endif /*_CUDA_FUNC_HH*/

/* ------------------ */

// #ifndef __CUDACC_EXTENDED_LAMBDA__
// #error "please compile with --expt-extended-lambda"
// #endif
// void saxpy(auto x, auto y, float a, int N)
// {
//     // using namespace thrust;
//     size_t gpuThreshold = 50;
//     auto r = thrust::counting_iterator<std::size_t>(0);

//     auto lambda = [=] __host__ __device__(int i)
//     {
//         y[i] = a * x[i] + y[i] + gpuThreshold;
//     };

//     if (N > gpuThreshold)
//         thrust::for_each(thrust::device, r, r + N, lambda);
//     else
//         thrust::for_each(thrust::host, r, r + N, lambda);
// }

/* ------------------ */

/* Not Working */
/* https://stackoverflow.com/questions/48547409/can-i-launch-a-cooperative-kernel-without-passing-an-array-of-pointers */
// namespace detail
// {

//     template <typename F, typename... Args>
//     void for_each_argument_address(F f, Args &&...args)
//     {
//         [](...) {}((f((void *)&std::forward<Args>(args)), 0)...);
//     }

// } // namespace detail

// template <typename KernelFunction, typename... KernelParameters>
// inline void cooperative_launch(
//     const KernelFunction &kernel_function,
//     cudaLaunchParams launch_configuration,
//     KernelParameters... parameters)
// {
//     void *arguments_ptrs[sizeof...(KernelParameters)];
//     auto arg_index = 0;
//     detail::for_each_argument_address(
//         [&](void *x)
//         { arguments_ptrs[arg_index++] = x; },
//         parameters...);
//     std::cout << (const size_t)arguments_ptrs[3] << std::endl;
//     cudaLaunchCooperativeKernel<KernelFunction>(
//         &kernel_function,
//         launch_configuration.gridDim,
//         launch_configuration.blockDim,
//         arguments_ptrs,
//         launch_configuration.sharedMem,
//         launch_configuration.stream);
// }

/* ------------------------------------- */
