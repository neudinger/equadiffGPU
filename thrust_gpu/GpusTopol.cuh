#pragma once

#if not defined(_GPUS_TOPOL)
#define _GPUS_TOPOL

#include <omp.h>
#include <type_traits> // std::remove_pointer

#define xstr(s) str(s)
#define str(s) #s

// ---------- CUDA ---------- //

#pragma message("__CUDA_ARCH__ " xstr(__CUDA_ARCH__))
#pragma message("__NVCC__ " xstr(__NVCC__))
#pragma message("__CUDACC__ " xstr(__CUDACC__))

#include <cuda.h>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <device_launch_parameters.h>

#include <thrust/for_each.h>
#include <thrust/async/for_each.h>
#include <thrust/system/cuda/execution_policy.h>

// ---------- CUDA SAFE CALL ---------- //
// hipError_t
#include <iostream>
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

#define CUDA_RT_CALL(val) check((val), #val, __FILE__, __LINE__)
template <typename T = cudaError_t>
inline void check(T cudaStatus,
                  const char *const func,
                  const char *const file,
                  const int line)
{
    if (cudaStatus not_eq cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << "\n"
                  << "ERROR Nb "
                  << "( " << cudaStatus << " )"
                  << cudaGetErrorString(cudaStatus) << " " << func << std::endl;
        exit(cudaStatus);
    }
}
// ---------- CUDA SAFE CALL ---------- //

// #if defined(__NVCC__) or defined(__CUDACC__)
// #if defined(__HCC__) or (defined(__clang__) && defined(__HIP__))
// #define __HIP_PLATFORM_HCC__
// #endif

// ------------------------------------------------------------ //

bool canAccessPeer(const int &id_top,
                   const int &id_curent,
                   const int &id_bottom);

// ------------------------------------------------------------ //

template <typename PTR_TYPE_COMPUTE>
struct GpuManager
{
    PTR_TYPE_COMPUTE device_memory;
    // hipStream_t
    cudaStream_t compute_stream;

    int id_top;
    int id;
    int id_bottom;

    size_t nbr_of_row_local;
    size_t nbr_of_col;
    size_t segment_size_device;

    GpuManager(size_t rows,
               size_t nbr_of_col,
               size_t num_devices) : nbr_of_col(nbr_of_col)
    {
        // OMP Thread ID and Device ID is the same
        id = omp_get_thread_num();
        // hipSetDevice
        CUDA_RT_CALL(cudaSetDevice(id));
        // hipSetDeviceFlags // hipDeviceScheduleSpin
        CUDA_RT_CALL(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        // hipFree
        CUDA_RT_CALL(cudaFree(0));
        CHECK_LAST_CUDA_ERROR();

        int row_local_low = rows / num_devices;
#define row_local_high (row_local_low + 1)
#define num_ranks_low (num_devices * row_local_low + num_devices - rows)
        nbr_of_row_local = id < num_ranks_low ? row_local_low : row_local_high;

        segment_size_device = nbr_of_col * nbr_of_row_local;

        id_top = id > 0 ? id - 1 : 0;
        id_bottom = id == num_devices - 1 ? num_devices - 1 : id + 1;
    }

    ~GpuManager() {}
};

template <typename PTR_TYPE_COMPUTE>
struct GPUsTopol
{

private:
    // typedef typename std::remove_pointer<PTR_TYPE_COMPUTE>::type TYPE_COMPUTE;
    using TYPE_COMPUTE = std::remove_pointer_t<PTR_TYPE_COMPUTE>;

    std::allocator<PTR_TYPE_COMPUTE> alloc_array_ptr;
    // hipEvent_t
    std::allocator<cudaEvent_t> alloc_event;

public:
    PTR_TYPE_COMPUTE *array_ptr_devices[2];
    cudaEvent_t *event_compute_done[2];

    int num_devices;

    void send_memory_on_devices(GpuManager<PTR_TYPE_COMPUTE> &curent_gpu,
                                PTR_TYPE_COMPUTE host)
    {
        CUDA_RT_CALL(cudaMalloc(&(curent_gpu.device_memory), curent_gpu.segment_size_device * sizeof(TYPE_COMPUTE) * 2));
        // CUDA_RT_CALL(cudaMemset(curent_gpu.device_memory, 0, curent_gpu.segment_size_device * sizeof(TYPE_COMPUTE) * 2));
        CHECK_LAST_CUDA_ERROR();

        array_ptr_devices[0][curent_gpu.id] = curent_gpu.device_memory;
        array_ptr_devices[1][curent_gpu.id] = curent_gpu.device_memory + curent_gpu.segment_size_device;

        // hipMemcpy
        CUDA_RT_CALL(cudaMemcpy(array_ptr_devices[0][curent_gpu.id],
                                host + (curent_gpu.id * curent_gpu.segment_size_device),
                                curent_gpu.segment_size_device * sizeof(TYPE_COMPUTE),
                                cudaMemcpyHostToDevice));
        // hipMemcpyAsync
        // Async
        CUDA_RT_CALL(cudaMemcpyAsync(array_ptr_devices[1][curent_gpu.id],
                                     array_ptr_devices[0][curent_gpu.id],
                                     curent_gpu.segment_size_device * sizeof(TYPE_COMPUTE),
                                     cudaMemcpyDeviceToDevice));
        //  curent_gpu.compute_stream
        CHECK_LAST_CUDA_ERROR();
        // hipGetLastError
        // hipDeviceSynchronize
        CUDA_RT_CALL(cudaDeviceSynchronize());
        // hipStreamCreate
        CUDA_RT_CALL(cudaStreamCreate(&curent_gpu.compute_stream));
        // hipEventCreateWithFlags
        CUDA_RT_CALL(cudaEventCreateWithFlags(&event_compute_done[0][curent_gpu.id],
                                              cudaEventDisableTiming));
        //   cudaEventCreateWithFlags
        CUDA_RT_CALL(cudaEventCreateWithFlags(&event_compute_done[1][curent_gpu.id],
                                              cudaEventDisableTiming));
        CHECK_LAST_CUDA_ERROR();
        /* May not be usefull */
        // CUDA_RT_CALL(cudaEventSynchronize(event_compute_done[0][curent_gpu.id]));
        // CUDA_RT_CALL(cudaEventSynchronize(event_compute_done[1][curent_gpu.id]));
        // CUDA_RT_CALL(cudaStreamSynchronize(curent_gpu.compute_stream));
        /* ------------------ */
        // hipDeviceSynchronize
        CUDA_RT_CALL(cudaDeviceSynchronize());
        CHECK_LAST_CUDA_ERROR();
    }

    void compute(GpuManager<PTR_TYPE_COMPUTE> &curent_gpu,
                 StencilOpP2P<TYPE_COMPUTE> &stencilOpFunctor,
                 const size_t &iter)
    {

#define iter_curr (iter % 2)
#define iter_prev ((iter + 1) % 2)
#define iter_next iter_prev

        // hipStreamWaitEvent
        CUDA_RT_CALL(cudaStreamWaitEvent(curent_gpu.compute_stream,
                                         event_compute_done[iter_prev][curent_gpu.id_top], 0));
        // hipStreamWaitEvent
        CUDA_RT_CALL(cudaStreamWaitEvent(curent_gpu.compute_stream,
                                         event_compute_done[iter_prev][curent_gpu.id_bottom], 0));
        CHECK_LAST_CUDA_ERROR();
        stencilOpFunctor.in_top = array_ptr_devices[iter_curr][curent_gpu.id_top];
        stencilOpFunctor.in = array_ptr_devices[iter_curr][curent_gpu.id];
        stencilOpFunctor.in_bottom = array_ptr_devices[iter_curr][curent_gpu.id_bottom];
        stencilOpFunctor.out = array_ptr_devices[iter_next][curent_gpu.id];

#define is_first_device (curent_gpu.id == 0)
#define is_last_device (curent_gpu.id == num_devices - 1)

        // thrust::hip
        // https://rocmdocs.amd.com/en/latest/ROCm_API_References/Thrust.html
        // async::
        auto stencil_loop = thrust::async::for_each(thrust::cuda::par.on(curent_gpu.compute_stream),
                                                    // Check Top boundaries condition
                                                    thrust::counting_iterator<std::size_t>(is_first_device ? curent_gpu.nbr_of_col : 0),
                                                    // Check Bottom boundaries condition
                                                    thrust::counting_iterator<std::size_t>(is_last_device ? curent_gpu.segment_size_device - curent_gpu.nbr_of_col : curent_gpu.segment_size_device),
                                                    stencilOpFunctor);
#undef is_first_device
#undef is_last_device

        // hipGetLastError
        CUDA_RT_CALL(cudaGetLastError());
        // hipEventRecord
        CUDA_RT_CALL(cudaEventRecord(event_compute_done[iter_curr][curent_gpu.id],
                                     curent_gpu.compute_stream));
        /* May not be usefull */
        CUDA_RT_CALL(cudaStreamSynchronize(curent_gpu.compute_stream));
        /* ----------------- */
    }

    void retrieve_memory_from_devices(GpuManager<PTR_TYPE_COMPUTE> &curent_gpu,
                                      PTR_TYPE_COMPUTE host, const size_t &iter)
    {
        // hipDeviceSynchronize
        CUDA_RT_CALL(cudaDeviceSynchronize());
        // hipMemcpy
        CUDA_RT_CALL(cudaMemcpy(host + (curent_gpu.id * curent_gpu.segment_size_device),
                                array_ptr_devices[iter_curr][curent_gpu.id],
                                curent_gpu.segment_size_device * sizeof(TYPE_COMPUTE),
                                cudaMemcpyDeviceToHost));
        // hipEventDestroy
        CUDA_RT_CALL(cudaEventDestroy(event_compute_done[1][curent_gpu.id]));

        // hipEventDestroy
        CUDA_RT_CALL(cudaEventDestroy(event_compute_done[0][curent_gpu.id]));

        // hipStreamDestroy
        CUDA_RT_CALL(cudaStreamDestroy(curent_gpu.compute_stream));

        // hipFree
        CUDA_RT_CALL(cudaFree(curent_gpu.device_memory));
        CHECK_LAST_CUDA_ERROR();
    }

    GPUsTopol(void)
    {
        // hipGetDeviceCount
        CUDA_RT_CALL(cudaGetDeviceCount(&num_devices));

        array_ptr_devices[0] = alloc_array_ptr.allocate(num_devices);
        array_ptr_devices[1] = alloc_array_ptr.allocate(num_devices);
        event_compute_done[0] = alloc_event.allocate(num_devices);
        event_compute_done[1] = alloc_event.allocate(num_devices);
    }
    ~GPUsTopol()
    {
        alloc_array_ptr.deallocate(array_ptr_devices[0], num_devices);
        alloc_array_ptr.deallocate(array_ptr_devices[1], num_devices);
        alloc_event.deallocate(event_compute_done[0], num_devices);
        alloc_event.deallocate(event_compute_done[1], num_devices);
    }

#undef iter_curr
#undef iter_prev
#undef iter_next
};

// ------------------------------------------------------------ //

// #endif /* __NVCC__ or __CUDACC__ */

#endif /* _GPUS_TOPOL */