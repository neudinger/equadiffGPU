#pragma once

#if not defined(_STENCIL_CUDA_CUH)
#define _STENCIL_CUDA_CUH

#include <algorithm> // std::swap

// -----------DEBUG NVTX-------------- //

#include <omp.h>

#if defined(_OPENMP)
#pragma message("CUDA OPENMP ON")
#endif // _OPENMP

#ifdef HAVE_CUB
#include <cub/block/block_reduce.cuh>
#endif // HAVE_CUB

#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};

#define num_colors (sizeof(colors) / sizeof(uint32_t))

#define PUSH_RANGE(name, cid)                              \
    {                                                      \
        int color_id = cid;                                \
        color_id = color_id % num_colors;                  \
        nvtxEventAttributes_t eventAttrib = {0};           \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.colorType = NVTX_COLOR_ARGB;           \
        eventAttrib.color = colors[color_id];              \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxRangePushEx(&eventAttrib);                     \
    }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

// -----------DEBUG NVTX-------------- //

#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/device_ptr.h>

#include <thrust/for_each.h>
#include <thrust/fill.h>
#include <thrust/copy.h>

#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>

#include <operations.inl>
#include <params.hh>

#include "GpusTopol.cuh"

template <typename COMPUTE_TYPE>
void stencil_compute_thrust_gpu(COMPUTE_PARAMS)
{
    const size_t size = (nbr_of_col * nbr_of_row);

    thrust::device_ptr<COMPUTE_TYPE> device_ptr_in = thrust::device_malloc<COMPUTE_TYPE>(size);
    thrust::device_ptr<COMPUTE_TYPE> device_ptr_out = thrust::device_malloc<COMPUTE_TYPE>(size);

    StencilOp<COMPUTE_TYPE> stencilOpFunctor(nbr_of_col, nbr_of_row);
    thrust::copy(buffer_in, buffer_in + size, device_ptr_in);
    thrust::copy(buffer_out, buffer_out + size, device_ptr_out);
    for (size_t iter = 0; iter < iter_max; iter++)
    {
        stencilOpFunctor.in = device_ptr_in.get();
        stencilOpFunctor.out = device_ptr_out.get();
        thrust::for_each(thrust::device,
                         thrust::counting_iterator<size_t>(first),
                         thrust::counting_iterator<size_t>(last),
                         stencilOpFunctor);
        thrust::swap(device_ptr_in, device_ptr_out);
    }
    thrust::copy(device_ptr_out, device_ptr_out + size, buffer_out);

    thrust::device_free(device_ptr_in);
    thrust::device_free(device_ptr_out);
}

template <typename COMPUTE_TYPE>
void stencil_compute_thrust_gpu_unmanaged(COMPUTE_PARAMS)
{

    StencilOp<COMPUTE_TYPE> stencilOpFunctor(nbr_of_col, nbr_of_row);
    for (size_t iter = 0; iter < iter_max; iter++)
    {
        stencilOpFunctor.in = buffer_in;
        stencilOpFunctor.out = buffer_out;

        thrust::for_each(thrust::device,
                         thrust::counting_iterator<size_t>(first),
                         thrust::counting_iterator<size_t>(last),
                         stencilOpFunctor);
        std::swap(buffer_in, buffer_out);
    }
}

template <typename COMPUTE_TYPE>
auto alloc_on_device(const size_t size) -> COMPUTE_TYPE *
{
    return thrust::device_malloc<COMPUTE_TYPE>(size).get();
}

template <typename COMPUTE_TYPE>
void copy_on_device(COMPUTE_TYPE *host_in, COMPUTE_TYPE *device_ptr_in, const size_t size)
{
    thrust::copy(host_in, host_in + size, thrust::device_pointer_cast<COMPUTE_TYPE>(device_ptr_in));
}

template <typename COMPUTE_TYPE>
void copy_on_host(COMPUTE_TYPE *device_ptr_in, COMPUTE_TYPE *host_out, const size_t size)
{
    thrust::device_ptr<COMPUTE_TYPE> ptr = thrust::device_pointer_cast<COMPUTE_TYPE>(device_ptr_in);
    thrust::copy(ptr, ptr + size, host_out);
}

template <typename COMPUTE_TYPE>
void free_on_device(COMPUTE_TYPE *buffer_in)
{
    thrust::device_free(thrust::device_pointer_cast<COMPUTE_TYPE>(buffer_in));
}

template <typename COMPUTE_TYPE>
void read_on_device(COMPUTE_TYPE *buffer_in, const size_t idx)
{
    std::cout << thrust::device_pointer_cast<COMPUTE_TYPE>(buffer_in)[idx] << std::endl;
}

#define ROW_MAJOR_IDX(i, j) ((i) * (nbr_of_col)) + (j)

template <typename COMPUTE_TYPE>
void stencil_compute_thrust_gpus_p2p(COMPUTE_PARAMS)
{
    GPUsTopol<COMPUTE_TYPE *> topol;
#pragma omp parallel num_threads(topol.num_devices) shared(topol)
    {

        GpuManager<COMPUTE_TYPE *> curent_gpu(nbr_of_row, nbr_of_col, topol.num_devices);
        StencilOpP2P<COMPUTE_TYPE> stencilOpFunctor(/* nbr_of_col = */ curent_gpu.nbr_of_col,
                                                    /* nbr_of_row_local = */ curent_gpu.nbr_of_row_local);

// Peer access ?
#pragma omp barrier
        if (canAccessPeer(curent_gpu.id_top, curent_gpu.id, curent_gpu.id_bottom))
        {

            topol.send_memory_on_devices(curent_gpu, buffer_in);

            PUSH_RANGE("Solve", 0)
            CHECK_LAST_CUDA_ERROR();
            for (size_t iter = 0; iter < iter_max; iter++)
            {
#pragma omp barrier
                topol.compute(curent_gpu, stencilOpFunctor, iter);
                // https://learn.microsoft.com/fr-fr/cpp/preprocessor/pragma-directives-and-the-pragma-keyword?view=msvc-170
                /* CUDA/DEVICES - MPI  */
#pragma region DEVICES - MPI

#pragma endregion DEVICES - MPI

/* HOST - MPI  */
// DEVICE -> HOST -> HOST -> DEVICE
#pragma region HOST - MPI 
// IF first device get first line
// DEVICE -> HOST
// MPI BARRIER / win_lock
// GET HALO LINE HOST -> HOST
// HOST -> DEVICE

// IF last device get last line
// SAME as first
#pragma endregion HOST - MPI

#pragma omp barrier
            }

            // #pragma omp critical
            //             {
            //                 thrust::device_ptr<double> ptr_thrust = thrust::device_pointer_cast<double>(topol.array_ptr_devices[1][curent_gpu.id]);

            //                 std::cout << curent_gpu.id << " " str(ptr_thrust) "\t" << ptr_thrust << std::endl;
            //                 for (size_t i = 0; i < curent_gpu.nbr_of_row_local; ++i)
            //                 {
            //                     for (size_t j = 0; j < nbr_of_col; ++j)
            //                         std::cout << ptr_thrust[ROW_MAJOR_IDX(i, j)] << "\t";
            //                     std::cout << "\n"
            //                               << std::flush;
            //                 }
            //                 std::cout << "\n\n"
            //                           << std::flush;
            //             }
            // POP_RANGE
            topol.retrieve_memory_from_devices(curent_gpu, buffer_out, iter_max);
        }
    }
}

#endif /*_STENCIL_CUDA_CUH*/