#pragma once

#if not defined(_DEFINITIONS_IN)
#define _DEFINITIONS_IN

#include <params.hh>

template <typename COMPUTE_TYPE>
void stencil_compute_omp(COMPUTE_PARAMS);

template <typename COMPUTE_TYPE>
void stencil_compute_thrust_seq(COMPUTE_PARAMS);
template <typename COMPUTE_TYPE>
void stencil_compute_thrust_omp(COMPUTE_PARAMS);
template <typename COMPUTE_TYPE>
void stencil_compute_thrust_tbb(COMPUTE_PARAMS);

template <typename COMPUTE_TYPE>
void stencil_compute_acc(COMPUTE_PARAMS);

template <typename COMPUTE_TYPE>
void stencil_compute_thrust_gpu(COMPUTE_PARAMS);
template <typename COMPUTE_TYPE>
void stencil_compute_thrust_gpu_unmanaged(COMPUTE_PARAMS);
template <typename COMPUTE_TYPE>
void stencil_compute_thrust_gpus_p2p(COMPUTE_PARAMS);


template <typename COMPUTE_TYPE>
void copy_on_device(COMPUTE_TYPE *buffer_in,
                    COMPUTE_TYPE *device_ptr_in,
                    const std::size_t size);
template <typename COMPUTE_TYPE>
void copy_on_host(COMPUTE_TYPE *device_ptr_in,
                  COMPUTE_TYPE *host_out,
                  const std::size_t size);
template <typename COMPUTE_TYPE>
auto alloc_on_device(const std::size_t size) -> COMPUTE_TYPE *;
template <typename COMPUTE_TYPE>
void free_on_device(COMPUTE_TYPE *buffer_in);
template <typename COMPUTE_TYPE>
void read_on_device(COMPUTE_TYPE *buffer_in,
                    const std::size_t idx);

template <COMPUTE_UNIT PU, typename COMPUTE_TYPE>
struct stencil_impl;

template <COMPUTE_UNIT PU, typename COMPUTE_TYPE>
static inline constexpr void stencil_compute(COMPUTE_PARAMS)
{
    return stencil_impl<PU, COMPUTE_TYPE>::_(COMPUTE_ARGS);
}

template <typename COMPUTE_TYPE>
struct stencil_impl<COMPUTE_UNIT::OMP, COMPUTE_TYPE>
{
    static inline constexpr void _(COMPUTE_PARAMS)
    {
        return stencil_compute_omp<COMPUTE_TYPE>(COMPUTE_ARGS);
    }
};

template <typename COMPUTE_TYPE>
struct stencil_impl<COMPUTE_UNIT::ACC, COMPUTE_TYPE>
{
    static inline constexpr void _(COMPUTE_PARAMS)
    {
        return stencil_compute_acc<COMPUTE_TYPE>(COMPUTE_ARGS);
    }
};


template <typename COMPUTE_TYPE>
struct stencil_impl<COMPUTE_UNIT::THRUST_SEQ, COMPUTE_TYPE>
{
    static inline constexpr void _(COMPUTE_PARAMS)
    {
        return stencil_compute_thrust_seq<COMPUTE_TYPE>(COMPUTE_ARGS);
    }
};

template <typename COMPUTE_TYPE>
struct stencil_impl<COMPUTE_UNIT::THRUST_OMP, COMPUTE_TYPE>
{
    static inline constexpr void _(COMPUTE_PARAMS)
    {
        return stencil_compute_thrust_omp<COMPUTE_TYPE>(COMPUTE_ARGS);
    }
};

template <typename COMPUTE_TYPE>
struct stencil_impl<COMPUTE_UNIT::THRUST_TBB, COMPUTE_TYPE>
{
    static inline constexpr void _(COMPUTE_PARAMS)
    {
        return stencil_compute_thrust_tbb<COMPUTE_TYPE>(COMPUTE_ARGS);
    }
};

template <typename COMPUTE_TYPE>
struct stencil_impl<COMPUTE_UNIT::THRUST_GPU, COMPUTE_TYPE>
{
    static inline constexpr void _(COMPUTE_PARAMS)
    {
        return stencil_compute_thrust_gpu<COMPUTE_TYPE>(COMPUTE_ARGS);
    }
};

template <typename COMPUTE_TYPE>
struct stencil_impl<COMPUTE_UNIT::THRUST_GPU_UNMANAGED, COMPUTE_TYPE>
{
    static inline constexpr void _(COMPUTE_PARAMS)
    {
        return stencil_compute_thrust_gpu_unmanaged<COMPUTE_TYPE>(COMPUTE_ARGS);
    }
};

template <typename COMPUTE_TYPE>
struct stencil_impl<COMPUTE_UNIT::THRUST_GPUS_P2P, COMPUTE_TYPE>
{
    static inline constexpr void _(COMPUTE_PARAMS)
    {
        return stencil_compute_thrust_gpus_p2p<COMPUTE_TYPE>(COMPUTE_ARGS);
    }
};

#endif /*_DEFINITIONS_IN*/
