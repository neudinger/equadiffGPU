#pragma once

#if not defined(_STENCIL_HH)
#define _STENCIL_HH

#include <thrust/for_each.h>

#include <thrust/system/cpp/execution_policy.h>

#if defined(_OPENMP)
#include <omp.h>
#include <thrust/system/omp/execution_policy.h>
#pragma message("THRUST OPENMP ON")
#endif
#if defined(_TBB)
#include <thrust/system/tbb/execution_policy.h>
#pragma message("THRUST TBB ON")
#endif

#include <operations.inl>
#include <params.hh>

template <typename COMPUTE_TYPE>
void stencil_compute_thrust_seq(COMPUTE_PARAMS)
{
    StencilOp<COMPUTE_TYPE> stencilOpFunctor(nbr_of_col, nbr_of_row);
    for (size_t iter = 0; iter < iter_max; iter++)
    {
        stencilOpFunctor.in = buffer_in;
        stencilOpFunctor.out = buffer_out;
        thrust::for_each(thrust::cpp::par,
                         thrust::counting_iterator<size_t>(first),
                         thrust::counting_iterator<size_t>(last),
                         stencilOpFunctor);
        std::swap(buffer_in, buffer_out);
    }
}

#if defined(_TBB)
template <typename COMPUTE_TYPE>
void stencil_compute_thrust_tbb(COMPUTE_PARAMS)
{
    // tbb::task_scheduler_init init(nthread);
    // tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);
    StencilOp<COMPUTE_TYPE> stencilOpFunctor(nbr_of_col, nbr_of_row);
    for (size_t iter = 0; iter < iter_max; iter++)
    {
        stencilOpFunctor.in = buffer_in;
        stencilOpFunctor.out = buffer_out;
        thrust::for_each(thrust::tbb::par,
                         thrust::counting_iterator<size_t>(first),
                         thrust::counting_iterator<size_t>(last),
                         stencilOpFunctor);
        std::swap(buffer_in, buffer_out);
    }
}
#endif

#if defined(_OPENMP)
template <typename COMPUTE_TYPE>
void stencil_compute_thrust_omp(COMPUTE_PARAMS)
{
    StencilOp<COMPUTE_TYPE> stencilOpFunctor(nbr_of_col, nbr_of_row);
    stencilOpFunctor.in = buffer_in;
    stencilOpFunctor.out = buffer_out;
    for (size_t iter = 0; iter < iter_max; iter++)
    {
        thrust::for_each(thrust::omp::par,
                         thrust::counting_iterator<size_t>(first),
                         thrust::counting_iterator<size_t>(last),
                         stencilOpFunctor);
        std::swap(buffer_in, buffer_out);
    }
}
#endif

#endif /*_STENCIL_HH*/
