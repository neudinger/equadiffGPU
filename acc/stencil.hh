#pragma once

#if not defined(_STENCIL_HH)
#define _STENCIL_HH

#include <openacc.h>

#include <algorithm> // std::swap
// #include <iostream> // std::cout

#include <params.hh>

template <typename COMPUTE_TYPE>
void stencil_compute_acc(COMPUTE_PARAMS)
{
#define size (nbr_of_col * nbr_of_row)
#pragma acc data create(buffer_out [0:size]) copy(buffer_in [0:size]) copyout(buffer_out [0:size])
    {
        size_t i, j;
        double *buffer_tmp;
        for (size_t niter = 0; niter < iter_max; niter++)
        {
#pragma acc parallel loop collapse(2)
            for (i = first; i < nbr_of_row - last; ++i)
                for (j = first; j < nbr_of_col - last; ++j)
                    buffer_out[ROW_MAJOR_IDX(i, j)] =
                        0.25 * (buffer_in[ROW_MAJOR_IDX(i + 1, j)] +
                                buffer_in[ROW_MAJOR_IDX(i - 1, j)] +
                                buffer_in[ROW_MAJOR_IDX(i, j + 1)] +
                                buffer_in[ROW_MAJOR_IDX(i, j - 1)] -
                                4 * buffer_in[ROW_MAJOR_IDX(i, j)]);
            buffer_tmp = buffer_in;
            buffer_in = buffer_out;
            buffer_out = buffer_tmp;
        }
    }
}

#endif /*_STENCIL_HH*/
