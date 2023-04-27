#pragma once

#if not defined(_PARAMS_HH)
#define _PARAMS_HH

#define COMPUTE_PARAMS COMPUTE_TYPE *buffer_in,                        \
                       COMPUTE_TYPE *buffer_out,                       \
                       const std::size_t &nbr_of_col,                  \
                       const std::size_t &nbr_of_row = 1,              \
                                         const std::size_t &first = 0, \
                                         const std::size_t &last = 0,  \
                                         const size_t &iter_max = 1

#define COMPUTE_ARGS buffer_in,  \
                     buffer_out, \
                     nbr_of_col, \
                     nbr_of_row, \
                     first,      \
                     last,       \
                     iter_max

typedef const enum COMPUTE_UNIT {
    OMP,
    ACC,
    THRUST_SEQ,
    THRUST_OMP,
    THRUST_TBB,
    THRUST_GPU,
    THRUST_GPU_UNMANAGED,
    THRUST_GPUS_P2P,
    GPU,
    TPU,
} _COMPUTE_UNIT;

#define ROW_MAJOR_IDX(i, j) ((i) * (nbr_of_col)) + (j)

#endif /*_PARAMS_HH*/
