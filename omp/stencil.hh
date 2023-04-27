#pragma once

#if not defined(_STENCIL_HH)
#define _STENCIL_HH

#include <algorithm> // std::swap

#if defined(_OPENMP)
#include <omp.h>
#pragma message("OPENMP ON")
#endif // _OPENMP

#include <params.hh>

constexpr size_t col_block_size = 64;
constexpr size_t row_block_size = col_block_size;

/* Compute with cache blocking and vectorisation 2 */

inline void compute_vectorized_N(
    double *in_host,
    double *out_host,
    const size_t &nbr_of_col,
    const size_t &nbr_of_row,
    const size_t &col_block_size, const size_t &row_block_size,
    const size_t &col_block, const size_t &row_block,
    const std::initializer_list<size_t> col_idx_slice_idx,
    const std::initializer_list<size_t> row_idx_slice_idx)
{
#define not_limit_left (col_idx > 0)
#define not_limit_right (col_idx < nbr_of_col - 1)
#define not_limit_up row_idx

    const std::initializer_list<size_t>::const_iterator
        col_slice_idx = col_idx_slice_idx.begin(),
        row_slice_idx = row_idx_slice_idx.begin();

#pragma omp parallel for
    for (size_t col_block_size_idx = col_slice_idx[0];
         col_block_size_idx < col_slice_idx[1];
         col_block_size_idx += col_block_size)
        for (size_t row_block_size_idx = row_slice_idx[0];
             row_block_size_idx < row_slice_idx[1];
             row_block_size_idx += row_block_size)
            for (size_t row_idx = row_block_size_idx;
                 (row_idx < std::min(nbr_of_row, row_block_size_idx + row_block_size));
                 ++row_idx)
                if (not_limit_up)
                    for (size_t col_idx = col_block_size_idx;
                         (col_idx < std::min(nbr_of_col, col_block_size_idx + col_block_size)); ++col_idx)
                        if (not_limit_right and not_limit_left)
                        {
                            const size_t idx = ROW_MAJOR_IDX(row_idx, col_idx);
                            out_host[idx] = 0.125 *
                                            (in_host[ROW_MAJOR_IDX(row_idx + 1, col_idx)] +
                                             in_host[ROW_MAJOR_IDX(row_idx - 1, col_idx)] +
                                             in_host[idx] +
                                             in_host[ROW_MAJOR_IDX(row_idx, col_idx + 1)] +
                                             in_host[ROW_MAJOR_IDX(row_idx, col_idx - 1)]);
                        }

#undef not_limit_left
#undef not_limit_right
#undef not_limit_up
}

inline void compute_vectorized_S(
    double *in_host,
    double *out_host,
    const size_t &nbr_of_col,
    const size_t &nbr_of_row,
    const size_t &col_block_size, const size_t &row_block_size,
    const size_t &col_block, const size_t &row_block,
    const std::initializer_list<size_t> col_idx_slice_idx,
    const std::initializer_list<size_t> row_idx_slice_idx)
{
#define not_limit_left (col_idx > 0)
#define not_limit_right (col_idx < nbr_of_col - 1)
#define limit_down (row_idx > nbr_of_row - 2)

    const std::initializer_list<size_t>::const_iterator
        col_slice_idx = col_idx_slice_idx.begin(),
        row_slice_idx = row_idx_slice_idx.begin();

#pragma omp parallel for
    for (size_t col_block_size_idx = col_slice_idx[0];
         col_block_size_idx < col_slice_idx[1];
         col_block_size_idx += col_block_size)
        for (size_t row_block_size_idx = row_slice_idx[0];
             row_block_size_idx < row_slice_idx[1];
             row_block_size_idx += row_block_size)
            for (size_t row_idx = row_block_size_idx;
                 (row_idx < std::min(nbr_of_row, row_block_size_idx + row_block_size));
                 ++row_idx)
                if (not limit_down)
                    for (size_t col_idx = col_block_size_idx;
                         (col_idx < std::min(nbr_of_col, col_block_size_idx + col_block_size)); ++col_idx)
                        if (not_limit_right and not_limit_left)
                        {
                            const size_t idx = ROW_MAJOR_IDX(row_idx, col_idx);
                            out_host[idx] = 0.125 *
                                            (in_host[ROW_MAJOR_IDX(row_idx + 1, col_idx)] +
                                             in_host[ROW_MAJOR_IDX(row_idx - 1, col_idx)] +
                                             in_host[idx] +
                                             in_host[ROW_MAJOR_IDX(row_idx, col_idx + 1)] +
                                             in_host[ROW_MAJOR_IDX(row_idx, col_idx - 1)]);
                        }

#undef limit_down
#undef not_limit_left
#undef not_limit_right
}

inline void compute_vectorized_E(
    double *in_host,
    double *out_host,
    const size_t &nbr_of_col,
    const size_t &nbr_of_row,
    const size_t &col_block_size, const size_t &row_block_size,
    const size_t &col_block, const size_t &row_block,
    const std::initializer_list<size_t> col_idx_slice_idx,
    const std::initializer_list<size_t> row_idx_slice_idx)
{
#define not_limit_right (col_idx < nbr_of_col - 1)

    const std::initializer_list<size_t>::const_iterator
        col_slice_idx = col_idx_slice_idx.begin(),
        row_slice_idx = row_idx_slice_idx.begin();

#pragma omp parallel for
    for (size_t col_block_size_idx = col_slice_idx[0];
         col_block_size_idx < col_slice_idx[1];
         col_block_size_idx += col_block_size)
        for (size_t row_block_size_idx = row_slice_idx[0];
             row_block_size_idx < row_slice_idx[1];
             row_block_size_idx += row_block_size)
            for (size_t row_idx = row_block_size_idx;
                 (row_idx < std::min(nbr_of_row, row_block_size_idx + row_block_size));
                 ++row_idx)
                for (size_t col_idx = col_block_size_idx;
                     (col_idx < std::min(nbr_of_col, col_block_size_idx + col_block_size)); ++col_idx)
                    if (not_limit_right)
                    {
                        const size_t idx = ROW_MAJOR_IDX(row_idx, col_idx);
                        out_host[idx] = 0.125 *
                                        (in_host[ROW_MAJOR_IDX(row_idx + 1, col_idx)] +
                                         in_host[ROW_MAJOR_IDX(row_idx - 1, col_idx)] +
                                         in_host[idx] +
                                         in_host[ROW_MAJOR_IDX(row_idx, col_idx + 1)] +
                                         in_host[ROW_MAJOR_IDX(row_idx, col_idx - 1)]);
                    }
#undef not_limit_right
}

inline void compute_vectorized_W(
    double *in_host,
    double *out_host,
    const size_t &nbr_of_col,
    const size_t &nbr_of_row,
    const size_t &col_block_size, const size_t &row_block_size,
    const size_t &col_block, const size_t &row_block,
    const std::initializer_list<size_t> col_idx_slice_idx,
    const std::initializer_list<size_t> row_idx_slice_idx)
{
#define not_limit_left (col_idx > 0)

    const std::initializer_list<size_t>::const_iterator
        col_slice_idx = col_idx_slice_idx.begin(),
        row_slice_idx = row_idx_slice_idx.begin();

#pragma omp parallel for
    for (size_t col_block_size_idx = col_slice_idx[0];
         col_block_size_idx < col_slice_idx[1];
         col_block_size_idx += col_block_size)
        for (size_t row_block_size_idx = row_slice_idx[0];
             row_block_size_idx < row_slice_idx[1];
             row_block_size_idx += row_block_size)
            for (size_t row_idx = row_block_size_idx;
                 (row_idx < std::min(nbr_of_row, row_block_size_idx + row_block_size));
                 ++row_idx)
                for (size_t col_idx = col_block_size_idx;
                     (col_idx < std::min(nbr_of_col, col_block_size_idx + col_block_size)); ++col_idx)
                    if (not_limit_left)
                    {
                        const size_t idx = ROW_MAJOR_IDX(row_idx, col_idx);
                        out_host[idx] = 0.125 *
                                        (in_host[ROW_MAJOR_IDX(row_idx + 1, col_idx)] +
                                         in_host[ROW_MAJOR_IDX(row_idx - 1, col_idx)] +
                                         in_host[idx] +
                                         in_host[ROW_MAJOR_IDX(row_idx, col_idx + 1)] +
                                         in_host[ROW_MAJOR_IDX(row_idx, col_idx - 1)]);
                    }

#undef not_limit_left
}

inline void compute_vectorized_I(
    double *in_host,
    double *out_host,
    const size_t &nbr_of_col,
    const size_t &nbr_of_row,
    const size_t &col_block_size, const size_t &row_block_size,
    const size_t &col_block, const size_t &row_block,
    const std::initializer_list<size_t> col_idx_slice_idx,
    const std::initializer_list<size_t> row_idx_slice_idx)
{

    const std::initializer_list<size_t>::const_iterator
        col_slice_idx = col_idx_slice_idx.begin(),
        row_slice_idx = row_idx_slice_idx.begin();

#pragma omp parallel for
    for (size_t col_block_size_idx = col_slice_idx[0];
         col_block_size_idx < col_slice_idx[1];
         col_block_size_idx += col_block_size)
        for (size_t row_block_size_idx = row_slice_idx[0];
             row_block_size_idx < row_slice_idx[1];
             row_block_size_idx += row_block_size)
            for (size_t row_idx = row_block_size_idx;
                 (row_idx < std::min(nbr_of_row, row_block_size_idx + row_block_size));
                 ++row_idx)
                for (size_t col_idx = col_block_size_idx;
                     (col_idx < std::min(nbr_of_col, col_block_size_idx + col_block_size)); ++col_idx)
                {
                    const size_t idx = ROW_MAJOR_IDX(row_idx, col_idx);
                    out_host[idx] = 0.125 *
                                    (in_host[ROW_MAJOR_IDX(row_idx + 1, col_idx)] +
                                     in_host[ROW_MAJOR_IDX(row_idx - 1, col_idx)] +
                                     in_host[idx] +
                                     in_host[ROW_MAJOR_IDX(row_idx, col_idx + 1)] +
                                     in_host[ROW_MAJOR_IDX(row_idx, col_idx - 1)]);
                }
}

inline void computes(
    double *in_host,
    double *out_host,
    const size_t &nbr_of_col,
    const size_t &nbr_of_row)
{
    const size_t col_block = (nbr_of_col + col_block_size - 1) / col_block_size;
    const size_t row_block = (nbr_of_row + row_block_size - 1) / row_block_size;

    /* IN_PART */
    compute_vectorized_I(in_host, out_host,
                         nbr_of_col, nbr_of_row,
                         col_block_size, row_block_size,
                         col_block, row_block,
                         {col_block_size, col_block_size * (col_block - 1)},
                         {row_block_size, row_block_size * (row_block - 1)});

    /* EAST_PART */
    compute_vectorized_E(in_host, out_host,
                         nbr_of_col, nbr_of_row,
                         col_block_size, row_block_size,
                         col_block, row_block,
                         {nbr_of_col - col_block_size, nbr_of_col},
                         {row_block_size, row_block_size * (row_block - 1)});

    /* WEST_PART */
    compute_vectorized_W(in_host, out_host,
                         nbr_of_col, nbr_of_row,
                         col_block_size, row_block_size,
                         col_block, row_block,
                         {0, col_block_size},
                         {row_block_size, row_block_size * (row_block - 1)});

    /* MPI Recovery should be done */
    /* NORTH_PART */
    compute_vectorized_N(in_host, out_host,
                         nbr_of_col, nbr_of_row,
                         col_block_size, row_block_size,
                         col_block, row_block,
                         {0, col_block_size * col_block},
                         {0, row_block_size});
    /* SOUTH_PART */
    compute_vectorized_S(in_host, out_host,
                         nbr_of_col, nbr_of_row,
                         col_block_size, row_block_size,
                         col_block, row_block,
                         {0, col_block_size * col_block},
                         {row_block_size * (row_block - 1),
                          row_block_size * row_block});
}

template <typename COMPUTE_TYPE>
void stencil_compute_omp(COMPUTE_PARAMS)
{
    for (size_t iter = 0; iter < iter_max; iter++)
    {
        computes(
            /* double *in_host = */ buffer_in,
            /* double *out_host = */ buffer_out,
            /* size_t nbr_of_col = */ nbr_of_col,
            /* size_t nbr_of_row = */ nbr_of_row);
        std::swap(buffer_in, buffer_out);
    }
}

#endif /*_STENCIL_HH*/
