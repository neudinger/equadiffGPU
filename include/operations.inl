#pragma once

#if not defined(_OPERATIONS_IN)
#define _OPERATIONS_IN

#include <iso646.h>

// ------------------------------------------------------------ //

// Boolean to check the Domain border Left/Right (boundaries condition)
#define not_limit_left (idx % nbr_of_col)
#define not_limit_right ((idx + 1) % nbr_of_col)

// ------------------------------------------------------------ //

#define right in[idx + 1]
#define left in[idx - 1]

// ------------------------------------------------------------ //

template <typename TYPE_COMPUTE>
struct StencilOpP2P
{
    const TYPE_COMPUTE *__restrict__ in_top;
    const TYPE_COMPUTE *__restrict__ in;
    const TYPE_COMPUTE *__restrict__ in_bottom;
    TYPE_COMPUTE *__restrict__ out;

    const size_t nbr_of_col;
    const size_t nbr_of_row_local;
    size_t size_of_chunk;

    StencilOpP2P(
        const size_t nbr_of_col,
        const size_t nbr_of_row_local = 1,
        const TYPE_COMPUTE *__restrict__ in = nullptr,
        TYPE_COMPUTE *__restrict__ out = nullptr,
        const TYPE_COMPUTE *__restrict__ in_top = nullptr,
        const TYPE_COMPUTE *__restrict__ in_bottom = nullptr)
        : nbr_of_col(nbr_of_col),
          nbr_of_row_local(nbr_of_row_local),
          in(in),
          out(out),
          in_top(in_top),
          in_bottom(in_bottom) { size_of_chunk = nbr_of_col * nbr_of_row_local; };

    template <typename IDX_TYPE>
    __device__ __host__ void operator()(IDX_TYPE &idx)
    {
        // Will get the value on the local device or the remote device regard to
        // (in_top / in_bottom) pointers and the curent idx
#define up_p2p                                          \
    ((in_top and in not_eq in_top and idx < nbr_of_col) \
         ? in_top[(size_of_chunk - nbr_of_col) + idx]   \
         : in[idx - nbr_of_col])
#define down_p2p                                                                \
    ((in_bottom and in not_eq in_bottom and idx > (size_of_chunk - nbr_of_col)) \
         ? in_bottom[nbr_of_col - (size_of_chunk - idx)]                        \
         : in[idx + nbr_of_col])

        if (not_limit_left and not_limit_right)
            out[idx] = 0.25 * (up_p2p +
                               left - 4 * in[idx] + right +
                               down_p2p);
#undef up_p2p
#undef down_p2p
    }
};

template <typename TYPE_COMPUTE>
struct StencilOp
{
    const TYPE_COMPUTE *__restrict__ in;
    TYPE_COMPUTE *__restrict__ out;

    const size_t nbr_of_col;
    const size_t nbr_of_row_local;
    size_t size_of_chunk;

    StencilOp(
        const size_t nbr_of_col,
        const size_t nbr_of_row_local = 1,
        const TYPE_COMPUTE *__restrict__ in = nullptr,
        TYPE_COMPUTE *__restrict__ out = nullptr)
        : nbr_of_col(nbr_of_col),
          nbr_of_row_local(nbr_of_row_local),
          in(in),
          out(out) { size_of_chunk = nbr_of_col * nbr_of_row_local; };

    template <typename IDX_TYPE>
    __device__ __host__ void operator()(IDX_TYPE &idx)
    {
        if (not_limit_left and not_limit_right)
            out[idx] = 0.25 * (in[idx - nbr_of_col] +
                               left - 4 * in[idx] + right +
                               in[idx + nbr_of_col]);
    }
};

template <typename TYPE_COMPUTE>
struct InitOp
{
    TYPE_COMPUTE *__restrict__ in;
    const TYPE_COMPUTE value;

    InitOp(TYPE_COMPUTE *__restrict__ in,
           const TYPE_COMPUTE &value) : in(in),
                                        value(value){};
    template <typename IDX_TYPE>
    __device__ __host__ void operator()(IDX_TYPE &idx)
    {
        in[idx] = value;
    }
};

template <typename TYPE_COMPUTE>
struct AddOp
{
    const TYPE_COMPUTE *__restrict__ const in;
    TYPE_COMPUTE *out;
    const size_t nbr_of_col;

    AddOp(const TYPE_COMPUTE *__restrict__ const in,
          TYPE_COMPUTE *out,
          const size_t nbr_of_col) : in(in),
                                     out(out),
                                     nbr_of_col(nbr_of_col){};
    template <typename IDX_TYPE>
    __device__ __host__ void operator()(IDX_TYPE &idx)
    {
        if (not_limit_left and not_limit_right)
            out[idx] = in[idx] + 1;
    }
};

#endif /*_OPERATIONS_IN*/
