#include "stencil_cpu.hh"
#include <macros.h>

#define stencil_compute_thrust_seq_macro(macrotype)                               \
    template void stencil_compute_thrust_seq<macrotype>(macrotype * buffer_in,    \
                                                        macrotype * buffer_out,   \
                                                        const size_t &nbr_of_col, \
                                                        const size_t &nbr_of_row, \
                                                        const size_t &first,      \
                                                        const size_t &last,       \
                                                        const size_t &iter_max);

FOR_EACH(stencil_compute_thrust_seq_macro, int, size_t, long, double)

// template void
// stencil_compute_thrust_seq<double>(double *buffer_in,
//                                    double *buffer_out,
//                                    const std::size_t &nbr_of_col,
//                                    const std::size_t &nbr_of_row,
//                                    const std::size_t &first,
//                                    const std::size_t &last,
//                                    const size_t &iter_max);

#if defined(_TBB)
#define stencil_compute_thrust_tbb_macro(macrotype)                               \
    template void stencil_compute_thrust_tbb<macrotype>(macrotype * buffer_in,    \
                                                        macrotype * buffer_out,   \
                                                        const size_t &nbr_of_col, \
                                                        const size_t &nbr_of_row, \
                                                        const size_t &first,      \
                                                        const size_t &last,       \
                                                        const size_t &iter_max);
FOR_EACH(stencil_compute_thrust_tbb_macro, int, size_t, long, double)

// template void
// stencil_compute_thrust_tbb<double>(double *buffer_in,
//                                    double *buffer_out,
//                                    const std::size_t &nbr_of_col,
//                                    const std::size_t &nbr_of_row,
//                                    const std::size_t &first,
//                                    const std::size_t &last,
                                //    const size_t &iter_max);
#endif
#if defined(_OPENMP)
#define stencil_compute_thrust_omp_macro(macrotype)                               \
    template void stencil_compute_thrust_omp<macrotype>(macrotype * buffer_in,    \
                                                        macrotype * buffer_out,   \
                                                        const size_t &nbr_of_col, \
                                                        const size_t &nbr_of_row, \
                                                        const size_t &first,      \
                                                        const size_t &last,       \
                                                        const size_t &iter_max);
FOR_EACH(stencil_compute_thrust_omp_macro, int, size_t, long, double)
// template void
// stencil_compute_thrust_omp<double>(double *buffer_in,
//                                    double *buffer_out,
//                                    const std::size_t &nbr_of_col,
//                                    const std::size_t &nbr_of_row,
//                                    const std::size_t &first,
//                                    const std::size_t &last,
//                                    const size_t &iter_max);
#endif