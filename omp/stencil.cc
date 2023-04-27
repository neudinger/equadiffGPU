#include "stencil.hh"
#include <macros.h>

#define stencil_compute_omp_macro(macrotype)                               \
    template void stencil_compute_omp<macrotype>(macrotype * buffer_in,    \
                                                 macrotype * buffer_out,   \
                                                 const size_t &nbr_of_col, \
                                                 const size_t &nbr_of_row, \
                                                 const size_t &first,      \
                                                 const size_t &last,       \
                                                 const size_t &iter_max);
FOR_EACH(stencil_compute_omp_macro, double)
