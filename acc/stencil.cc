#include "stencil.hh"

template void
stencil_compute_acc<double>(double *buffer_in,
                            double *buffer_out,
                            const std::size_t &nbr_of_col,
                            const std::size_t &nbr_of_row,
                            const std::size_t &first,
                            const std::size_t &last,
                            const size_t &iter_max);
