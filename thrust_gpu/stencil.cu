#include "stencil.cuh"
#include <macros.h>

bool canAccessPeer(const int &id_top,
                   const int &id_curent,
                   const int &id_bottom)
{
    bool p2p_works = true;
    int canAccessPeer = 0;
    if (id_curent not_eq id_top)
    {
        canAccessPeer = 0;
        // hipDeviceEnablePeerAccess
        CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, id_curent, id_top));
        if (canAccessPeer)
        {
            // hipDeviceEnablePeerAccess
            CUDA_RT_CALL(cudaDeviceEnablePeerAccess(id_top, 0));
        }
        else
        {
            std::cerr << "P2P access required from " << id_curent << " to " << id_top << std::endl;
// The omp critical directive identifies a section of code that must be executed by a
// single thread at a time.
#pragma omp critical
            if (p2p_works)
                p2p_works = false;
        }
    }
    if (id_curent not_eq id_bottom)
    {
        canAccessPeer = 0;
        CUDA_RT_CALL(cudaDeviceCanAccessPeer(&canAccessPeer, id_curent, id_bottom));
        if (canAccessPeer)
        {
            CUDA_RT_CALL(cudaDeviceEnablePeerAccess(id_bottom, 0));
        }
        else
        {
            std::cerr << "P2P access required from " << id_curent << " to " << id_bottom
                      << std::endl;
#pragma omp critical
            if (p2p_works)
                p2p_works = false;
        }
    }
    return p2p_works;
}

#define stencil_compute_thrust_gpu_macro(macrotype)                               \
    template void stencil_compute_thrust_gpu<macrotype>(macrotype * buffer_in,    \
                                                        macrotype * buffer_out,   \
                                                        const size_t &nbr_of_col, \
                                                        const size_t &nbr_of_row, \
                                                        const size_t &first,      \
                                                        const size_t &last,       \
                                                        const size_t &iter_max);

FOR_EACH(stencil_compute_thrust_gpu_macro, double)

// template void
// stencil_compute_thrust_gpus_p2p<double>(double *device_in,
//                                         double *device_out,
//                                         const size_t &nbr_of_col,
//                                         const size_t &nbr_of_row,
//                                         const size_t &first,
//                                         const size_t &last,
//                                         const size_t &iter_max);

template void
stencil_compute_thrust_gpus_p2p<double>(double *buffer_in,
                                        double *buffer_out,
                                        const size_t &nbr_of_col,
                                        const size_t &nbr_of_row,
                                        const size_t &first,
                                        const size_t &last,
                                        const size_t &iter_max);

template void
stencil_compute_thrust_gpu_unmanaged<double>(double *buffer_in,
                                             double *buffer_out,
                                             const size_t &nbr_of_col,
                                             const size_t &nbr_of_row,
                                             const size_t &first,
                                             const size_t &last,
                                             const size_t &iter_max);

template double *alloc_on_device<double>(const size_t size);
template void free_on_device<double>(double *buffer_in);
template void read_on_device<double>(double *device_ptr_in,
                                     const size_t idx);
template void copy_on_device<double>(double *host_in,
                                     double *device_ptr_in,
                                     const size_t size);
template void copy_on_host<double>(double *device_ptr_in,
                                   double *host_out,
                                   const size_t size);
