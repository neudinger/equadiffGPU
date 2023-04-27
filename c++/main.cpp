#include <iostream>
#include <vector>
#include <array>

#include <chrono>

using TimeVar = typename std::chrono::high_resolution_clock::time_point;
#define duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

#include <structargs.hh>

#include <definitions.inl>

#include <cuda.h>
#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/async/for_each.h>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>

#if defined(_OPENMP)
#include <omp.h>
#pragma message("MAIN OPENMP ON")
#endif // _OPENMP

// Dont need thrust but you can use it
// #include <thrust/device_ptr.h>
// #include <thrust/fill.h>
// #include <thrust/device_malloc.h>
// #include <thrust/device_free.h>
// #include <thrust/device_ptr.h>
// #include <thrust/execution_policy.h>
// #include <thrust/for_each.h>
// #include <thrust/copy.h>
// #include <thrust/system/cuda/execution_policy.h>

#if defined(ROW_MAJOR_IDX)
#undef ROW_MAJOR_IDX
#endif

#define ROW_MAJOR_IDX(i, j) ((i) * (parameters.bloc_size)) + (j)

#pragma pack(1)
TEMPLATESTRUCT(/* Struct Name = */ arguments,
               /* ARG_TYPE = */ ulong,
               ARG_TYPE, bloc_size, 0,
               ARG_TYPE, iter_max, 0)
#pragma pack(0)

int main(int argc, char const *argv[])
{
  // #pragma message("foo")
  // #warning "warning"
  // #error "warning"
  // std::cout << _OPENMP << std::endl;
  arguments<ulong> parameters;
  ARGTOSTRUCT(parameters.bloc_size, ulong, argv[1],
              parameters.iter_max, ulong, argv[2])
  TimeVar time_checkpoint_start, time_checkpoint_end;

  const ulong size = parameters.bloc_size * parameters.bloc_size;
  std::array timers{0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL, 0UL};
  // std::cout << parameters.bloc_size << std::endl;
  // std::cout << parameters.bloc_size << std::endl;

  thrust::host_vector<double> in_host(size, 1);
  thrust::host_vector<double> out_host(size, 1);

  // std::vector<double> in_host(size, 1);
  // std::vector<double> out_host(size, 1);

  time_checkpoint_start = timeNow();
  stencil_compute<COMPUTE_UNIT::OMP, double>(/* const COMPUTE_TYPE *buffer_in: */ in_host.data(),
                                             /* COMPUTE_TYPE *buffer_out: */ out_host.data(),
                                             /* const std::size_t nbr_of_col */ parameters.bloc_size,
                                             /* const std::size_t nbr_of_row */ parameters.bloc_size,
                                             /* std::size_t first: */ 1,
                                             /* std::size_t last: */ 1,
                                             /* const size_t &iter_max */ parameters.iter_max);
  time_checkpoint_end = timeNow();
  timers[0] = duration(time_checkpoint_end - time_checkpoint_start);
  std::cout << "OMP = " << timers[0] << " in ms" << std::endl;

  time_checkpoint_start = timeNow();
  stencil_compute<COMPUTE_UNIT::ACC, double>(/* const COMPUTE_TYPE *buffer_in: */ in_host.data(),
                                             /* COMPUTE_TYPE *buffer_out: */ out_host.data(),
                                             /* const std::size_t nbr_of_col */ parameters.bloc_size,
                                             /* const std::size_t nbr_of_row */ parameters.bloc_size,
                                             /* std::size_t first: */ 1,
                                             /* std::size_t last: */ 1,
                                             /* const size_t &parameters.iter_max */ parameters.iter_max);
  time_checkpoint_end = timeNow();
  timers[1] = duration(time_checkpoint_end - time_checkpoint_start);
  std::cout << "ACC = " << timers[1] << " in ms" << std::endl;

  time_checkpoint_start = timeNow();
  stencil_compute<COMPUTE_UNIT::THRUST_SEQ, double>(/* const COMPUTE_TYPE *buffer_in: */ in_host.data(),
                                             /* COMPUTE_TYPE *buffer_out: */ out_host.data(),
                                             /* const std::size_t nbr_of_col */ parameters.bloc_size,
                                             /* const std::size_t nbr_of_row */ parameters.bloc_size,
                                             /* std::size_t first: */ parameters.bloc_size,
                                             /* std::size_t last: */ size - parameters.bloc_size);
  time_checkpoint_end = timeNow();
  timers[1] = duration(time_checkpoint_end - time_checkpoint_start);

  time_checkpoint_start = timeNow();
  stencil_compute<COMPUTE_UNIT::THRUST_OMP, double>(/* const COMPUTE_TYPE *buffer_in: */ in_host.data(),
                                                    /* COMPUTE_TYPE *buffer_out: */ out_host.data(),
                                                    /* const std::size_t nbr_of_col */ parameters.bloc_size,
                                                    /* const std::size_t nbr_of_row */ parameters.bloc_size,
                                                    /* std::size_t first: */ parameters.bloc_size,
                                                    /* std::size_t last: */ size - parameters.bloc_size,
                                                    /* const size_t &iter_max */ parameters.iter_max);
  time_checkpoint_end = timeNow();
  timers[2] = duration(time_checkpoint_end - time_checkpoint_start);
  std::cout << "THRUST_OMP = " << timers[2] << std::endl;

  time_checkpoint_start = timeNow();
  stencil_compute<COMPUTE_UNIT::THRUST_TBB, double>(/* const COMPUTE_TYPE *buffer_in: */ in_host.data(),
                                                    /* COMPUTE_TYPE *buffer_out: */ out_host.data(),
                                                    /* const std::size_t nbr_of_col */ parameters.bloc_size,
                                                    /* const std::size_t nbr_of_row */ parameters.bloc_size,
                                                    /* std::size_t first: */ parameters.bloc_size,
                                                    /* std::size_t last: */ size - parameters.bloc_size,
                                                    /* const size_t &iter_max */ parameters.iter_max);
  time_checkpoint_end = timeNow();
  timers[3] = duration(time_checkpoint_end - time_checkpoint_start);
  std::cout << "THRUST_TBB = " << timers[3] << std::endl;

  time_checkpoint_start = timeNow();
  stencil_compute<COMPUTE_UNIT::THRUST_GPU, double>(/* const COMPUTE_TYPE *buffer_in: */ in_host.data(),
                                                    /* COMPUTE_TYPE *buffer_out: */ out_host.data(),
                                                    /* const std::size_t nbr_of_col */ parameters.bloc_size,
                                                    /* const std::size_t nbr_of_row */ parameters.bloc_size,
                                                    /* std::size_t first: */ parameters.bloc_size,
                                                    /* std::size_t last: */ size - parameters.bloc_size,
                                                    /* const size_t &iter_max */ parameters.iter_max);
  time_checkpoint_end = timeNow();
  timers[4] = duration(time_checkpoint_end - time_checkpoint_start);
  std::cout << "THRUST_GPU = " << timers[4] << " in ms" << std::endl;

  double *in_device = alloc_on_device<double>(size); // cudaMalloc((void **) &in_device, size * sizeof(double));
  double *out_device = alloc_on_device<double>(size);
  copy_on_device<double>(in_host.data(), in_device, size);   // cudaMemcpyHostToDevice
  copy_on_device<double>(out_host.data(), out_device, size); // cudaMemcpyHostToDevice
  // read_on_device<double>(out_device, 1);                     // for debug
  
  time_checkpoint_start = timeNow();
  stencil_compute<COMPUTE_UNIT::THRUST_GPU_UNMANAGED, double>(/* const COMPUTE_TYPE *buffer_in: */ in_device,
                                                       /* COMPUTE_TYPE *buffer_out: */ out_device,
                                                       /* const std::size_t nbr_of_col */ parameters.bloc_size,
                                                       /* const std::size_t nbr_of_row */ parameters.bloc_size,
                                                       /* std::size_t first: */ parameters.bloc_size,
                                                       /* std::size_t last: */ size - parameters.bloc_size);
  // read_on_device<double>(out_device, 1); // for debug
  std::swap(in_device, out_device);
  stencil_compute<COMPUTE_UNIT::THRUST_GPU_UNMANAGED, double>(/* const COMPUTE_TYPE *buffer_in: */ in_device,
                                                       /* COMPUTE_TYPE *buffer_out: */ out_device,
                                                       /* const std::size_t nbr_of_col */ parameters.bloc_size,
                                                       /* const std::size_t nbr_of_row */ parameters.bloc_size,
                                                       /* std::size_t first: */ parameters.bloc_size,
                                                       /* std::size_t last: */ size - parameters.bloc_size);
  timers[5] = duration(time_checkpoint_end - time_checkpoint_start);
  std::cout << "THRUST_GPU_UNMANAGED = " << timers[5] << " in ms" << std::endl;
  copy_on_host<double>(in_device, in_host.data(), size);   // cudaMemcpyDeviceToHost
  copy_on_host<double>(out_device, out_host.data(), size); // cudaMemcpyDeviceToHost
  free_on_device<double>(in_device);                       // cudaFree(in_device);
  free_on_device<double>(out_device);                      // cudaFree(in_device);

  time_checkpoint_start = timeNow();
  stencil_compute<COMPUTE_UNIT::THRUST_GPUS_P2P, double>(/* COMPUTE_TYPE *buffer_in: */ in_host.data(),
                                                         /* COMPUTE_TYPE *buffer_out: */ out_host.data(),
                                                         /* const size_t nbr_of_col */ parameters.bloc_size,
                                                         /* const size_t nbr_of_row */ parameters.bloc_size,
                                                         /* size_t first: */ parameters.bloc_size,
                                                         /* size_t last: */ size - parameters.bloc_size,
                                                         /* const size_t &iter_max */ parameters.iter_max);
  time_checkpoint_end = timeNow();
  timers[6] = duration(time_checkpoint_end - time_checkpoint_start);
  std::cout << "THRUST_GPU_P2P = " << timers[6] << " in ms" << std::endl;
  // std::cout << "\n"
  //           << std::endl;
  // for (size_t i = 0; i < parameters.bloc_size; ++i)
  // {
  //     for (size_t j = 0; j < parameters.bloc_size; ++j)
  //         std::cout << out_host[ROW_MAJOR_IDX(i, j)] << "\t";
  //     std::cout << "\n";
  // }
  in_host.clear();
  out_host.clear();
  return 0;
}
