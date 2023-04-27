
#include "cuda_func.cuh"
#include "structargs.hh"

#include <chrono>

using TimeVar = typename std::chrono::high_resolution_clock::time_point;
#define duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now()

#pragma pack(1)
TEMPLATESTRUCT(/* Struct Name = */ arguments,
               /* ARG_TYPE = */ ulong,
               ARG_TYPE, bloc_size, 0,
               ARG_TYPE, iter_max, 0)
#pragma pack(0)


// On Linux, you can now put this in /etc/modprobe.d/diable-nvlink.conf:
// options nvidia NVreg_NvLinkDisable=1

int main(int argc, char const *argv[])
{
    arguments<ulong> parameters;
    ARGTOSTRUCT(parameters.bloc_size, ulong, argv[1],
                parameters.iter_max, ulong, argv[2])
    TimeVar time_checkpoint_start, time_checkpoint_end;

    long double cpu_time[] = {0, 0, 0};
    CUDA_RT_CALL(cudaFree(0));

    int numGpus = 0;
    CUDA_RT_CALL(cudaGetDeviceCount(&numGpus));
    // std::cout << "numGpus " << numGpus << std::endl;
    // Since all devices must be of the same compute capability and have the same launch configuration
    // it is sufficient to query device 0 here
    cudaDeviceProp deviceProp;
    CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, 0));

    int numSMs;
    // deviceProp.pciDeviceID
    CUDA_RT_CALL(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
    std::cout << "device numSMs available " << numSMs << std::endl;
    printf("The warp size is %d.\n", deviceProp.warpSize);

    if (!deviceProp.managedMemory)
    {
        // This sample requires being run on a device that supports Unified Memory
        std::cerr << "Unified Memory not supported on this device" << std::endl;
        exit(EXIT_FAILURE);
    }

    // This sample requires being run on a device that supports Cooperative Kernel
    // Launch
    if (!deviceProp.cooperativeLaunch)
    {
        std::cerr << "\nSelected GPU ( " << 0 << " ) does not support Cooperative Kernel Launch, "
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    // printf(
    // "> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
    // deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
    std::cout << "maxThreadsPerMultiProcessor " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "sharedMemPerBlock " << deviceProp.sharedMemPerBlock << std::endl;
    std::cout << "sharedMemPerMultiprocessor " << deviceProp.sharedMemPerMultiprocessor << std::endl;
    std::cout << "reservedSharedMemPerBlock " << deviceProp.reservedSharedMemPerBlock << std::endl;
    std::cout << "maxGridSize " << deviceProp.maxGridSize[0] << " : " << deviceProp.maxGridSize[1] << " : " << deviceProp.maxGridSize[2] << std::endl;

    cudaStream_t s1;
    CUDA_RT_CALL(cudaStreamCreate(&s1));

    float gpu_time[] = {0, 0, 0, 0};
    cudaEvent_t start, stop;

    const size_t nbr_of_col = parameters.bloc_size,
                 nbr_of_row = parameters.bloc_size,
                 domain_size = nbr_of_col * nbr_of_row,
                 offset_end = domain_size - nbr_of_col,
                 threads_per_block = deviceProp.maxThreadsPerBlock; /* deviceProp.maxThreadsPerBlock */

    std::cout << "threads_per_block = " << threads_per_block << " <= " << deviceProp.maxThreadsPerBlock << std::endl;

    cudaLaunchParams launch_configuration;
    launch_configuration.sharedMem = 0;
    // launch_configuration.sharedMem = sizeof(double) *
    //                                  ((threads_per_block / deviceProp.maxBlocksPerMultiProcessor /* 32 <= deviceProp.maxBlocksPerMultiProcessor */) +
    //                                   1);
    int numBlocksPerSm = 0;

    launch_configuration.stream = NULL;
    launch_configuration.func = (void *)stencil_cg;
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                               launch_configuration.func,
                                                               threads_per_block,
                                                               launch_configuration.sharedMem));

    launch_configuration.blockDim = dim3(threads_per_block, 1, 1);
    launch_configuration.gridDim = dim3(deviceProp.multiProcessorCount * numBlocksPerSm, 1, 1);

    std::cout << "launch_configuration.gridDim.x = " << launch_configuration.gridDim.x << std::endl;
    std::cout << "launch_configuration.blockDim.x = " << launch_configuration.blockDim.x << std::endl;

    // thrust::host_vector<double> h_A(domain_size);
    // for (size_t i = 0; i < h_A.size(); i++)
    // h_A[i] = 1.0;
    // thrust::device_vector<double> d_A = h_A;
    // thrust::device_vector<double> d_B = h_A;
    // thrust::host_vector<double> d_A(domain_size, 1);
    // thrust::host_vector<double> d_B(domain_size, 1);

    thrust::device_vector<double> d_A(domain_size, 1);
    thrust::device_vector<double> d_B(domain_size, 1);

    double *kernel_ptr_A = thrust::raw_pointer_cast(d_A.data());
    double *kernel_ptr_B = thrust::raw_pointer_cast(d_B.data());

    std::nullptr_t top = nullptr,
                   down = nullptr;

    size_t size_of_chunk = 0;

    // The kernel arguments are copied over during launch
    // Its also possible to have individual copies of kernel arguments per device, but
    // the signature and name of the function/kernel must be the same.

    time_checkpoint_start = timeNow();
    void *kernelArgs[] = {(void *)&kernel_ptr_A,
                          (void *)&kernel_ptr_B,
                          (void *)&nbr_of_col,
                          (void *)&offset_end,
                          (void *)&domain_size,
                          (void *)&parameters.iter_max,
                          (void *)&size_of_chunk,
                          (void *)&top,
                          (void *)&down};

    CUDA_RT_CALL(cudaEventCreate(&start));
    CUDA_RT_CALL(cudaEventCreate(&stop));
    CUDA_RT_CALL(cudaEventRecord(start, 0));
    CUDA_RT_CALL(cudaLaunchCooperativeKernel(launch_configuration.func,
                                             launch_configuration.gridDim,
                                             launch_configuration.blockDim,
                                             kernelArgs,
                                             launch_configuration.sharedMem,
                                             launch_configuration.stream));
    CUDA_RT_CALL(cudaStreamSynchronize(s1));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    time_checkpoint_end = timeNow();
    CUDA_RT_CALL(cudaEventRecord(stop, 0));
    CUDA_RT_CALL(cudaEventSynchronize(stop));
    CUDA_RT_CALL(cudaEventElapsedTime(&gpu_time[0], start, stop));
    CUDA_RT_CALL(cudaEventDestroy(start));
    CUDA_RT_CALL(cudaEventDestroy(stop));
    CHECK_LAST_CUDA_ERROR();
    cpu_time[0] = duration(time_checkpoint_end - time_checkpoint_start);

    std::cout << std::endl
              << "Timing: \n"
              << "cooperativeKernel_time\n TIME : " << gpu_time[0] << " in ms" << std::endl;
    /*  */

    launch_configuration.func = (void *)stencil_host_cg;
    CUDA_RT_CALL(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm,
                                                               launch_configuration.func,
                                                               threads_per_block,
                                                               launch_configuration.sharedMem));
    time_checkpoint_start = timeNow();

    CUDA_RT_CALL(cudaEventCreate(&start));
    CUDA_RT_CALL(cudaEventCreate(&stop));
    CUDA_RT_CALL(cudaEventRecord(start, 0));
    for (size_t i = 0; i < parameters.iter_max; i++)
    {
        void *kernelArgs[] = {(void *)&kernel_ptr_A,
                              (void *)&kernel_ptr_B,
                              (void *)&nbr_of_col,
                              (void *)&offset_end,
                              (void *)&domain_size,
                              (void *)&parameters.iter_max,
                              (void *)&size_of_chunk,
                              (void *)&top,
                              (void *)&down};
        CUDA_RT_CALL(cudaLaunchCooperativeKernel(launch_configuration.func,
                                                 launch_configuration.gridDim,
                                                 launch_configuration.blockDim,
                                                 kernelArgs,
                                                 launch_configuration.sharedMem,
                                                 launch_configuration.stream));
        CUDA_RT_CALL(cudaDeviceSynchronize());
        CHECK_LAST_CUDA_ERROR();
        thrust::swap(kernel_ptr_A, kernel_ptr_B);
    }
    CUDA_RT_CALL(cudaStreamSynchronize(s1));
    CUDA_RT_CALL(cudaDeviceSynchronize());
    time_checkpoint_end = timeNow();
    CUDA_RT_CALL(cudaEventRecord(stop, 0));
    CUDA_RT_CALL(cudaEventSynchronize(stop));
    CUDA_RT_CALL(cudaEventElapsedTime(&gpu_time[1], start, stop));
    CUDA_RT_CALL(cudaEventDestroy(start));
    CUDA_RT_CALL(cudaEventDestroy(stop));
    CHECK_LAST_CUDA_ERROR();
    cpu_time[1] = duration(time_checkpoint_end - time_checkpoint_start);
    std::cout << std::endl
              << "Timing: \n"
              << "cooperativeKernel_time HOST \n TIME : " << gpu_time[1] << " in ms" << std::endl;
    /*  */
    // std::ofstream CooperativeKernel_A("CooperativeKernel_A.out");
    // thrust::copy_n(d_A.begin(), domain_size, std::ostream_iterator<double>(CooperativeKernel_A, ","));
    // CooperativeKernel_A.close();
    // std::ofstream CooperativeKernel_B("CooperativeKernel_B.out");
    // thrust::copy_n(d_B.begin(), domain_size, std::ostream_iterator<double>(CooperativeKernel_B, ","));
    // CooperativeKernel_B.close();

    thrust::fill(d_A.begin(), d_A.end(), 1);
    thrust::fill(d_B.begin(), d_B.end(), 1);

    // std::cout << "numBlocksPerSm = " << numBlocksPerSm << " <= " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
    auto threads = std::sqrt(threads_per_block);
    // launch_configuration.blockDim = dim3(threads, threads);
    // launch_configuration.gridDim = dim3((nbr_of_col + threads - 1) / threads,
    // (nbr_of_row + threads - 1) / threads);
    // launch_configuration.blockDim = dim3(threads_per_block, 1, 1);

    launch_configuration.blockDim = dim3(threads, threads);
    launch_configuration.gridDim = dim3((nbr_of_col + threads - 1) / threads,
                                        (nbr_of_row + threads - 1) / threads);
    // launch_configuration.blockDim = dim3(threads_per_block);
    // launch_configuration.gridDim = dim3((nbr_of_col + threads_per_block - 1) / threads_per_block);
    time_checkpoint_start = timeNow();
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (size_t i = 0; i < parameters.iter_max; i++)
    {
        stencil<<<launch_configuration.gridDim,
                  launch_configuration.blockDim,
                  launch_configuration.sharedMem>>>(thrust::raw_pointer_cast(d_A.data()),
                                                    thrust::raw_pointer_cast(d_B.data()),
                                                    nbr_of_col,
                                                    domain_size,
                                                    domain_size - nbr_of_col,
                                                    parameters.iter_max);
        CUDA_RT_CALL(cudaDeviceSynchronize());
        CHECK_LAST_CUDA_ERROR();
        thrust::swap(d_A, d_B);
    }
    CUDA_RT_CALL(cudaDeviceSynchronize());
    CHECK_LAST_CUDA_ERROR();
    CUDA_RT_CALL(cudaStreamSynchronize(s1));
    time_checkpoint_end = timeNow();
    CUDA_RT_CALL(cudaEventRecord(stop, 0));
    CUDA_RT_CALL(cudaEventSynchronize(stop));
    CUDA_RT_CALL(cudaEventElapsedTime(&gpu_time[2], start, stop));
    CUDA_RT_CALL(cudaEventDestroy(start));
    CUDA_RT_CALL(cudaEventDestroy(stop));
    cpu_time[2] = duration(time_checkpoint_end - time_checkpoint_start);
    std::cout << std::endl
              << "Timing: \n"
              << "kernel_time HOST \n TIME : " << gpu_time[2] << " in ms" << std::endl;
    // std::ofstream Kernel_A("Kernel_A.out");
    // thrust::copy_n(d_A.begin(), domain_size, std::ostream_iterator<double>(Kernel_A, ","));
    // Kernel_A.close();
    // std::ofstream Kernel_B("Kernel_B.out");
    // thrust::copy_n(d_B.begin(), domain_size, std::ostream_iterator<double>(Kernel_B, ","));
    // Kernel_B.close();

    thrust::fill(d_A.begin(), d_A.end(), 1);
    thrust::fill(d_B.begin(), d_B.end(), 1);
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    time_checkpoint_start = timeNow();
    for (size_t i = 0; i < parameters.iter_max; i++)
    {
        // thrust::cuda::par.on(s1),
        auto o = thrust::for_each(thrust::counting_iterator<std::size_t>(nbr_of_col),
                                  thrust::counting_iterator<std::size_t>(domain_size - nbr_of_col),
                                  stencil_stuct(thrust::raw_pointer_cast(d_A.data()),
                                                thrust::raw_pointer_cast(d_B.data()),
                                                nbr_of_col, parameters.iter_max));
        thrust::swap(d_A, d_B);
    }
    CUDA_RT_CALL(cudaStreamSynchronize(s1));
    time_checkpoint_end = timeNow();
    CUDA_RT_CALL(cudaEventRecord(stop, 0));
    CUDA_RT_CALL(cudaEventSynchronize(stop));
    CUDA_RT_CALL(cudaEventElapsedTime(&gpu_time[3], start, stop));
    CUDA_RT_CALL(cudaEventDestroy(start));
    CUDA_RT_CALL(cudaEventDestroy(stop));
    cpu_time[3] = duration(time_checkpoint_end - time_checkpoint_start);
    std::cout << std::endl
              << "Timing: \n"
              << "Thrust \n TIME : " << gpu_time[3] << " in ms" << std::endl;
    // std::ofstream Thrust_A("Thrust_A.out");
    // thrust::copy_n(d_A.begin(), domain_size, std::ostream_iterator<double>(Thrust_A, ","));
    // Thrust_A.close();
    // std::ofstream Thrust_B("Thrust_B.out");
    // thrust::copy_n(d_B.begin(), domain_size, std::ostream_iterator<double>(Thrust_B, ","));
    // Thrust_B.close();

    cudaStreamDestroy(s1);

    /* Debug */
    // std::cout << std::endl;
    // for (size_t i = 0; i < nbr_of_row; ++i)
    // {
    //     for (size_t j = 0; j < nbr_of_col; ++j)
    //         std::cout << thrust::device_pointer_cast<double>(d_B.data())[ROW_MAJOR_IDX(i, j)] << "\t";
    //     std::cout << "\n"
    //               << std::flush;
    // }
    // std::cout << std::endl;
    /* End Debug */

    d_A.clear();
    thrust::device_vector<double>().swap(d_A);
    d_B.clear();
    thrust::device_vector<double>().swap(d_B);

    return 0;

    // // auto x = thrust::async::for_each(thrust::cuda::par.on(s1),
    // //                                  thrust::counting_iterator<std::size_t>(nbr_of_col),
    // //                                  thrust::counting_iterator<std::size_t>(size - nbr_of_col),
    // //                                  f(thrust::raw_pointer_cast(d_A.data()), thrust::raw_pointer_cast(d_B.data()), nbr_of_col));

    // //  __host__ __device__
    // // thrust::cuda::par.on(s1),

    // unsigned long long dt3 = dtime_usec(dt0);
    // auto myfunc = [ in = thrust::raw_pointer_cast(d_C.data()),
    //                 out = thrust::raw_pointer_cast(d_B.data()),
    //                 nbr_of_col = nbr_of_col ](auto idx) -> auto
    // {
    //     if (not_limit_left and not_limit_right)
    //         out[idx] = 0.25 * (in[idx - nbr_of_col] +
    //                            in[idx - 1] + 2 * in[idx] + in[idx + 1] +
    //                            in[idx + nbr_of_col]);
    // };

    // // auto c = thrust::async::for_each(thrust::cuda::par.on(s1),
    // //   thrust::counting_iterator<std::size_t>(nbr_of_col),
    // //   thrust::counting_iterator<std::size_t>(size - nbr_of_col),
    // //   myfunc);

    // for (size_t i = 0; i < nbr_of_row; ++i)
    // {
    //     for (size_t j = 0; j < nbr_of_col; ++j)
    //         std::cout << thrust::device_pointer_cast<double>(d_B.data())[ROW_MAJOR_IDX(i, j)] << "\t";
    //     std::cout << "\n"
    //               << std::flush;
    // }
    // std::cout << std::endl;
    // for (size_t i = 0; i < nbr_of_row; ++i)
    // {
    //     for (size_t j = 0; j < nbr_of_col; ++j)
    //         std::cout << thrust::device_pointer_cast<double>(d_C.data())[ROW_MAJOR_IDX(i, j)] << "\t";
    //     std::cout << "\n"
    //               << std::flush;
    // }
    // std::cout << std::endl;

    // // for (size_t i = 0; i < nbr_of_row; ++i)
    // // {
    // //     for (size_t j = 0; j < nbr_of_col; ++j)
    // //         std::cout << (thrust::device_pointer_cast<double>(d_B.data())[ROW_MAJOR_IDX(i, j)]  == thrust::device_pointer_cast<double>(d_A.data())[ROW_MAJOR_IDX(i, j)]) << "\t";
    // //     std::cout << "\n"
    // //               << std::flush;
    // // }
    // // std::cout << std::endl;

    // // for (size_t i = 0; i < nbr_of_row; ++i)
    // // {
    // //     for (size_t j = 0; j < nbr_of_col; ++j)
    // //         std::cout << thrust::device_pointer_cast<double>(d_B.data())[ROW_MAJOR_IDX(i, j)] << "\t";
    // //     std::cout << "\n"
    // //               << std::flush;
    // // }
    // // std::cout << std::endl;
}
