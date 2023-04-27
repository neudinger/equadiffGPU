# Parallel Heterogeneous CPU/GPU computing on diffusion equation with OpenMP, CUDA, Thrust, OpenACC, TBB

This project can use interconnected GPUs by PCIe or Nvlink with P2P connection.

## Requirement

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black) ![C++](https://img.shields.io/badge/c++-%2300599C.svg?style=for-the-badge&logo=c%2B%2B&logoColor=white) ![C](https://img.shields.io/badge/c-%2300599C.svg?style=for-the-badge&logo=c&logoColor=white)


![NVIDIA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![Thrust](https://img.shields.io/badge/Thrust-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![OpenACC](https://img.shields.io/badge/OpenACC-76B900?style=for-the-badge&logo=nvidia&logoColor=white) ![NVLINK](https://img.shields.io/badge/NVLINK-76B900?style=for-the-badge&logo=nvidia&logoColor=white) 


![OMP](https://img.shields.io/badge/OpenMp-%23316192.svg?style=for-the-badge) ![TBB](https://img.shields.io/badge/TBB-0071C5?style=for-the-badge&logo=intel&logoColor=white)


### CUDA / Thrust / OpenACC

Install NVIDIA HPC kit

https://developer.nvidia.com/hpc-sdk

And setup the CUDA_PATH toward the hpc kit directory

Example:
```bash
export CUDA_PATH=/opt/nvidia/hpc_sdk/Linux_x86_64/22.2/cuda
```

- nvcc required for CUDA and thrust
- nvc++ required for openACC

### Thread building-blocks

```bash
sudo apt install libtbb-dev
```

or

Follow this : <https://www.intel.com/content/www/us/en/developer/articles/guide/get-started-with-tbb.html>


### OpenMP

Any compatible compiler like GNU g++ or LLVM clang++

## Build

Each project can be built as library separately :

- OpenACC in directory [acc](./acc)
- OpenMP in directory [omp](./omp)
- Thrust CPU (TBB/OpenMP) in directory [thrust_cpu](./thrust_cpu)
- Thrust GPU (CUDA) in directory [thrust_gpu](./thrust_gpu)
- OpenACC in directory [acc](./acc)

All computation model and library can be built in one cmake but every dependencies is required and the binary will be able to be executed

The cmake will select the specific required compiler for each subproject (g++, clang++, nvcc, nvc++)

C++ 17 was used due to usage of thrust template and the usage of [SFINAE](https://en.cppreference.com/w/cpp/language/sfinae) template technical style.

If you have configured clang++ to be able to compile cuda code you can replace

```bash
-DCMAKE_CUDA_COMPILER=nvcc
by
-DCMAKE_CUDA_COMPILER=clang++
```

### Build

```bash
cd c++ && \
cmake \
-DCMAKE_BUILD_TYPE=RelWithDebInfo \
-DCMAKE_CUDA_COMPILER=nvcc \
-B build -S . && \
cmake --build build
```

## Execute

```bash
./build/bin/stencil 10000 10000
```