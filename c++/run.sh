#!/bin/env bash

if (command -v module &> /dev/null)
then
module load cmake gcc tbb cuda
fi

# https://linux.die.net/man/1/hwloc-bind
# https://www.open-mpi.org/projects/hwloc/tutorials/20160606-PATC-hwloc-tutorial.pdf

# hwloc-bind pci=01:00.0 echo hello
# To run on a core near the network interface named eth0:
# hwloc-bind os=eth0 echo hello
# To run on a core near the PCI device whose bus ID is 0000:01:02.0:
# hwloc-bind pci=0000:01:02.0 echo hello 

# hwloc-bind --physical  socket:0.pu:3 ./laplace

export OMP_PROC_BIND=TRUE
export OMP_NUM_THREADS=20

set -x

# -DCMAKE_CUDA_COMPILER=clang
# -DCMAKE_BUILD_TYPE=Release
# -DCMAKE_BUILD_TYPE=Debug

# rm -rf build/ && \
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_COMPILER=nvcc -B build -S . && \
cmake --build build && \
time ./build/bin/stencil 10000 1000

# ${1} ${2}
# nsys profile --trace=cuda,nvtx --force-overwrite true -o out
# --verbose