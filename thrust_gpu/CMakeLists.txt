cmake_minimum_required(VERSION  3.19...3.23)
set(TARGET "stencil_thrustlib_gpu")
set(LINK_TYPE STATIC)
set(TARGET_VERSION 1.0)

project(${TARGET} LANGUAGES CUDA VERSION ${TARGET_VERSION})

set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CUDA_STANDARD_REQUIRED ON)
set(CUDA_EXTENSIONS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message(STATUS ${TARGET} " LINK TYPE: " ${LINK_TYPE})
message(STATUS ${TARGET} " CUDA_PATH: " $ENV{CUDA_PATH})

set(CMAKE_${LINK_TYPE}_LIBRARY_PREFIX "" CACHE STRING "" FORCE)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

get_filename_component(HOME_DIR ${PROJECT_SOURCE_DIR} PATH)
set(WORKDIR ${CMAKE_CURRENT_LIST_DIR})
set(INCLUDE_DIR ${HOME_DIR}/include)
message(STATUS ${TARGET} " INCLUDE_DIR in: " ${INCLUDE_DIR})
message(STATUS ${TARGET} " WORKDIR in: " ${WORKDIR})

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif(NOT CMAKE_BUILD_TYPE)

message(STATUS ${TARGET} " CMAKE_BUILD_TYPE is " ${CMAKE_BUILD_TYPE})
message(STATUS ${TARGET} " CXX = " $ENV{CXX} " of " ${TARGET})
message(STATUS ${TARGET} " CXXFLAGS = " $ENV{CXXFLAGS} " of " ${TARGET})


string(APPEND CMAKE_CXX_FLAGS " -fopenmp -march=native ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable -Wunused-parameter")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wextra -Wreorder -Wconversion -Wsign-conversion")

# https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 61 CACHE STRING "CUDA architectures" FORCE)
endif()


if (${CMAKE_CUDA_COMPILER_ID} STREQUAL "Clang")
  string(APPEND CMAKE_CUDA_FLAGS " -fopenmp -fcompare-debug-second --cuda-gpu-arch=sm_61") # WITH CLANG
  # https://llvm.org/docs/CompileCudaWithLLVM.html
  # string(APPEND CMAKE_CUDA_FLAGS " -fopenmp-targets=nvptx64-nvidia-cuda -Xopenmp-target -march=sm_61 -L$ENV{CUDA_PATH} -lcudart") # WITH CLANG
elseif(${CMAKE_CUDA_COMPILER_ID} STREQUAL "NVIDIA")
  string(APPEND CMAKE_CUDA_FLAGS "-fopenmp  -fcompare-debug-second ")
  string(APPEND CMAKE_CUDA_FLAGS "-lineinfo -lnvToolsExt ") # WITH NVCC  
  string(APPEND CMAKE_CUDA_FLAGS "-DHAVE_CUB -DUSE_NVTX ") # WITH NVCC
  string(APPEND CMAKE_CUDA_FLAGS "-Xptxas --optimize-float-atomics ") # WITH NVCC
else()
  message(FATAL_ERROR "NO cuda compiler compatible found: CMAKE_CUDA_COMPILER_ID = " ${CMAKE_CUDA_COMPILER_ID})
endif()

include_directories(${INCLUDE_DIR})
# /opt/nvidia/hpc_sdk/Linux_x86_64/22.2/math_libs/11.6/targets/x86_64-linux/include/
include_directories($ENV{CUDA_PATH}/include)
include_directories($ENV{CUDA_PATH}/targets/x86_64-linux/include/)

file(GLOB SRC
    "${WORKDIR}/*.cu"
)

add_library(${TARGET}
            ${LINK_TYPE}
            ${SRC}
)

set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
target_compile_features(${TARGET} PRIVATE cxx_std_${CMAKE_CXX_STANDARD})

set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})
set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)

unset(CMAKE_CUDA_FLAGS CACHE)
unset(CMAKE_CUDA_ARCHITECTURES CACHE)
