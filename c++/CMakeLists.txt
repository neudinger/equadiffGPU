cmake_minimum_required(VERSION 3.19...3.23 FATAL_ERROR)

set(TARGET "stencil")
set(TARGET_VERSION 1.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Sadly nvc++ compiler does not accept -lto so either you do not use -lto or you comment acc project
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # It is -lto link table optimisation flag optimisation
project(${TARGET} LANGUAGES CXX VERSION ${TARGET_VERSION})

if(NOT CMAKE_BUILD_TYPE)
  # set(CMAKE_BUILD_TYPE Debug)
  set(CMAKE_BUILD_TYPE Release)
  message("CMAKE_BUILD_TYPE is " ${CMAKE_BUILD_TYPE})
endif(NOT CMAKE_BUILD_TYPE)

include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)

option(TBB_LIB "Add TBB" TRUE)

message(STATUS "CXX = \"" $ENV{CXX} "\" of ${TARGET}")

# Compile with nvc++
# string(APPEND CMAKE_CXX_FLAGS "-acc=gpu -ta=nvidia ")
# string(APPEND CMAKE_CXX_FLAGS "-acc=gpu -gpu=cc61 ")

# string(APPEND CMAKE_CXX_FLAGS "-ffast-math -mavx2 -march=native -fopenmp ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast -Wunused-variable -Wunused-parameter")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wextra -Wreorder -Wconversion -Wsign-conversion")
get_filename_component(HOME_DIR ${PROJECT_SOURCE_DIR} PATH)
set(WORKDIR ${CMAKE_CURRENT_LIST_DIR})
set(INCLUDE_DIR ${HOME_DIR}/include)
message(STATUS "INCLUDE_DIR in: ${INCLUDE_DIR}")
message(STATUS "WORKDIR in: ${WORKDIR}")
message(STATUS "PROJECT_SOURCE_DIR in: ${PROJECT_SOURCE_DIR}")

include_directories(${INCLUDE_DIR})
include_directories($ENV{CUDA_PATH}/include)
include_directories($ENV{CUDA_PATH}/targets/x86_64-linux/include/)

file(GLOB SRC
  "${WORKDIR}/*.cpp"
)

include(FetchContent)

FetchContent_Declare(
  acc_gpu
  URL ${HOME_DIR}/acc
)

FetchContent_Declare(
  omp_loop
  URL ${HOME_DIR}/omp
)

FetchContent_Declare(
  gpu_thrustlib
  URL ${HOME_DIR}/thrust_gpu
)

FetchContent_Declare(
  cpu_thrustlib
  URL ${HOME_DIR}/thrust_cpu
)

FetchContent_MakeAvailable(acc_gpu)
FetchContent_MakeAvailable(omp_loop)
FetchContent_MakeAvailable(gpu_thrustlib)
FetchContent_MakeAvailable(cpu_thrustlib)

add_executable(
  ${TARGET}
  ${SRC}
)

set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
target_compile_features(${TARGET} PRIVATE cxx_std_${CMAKE_CXX_STANDARD})

set_target_properties(
  ${TARGET}
  PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

file(GLOB CUDA_LIBS
  "$ENV{CUDA_PATH}/lib64/*cudart*.so"
  # "$ENV{CUDA_PATH}/lib64/*.so"
  # "$ENV{CUDA_PATH}/lib64/*.a"
  "$ENV{CUDA_PATH}/lib64/libnvToolsExt.so"
)

# message(${CUDA_LIBS})
# Sometime the kit is instaled here
link_directories("${CUDA_TOOLKIT_ROOT_DIR}/lib64")
target_link_libraries(
  ${TARGET}
  PRIVATE
  stencil_acc
  stencil_omp
  stencil_thrustlib_gpu
  stencil_thrustlib_cpu
  pthread
  dl
  rt
  tbb
  ${CUDA_LIBS}
)

# Be sure tha -lto is on on this target
# if(supported)
#   message(STATUS "IPO / LTO enabled")
#   set_property(TARGET ${TARGET} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
# else()
#   message(STATUS "IPO / LTO not supported: <${error}>")
# endif()

find_package(OpenMP REQUIRED)

if(OpenMP_FOUND)
  string(APPEND CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  string(APPEND CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
  # -lgomp
endif(OpenMP_FOUND)

find_package(TBB REQUIRED)

if(TBB_LIB)
  file(GLOB TBB_LIBS
    "$ENV{TBB_PATH}/libtbb.so")
  add_compile_definitions(_TBB=True)
  target_link_libraries(${TARGET} PRIVATE ${TBB_LIBS})
  # -ltbb
  # TBB::tbb
endif(TBB_LIB)
