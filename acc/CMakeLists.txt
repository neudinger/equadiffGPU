cmake_minimum_required(VERSION 3.19...3.23 FATAL_ERROR)
# SET(CMAKE_C_COMPILER_WORKS 1)
# SET(CMAKE_CXX_COMPILER_WORKS 1)

set(TARGET "stencil_acc")
set(LINK_TYPE STATIC)
set(TARGET_VERSION 1.0)

project(${TARGET} LANGUAGES CXX VERSION ${TARGET_VERSION})

set(CMAKE_CXX_STANDARD 17)
message(STATUS ${TARGET} " LINK TYPE IS " ${LINK_TYPE})

set(CMAKE_CXX_COMPILER "nvc++")

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
message(STATUS ${TARGET} " CXX = \"" $ENV{CXX} "\" of " ${TARGET})
message(STATUS ${TARGET} " CXXFLAGS = \"" $ENV{CXXFLAGS} "\" of " ${TARGET})

# string(APPEND CMAKE_CXX_FLAGS "-acc=gpu -gpu=cc61 ")
# string(APPEND CMAKE_CXX_FLAGS "-acc=gpu -ta=nvidia ")

string(APPEND CMAKE_CXX_FLAGS "-march=native ")
if (${CMAKE_CXX_COMPILER} STREQUAL "GNU")
  string(APPEND CMAKE_CXX_FLAGS "-fcompare-debug-second ")
endif(${CMAKE_CXX_COMPILER} STREQUAL "GNU")

include_directories(${INCLUDE_DIR})
include_directories($ENV{CUDA_PATH}/include)
include_directories($ENV{CUDA_PATH}/targets/x86_64-linux/include/)

file(GLOB SRC
    "${WORKDIR}/*.cc"
)

add_library(${TARGET}
            ${LINK_TYPE}
            ${SRC}
)


file(GLOB CUDA_LIBS
    "$ENV{CUDA_PATH}/lib64/*.so"
)

target_link_libraries(
    ${TARGET}
    PRIVATE
    ${CUDA_LIBS}
)

set_property(TARGET ${TARGET} PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)

# set_source_files_properties(${TARGET} PROPERTIES COMPILE_OPTIONS "-lm -acc=gpu")
# add_compile_options(-lm)
# add_compile_options(-acc=gpu)
# add_compile_definitions(lm)
# add_compile_definitions(acc=gpu)