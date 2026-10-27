cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-2201" CACHE STRING "Target NPU arch")
set(CMAKE_ASC_RUN_MODE "sim" CACHE STRING "Run mode: sim/cpu/npu")

find_package(ASC REQUIRED)
project(vecadd_project LANGUAGES ASC CXX)

# Kernel (.asc) + Host (.cpp)
add_executable(vecadd_app kernel_vecadd.asc host_main.cpp)

target_compile_options(vecadd_app PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
target_compile_options(vecadd_app PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-std=c++17>
)

target_link_libraries(vecadd_app PRIVATE tiling_api platform)
