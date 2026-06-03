cmake_minimum_required(VERSION 3.18)

project(vecAddRed LANGUAGES CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(vecAddRed
    cuda/main.cu
)
