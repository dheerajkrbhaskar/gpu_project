cmake_minimum_required(VERSION 3.20)

project(GPUAcceleratedCrackDetection
    VERSION 1.0
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)

find_program(GPU_CRACK_NVCC nvcc)
set(GPU_CRACK_HAS_CUDA OFF)

if(GPU_CRACK_NVCC)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()
    find_package(CUDAToolkit REQUIRED)
    set(GPU_CRACK_HAS_CUDA ON)
endif()

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_executable(gpu_crack_detection
    src/main.cpp
    src/cpu_pipeline.cpp
    $<$<BOOL:${GPU_CRACK_HAS_CUDA}>:src/gpu_pipeline.cu>
    $<$<NOT:$<BOOL:${GPU_CRACK_HAS_CUDA}>>:src/gpu_pipeline_stub.cpp>)

target_include_directories(gpu_crack_detection PRIVATE src)
target_link_libraries(gpu_crack_detection PRIVATE)
target_compile_definitions(gpu_crack_detection PRIVATE GPU_CRACK_PYTHON_EXECUTABLE="${Python3_EXECUTABLE}")

if(GPU_CRACK_HAS_CUDA)
    target_link_libraries(gpu_crack_detection PRIVATE CUDA::cudart)
    target_compile_definitions(gpu_crack_detection PRIVATE GPU_CRACK_HAS_CUDA=1)
    target_compile_options(gpu_crack_detection PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
    set_target_properties(gpu_crack_detection PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
else()
    target_compile_definitions(gpu_crack_detection PRIVATE GPU_CRACK_HAS_CUDA=0)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(gpu_crack_detection PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(gpu_crack_detection PRIVATE GPU_CRACK_HAVE_OPENMP=1)
endif()
