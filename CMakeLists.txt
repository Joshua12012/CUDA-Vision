cmake_minimum_required(VERSION 3.18)
project(cuda_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP for CPU multi-threading
find_package(OpenMP REQUIRED)

# Check for CUDA Toolkit support
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA compiler found: ${CMAKE_CUDA_COMPILER}")
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA not found. Configuring CPU Heterogeneous mode.")
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

set(CORE_SOURCES
    src/image_buffer.cpp
    src/signal_buffer.cpp
    src/cpu/cpu_engine.cpp
    src/benchmark.cpp
    src/main.cpp
)

if(CUDA_AVAILABLE)
    add_executable(cuda_vision
        ${CORE_SOURCES}
        src/cuda/cuda_kernels.cu
    )
    target_compile_definitions(cuda_vision PRIVATE CUDA_ENABLED)
    target_link_libraries(cuda_vision PRIVATE OpenMP::OpenMP_CXX)
else()
    add_executable(cuda_vision
        ${CORE_SOURCES}
    )
    target_link_libraries(cuda_vision PRIVATE OpenMP::OpenMP_CXX)
endif()

# Installation target
install(TARGETS cuda_vision DESTINATION bin)
