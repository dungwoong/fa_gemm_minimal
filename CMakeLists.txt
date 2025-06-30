cmake_minimum_required(VERSION 3.22.1)

project(CUDA_AMPERE LANGUAGES CXX CUDA) # specify project uses C++ and CUDA

# fetch the CUDA library
find_package(CUDA REQUIRED)

# Make sure CUDA language is available
include(CheckLanguage)
check_language(CUDA)

# This makes PTX error
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-g -G")  # enable cuda-gdb
# set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} "-lineinfo")

# KITTENS_4090 is ONLY used to figure out TK smem size
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} \
    -DKITTENS_4090 \
    -lineinfo \
    -std=c++20 \
    -DNDEBUG \
    -Xcompiler=-Wno-psabi \
    -Xcompiler=-fno-strict-aliasing \
    --expt-extended-lambda \
    --expt-relaxed-constexpr \
    -forward-unknown-to-host-compiler \
    --use_fast_math \
    -Xnvlink=--verbose \
    -Xptxas=--verbose \
    -Xptxas=--warn-on-spills \
")

# Set some variables
set(CMAKE_CXX_STANDARD 20)
set(CUDA_COMPUTE_CAPABILITY 120)
set(CMAKE_CUDA_ARCHITECTURES 120)

# CUDA_INCLUDE_DIRS is set by find_package(CUDA)
include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(${PROJECT_SOURCE_DIR}/src) # includes src in search path
include_directories(${PROJECT_SOURCE_DIR}/ThunderKittens/include)
include_directories(${PROJECT_SOURCE_DIR}/cuTK/include)

add_executable(fa4090ref fa2_4090_ref.cu)
set_target_properties(fa4090ref PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(fa4090ref ${CUDA_LIBRARIES} cuda)

add_executable(fa_ampere_cuda fa_ampere_cuda.cu)
set_target_properties(fa_ampere_cuda PROPERTIES CUDA_ARCHITECTURES ${CUDA_COMPUTE_CAPABILITY})
target_link_libraries(fa_ampere_cuda ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES} cuda)