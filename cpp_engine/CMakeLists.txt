cmake_minimum_required(VERSION 3.18)

project(dsv4_cpp_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

add_library(dsv4_cpp_core STATIC
    src/json_lite.cpp
    src/gguf_reader.cpp
    src/safetensors_reader.cpp
    src/safetensors_model.cpp
    src/model_config.cpp
    src/tensor.cpp
    src/dsv4_engine.cpp
    src/sampler.cpp
    cuda/cuda_ops.cu
    cuda/bf16_ops.cu
    cuda/quant_gemm.cu
    cuda/fp4_ops.cu
    cuda/fp8_ops.cu
    cuda/rmsnorm_rope.cu
)

target_include_directories(dsv4_cpp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(dsv4_cpp_core PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(dsv4_cpp_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(dsv4_cpp_engine src/main.cpp)
target_link_libraries(dsv4_cpp_engine PRIVATE dsv4_cpp_core)

add_executable(inspect_gguf tools/inspect_gguf.cpp)
target_link_libraries(inspect_gguf PRIVATE dsv4_cpp_core)
set_target_properties(inspect_gguf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)

add_executable(inspect_safetensors tools/inspect_safetensors.cpp)
target_link_libraries(inspect_safetensors PRIVATE dsv4_cpp_core)
set_target_properties(inspect_safetensors PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)

add_executable(test_gguf_reader tests/test_gguf_reader.cpp)
target_link_libraries(test_gguf_reader PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_reader PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_safetensors_reader tests/test_safetensors_reader.cpp)
target_link_libraries(test_safetensors_reader PRIVATE dsv4_cpp_core)
set_target_properties(test_safetensors_reader PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_safetensors_model tests/test_safetensors_model.cpp)
target_link_libraries(test_safetensors_model PRIVATE dsv4_cpp_core)
set_target_properties(test_safetensors_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_quant_gemm tests/test_quant_gemm.cpp)
target_link_libraries(test_quant_gemm PRIVATE dsv4_cpp_core)
set_target_properties(test_quant_gemm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_fp4_matvec tests/test_fp4_matvec.cpp)
target_link_libraries(test_fp4_matvec PRIVATE dsv4_cpp_core)
set_target_properties(test_fp4_matvec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_fp8_matvec tests/test_fp8_matvec.cpp)
target_link_libraries(test_fp8_matvec PRIVATE dsv4_cpp_core)
set_target_properties(test_fp8_matvec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_rmsnorm tests/test_rmsnorm.cpp)
target_link_libraries(test_rmsnorm PRIVATE dsv4_cpp_core)
set_target_properties(test_rmsnorm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_attention_projection tests/test_attention_projection.cpp)
target_link_libraries(test_attention_projection PRIVATE dsv4_cpp_core)
set_target_properties(test_attention_projection PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_single_expert tests/test_moe_single_expert.cpp)
target_link_libraries(test_moe_single_expert PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_single_expert PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_embedding_head tests/test_embedding_head.cpp)
target_link_libraries(test_embedding_head PRIVATE dsv4_cpp_core)
set_target_properties(test_embedding_head PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_min_layer_smoke tests/test_min_layer_smoke.cpp)
target_link_libraries(test_min_layer_smoke PRIVATE dsv4_cpp_core)
set_target_properties(test_min_layer_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
