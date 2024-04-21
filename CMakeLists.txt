cmake_minimum_required(VERSION 3.25 FATAL_ERROR)

project(model-runner LANGUAGES CXX CUDA)

add_library(runner STATIC src/runner.cpp src/ops.cu)
target_compile_features(runner PUBLIC cxx_std_20)
set_target_properties( runner PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(model_sample src/sample/main.cpp)
target_link_libraries(model_sample PRIVATE runner)
