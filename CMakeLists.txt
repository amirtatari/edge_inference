cmake_minimum_required(VERSION 3.10)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(CMAKE_CXX_STANDARD 17)

project(inference C CXX)

set(TENSORFLOW_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow")

add_subdirectory(
  "${TENSORFLOW_SOURCE_DIR}/tensorflow/lite"
  "${CMAKE_CURRENT_BINARY_DIR}/tensorflow-lite"
  EXCLUDE_FROM_ALL
)

add_executable(inference main.cpp inference.cpp)

target_link_libraries(inference
    tensorflow-lite
)
