cmake_minimum_required(VERSION 3.16)

project(edge_inference VERSION 1.0.0) 

# define c++ standard and compiler flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt")

# add debug flag if the build type is DEBUG
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE)
if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    # Check if -g flag is already present
    if(NOT CMAKE_CXX_FLAGS_DEBUG MATCHES "-g")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
    endif()
endif()

# define delegate type
#set(DELEGATE_TYPE "")
string(TOUPPER ${DELEGATE_TYPE} DELEGATE_TYPE)

# define libs directory
set(LIBS_DIR "${CMAKE_SOURCE_DIR}/libs")

# define tflite source directory
set(TF_SRC_DIR "${LIBS_DIR}/tensorflow_src")

# define src link_directories
set(SRC_DIRS
  ${CMAKE_CURRENT_SOURCE}/base
)

# define tflite lib dir
if(DELEGATE_TYPE STREQUAL "CPU")
  add_compile_definitions(DELEGATE_CPU)
  set(TFLITE_DIR "${LIBS_DIR}/tflite_x86_cpu")
elseif(DELEGATE_TYPE STREQUAL "GPU")
  add_compile_definitions(DELEGATE_GPU)
  set(TFLITE_DIR "${LIBS_DIR}/tflite_x86_gpu")
else()
  message(FATAL_ERROR "Invalid delegate type: ${DELEGATE_TYPE}, Available options: CPU, GPU")
endif()

# add tflite lib
add_library(tflite SHARED IMPORTED)
set_property(TARGET tflite PROPERTY IMPORTED_LOCATION "${TFLITE_DIR}/libtensorflow-lite.so")

include_directories(
  ${SRC_DIRS}
  ${TF_SRC_DIR}
)

add_executable(${PROJECT_NAME} 
  main.cpp 
  base/engine.cpp
  base/model.cpp
)

target_link_libraries(${PROJECT_NAME} 
  PRIVATE
  tflite
)
