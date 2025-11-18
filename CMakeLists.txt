cmake_minimum_required(VERSION 3.16)

project(edge_inference VERSION 0.1.0) 

# find opencv
find_package(OpenCV REQUIRED)

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


# define libs directory
set(LIBS_DIR "${CMAKE_SOURCE_DIR}/libs")

# define tflite source directory
set(TF_SRC_DIR "${LIBS_DIR}/tensorflow_src")

# define delegate type
string(TOUPPER ${DELEGATE_TYPE} DELEGATE_TYPE)

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

# project src files
add_library(project_logic
  frameworks/tflite/engineLite.cpp
  frameworks/tensorRT/engineRt.cpp
  frameworks/openVino/engineVino.cpp
  utils/profiler/profiler.cpp
)

target_include_directories(project_logic PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/tflite
  ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/tensorRT
  ${CMAKE_CURRENT_SOURCE_DIR}/frameworks/openVino
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/profiler
  ${TF_SRC_DIR}
  ${OpenCV_INCLUDE_DIRS}
  ${LIBS_DIR}
)

add_executable(${PROJECT_NAME} 
  main.cpp 
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  project_logic
  tflite
  ${OpenCV_LIBS}
  )

enable_testing()
add_subdirectory(tests)
