cmake_minimum_required(VERSION 3.16)

project(edge_inference VERSION 0.2.0) 

# define c++ standard and compiler flags
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lrt")

# Optimization flags for Edge Deployment
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# add debug flag if the build type is DEBUG
string(TOUPPER "${CMAKE_BUILD_TYPE}" CMAKE_BUILD_TYPE)
if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
endif()

# define libs directory
set(LIBS_DIR "${CMAKE_SOURCE_DIR}/libs")

# opencv
include(${LIBS_DIR}/opencv/opencv.cmake)

# tflite
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/libs/tflite")
include(${LIBS_DIR}/tflite/tflite.cmake)

# project src files
add_library(src STATIC
  testBench/testBench.cpp
  engine/base.cpp
  engine/tfLite.cpp
  engine/tensorRt.cpp
  engine/openVino.cpp
  utils/profiler/profiler.cpp
  utils/config/config.cpp
  libs/pugi/pugixml.cpp
)

# create a testable library from src
target_link_libraries(src PUBLIC 
  tensorflow-lite
  ${OpenCV_LIBS}
)
target_include_directories(src PUBLIC 
  ${CMAKE_CURRENT_SOURCE_DIR}/testBench
  ${CMAKE_CURRENT_SOURCE_DIR}/engine
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/profiler
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/config
  ${LIBS_DIR}
  ${TFLITE_SOURCE_DIR}

  # opencv includes
  ${opencv_SOURCE_DIR}/include                    
  ${opencv_SOURCE_DIR}/modules/core/include       
  ${opencv_SOURCE_DIR}/modules/imgproc/include    
  ${opencv_SOURCE_DIR}/modules/imgcodecs/include  
  ${opencv_SOURCE_DIR}/modules/highgui/include    
  ${opencv_SOURCE_DIR}/modules/videoio/include
  
  # opencv generated files
  ${opencv_BINARY_DIR}   
  ${OPENCV_CONFIG_FILE_INCLUDE_DIR}
)

add_executable(${PROJECT_NAME} 
  main.cpp 
)

target_link_libraries(${PROJECT_NAME} PRIVATE
  src 
)

enable_testing()
add_subdirectory(tests)
