include(FetchContent)

# disable DNN and Protobuf to avoid collision with TFLite
set(BUILD_opencv_dnn OFF CACHE BOOL "Disable DNN to avoid Protobuf collision" FORCE)
set(BUILD_PROTOBUF OFF CACHE BOOL "Do not build internal Protobuf" FORCE)
set(PROTOBUF_UPDATE_FILES OFF CACHE BOOL "Do not update protobuf files" FORCE)

# force these to OFF to speed up the build and reduce binary size
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build Shared Libraries" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "No test" FORCE)
set(BUILD_PERF_TESTS OFF CACHE BOOL "Build perf tests" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "Build examples" FORCE)
set(BUILD_DOCS OFF CACHE BOOL "Build docs" FORCE)
set(BUILD_opencv_python3 OFF CACHE BOOL "Build Python bindings" FORCE)
set(BUILD_opencv_java OFF CACHE BOOL "Build Java bindings" FORCE)
set(BUILD_opencv_apps OFF CACHE BOOL "Build apps" FORCE)
set(BUILD_opencv_gapi OFF CACHE BOOL "Disable G-API" FORCE)

set(BUILD_JPEG ON CACHE BOOL "Build jpeg from source" FORCE)
set(BUILD_PNG ON CACHE BOOL "Build png from source" FORCE)


FetchContent_Declare(
  opencv
  GIT_REPOSITORY https://github.com/opencv/opencv.git
  GIT_TAG        4.10.0  
  GIT_SHALLOW    TRUE   
)

set(CMAKE_CXX_FLAGS_SAVE "${CMAKE_CXX_FLAGS}")
set(CMAKE_C_FLAGS_SAVE "${CMAKE_C_FLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -w")

FetchContent_MakeAvailable(opencv)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SAVE}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS_SAVE}")

set(OpenCV_LIBS 
    opencv_core 
    opencv_imgproc 
    opencv_highgui 
    opencv_imgcodecs 
    opencv_videoio
)