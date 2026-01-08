include(FetchContent)

# set TFLite Configuration Flags 
set(TFLITE_ENABLE_XNNPACK ON CACHE BOOL "Enable XNNPACK" FORCE)

# avoid building TFLite tests to save time and disk space
set(TFLITE_ENABLE_TESTS OFF CACHE BOOL "Disable TFLite internal tests" FORCE)

set(ABSL_PROPAGATE_CXX_STD ON)

set(TAG_VERSION "v2.18.0")

message(STATUS "Fetching TensorFlow Lite ${TAG_VERSION}...")

FetchContent_Declare(
  tensorflow-lite
  GIT_REPOSITORY https://github.com/tensorflow/tensorflow.git
  GIT_TAG        ${TAG_VERSION}
  GIT_PROGRESS   TRUE
  SOURCE_SUBDIR  tensorflow/lite
)

FetchContent_MakeAvailable(tensorflow-lite)

set(TFLITE_SOURCE_DIR ${tensorflow-lite_SOURCE_DIR})