set(TFLITE_LIB "${SDK_PATH}/lib/libtensorflow-lite.so")

if(NOT EXISTS "${TFLITE_LIB}")
    message(FATAL_ERROR
        "TFLite shared library not found at ${TFLITE_LIB}. "
        "Check that SDK_PATH is set correctly and TFLite shared artifacts "
        "were copied into ${SDK_PATH}/lib in the SDK image build.")
endif()

add_library(tensorflow-lite SHARED IMPORTED GLOBAL)
set_target_properties(tensorflow-lite PROPERTIES
    IMPORTED_LOCATION             "${TFLITE_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${SDK_PATH}/include"
)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_property(TARGET tensorflow-lite APPEND PROPERTY
        INTERFACE_LINK_OPTIONS "-Wl,-rpath-link,${SDK_PATH}/lib")
endif()

message(STATUS "TFLite found (prebuilt shared): ${TFLITE_LIB}")
