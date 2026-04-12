find_package(OpenCV REQUIRED
    COMPONENTS core imgproc highgui imgcodecs videoio
    PATHS "${SDK_PATH}/lib/cmake/opencv4"
    NO_DEFAULT_PATH
)

message(STATUS "OpenCV found: ${OpenCV_VERSION}  libs=${OpenCV_LIBS}")