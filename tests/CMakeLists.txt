include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  )

# specific setting to prevent MSVC runtime library conflicts
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# 2. Create the Test Executable
add_executable(tests
  tfliteEngine_test.cpp)

# 3. Link Libraries
target_link_libraries(tests PRIVATE
    gtest_main     
    src
)

# 4. Enable Test Discovery (integration with CTest/IDE)
include(GoogleTest)
gtest_discover_tests(tests)
