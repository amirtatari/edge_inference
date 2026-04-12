add_executable(tests
  tflite_test.cpp
  queue_test.cpp
  config_test.cpp
)

target_link_libraries(tests PRIVATE    
    src
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tests)
