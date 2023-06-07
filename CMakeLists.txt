cmake_minimum_required(VERSION 3.0.0)
project(preprocessor VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()
find_package(OpenCV REQUIRED)

include_directories(AFTER /usr/local/include/opencv2/)

add_executable(preprocessor src/preprocessor.cpp src/utility.cpp)

target_link_libraries(preprocessor opencv_core opencv_imgcodecs opencv_imgproc)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
