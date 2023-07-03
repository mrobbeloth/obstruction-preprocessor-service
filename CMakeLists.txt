cmake_minimum_required(VERSION 3.0.0)
project(preprocessor VERSION 0.1.0 LANGUAGES C CXX)
set (CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include(CTest)
enable_testing()
find_package(OpenCV REQUIRED)

# include SuperString
add_subdirectory(ext/SuperString)

# add SuperString headers to include directory
include_directories(AFTER ext/SuperString/include)
include_directories(AFTER ext/ThreadPool/include)
include_directories(AFTER ext/ThreadPool-master-cpp-11/ThreadPool-master/include)

cuda_add_executable(preprocessor src/preprocessor.cu src/utility.cpp src/BS_thread_pool.cpp)

# link your executable against SuperString w/ SuperString
target_link_libraries(preprocessor SuperString opencv_core opencv_imgcodecs opencv_imgproc)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)