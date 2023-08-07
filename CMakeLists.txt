cmake_minimum_required(VERSION 3.0.0)
project(preprocessor VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

include_directories(AFTER /usr/local/include/opencv2/)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lopencv_cudafeatures2d -lopencv_cudaimgproc -lopencv_cudafilters")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} opencv_cudafeatures2d opencv_cudaimgproc")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(preprocessor src/preprocessor.cpp src/utility.cpp src/CompositeMat.cpp)

target_link_libraries(preprocessor opencv_core opencv_imgcodecs opencv_imgproc opencv_cudaimgproc opencv_cudafilters)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
