cmake_minimum_required(VERSION 3.10)
project(preprocessor VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

option(ENABLE_CUDA "Enable CUDA GPU acceleration (requires CUDA toolkit and OpenCV CUDA modules)" ON)

find_package(OpenCV REQUIRED)

set(CUDA_AVAILABLE FALSE)
if(ENABLE_CUDA)
  find_package(CUDA QUIET)
  if(CUDA_FOUND)
    set(CUDA_AVAILABLE TRUE)
    message(STATUS "CUDA found — GPU acceleration enabled")
  else()
    message(STATUS "CUDA not found — building CPU-only")
  endif()
else()
  message(STATUS "CUDA disabled by user — building CPU-only")
endif()

add_executable(preprocessor src/preprocessor.cpp src/utility.cpp src/CompositeMat.cpp src/kMeansNGBContainer.cpp)
target_include_directories(preprocessor PRIVATE ${OpenCV_INCLUDE_DIRS})

if(CUDA_AVAILABLE)
  target_compile_definitions(preprocessor PRIVATE HAVE_CUDA)
  target_link_libraries(preprocessor ${OpenCV_LIBS} opencv_cudaimgproc opencv_cudafilters opencv_cudaarithm)
else()
  target_link_libraries(preprocessor ${OpenCV_LIBS})
endif()

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
