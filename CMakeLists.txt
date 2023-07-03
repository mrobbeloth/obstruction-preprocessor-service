cmake_minimum_required(VERSION 3.0.0)
project(preprocessor VERSION 0.1.0 LANGUAGES C CXX)
set (CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include(CTest)
enable_testing()
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# include SuperString
add_subdirectory(ext/SuperString)

# add SuperString headers to include directory
include_directories(AFTER ext/SuperString/include)
include_directories(AFTER ext/ThreadPool/include)
include_directories(AFTER ext/ThreadPool-master-cpp-11/ThreadPool-master/include)

# Get CUDA architectures
execute_process(
  COMMAND ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc --run
          -Xcompiler -v
          ${CUDA_SOURCE_FILES}
  RESULT_VARIABLE nvcc_result
  OUTPUT_VARIABLE nvcc_output
  ERROR_VARIABLE nvcc_error
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)

string(REGEX MATCHALL "compute_([0-9]+)" nvcc_arch_list "${nvcc_output}")
list(REMOVE_DUPLICATES nvcc_arch_list)

set(CUDA_ARCHITECTURES "")
foreach(nvcc_arch ${nvcc_arch_list})
  string(REGEX REPLACE "compute_([0-9]+)" "\\1" sm_ver ${nvcc_arch})
  set(CUDA_ARCHITECTURES "${CUDA_ARCHITECTURES} -gencode arch=compute_${sm_ver},code=sm_${sm_ver}")
endforeach()

# Add CUDA source files
set(CUDA_SOURCE_FILES 
    src/preprocessorKernels.h 
    src/preprocessor.cu 
    src/utility.cpp
)

# Add CUDA executable
cuda_add_executable(preprocessor ${CUDA_SOURCE_FILES})

# Set CUDA flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 ${CUDA_ARCHITECTURES}")

# link your executable against SuperString w/ SuperString
target_link_libraries(preprocessor SuperString opencv_core opencv_imgcodecs opencv_imgproc)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)