
cmake_minimum_required(VERSION 3.0.0)
project(preprocessor VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Downloaded Libraries
find_package(rapidfuzz REQUIRED)
find_package(OpenXLSX REQUIRED)
find_package(plplot REQUIRED)

include_directories(AFTER /usr/local/include/opencv2)
include_directories(AFTER /usr/local/cuda/targets/x86_64-linux/include) 
include_directories(AFTER /usr/local/lib)
include_directories(AFTER /usr/local/include)
include_directories(AFTER ext)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories( ${OpenCV_INCLUDE_DIRS} )

file( GLOB APP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp )
foreach( sourcefile ${APP_SOURCES} )
    file(RELATIVE_PATH filename ${CMAKE_CURRENT_SOURCE_DIR} ${sourcefile})
    string( REPLACE ".cpp" "" file ${filename} )
    add_executable( ${file} ${sourcefile} )
    target_link_libraries( ${file} ${OpenCV_LIBS} )
endforeach( sourcefile ${APP_SOURCES} )


# Preprocessor.cpp
# add_executable(preprocessor src/includes.h src/preprocessor.cpp src/ProjectUtilities.cpp)
# target_link_libraries(preprocessor plplot OpenXLSX::OpenXLSX rapidfuzz::rapidfuzz opencv_core opencv_imgcodecs opencv_imgproc opencv_cudaimgproc opencv_cudafilters)

#[[
add_executable(preprocessor 
  src/includes.h src/ProjectUtilities.h src/preprocessor.cpp 
  src/LGAlgorithm.hpp src/ProjectUtilities.cpp src/kMeansNGBContainer.hpp
  src/LGNode.hpp src/CurveLineSegmentMetaData.hpp src/ChainCodingContainer.hpp
  src/Direction.hpp src/LineSegmentContainer.hpp src/TestFunctions.hpp)
]]
add_executable(preprocessor 
src/includes.h src/ProjectUtilities.h src/preprocessor.cpp 
src/ProjectUtilities.cpp src/kMeansNGBContainer.hpp
src/LGNode.hpp src/CurveLineSegmentMetaData.hpp src/ChainCodingContainer.hpp
src/Direction.hpp src/LineSegmentContainer.hpp src/TestFunctions.hpp)
#[[target_link_libraries(preprocessor plplot OpenXLSX::OpenXLSX rapidfuzz::rapidfuzz opencv_core opencv_imgcodecs opencv_imgproc opencv_cudaimgproc opencv_cudafilters)
]]
target_link_libraries(preprocessor plplot OpenXLSX::OpenXLSX rapidfuzz::rapidfuzz ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_BUILD_TYPE Debug)
include(CPack)



#[[
cmake_minimum_required(VERSION 3.0.0)
project(kernel VERSION 0.1.0 LANGUAGES C CXX)
set (CMAKE_CXX_STANDARD 17)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include(CTest)
enable_testing()
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(rapidfuzz REQUIRED)

include_directories(AFTER /usr/local/include/opencv2/)
include_directories(AFTER /usr/local/lib)
include_directories(AFTER /usr/local/cuda/include)
include_directories(AFTER obstruction-preprocessor-service/ext)

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
    src/preprocessor.cpp
    src/ProjectUtilities.h
    src/ProjectUtilities.cpp
    src/TestFunctions.hpp
    src/includes.h
    src/kMeansNGBContainer.hpp
)

# Add CUDA executable
cuda_add_executable(kernel ${CUDA_SOURCE_FILES})

target_link_libraries(kernel rapidfuzz::rapidfuzz opencv_core opencv_imgcodecs opencv_imgproc opencv_cudaimgproc opencv_cudafilters)

# Set CUDA flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 ${CUDA_ARCHITECTURES}")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
]]