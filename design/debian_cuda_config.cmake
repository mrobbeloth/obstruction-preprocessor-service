set(WITH_CUDA ON CACHE BOOL "")
set(WITH_CUDNN ON CACHE BOOL "")
set(OPENCV_DNN_CUDA ON CACHE BOOL "")
set(CUDA_ARCH_BIN "5.2 6.1 7.5 8.6" CACHE STRING "")
# on GTX 970 systems, use set(CUDA_NVCC_FLAGS "-ccbin /usr/bin/gcc-11")
# Add other flags here using the 'set(... CACHE ...)' syntax
