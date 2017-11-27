message("${CMAKE_ROOT} + modules jesus christ")
#include("${CMAKE_ROOT}/Modules/FindCuda.cmake")

file(GLOB local_headers ./*.h)
set(gpu gpu.cpp ${local_headers})

add_executable(gpu gpu.cpp ${HEADER_FILES})

include_directories("${CUDA_TOOLKIT_INCLUDE}/")
target_link_libraries(gpu "${CUDA_CUDA_LIBRARY}")
target_link_libraries(gpu "${CUDA_cudart_static_LIBRARY}")
target_link_libraries(gpu "${CUDA_cublas_LIBRARY}")
target_link_libraries(gpu "${CUDA_cusparse_LIBRARY}")