cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

# VS name
project(neuralnet_locomotion)

#sets cpp standard to "std=c++14"
set(CMAKE_CXX_STANDARD 14)

#look for include files in
include_directories("${PROJECT_SOURCE_DIR}/lib/Eigen"
 "${PROJECT_SOURCE_DIR}/neuralnet"
 "${PROJECT_SOURCE_DIR}/evolutionaryalgorithm"
 "${PROJECT_SOURCE_DIR}/utility"
 "${PROJECT_SOURCE_DIR}/testclasses"
 "${PROJECT_SOURCE_DIR}/lib/tinyxml2")

#add library subdirectories -> executes subs CMakeLists.txt
add_subdirectory("${PROJECT_SOURCE_DIR}/neuralnet")
add_subdirectory("${PROJECT_SOURCE_DIR}/evolutionaryalgorithm")
add_subdirectory("${PROJECT_SOURCE_DIR}/lib/tinyxml2")
add_subdirectory("${PROJECT_SOURCE_DIR}/utility")

#add libraries LIBS macro
set(LIBS ${LIBS} neuralnet evo_alg tinyxml utility)

#Turn on options by invoking $cmake -D<option_name>=ON <src path>
#aka $cmake -DDEBUG=ON ..
option(DEBUG "Flag for preprocessor to enable DEBUG macros" OFF)

#reads config.h.in, outputs config.h
configure_file (
  "${PROJECT_SOURCE_DIR}/config.h.in"
  "${PROJECT_BINARY_DIR}/config.h"
  )

#to find config file when compiling
include_directories("${PROJECT_BINARY_DIR}")

#add _test directories which build locally
add_subdirectory("${PROJECT_SOURCE_DIR}/evolutionaryalgorithm_test")
add_subdirectory("${PROJECT_SOURCE_DIR}/neuralnet_test")
