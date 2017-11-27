cmake_minimum_required(VERSION 3.9 FATAL_ERROR)

# VS name
project(neuralnet_locomotion)

#sets cpp standard to "std=c++14"
set(CMAKE_CXX_STANDARD 14)

#Compiler flags
option(USE_AVX_INSTRUCTIONS  "Compile your program with AVX instructions"  ON)
IF(USE_AVX_INSTRUCTIONS)
	IF(MSVC)
		add_definitions(/arch:AVX)
	ELSE()
		add_definitions (-mavx)
	ENDIF()
ENDIF()


#look for include files in
include_directories("${PROJECT_SOURCE_DIR}/lib/Eigen"
 "${PROJECT_SOURCE_DIR}/bullet"
 "${PROJECT_SOURCE_DIR}/gpu"
 "${PROJECT_SOURCE_DIR}/neuralnet"
 "${PROJECT_SOURCE_DIR}/evolutionaryalgorithm"
 "${PROJECT_SOURCE_DIR}/utility"
 "${PROJECT_SOURCE_DIR}/testframework"
 "${PROJECT_SOURCE_DIR}/lib/tinyxml2"
 "${PROJECT_SOURCE_DIR}/rl"
 "${PROJECT_SOURCE_DIR}/rl_test"
 "${PROJECT_SOURCE_DIR}"
 "${PROJECT_SOURCE_DIR}/mj_render"
 "${PROJECT_SOURCE_DIR}/mjenvironment")

#add library subdirectories -> executes subs CMakeLists.txt
add_subdirectory("${PROJECT_SOURCE_DIR}/bullet")
add_subdirectory("${PROJECT_SOURCE_DIR}/neuralnet")
add_subdirectory("${PROJECT_SOURCE_DIR}/evolutionaryalgorithm")
add_subdirectory("${PROJECT_SOURCE_DIR}/lib/tinyxml2")
add_subdirectory("${PROJECT_SOURCE_DIR}/utility")
add_subdirectory("${PROJECT_SOURCE_DIR}/testframework")
add_subdirectory("${PROJECT_SOURCE_DIR}/rl")


#add libraries LIBS macro
set(LIBS ${LIBS} ${BULLET_LIBRARIES} neuralnet evo_alg tinyxml utility testframework rl)

#Turn on options by invoking $cmake -D<option_name>=ON <src path>
#aka $cmake -D_DEBUG=ON ..
option(_DEBUG "Flag for preprocessor to enable DEBUG macros" ON)

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
add_subdirectory("${PROJECT_SOURCE_DIR}/rl_test")

#Make visual studio specific projects
IF(MSVC)
	#link_directories("${PROJECT_SOURCE_DIR}/lib/mjpro150/bin")
	add_subdirectory("${PROJECT_SOURCE_DIR}/mjenvironment")
	add_subdirectory("${PROJECT_SOURCE_DIR}/mj_render")
ENDIF()

option(GENERATE_GPU  "Make gpu library"  ON)
IF(GENERATE_GPU)
	add_subdirectory("${PROJECT_SOURCE_DIR}/gpu")
ENDIF()

