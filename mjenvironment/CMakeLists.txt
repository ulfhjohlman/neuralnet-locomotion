add_executable(mjenvironment mjenvironment.cpp)
include_directories("${PROJECT_SOURCE_DIR}/lib/mjpro150/include")

target_link_libraries (mjenvironment mujoco150.lib)
target_link_libraries (mjenvironment mujoco150.dll)
target_link_libraries (mjenvironment glfw3.lib)
target_link_libraries (mjenvironment glfw3.dll)
target_link_libraries (mjenvironment ${LIBS})

message(STATUS "add to environment variable PATH=%PATH%;$(ProjectDir)/../lib/mjpro150/bin/     assuming you have generated vs in source repository")