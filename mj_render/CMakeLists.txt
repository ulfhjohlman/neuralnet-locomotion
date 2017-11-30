include_directories("${PROJECT_SOURCE_DIR}/lib/mjpro150/include")

file(GLOB local_headers ./*.h)
set(mj_render mj_render.cpp ${local_headers})
add_executable(mj_render mj_render.cpp ${local_headers})

find_library(MUJOCO mujoco150 "${PROJECT_SOURCE_DIR}/lib/mjpro150/bin")
find_library(GLFW3 glfw3 "${PROJECT_SOURCE_DIR}/lib/mjpro150/bin")

if (MUJOCO)
	message(STATUS "MUJOCO LIB FOUND")
endif()

if (GLFW3)
	message(STATUS "GLFW3 LIB FOUND")
endif()

target_link_libraries(mj_render ${MUJOCO} ${GLFW3} ${LIBS})

#COPYING OF NECESSARY FILES
add_custom_command(TARGET mj_render POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different 
        "${PROJECT_SOURCE_DIR}/lib/mjpro150/bin/mujoco150.dll"   
        $<TARGET_FILE_DIR:mj_render>)    

add_custom_command(TARGET mj_render POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${PROJECT_SOURCE_DIR}/lib/mjpro150/bin/glfw3.dll"  
        $<TARGET_FILE_DIR:mj_render>) 

add_custom_command(TARGET mj_render POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${PROJECT_SOURCE_DIR}/mjenvironment/mjkey.txt"  
        $<TARGET_FILE_DIR:mj_render>/..) 

add_custom_command(TARGET mj_render POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${PROJECT_SOURCE_DIR}/mjenvironment/humanoid100.xml"  
        $<TARGET_FILE_DIR:mj_render>/..) 

add_custom_command(TARGET mj_render POST_BUILD 
    COMMAND ${CMAKE_COMMAND} -E copy_if_different  
        "${PROJECT_SOURCE_DIR}/mjenvironment/humanoid.xml"  
        $<TARGET_FILE_DIR:mj_render>/..)