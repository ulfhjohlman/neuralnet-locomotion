file(GLOB local_headers ./*.h)
add_library(testframework TestFramework.cpp ${local_headers})
