file(GLOB local_headers ./*.h)
add_library(utility Dummyfile.cpp ${local_headers})
