file(GLOB local_headers ./*.h)
add_library(rl Dummyfile.cpp ${local_headers})
