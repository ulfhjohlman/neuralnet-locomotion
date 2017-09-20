file(GLOB local_headers ./*.h)
add_library(neuralnet Dataset.cpp ${local_headers})
