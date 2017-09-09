#include <iostream>
#include <string>

#include "odeinttests.h"
#include "vairoustests.h"

//This project is so fucking broken.
int main() {
	eigen_test();
	list_platforms();
	gpu_test1();

	std::cout << "Done. Press any key to exit.\n";
	std::cin.get();
}

