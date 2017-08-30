#include <iostream>

#include "odeinttests.h"
#include "vairoustests.h"

int main() {
	test_random_engines();
	strange_attractor();
	

	std::cout << "Done. Press any key to exit.\n";
	std::cin.get();
}

