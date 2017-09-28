#include <iostream>
#include "RandomEngineFactory.h"
#include "Generator.h"

int main()
{
	RandomEngineFactory::initialize(); //optional
	example::uniformrealdist_hist_example();
	example::normaldist_histogram_example();
	example::exponentialdist_hist_example();
	example::generator_example();

	return 0;
}
