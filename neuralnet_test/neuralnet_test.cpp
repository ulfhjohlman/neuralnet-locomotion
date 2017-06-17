#include "stdafx.h"
#include "../neuralnet/Dataset.h"
#include "../neuralnet/Stopwatch.h"

#ifdef _DEBUG
#pragma comment(lib, "../Debug/neuralnet.lib")
#else
#pragma comment(lib, "../Release/neuralnet.lib")
#endif // _DEBUG

#include <iostream>

//Function prototypes
void stopwatch_test();
void dataset_test();

int main()
{
	stopwatch_test();
	dataset_test();
	return 0;
}


void stopwatch_test()
{
	Stopwatch<std::milli> sw;
	std::cout << sw.getAbsoluteTime() << std::endl;
	std::cout << "Stopwatch test passed" << std::endl;
}

void dataset_test()
{
	Dataset d;
	std::cout << "Dataset test passed" << std::endl;
}