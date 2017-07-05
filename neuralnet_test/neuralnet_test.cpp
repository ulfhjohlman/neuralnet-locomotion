#include "stdafx.h"

#include "../neuralnet/Dataset.h"
#include "../neuralnet/Stopwatch.h"
#include "../neuralnet/DatasetException.h"
#include "../neuralnet/XMLException.h"
#include "../neuralnet/DataPrinter.h"

#include "../testclasses/ThreadsafeQueue.h"
#include "../testclasses/utilityfunctions.h"

#ifdef _DEBUG
#pragma comment(lib, "../x64/Debug/neuralnet.lib")
#pragma comment(lib, "../x64/Debug/testclasses.lib")
#else
#pragma comment(lib, "../x64/Release/neuralnet.lib")
#pragma comment(lib, "../x64//Release/testclasses.lib")
#endif // _DEBUG

#include <iostream>
#include <exception>
#include <vector>
#include <thread>
#include <chrono>
#include <functional>

//Function prototypes
void stopwatch_test();
void dataset_test();
void dataprinter_test();

int main()
{
	std::cout.sync_with_stdio(true); // make cout thread-safe

	dataprinter_test();
	stopwatch_test();
	dataset_test();

	std::cout << "Neural net tests done." << std::endl;
	std::cin.get();
	return 0;
}

void stopwatch_test()
{
	using namespace std::chrono_literals;
	Stopwatch<std::milli> sw;
	sw.getLapTime();
	auto start = std::chrono::high_resolution_clock::now();
	std::this_thread::sleep_for(500ms);
	auto end = std::chrono::high_resolution_clock::now();

	double stopwatchTime = sw.getLapTime();

	std::chrono::duration<double, std::milli> elapsed = end - start;
	std::cout << "Waited " << elapsed.count() << " ms, " << " Stopwatch lap time=" << stopwatchTime << std::endl;

	const double absoluteErrorThreshold = 0.02; //20 micro sec
	if (std::abs(elapsed.count() - stopwatchTime) < absoluteErrorThreshold)
		std::cout << "Stopwatch test passed" << std::endl;
	else
		std::cout << "Stopwatch test failed" << std::endl;
}

void dataset_test()
{
	Dataset d;
	try {
		d.clearDocument();
		d.insertNewRoot("XML");
		d.insertNewRoot("newRoot");
		d.insertNewElement("int", 5);
		std::vector<float> floats = { 1, 2, 3, 4, 5, 6.005f, 42.01f }; //Will not be exact representation
		d.insertNewElements("floats", floats);
		d.insertDate();

		d.insertNewNode("newNode");
		d.selectRootNode("newNode");
		d.insertNewElement("int", 5);
		d.print();
		std::cout << "Dataset test passed" << std::endl;
		return;
	}
	catch (XMLException& e) {
		std::cerr << e.what() << std::endl;
	}
	catch (DatasetException& e) {
		std::cerr << e.what() << std::endl;
	}
	std::cerr << "Dataset test failed." << std::endl;
}

void dataprinter_test()
{
	DataPrinter dp;
	std::vector<float> floats = { 1, 2, 3, 4, 5, 6.00005f, 42.0231231231231f }; //Will not be exact representation
	dp.write(floats);
	std::cout << dp.getString() << std::endl;
	std::cout << "DataPrinter test passed" << std::endl;
}