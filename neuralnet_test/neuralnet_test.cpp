#include "stdafx.h"

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <sstream>

#include "../neuralnet/Dataset.h"
#include "../neuralnet/DatasetException.h"

#include "../utility/Stopwatch.h"
#include "../utility/XMLException.h"
#include "../utility/DataPrinter.h"
#include "../utility/XMLWrapper.h"
#include "../utility/utilityfunctions.h"
#include "../utility/ThreadPoolv2.h"

#include "../testclasses/StopwatchTest.h"

#ifdef _DEBUG
#pragma comment(lib, "../x64/Debug/neuralnet.lib")
#pragma comment(lib, "../x64/Debug/testclasses.lib")
#pragma comment(lib, "../x64/Debug/utility.lib")
#else
#pragma comment(lib, "../x64/Release/neuralnet.lib")
#pragma comment(lib, "../x64//Release/testclasses.lib")
#pragma comment(lib, "../x64/Release/utility.lib")
#endif // _DEBUG

//c++ standard includes
#include <iostream>
#include <exception>
#include <vector>
#include <thread>
#include <chrono>
#include <functional>
#include <sstream> //ostringstream
#include <iomanip> //std::get_time

//Function prototypes
void single_threaded_tests();
std::string XMLWrapper_test();
std::string dataprinter_test();
std::string dataset_test();

int main()
{
	std::cout.sync_with_stdio(true); // make cout thread-safe
	
	//test 
	single_threaded_tests();

	std::cout << "Neural net tests done." << std::endl;
	std::cin.get();
	return 0;
}

/// <summary>
/// 
/// </summary>
void single_threaded_tests()
{
	ThreadPool pool;

	//methods, pass: &function == temporary pointer to function.
	auto test_dataprinter = pool.submit(&dataprinter_test);
	auto test_XMLWrapper = pool.submit(&XMLWrapper_test);
	auto test_dataset = pool.submit(&dataset_test);

	//TestFramework
	StopwatchTest sw_test;
	pool.addWork(sw_test);

	while (!pool.isDone()) { pool.help(); }

	auto print = [](const std::string& message) { std::cout << message << std::endl; };
	//methods result
	print(test_dataprinter.get());
	print(test_XMLWrapper.get());
	print(test_dataset.get());

	//TestFramework classes
	sw_test.print();
}

std::string XMLWrapper_test()
{
	XMLWrapper d;
	std::ostringstream output;
	output << "XMLWrapper test:" << std::endl;
	try {
		d.clearDocument();
		d.insertNewRoot("XML");
		d.insertNewRoot("newRoot"); //overwrite XML
		d.insertNewElement<int>("int", 5);
		std::vector<float> floats = { 1, 2, 3, 4, 5, 6.005f, 42.01f }; //Will not be exact representation
		d.insertNewElements<float>("floats", floats);
		d.insertDate();
		
		d.selectRootNode("newRoot");
		d.selectCurrentElement("floats");
		output << "item count: " << d.getNumberOfItems() << "==" << floats.size() << std::endl;

		//d.print();
		output << "Dataset test passed" << std::endl;
		return output.str();
	}
	catch (XMLException& e) {
		output << e.what() << std::endl;
	}
	catch (DatasetException& e) {
		output << e.what() << std::endl;
	}
	output << "Dataset test failed." << std::endl;
	return output.str();
}

std::string dataprinter_test()
{
	DataPrinter dp;
	std::ostringstream output;
	std::vector<float> floats = { 1, 2, 3, 4, 5, 6.00005f, 42.0231231231231f }; //Will not be exact representation
	dp.write<float>(floats);
	std::string data = dp.getString();

	output << "DataPrinter test:" << std::endl;
	output << data << std::endl;
	output << "DataPrinter test passed" << std::endl;

	return output.str();
}

std::string dataset_test()
{
	Dataset d;
	std::ostringstream output;
	output << "Dataset test:" << std::endl;
	try
	{
		d.createDataset("alpha");
		d.setDescription("Greek alphabet");
		d.setInputInfo(2, 50);


		DataPrinter dp;
		std::vector<float> x(50);
		std::vector<float> y(50);
		int i = 0;
		std::generate(x.begin(), x.end(), [&i] { return ++i; });
		std::transform(x.begin(), x.end(), y.begin(), [](float f) { return sin(f); });
		
		dp.write<float>(x);
		dp.write<float>(y);
		d.setInputData(dp.getString().c_str(), "float");
		d.getInputInfo();

		try { d.getOutputInfo(); }
		catch (std::exception e) { output << "caught expected std::exception: " << e.what() << std::endl; }

		output << "Dataset test passed" << std::endl;
		return output.str();
	}
	catch (XMLException e)
	{
		output << e.what() << std::endl;
	}
	catch (DatasetException e)
	{
		output << e.what() << std::endl;
	}
	catch (std::exception e)
	{
		output << e.what() << std::endl;
	}
	output << "Dataset test failed!" << std::endl;
	return output.str();
}