#include "stdafx.h"

#include <algorithm>
#include <numeric>
#include <type_traits>
#include <sstream>

#include "../neuralnet/Dataset.h"
#include "../neuralnet/DatasetException.h"
#include "../neuralnet/NeuralNet.h"
#include "../neuralnet/FeedForwardNeuralNet.h"
#include "../neuralnet/LayeredTopology.h"
#include "../neuralnet/CascadeNeuralNet.h"
#include "../neuralnet/RecurrentTopology.h"

#include "../lib/Eigen/dense"

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
std::string feedForwardNeuralNet_test();
std::string cascadeNeuralNet_test();

void multi_threaded_tests();
void parallel_for_test();


int main()
{
	std::cout.sync_with_stdio(true); // make cout thread-safe

	std::cout << cascadeNeuralNet_test() << std::endl;
	//std::cout << feedForwardNeuralNet_test();
	
	//test 
	//single_threaded_tests();
	//multi_threaded_tests();

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
	auto test_dataprinter      = pool.submit(&dataprinter_test);
	auto test_XMLWrapper       = pool.submit(&XMLWrapper_test);
	auto test_dataset          = pool.submit(&dataset_test);

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

std::string feedForwardNeuralNet_test() {
	std::ostringstream output;
	const int n_inputs = 2;
	LayeredTopology * top = new LayeredTopology{ n_inputs, 2, 3, 1 };

	FeedForwardNeuralNet ffnn(top);
	ffnn.initializeRandomWeights();

	MatrixType m(n_inputs, 1);
	m.setRandom();

	ffnn.input(m);
	output << ffnn.output() << std::endl; 

	output << "FFNN test passed.\n";
	return output.str();
}

std::string cascadeNeuralNet_test() {
	std::ostringstream output;
	const int n_inputs = 2;

	try {
		CascadeTopology * top = new CascadeTopology{ n_inputs, 2, 2, 1 };
		top->addLayerConnection(1, { 0 });
		top->addLayerConnection(2, { 0, 1 });
		top->addLayerConnection(3, { 0, 1, 2 });

		CascadeNeuralNet cnn(top);
		cnn.initializeRandomWeights();

		MatrixType m(n_inputs, 1);
		m.setRandom();

		cnn.input(m);
		output << cnn.output() << std::endl;


		output << "CNN test passed.\n";
		return output.str();
	}
	catch (NeuralNetException e) {
		output << e.what() << std::endl;
	}
	catch (std::invalid_argument e) {
		output << e.what() << std::endl;
	}
	output << "Cascade test failed.\n";
	return output.str();
}

template<typename T>
T calculate_error(const std::vector<T>& v1, const std::vector<T>& v2)
{
	if (v1.size() != v2.size())
		throw std::invalid_argument("vectors is of different size, calculate_error(v1,v2)");

	T result = 0;
	for (size_t i = 0; i < v1.size(); i++)
	{
		T e = v1[i] - v2[i];
		result += std::abs(e);
	}
	return result;
}

void parallel_for_test() {
	auto print_time = [](const double message) { std::cout << message << std::endl; };
	auto f = [](double& u) { u = std::sin(u) + std::tanh(u); };
	Stopwatch<> timer;

	const size_t N = 2e8;
	std::vector<double> x(N);

	//generate and copy data.
	std::cout << "generating...\n";
	std::generate(x.begin(), x.end(), [] { return (double)std::rand() / (double)RAND_MAX - 0.5; });
	std::vector<double> y(x);
	std::vector<double> z(x);
	std::cout << "for_each...\n";

	//time single threaded
	timer.getLapTime();
	for_each(x.begin(), x.end(), f);
	double single_thread_time = timer.getLapTime();

	//time thread pool
	timer.getLapTime();
	parallel_for_each(y.begin(), y.end(), f, 1000000);
	double multi_thread_time = timer.getLapTime();

	//time open mp
	timer.getLapTime();
#pragma omp parallel for
	for (int i = 0; i < z.size(); i++) {
		z[i] = std::sin(z[i]) + std::tanh(z[i]); 
	}
	double openmp_time = timer.getLapTime();

	std::cout << "single to multi speed up: " << single_thread_time / multi_thread_time << "x" << "\nopenmp to multi speed up: " << openmp_time / multi_thread_time << "x" << std::endl;

	if (std::equal(x.begin(), x.end(), y.begin()))
		std::cout << "single == multi, success\n";
	else {
		std::cout << "single != multi, fail\n";
		std::cout << calculate_error<double>(x, y) << " total multi error\n";
	}

	if (std::equal(x.begin(), x.end(), z.begin()))
		std::cout << "single == openmp, success\n";
	else {
		std::cout << "single != openmp, fail\n";
		//Error only in release build of 10^-9.
		std::cout << calculate_error<double>(x, z) << " total open mp error, where the hell is this error coming from?\n";
	}
}

void multi_threaded_tests()
{
	parallel_for_test();
}
