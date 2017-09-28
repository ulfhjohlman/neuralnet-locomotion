#include <algorithm>
#include <numeric>
#include <type_traits>
#include <sstream>

#include "config.h"
#include "Dataset.h"
#include "DatasetException.h"
#include "NeuralNet.h"
#include "FeedForwardNeuralNet.h"
#include "LayeredTopology.h"
#include "CascadeNeuralNet.h"
#include "RecurrentTopology.h"



#include "Stopwatch.h"
#include "XMLException.h"
#include "DataPrinter.h"
#include "XMLWrapper.h"
#include "utilityfunctions.h"
#include "ThreadPoolv2.h"

#include "StopwatchTest.h"

//c++ standard includes
#include <iostream>
#include <exception>
#include <vector>
#include <thread>
#include <chrono>
#include <functional>
#include <utility>
#include <cstdlib>
#include <sstream> //ostringstream
#include <iomanip> //std::get_time

//Function prototypes
void single_threaded_tests();
std::string XMLWrapper_test();
std::string dataprinter_test();
std::string dataset_test();
std::string feedForwardNeuralNet_test();
std::string cascadeNeuralNet_test();
std::string relu_ff_test();
std::string training_XOR_test();

void multi_threaded_tests();
void parallel_for_test();


int main()
{
	std::cout.sync_with_stdio(true); // make cout thread-safe

	#ifdef _DEBUG
			std::cout << "_DEBUG FLAG ON\n";
	#else
			std::cout << "_DEBUG FLAG OFF\n";
	#endif
	#ifdef _NEURALNET_DEBUG
			std::cout << "_NEURALNET_DEBUG FLAG ON" << std::endl;
	#else
			std::cout << "_NEURALNET_DEBUG FLAG OFF" << std::endl;
	#endif
	//std::cout << cascadeNeuralNet_test() << std::endl;
	//std::cout << feedForwardNeuralNet_test();
	//std::cout << relu_ff_test() << std::endl;
	std::cout << training_XOR_test() << std::endl;
	//test
	//single_threaded_tests();
	//multi_threaded_tests();

	std::cout << "Tests done." << std::endl;
	std::cin.get();
	return 0;
}

std::string relu_ff_test()
// generic test using softmax, relu layers and calling forward/backwardpass
{
	std::ostringstream output;

	try
	{
		std::vector<int> layerSizes {2,8,16,4};
		int relu = Layer::LayerType::relu;
		int inputLayer = Layer::LayerType::inputLayer;
		int softmax = Layer::LayerType::softmax;
		std::vector<int> layerTypes {inputLayer,relu,relu,softmax};
		LayeredTopology* top = new LayeredTopology(layerSizes,layerTypes);

		//new random seed every run
		std::srand(static_cast<unsigned int>(time(0)));
		FeedForwardNeuralNet ffnn(top);
		ffnn.initializeRandomWeights();
		MatrixType input_matrix(layerSizes[0], 1);
		input_matrix.setRandom();
		output << "Input:\n " << input_matrix << std::endl;
		ffnn.input(input_matrix);
		output << "Output:\n " << ffnn.output() <<std::endl;
		output << "Total Prob = " << ffnn.output().sum() << std::endl;

		MatrixType x;
		x.resize(layerSizes.back(),1);
		x.setRandom();
		output << "Backproping random gradients:\n" << x <<"\n";
		ffnn.backprop( x );
		output << "Updating weights\n";
		double learning_rate = 10e-4;
		ffnn.updateWeights(learning_rate);

		output << "relu_ff_test() test ended\n";
		return output.str();
	}
	catch (NeuralNetException e) {
		output << e.what() << std::endl;
	}
	catch (const FactoryException e) {
		output << e.what() << std::endl;
	}
	catch (std::invalid_argument e) {
		output << e.what() << std::endl;
	}
	output << "relu_ff_test() failed\n";
	return output.str();
}
std::string training_XOR_test()
// training single hiddenlayer network to learn nonlinear XOR function
{
	std::ostringstream output;
	output << "XOR training started\n";
	try
	{
		std::vector<int> layerSizes {2,5,1};
		int relu = Layer::LayerType::relu;
		int inputLayer = Layer::LayerType::inputLayer;
		int noactiv = Layer::noActivation;
		int softmax = Layer::LayerType::softmax;
		int sigmoid= Layer::LayerType::sigmoid;
		std::vector<int> layerTypes {inputLayer,noactiv,sigmoid};
		LayeredTopology* top = new LayeredTopology(layerSizes,layerTypes);

		//new random seed every run
		std::srand(static_cast<unsigned int>(time(0)));
		FeedForwardNeuralNet ffnn(top);
		ffnn.initializeRandomWeights();
		MatrixType input_matrix(layerSizes[0], 1);
		MatrixType error_gradient(layerSizes.back(),1);
		int sample;
		float error;
		double running_error = 1;
		double learning_rate = 0.001;
		double ffnn_out;
		int selected_out;

		// Training Data:
		double train_data[4][3] {{0,0,0},{0,1,1},{1,0,1},{1,1,0}};
		for(int i = 0; i < 1000 && running_error>0.1 ; i++)
		{
			//forwardpass
			sample = std::rand() % 4;
			//sample = 1;
			input_matrix << train_data[sample][0],
							train_data[sample][1];
			ffnn.input(input_matrix);

			//backpass
			ffnn_out =ffnn.output()(0,0); //likelihood of 1

			selected_out = (std::rand()%1000 < ffnn_out*1000); //random selection,

			error = pow(selected_out - train_data[sample][2], 2); // L2 norm error

			running_error = running_error*0.99 + error;
			error_gradient << 2.0 * (ffnn_out - train_data[sample][2]);

			ffnn.backprop( error_gradient );
			ffnn.updateWeights(learning_rate);
			//ffnn.print_layer_outputs();
			//ffnn.print_layer_weights();
			//ffnn.print_layer_bias();
			//ffnn.print_layer_weight_gradients();
			//ffnn.print_layer_input_gradients();
			output << "Iteration: " << i << ". Running_error = " << running_error << "\n";
		}
		ffnn.printLayerOutputs();
		ffnn.printLayerWeights();
		ffnn.printLayerBias();
		output << "training_XOR_test() test ended\n";
		return output.str();
	}
	catch (NeuralNetException e) {
		output << e.what() << std::endl;
	}
	catch (const FactoryException e) {
		output << e.what() << std::endl;
	}
	catch (std::invalid_argument e) {
		output << e.what() << std::endl;
	}
	output << "training_XOR_test failed\n";
	return output.str();
}

/// <summary>
///	
/// </summary>
void single_threaded_tests()
{
	ThreadPool pool;

	StopwatchTest sw_test;
	//methods, pass: &function == temporary pointer to function.
	auto test_dataprinter      = pool.submit(&dataprinter_test);
	auto test_XMLWrapper       = pool.submit(&XMLWrapper_test);
	auto test_dataset          = pool.submit(&dataset_test);


	//TestFramework

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

		d.print();
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

	try {
		std::vector<int> layerSizes = { n_inputs,		   2, 3, 1 };
		std::vector<int> layerTypes = { Layer::inputLayer, 0, 0, 0 };
		LayeredTopology * top = new LayeredTopology(layerSizes, layerTypes);

		FeedForwardNeuralNet ffnn(top);
		ffnn.initializeRandomWeights();

		MatrixType m(n_inputs, 1);
		m.setRandom();

		ffnn.input(m);
		output << ffnn.output() << std::endl;

		output << "FFNN test passed.\n";
		return output.str();
	}
	catch (NeuralNetException e) {
		output << e.what() << std::endl;
	}
	catch (const FactoryException e) {
		output << e.what() << std::endl;
	}
	catch (std::invalid_argument e) {
		output << e.what() << std::endl;
	}

	output << "FFNN failed.\n";
	return output.str();
}

std::string cascadeNeuralNet_test() {
	std::ostringstream output;
	const int n_inputs = 2;

	try {
		std::vector<int> layerSizes = { n_inputs,		   2, 3, 1 };
		std::vector<int> layerTypes = { Layer::inputLayer, 0, 0, 0 };
		CascadeTopology * top = new CascadeTopology(layerSizes, layerTypes);
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
	catch (FactoryException e) {
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
	//auto print_time = [](const double message) { std::cout << message << std::endl; };
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

	std::cout << "single to multi speed up: " << single_thread_time / multi_thread_time <<
	 "x" << "\nopenmp to multi speed up: " << openmp_time / multi_thread_time << "x" << std::endl;

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
