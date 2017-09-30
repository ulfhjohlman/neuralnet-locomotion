#include <iostream>
#include "RandomEngineFactory.h"
#include "Generator.h"
#include "NeuralNetGenome.h"

#include "NNController.h"
#include "Mutation.h"

int main()
{
	RandomEngineFactory::initialize(); //optional

	//example::uniformrealdist_hist_example();
	//example::normaldist_histogram_example();
	//example::exponentialdist_hist_example();
	//example::generator_example();
	Mutation<FeedForwardNeuralNet> mutation_op(0.2);

	std::shared_ptr<Individual<FeedForwardNeuralNet>> controller ( new NNController(5, 5));
	MatrixType x(5, 1);
	x.setRandom();

	FeedForwardNeuralNet* ffnn = controller->decode();

	ffnn->input(x);
	std::cout << ffnn->output() << std::endl <<std::endl;
	
	mutation_op >> controller;

	ffnn->input(x);
	std::cout << ffnn->output() << std::endl;

	std::cin.get();
	return 0;
}
