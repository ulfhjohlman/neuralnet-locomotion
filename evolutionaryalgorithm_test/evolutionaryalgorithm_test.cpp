#include <iostream>

#include "RandomEngineFactory.h"
#include "Generator.h"


#include "NeuralNetGenome.h"
#include "NeuralNetChromosome.h"
#include "Mutation.h"
#include "Population.h"

int main()
{
	RandomEngineFactory::initialize(); //optional

	Population<LayeredNeuralNet> population;
	Mutation<LayeredNeuralNet> mutation_op(0.05);
	

	std::unique_ptr<NeuralNetChromosome> member ( new NeuralNetChromosome(5, 5) );
	LayeredNeuralNet* ffnn = member->decode();
	
	population.members.push_back( std::move( member) );

	MatrixType x(5, 1);
	x.setRandom();

	ffnn->input(x);
	std::cout << ffnn->output() << std::endl <<std::endl;
	
	mutation_op >> population;

	ffnn->input(x);
	std::cout << ffnn->output() << std::endl;

	std::cin.get();
	return 0;
}
