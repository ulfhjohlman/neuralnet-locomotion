#include <iostream>

#include "RandomEngineFactory.h"
#include "Generator.h"
#include "utilityfunctions.h"
#include "ThreadsafeQueue.h"

#include "NeuralNetGenome.h"
#include "NeuralNetChromosome.h"
#include "Mutation.h"
#include "Population.h"

#include <future>
#include <thread>


std::promise<int> prom;                      // create promise
std::future<int> contract() {
	return prom.get_future();
}

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

	using namespace std::chrono_literals;
	
	int a = 1;
	ThreadsafeQueue<int> q;
	parallel_for(0, 100, 8, [a, &q](int i, int extra) { q.push(a + i + extra); }, -1);

	int N = q.size();
	for (int i = 0; i < N; i++)
		std::cout << q.sequential_pop() << std::endl;

	std::cin.get();
	return 0;
}
