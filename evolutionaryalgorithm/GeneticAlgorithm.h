#pragma once
#include "Population.h"
#include "Individual.h"
#include "NeuralNetChromosome.h"
#include "Mutation.h"

//Utility
#include "ThreadsafeQueue.h"
#include "utilityfunctions.h"

//env
#include "../mjenvironment/mjEnvironment.h"

//stl
#include <future>
#include <thread>
#include <tuple>

//Design genome representation for problem.
//Encode genome
//Set instructions and operators
//define fitness function

//Algorithm ----------------
//1. Initialize population

//2. decode population

//3. calculate fitness

//4. apply any evolutionary operators
//4.1 selection for crossover
//4.2 crossover : uniform, 2-3-n point
//4.3* mutation : may be different kinds of mutation
//4.4* replacement : replace, dynamic population, worst individual killed
//4.5 elitism

//check for stop condition(s)

class GeneticAlgorithm {
public:
	GeneticAlgorithm(int population_size, int n_inputs, int n_outputs) : mutation(0.05), m_generation(0) {
		std::cout << "Generating " << population_size << " controllers... ";

		//Setup structure for generation
		ThreadsafeQueue<std::unique_ptr< NeuralNetChromosome > > container;
		auto generate_individual = [&container, n_inputs, n_outputs](int i){
			std::unique_ptr< NeuralNetChromosome > member(new NeuralNetChromosome(n_inputs, n_outputs));
			container.push(std::move(member));
		};

		//Generate across multiple threads
		parallel_for(0, population_size, 1, generate_individual);

		//Move pointers to population
		population.members.reserve(container.size());
		while (!container.empty())
			population.members.push_back(std::move(container.sequential_pop()));
			
		std::cout << "done\n";
	}
	void setEnvironment(mjEnvironment* environment) {
		checkEnvironment(environment);

		m_environment = environment;

		for (size_t i = 0; i < population.members.size(); i++)
			m_environment->evaluateController(population.members[i]->decode(), i);
	}

	void run() {
		checkEnvironment(m_environment);

		//std::cout << "simulation start...";
		m_environment->simulate();
		//std::cout << "end.\n";

		std::pair<size_t, double> fitness;
		while (m_environment->tryGetResult(fitness)) {
			population.members[fitness.first]->setFitness(fitness.second);
		}

		if (m_environment->all_done()) {
			auto cmp_by_fitness = [](const std::unique_ptr<Individual<LayeredNeuralNet>>& a, const std::unique_ptr<Individual<LayeredNeuralNet>>& b)
			{
				return a->getFitness() > b->getFitness();
			};

			std::sort( population.members.begin(), population.members.end(), cmp_by_fitness);

			const size_t duplications = 4;

			size_t it = population.members.size() / duplications;
			for (size_t i = 0; i < population.members.size() / 4; i++) {
				for(size_t j = 0; j < (duplications-1); j++)
					*population.members[it+j] = *population.members[i];

				it += duplications-1;
			}

			mutation >> population;
			
			for (size_t i = 0; i < population.members.size(); i++)
				m_environment->evaluateController(population.members[i]->decode(), i);

			m_generation++;
			double mean = 0;
			for (auto & member : population.members)
				mean += member->getFitness();
			mean /= population.members.size();

			std::cout << "Generation: " << m_generation << ", mean=" << mean << ", best=" << population.members[0]->getFitness() << std::endl;
		}
	}

protected:
	void checkEnvironment(mjEnvironment* environment) {
#ifdef _DEBUG
		if (!environment)
			throw std::runtime_error("no environment.");
#endif // _DEBUG
	}


private:
	size_t m_generation;

	Population<LayeredNeuralNet> population;
	Mutation<LayeredNeuralNet> mutation;

	mjEnvironment* m_environment;
};