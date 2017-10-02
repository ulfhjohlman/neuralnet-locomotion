#pragma once
#include "Population.h"
#include "Individual.h"
#include "NeuralNetChromosome.h"
#include "Mutation.h"

#include "../mjenvironment/mjEnvironment.h"

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
	GeneticAlgorithm(int population_size, int n_inputs, int n_outputs) : mutation(0.05) {
		std::cout << "Generating " << population_size << " controllers... ";
#pragma omp parallel for
		for (int i = 0; i < population_size; i++) {
			std::unique_ptr< NeuralNetChromosome > member( new NeuralNetChromosome(n_inputs, n_outputs));
			population.members.push_back(std::move(member));
		}
		std::cout << "done\n";
	}
	void setEnvironment(mjEnvironment* env) {
		if (!env)
			throw std::runtime_error("no environment.");
		m_environment = env;
	}

	void run() {
		if (!m_environment)
			throw std::runtime_error("no environment.");

		for (auto & member : population.members)
			m_environment->evaluateController(member->decode());
	}




private:
	Population<LayeredNeuralNet> population;
	Mutation<LayeredNeuralNet> mutation;

	mjEnvironment* m_environment;
};