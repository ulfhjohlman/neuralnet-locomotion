#pragma once
#include "Population.h"
#include "Individual.h"
#include "NeuralNetChromosome.h"
#include "Mutation.h"
#include "Selection.h"
#include "Crossover.h"
#include "Replacement.h"

//Utility
#include "ThreadsafeQueue.h"
#include "utilityfunctions.h"

//env
#include "mjEnvironment.h"

//stl
#include <future>
#include <thread>
#include <tuple>
#include <experimental/filesystem>

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

const int g_population_size = 128;
const double g_mutation_probability = 0.01;
double g_crossover_probability = 0.25;
const double g_ptour = 0.75;
const int g_tournamentSize = 10;

const double g_start_survival_percentage = 0.5;
const double g_survivor_fraction = 0.3; //top x%

const int g_elitism_count = 4;
 
double pmut = 0.01;

class GeneticAlgorithm {
public:
	GeneticAlgorithm(int population_size, int n_inputs, int n_outputs) : m_mutation(pmut, g_elitism_count), m_generation(0),
		m_selection(g_ptour, g_tournamentSize), m_ninputs(n_inputs), m_nouputs(n_outputs)
	{
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
		m_population.members.reserve(container.size());
		while (!container.empty())
			m_population.members.push_back(std::move(container.sequential_pop()));
			
		std::cout << "done\n";
	}

	void setEnvironment(mjEnvironment* environment) {
		checkEnvironment(environment);

		m_environment = environment;

		evaluatePopulation();
	}

	void run() {
		checkEnvironment(m_environment);

		//std::cout << "simulation start...";
		m_environment->simulate();
		//std::cout << "end.\n";

		std::pair<size_t, double> fitness;
		while (m_environment->tryGetResult(fitness)) {
			m_population.members[fitness.first]->setFitness(fitness.second);
		}

		if (m_environment->all_done()) {
			m_population.sort();

			if (m_generation % 1000 == 0) {
				m_population.save(m_generation, "ant1");
			}

			for (size_t i = 0; i < 5; i++) {
				std::cout << m_population[i]->getFitness() << std::endl;
			}

			double mean = m_population.meanFitness();
			std::cout << "Generation: " << m_generation << ", mean=" << mean << ", best=" << m_population.members[0]->getFitness() << " pmut=" << pmut << " ";
			std::cout << "Population size: " << m_population.size() << std::endl;
			m_generation++;

			const double start_kill_index = g_survivor_fraction*double(g_population_size);
			Death::linearDeath(m_population, m_object_pool, start_kill_index, g_start_survival_percentage);
			//Duplicate::duplicate(m_population);

			applyCrossover();

			applyMutation();

			evaluatePopulation();
		}
	}

	void applyCrossover()
	{
		Generator g;
		int n_mated = g.generate_binomial(g_population_size, g_crossover_probability);
		//std::cout << "born: " << n_mated << std::endl;
		ThreadsafeQueue< std::unique_ptr< Individual<LayeredNeuralNet> > > container;
		auto cross = [this, &container](int i) {
			std::pair<int, int> mates = m_selection.selectPair(m_population);

			std::unique_ptr<Individual<LayeredNeuralNet>> newIndividual = nullptr;
			bool has_stored_individual = m_object_pool.try_pop(newIndividual);
			if (!has_stored_individual)
				newIndividual = std::unique_ptr<Individual<LayeredNeuralNet>>(new NeuralNetChromosome(m_ninputs, m_nouputs));

			Crossover::uniformCrossover<LayeredNeuralNet>(m_population, mates.first, mates.second, newIndividual);
			container.push(std::move(newIndividual));
		};
		
		parallel_for(0, n_mated, 2, cross);

		//Move pointers to population
		while (!container.empty())
			m_population.members.push_back(std::move(container.sequential_pop()));
	}

	void evaluatePopulation()
	{
		for (size_t i = 0; i < m_population.members.size(); i++)
			m_environment->evaluateController(m_population.members[i]->decode(), i);
	}

	void applyMutation()
	{
		const ScalarType T = 1000;
		pmut = 0.05;
		pmut *= std::exp(-ScalarType(m_generation) / T);
		if (pmut < 0.0005)
			pmut = 0.0005;
		m_mutation.setMutationProbability(pmut);
		m_mutation >> m_population;
	}

	size_t m_generation;
protected:
	void checkEnvironment(mjEnvironment* environment) {
#ifdef _DEBUG
		if (!environment)
			throw std::runtime_error("no environment.");
#endif // _DEBUG
	}

	
private:
	int m_ninputs, m_nouputs;

	Population<LayeredNeuralNet> m_population;
	Mutation<LayeredNeuralNet> m_mutation;
	Crossover m_crossover;
	TournamentSelection m_selection;


	ThreadsafeQueue<std::unique_ptr<Individual<LayeredNeuralNet>>> m_object_pool;

	mjEnvironment* m_environment;
};

//for (auto& member : m_population.members) {
//	size_t num_layers = member->decode()->getTopology()->getNumberOfLayers();
//	ScalarType l2NormWeight = 0;
//	ScalarType l2NormBias = 0;

//	ScalarType C = 0; //regularization factor
//	for (size_t i = 1; i < num_layers; i++) { //skip input layer
//		l2NormWeight += member->decode()->getLayer(i)->getWeights().squaredNorm();
//		//l2NormBias += member->decode()->getLayer(i)->getBias().squaredNorm();
//		//C += member->decode()->getTopology()->getLayerSize(i);
//	}
//	
//	C = 0.0000;
//	member->setFitness(member->getFitness() - C * (l2NormWeight + l2NormBias));
//}