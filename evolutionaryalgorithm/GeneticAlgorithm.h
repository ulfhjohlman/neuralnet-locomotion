#pragma once
#include "Population.h"
#include "Individual.h"
#include "NeuralNetChromosome.h"
#include "Mutation.h"
#include "Selection.h"
#include "Crossover.h"
#include "Replacement.h"
#include "NicheSet.h"
#include "Elitism.h"

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
const double g_mutation_probability = 0.05;
double g_crossover_probability = 0.12;
double g_niche_crossover_probability = 0.25;
const double g_ptour = 0.7;
const int g_tournamentSize = 10;
const int g_elitism_count = 1;

const double g_start_survival_percentage = 4.0 / 7.0; //half dies if with 0.3 top survivors
const double g_start_extinction_survival_percentage = 0.5;
const double g_disease_percent = 0.2;
const double g_survivor_fraction = 0.3; //top x%

ScalarType recurrent_initial_noise_sigma = 0.05;
 
double pmut = g_mutation_probability;

class GeneticAlgorithm {
public:
	GeneticAlgorithm(int population_size, int n_inputs, int n_outputs) : m_mutation(pmut, g_elitism_count), m_generation(0),
		m_selection(g_ptour, g_tournamentSize), m_ninputs(n_inputs), m_nouputs(n_outputs)
	{
		std::cout << "Generating " << population_size << " controllers... ";

		//Setup structure for generation
		ThreadsafeQueue<std::shared_ptr< NeuralNetChromosome > > container;
		auto generate_individual = [&container, n_inputs, n_outputs](int i){
			std::shared_ptr< NeuralNetChromosome > member(new NeuralNetChromosome(n_inputs, n_outputs));
			container.push(std::move(member));
		};

		//Generate across multiple threads
		parallel_for(0, population_size, 1, generate_individual);

		//Move pointers to population
		m_population.members.reserve(container.size());
		while (!container.empty())
			m_population.members.push_back(std::move(container.sequential_pop()));

		m_niche_set.reset(m_population);
			
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
			m_population[fitness.first]->setFitness(fitness.second);
		}

		if (m_environment->all_done()) {
			m_population.sort();

			if (m_generation % 1000 == 0) {
				m_population.save(m_generation, "invdp2d");
			}

			for (size_t i = 0; i < 5; i++) {
				std::cout << m_population[i]->getFitness() << std::endl;
			}

			double mean = m_population.meanFitness();
			std::cout << "Generation: " << m_generation << ", mean=" << mean << ", best=" << m_population.members[0]->getFitness() << " pmut=" << pmut << " ";
			std::cout << "Population size: " << m_population.size() << std::endl;
			std::cout << "Niches: " << m_niche_set.size() << std::endl;
			m_generation++;

			m_niche_set.printNicheSizes();
			m_niche_set.update();

			applyReplacement();

			ThreadsafeQueue< std::shared_ptr< Individual<LayeredNeuralNet> > > new_generation;
			applyCrossover(new_generation);

			applyElitism();

			applyMutation();

			//Move pointers to population
			while (!new_generation.empty()) {
				m_population.members.push_back(std::move(new_generation.sequential_pop()));
				m_niche_set.addMember(m_population.back());
			}

			evaluatePopulation();
		}
	}

	void applyReplacement() {
		//const double start_kill_index = g_survivor_fraction*double(g_population_size);
		//Death::linearDeath(m_population, m_niche_set, m_object_pool, start_kill_index, g_start_survival_percentage);
		//Duplicate::duplicate(m_population);

		auto decimate_subpopulation = [this](int i) {
			auto & population = m_niche_set[i];
			int start_kill_index = 0;
			if (population.size() > 120)
				start_kill_index = g_elitism_count + 30;
			else
				start_kill_index = g_elitism_count + 50;
			Death::linearDeath(population, start_kill_index, g_start_survival_percentage);
		};

		parallel_for<int>(0, m_niche_set.size(), 2, decimate_subpopulation);

		m_niche_set.clearEmptyNiches();

		if(m_population.size() > 300)
			Death::extinction(m_niche_set, g_start_extinction_survival_percentage);
		if (m_population.size() > 600)
			Death::disease(m_niche_set, g_elitism_count, g_disease_percent);
		if (m_population.size() > 800)
			Death::linearDeath(m_population, m_niche_set, m_object_pool, 10, 0.5);

		for (size_t i = 0; i < m_population.size(); i++) {
			if (m_population[i].use_count() == 1) {
				m_object_pool.push(std::move(m_population[i]));
				m_population.erase(i);
				i--;
			}
		}
	}

	void applyElitism() {
		for (size_t i = 0; i < m_niche_set.size(); i++) {
			auto & population = m_niche_set[i];
			Elitism::decayMomentum(population, g_elitism_count);
		}
	}

	void applyCrossover(ThreadsafeQueue< std::shared_ptr< Individual<LayeredNeuralNet> > >& container)
	{
		for (size_t i = 0; i < m_niche_set.size(); i++) {
			auto & population = m_niche_set[i];
			Duplicate::asexualReproduction(population, m_object_pool, container, 2, m_ninputs, m_nouputs);
			if(population.size() < 2)
				continue;

			int n_mated = Generator::generate_binomial_shared<int>(population.size(), g_niche_crossover_probability);
			auto cross = [this, &population, &container](int i) {
				std::pair<int, int> mates = m_selection.selectPair(population);

				std::shared_ptr<Individual<LayeredNeuralNet>> newIndividual = nullptr;
				bool has_stored_individual = m_object_pool.try_pop(newIndividual);
				if (!has_stored_individual)
					newIndividual = std::shared_ptr<Individual<LayeredNeuralNet>>(new NeuralNetChromosome(m_ninputs, m_nouputs));

				Crossover::directionalCrossover<LayeredNeuralNet>(population, mates.first, mates.second, newIndividual);
				container.push(std::move(newIndividual));
			};
			parallel_for(0, n_mated, 2, std::move(cross));
		}

		int n_mated_interspecies = Generator::generate_binomial_shared<int>(g_population_size, g_crossover_probability);
		auto cross_interspecies = [this, &container](int i) {
			std::pair<int, int> mates = m_selection.selectPair(m_population);

			std::shared_ptr<Individual<LayeredNeuralNet>> newIndividual = nullptr;
			bool has_stored_individual = m_object_pool.try_pop(newIndividual);
			if (!has_stored_individual)
				newIndividual = std::shared_ptr<Individual<LayeredNeuralNet>>(new NeuralNetChromosome(m_ninputs, m_nouputs));

			Crossover::directionalCrossover<LayeredNeuralNet>(m_population, mates.first, mates.second, newIndividual);
			container.push(std::move(newIndividual));
		};
		
		parallel_for(0, n_mated_interspecies, 2, std::move(cross_interspecies));
	}

	void evaluatePopulation()
	{
		for (size_t i = 0; i < m_population.members.size(); i++) {
			m_population.members[i]->decode()->clearInternalStates(recurrent_initial_noise_sigma);
			m_environment->evaluateController(m_population.members[i]->decode(), i);
		}
	}

	void applyMutation()
	{
		const ScalarType T = 4000;
		pmut = g_mutation_probability;
		pmut *= std::exp(-ScalarType(m_generation) / T);
		if (pmut < 0.0005)
			pmut = 0.0005;
		m_mutation.setMutationProbability(pmut);

		auto mutate_subpopulation = [this](int i) {
			auto & population = m_niche_set[i];
			m_mutation >> population; //Mutates twice, has internal flag for this
		};

		parallel_for<int>(0, m_niche_set.size(), 1, mutate_subpopulation);
		m_population.clearMutationFlag(); //reset mutation states
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
	NicheSet<LayeredNeuralNet> m_niche_set;
	Mutation<LayeredNeuralNet> m_mutation;
	Crossover m_crossover;
	TournamentSelection m_selection;

	ThreadsafeQueue<std::shared_ptr<Individual<LayeredNeuralNet>>> m_object_pool;

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