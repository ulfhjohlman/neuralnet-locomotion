#pragma once

#include "Population.h"
#include "ThreadsafeQueue.h"
#include "Individual.h"
#include "Generator.h"
#include "NicheSet.h"
#include <memory>

//Replacement operator
class Death 
{
public:
	Death() = default;
	~Death() = default;

	template<typename T>
	static void linearDeath(Population<T>& population, NicheSet<T>& niches, ThreadsafeQueue< std::shared_ptr< Individual<T> > >& kill_list, int start_index, ScalarType start_survival_rate) {
		ScalarType k = start_survival_rate / ScalarType(population.size() - start_index); //dy/dx
		Generator g;
		int deathcounter = 0;
		for (int i = start_index; i < population.size(); i++) {
			ScalarType r = g.generate_uniform<ScalarType>(0.0f, 1.0f);
			ScalarType threshold = -k*ScalarType(i + deathcounter - start_index) + start_survival_rate;
			if (r > threshold) {
				niches.remove(population[i]);
				kill_list.push(std::move(population[i]));
				population.erase(i);
				i--;
				deathcounter++;
			}
		}
	}

	template<typename T>
	static void linearDeath(Population<T>& population, int start_index, ScalarType start_survival_rate) {
		ScalarType k = start_survival_rate / ScalarType(population.size() - start_index); //dy/dx
		Generator g;
		int deathcounter = 0;
		for (int i = start_index; i < population.size(); i++) {
			ScalarType r = g.generate_uniform<ScalarType>(0.0f, 1.0f);
			ScalarType threshold = -k*ScalarType(i + deathcounter - start_index) + start_survival_rate;
			if (r > threshold) {
				population.erase(i);
				i--;
				deathcounter++;
			}
		}
	}

	//Removes unfit niches
	template<typename T>
	static void extinction( 
		NicheSet<T>& niches, 
		ScalarType start_survival_rate) {

		niches.sortNiches();

		size_t start_index = niches.size() / 4;
		start_index = std::max<size_t>(nominal_number_of_niches, start_index);
		ScalarType k = start_survival_rate / ScalarType(niches.size() - start_index); //dy/dx
		Generator g;
		int deathcounter = 0;
		for (int i = start_index; i < niches.size(); i++) {
			ScalarType r = g.generate_uniform<ScalarType>(0.0f, 1.0f);
			ScalarType threshold = -k*ScalarType(i + deathcounter - start_index) + start_survival_rate;
			if (r > threshold) {
				niches.removeNiche(i);
				deathcounter++;
				i--;
			}
		}
	}

	//portion of subpopulation is erased.
	template<typename T>
	static void disease(
		NicheSet<T>& niches, size_t elitism_count,
		ScalarType survival_rate, size_t min_population_size = 60) {

		auto decimate_subpopulation = [&niches, elitism_count, survival_rate, min_population_size](int i) {
			Generator g;
			auto & population = niches[i];
			if (population.size() < min_population_size)
				return;
			for (size_t i = elitism_count; i < population.size(); i++) {
				ScalarType r = g.generate_uniform<ScalarType>(0.0f, 1.0f);
				if (r > survival_rate) {
					population.erase(i);
					i--;
				}
			}
		};

		parallel_for<size_t>(0, niches.size(), 4, decimate_subpopulation);
	}


	
private:
	
};

class Duplicate {
public:
	template<typename T>
	static void duplicate(Population<T>& population) {
		const size_t duplications = 4;

		size_t it = population.members.size() / duplications;
		for (size_t i = 0; i < population.members.size() / 4; i++) {
			for (size_t j = 0; j < (duplications - 1); j++)
				*population.members[it + j] = *population.members[i];

			it += duplications - 1;
		}
	}

	template<typename T>
	static void asexualReproduction(Population<T>& population, 
		ThreadsafeQueue< std::shared_ptr< Individual<T> > >& pool, 
		ThreadsafeQueue< std::shared_ptr< Individual<T> > >& container, 
		size_t copies, int m_ninputs, int m_nouputs)
	{
		for (int i = 0; i < copies; i++) {
			std::shared_ptr<Individual<T>> newIndividual = nullptr;
			bool has_stored_individual = pool.try_pop(newIndividual);
			if (!has_stored_individual)
				newIndividual = std::shared_ptr<Individual<T>>(new NeuralNetChromosome(m_ninputs, m_nouputs));

			*newIndividual = *population[0];

			container.push(newIndividual);
		}
	}
};