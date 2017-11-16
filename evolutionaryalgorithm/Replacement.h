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
};