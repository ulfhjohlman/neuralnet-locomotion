#pragma once

#include "Population.h"
#include "Individual.h"
#include "NeuralNetGenome.h"
#include "utilityfunctions.h"

#include <tuple>

class Crossover 
{
public:
	Crossover() { }
	~Crossover() = default;


	template<typename T>
	static void uniformCrossover( Population<T>& population, int mate1, int mate2, std::unique_ptr<Individual<T>>& child) {
		*child = *population[mate1];

		child->getGenome()->uniformCrossover(*population[mate2]->getGenome());
	}

	template<typename T>
	static void directionalCrossover(Population<T>& population, int mate1, int mate2, std::unique_ptr<Individual<T>>& child) {
		*child = *population[mate1];

		child->getGenome()->directionalCrossover(*population[mate2]->getGenome());
	}
private:
};