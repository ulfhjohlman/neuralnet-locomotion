#pragma once


#include "Population.h"

class Elitism 
{
public:
	Elitism() = default;
	~Elitism() = default;

	template<typename T>
	static void decayMomentum(Population<T>& population, size_t elitism_count = 1) {
		size_t N = std::min(elitism_count, population.size());
		for (size_t i = 0; i < N; i++)
			population.members[i]->getGenome()->decayMomentum();
	}
	
private:
	
};