#pragma once
#include "Population.h"
#include "Individual.h"
#include "NeuralNetGenome.h"

//Move implementation of operation here in future.
//Add template INSTANTIATION for genome such that it can apply mutations
//for this specific GA.
template<typename T>
class Mutation 
{
public:
	Mutation(ScalarType mutation_probability) : m_mutation_probability(mutation_probability) {}
	~Mutation() = default;

	void operator>>(Population<T>& population) {
		for(size_t i = 4; i < population.members.size(); i++)
			population.members[i]->getGenome()->mutate(m_mutation_probability);
	}
private:
	ScalarType m_mutation_probability;
};
