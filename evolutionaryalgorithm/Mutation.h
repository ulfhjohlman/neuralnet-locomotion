#pragma once
#include "Individual.h"
#include "NeuralNetGenome.h"

template<typename T>
class Mutation 
{
public:
	Mutation(ScalarType mutation_probability) : m_mutation_probability(mutation_probability) {}
	~Mutation() = default;

	void operator>>(std::shared_ptr< Individual<T> > mutate_this) {
		mutate_this->getGenome()->mutate(m_mutation_probability);
	}
private:
	ScalarType m_mutation_probability;
};
