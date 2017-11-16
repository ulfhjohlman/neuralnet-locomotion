#pragma once
#include "Population.h"
#include "Individual.h"
#include "NeuralNetGenome.h"
#include "utilityfunctions.h"
//Move implementation of operation here in future.
//Add template INSTANTIATION for genome such that it can apply mutations
//for this specific GA.
template<typename T>
class Mutation 
{
public:
	Mutation(ScalarType mutation_probability, size_t nelites) :
		m_mutation_probability(mutation_probability),
		m_elitism_count(nelites) { 
	}
	~Mutation() = default;

	void setMutationProbability(ScalarType mutation_probability) {
		m_mutation_probability = mutation_probability;
	}
	void setElitism(const size_t nelites) {
		m_elitism_count = nelites;
	}

	void operator>>(Population<T>& population) {
		auto start_index = std::min(m_elitism_count, population.size());

		auto f = [&population, this](int i) {
			population.members[i]->getGenome()->mutate(m_mutation_probability);
		};

		parallel_for<size_t>(start_index, population.members.size(), 16, f);
		//for(size_t i = m_elitism_count; i < population.members.size(); i++)
		//	population.members[i]->getGenome()->mutate(m_mutation_probability);
	}
private:
	ScalarType m_mutation_probability;
	size_t m_elitism_count;
};
