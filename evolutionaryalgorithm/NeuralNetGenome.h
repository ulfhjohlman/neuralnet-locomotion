#pragma once
#include "NeuralNet.h"
#include "Genome.h"

class NeuralNetGenome : public Genome
{
public:
	NeuralNetGenome() = default;
	~NeuralNetGenome() = default;

	virtual void mutate(float mutation_probability) {
		for (int i = 0; i < m_weight_size.size(); i++) {
			auto L = m_weight_size[i];
			auto dataPointer = m_weight_data[i];


		}
	}
	virtual std::vector<Genome*> cut(int cuts) { return std::vector<Genome*>(); }

private:
	std::vector<ScalarType*> m_weight_data; //dangling danger
	std::vector<int> m_weight_size; //dangling danger
};