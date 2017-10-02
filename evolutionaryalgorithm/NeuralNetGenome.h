#pragma once
#include "NeuralNet.h"
#include "LayeredNeuralNet.h"
#include "Genome.h"
#include "Generator.h"

#include <stdexcept>
#include <unordered_set>

class NeuralNetGenome : public Genome
{
public:
	NeuralNetGenome(LayeredNeuralNet * net) {
		checkNN(net);

		LayeredTopology* top = net->getTopology();
		int number_of_layers = top->getNumberOfLayers();

		m_weight_data.reserve(number_of_layers);
		m_weight_size.reserve(number_of_layers);

		//Hook up data pointers for modification
		for (int i = 0; i < number_of_layers; i++) {
			Layer* layer = net->getLayer(i);
			m_weight_data.push_back(layer->data());
			m_weight_size.push_back(top->getLayerSize(i));
		}
	}
	virtual ~NeuralNetGenome() = default;

	virtual void mutate(ScalarType mutation_probability) {
		checkMutationRate(mutation_probability);

		Generator generator;
		for (int i = 1; i < m_weight_size.size(); i++) {
			const auto L = m_weight_size[i];
			auto dataPointer = m_weight_data[i];

			//Get the binomially distributed variable ~B(n_trails, p_probability)
			int number_of_mutations = generator.generate_binomial(L, (double)mutation_probability);

			//Generate mutation sites. Assumed low mutation prob for now.
			//Problem is worst case scenario when same index is rolled.
			std::unordered_set<int> indexes(number_of_mutations);
			while (indexes.size() < number_of_mutations) {
				int index = generator.generate_uniform_int(0, L - 1);
				indexes.insert(index);
			}

			for (const auto & index : indexes) {
				dataPointer[index] += generator.generate_normal<ScalarType>( 0, 0.15 );
			}
		}

	}

protected:
	void checkMutationRate(ScalarType mutation_probability) {
#ifdef _NEURALNET_DEBUG
		if (mutation_probability < 0.0 || mutation_probability > 1.0) {
			throw std::invalid_argument("Mutation probability not in [0, 0.5]");
		}
#endif // _DEBUG
	}
	void checkNN(LayeredNeuralNet * net)
	{
#ifdef _NEURALNET_DEBUG
		if (net == nullptr)
			throw std::runtime_error("NeuralNetGenome received nullptr NN.");
#endif // _NEURALNET_DEBUG
	}

private:
	std::vector<ScalarType*> m_weight_data; //dangling pointer danger
	std::vector<int> m_weight_size; //dangling pointer danger
};