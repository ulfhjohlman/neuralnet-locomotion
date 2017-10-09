#pragma once
#include "NeuralNet.h"
#include "LayeredNeuralNet.h"
#include "Genome.h"
#include "Generator.h"
#include "dense"

#include <stdexcept>
#include <unordered_set>

class NeuralNetGenome : public Genome
{
public:
	NeuralNetGenome(LayeredNeuralNet * net) {
		checkNN(net);

		LayeredTopology* top = net->getTopology();
		size_t number_of_layers = top->getNumberOfLayers();

		this->reserve(number_of_layers);

		//Hook up data pointers for modification
		for (size_t i = 1; i < number_of_layers; i++) {
			//Get data field size
			m_layer_size.push_back(top->getLayerSize(i));
			

			//Get memory location of data.
			Layer* layer = net->getLayer(i);
			m_weight_data.push_back(layer->weightData());
			m_bias_data.push_back(layer->biasData());
		}

		Generator generator;
		for (size_t i = 0; i < m_layer_size.size(); i++) {
			const int N = m_layer_size[i];
			VectorType v;
			v.resize(N);
			m_moment_bias.push_back(v);
			m_moment_weight.push_back(v);
			m_mutation_resistance.push_back(v);

			for (int j = 0; j < m_layer_size[i]; j++) {
				m_moment_bias.back()(j) = generator.generate_normal<ScalarType>(0.0, 0.01);
				m_moment_weight.back()(j) = generator.generate_normal<ScalarType>(0.0, 0.01);
				m_mutation_resistance.back()(j) = generator.generate_normal<ScalarType>(2.0, 1.0);
			}

		}

	}



	virtual ~NeuralNetGenome() = default;

	NeuralNetGenome& operator=(const NeuralNetGenome& copy_this) {
		//std::cout << "Neural genome copy" << std::endl;
		for (int i = 0; i < m_weight_data.size(); i++) {
			std::copy(copy_this.m_weight_data[i], copy_this.m_weight_data[i] + copy_this.m_layer_size[i], m_weight_data[i]);
			std::copy(copy_this.m_bias_data[i],   copy_this.m_bias_data[i] +   copy_this.m_layer_size[i], m_bias_data[i]);
			// m_layer_size; // identical

			m_moment_weight[i] = copy_this.m_moment_weight[i];
			m_moment_bias[i] = copy_this.m_moment_bias[i];
			m_mutation_resistance[i] = copy_this.m_mutation_resistance[i];
		}
		return *this;
	}

	virtual void mutate(ScalarType mutation_probability) {
		checkMutationRate(mutation_probability);

		Generator generator;
		for (int i = 1; i < m_layer_size.size(); i++) {
			const auto size = m_layer_size[i];
			auto weightPointer = m_weight_data[i];
			auto biasPointer = m_bias_data[i];

			auto weightMomentum = m_moment_weight[i].data();
			auto biasMomentum = m_moment_bias[i].data();
			auto resistance = m_mutation_resistance[i].data();

			/*std::cout << m_mutation_resistance[i].mean() << " ";
			std::cout << m_moment_bias[i].mean() << " ";
			std::cout << m_moment_weight[i].mean() << std::endl;*/

			//Get the binomially distributed variable ~B(n_trails, p_probability)
			int number_of_mutations_weights = generator.generate_binomial(size, (double)mutation_probability);
			int number_of_mutations_bias = generator.generate_binomial(size, (double)mutation_probability);

			creepMutation(number_of_mutations_weights, generator, size, weightPointer, weightMomentum, resistance);
			creepMutation(number_of_mutations_weights, generator, size, biasPointer, biasMomentum, resistance);
		}

	}

	void creepMutation(int number_of_mutations, Generator &generator, const  int size, ScalarType * data, ScalarType * momentum, ScalarType * resistance)
	{
		//Generate mutation sites. Assumed low mutation prob for now.
		//Problem is worst case scenario when same index is rolled.
		//Roll from shrinking list for high p_mut
		std::unordered_set<int> indexes(number_of_mutations);
		while (indexes.size() < number_of_mutations) {
			int index = generator.generate_uniform_int(0, size - 1);
			indexes.insert(index);
		}

		//Add regulator gene!
		for (const auto & index : indexes) {
			resistance[index] += generator.generate_normal<ScalarType>(0.0f, 0.1f);
			resistance[index] = std::max<ScalarType>(0.3f, resistance[index]);
			resistance[index] = std::min<ScalarType>(10.0f, resistance[index]);

			momentum[index] += generator.generate_normal<ScalarType>(0.0f, resistance[index]*0.01f);
			momentum[index] = std::max<ScalarType>(-0.005f, momentum[index]);
			momentum[index] = std::min<ScalarType>(0.005f, momentum[index]);

			data[index] += generator.generate_normal<ScalarType>(momentum[index], resistance[index]*0.03f);
		}
	}

protected:
	void reserve(size_t number_of_layers)
	{
		m_bias_data.reserve(number_of_layers);
		m_weight_data.reserve(number_of_layers);
		m_layer_size.reserve(number_of_layers);

		m_moment_bias.reserve(number_of_layers);
		m_moment_weight.reserve(number_of_layers);
		m_mutation_resistance.reserve(number_of_layers);
	}


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
	std::vector<ScalarType*> m_bias_data; //dangling pointer danger

	std::vector<VectorType> m_moment_weight;
	std::vector<VectorType> m_moment_bias;
	std::vector<VectorType> m_mutation_resistance;

	std::vector<int>	     m_layer_size; //dangling pointer danger
};